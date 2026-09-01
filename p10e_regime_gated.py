"""P10e：regime-gated MLP（用「近期 alpha 质量」闸门防御 W4 衰减）。

P10d 结论：MLP 优势真实（持续至 W3/2025），但高 beta——盈利期领先、最新逆风期(W4/2026H1)
放大亏损；且全家桶在 W4 失效。探查发现 run_walk_forward 自带的 market_exposure 全程恒=1.0
（从未触发），朴素 breadth/trend 信号误杀盈利期。故 P10e 改用**模型自身近期 alpha 质量**
作 regime 信号：

    trailing_mlp_ic = MLP 分数 rank-IC 的 60 日滚动均值（因果，仅用历史）
    alpha_dead = trailing_mlp_ic < 0   （近期 alpha 转负 → 激进模型在送钱）

门控策略（均喂入 run_walk_forward，liquid_mask=None，与 P10c 同执行框架）：
  - P10e_switch : alpha_dead → 用防御性 linear 分；否则用 mlp 分（满仓）
  - P10e_defense: 同上切换，且 alpha_dead 时把 gross 暴露降至 EXPOSURE_FLOOR（降仓）
  - P10e_soft   : 连续混合 adv = clip((THR_HI - trailing)/(THR_HI-THR_LO),0,1)，
                  score = adv·linear + (1-adv)·mlp（满仓）

对比基线（来自 P10c metrics.json）：incumbent / mlp_only / ensemble_ew。
并在 4 AUM 上跑，做 W1–W4 OOS 分段，验证 W4 衰减是否被消除。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import next_open_return_label
from run_backtest import load_prices, pivot_prices, sharpe_like
from train_next_open_rank_model import (
    MIN_LIVE_SYMBOLS, build_features, calculate_walk_forward_metrics, clean_matrix,
    daily_ic, load_market_exposure,
)
from p10c_ensemble import (
    select_positive, build_linear_mlp_scores, MLP, MIN_LIVE_SYMBOLS as _M,
    TRAIN_DAYS, MTH, LIQ_LOOKBACK, THR as LIQ_THR, RETRAIN,
)

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
P10C_METRICS = HERE / "outputs" / "p10c_ensemble" / "metrics.json"
OUT = HERE / "outputs" / "p10e_regime_gated"
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"

MAX_ABS = 0.22
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
TOP_N = 20
REBALANCE = 1
AUMS = [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]
SPLIT = 440

# regime 参数
IC_WIN = 60
IC_MIN = 40
THR_DEAD = 0.0          # trailing IC <= 0 → alpha 死
THR_HI = 0.03           # 健康阈值（soft 混合上界）
THR_LO = 0.0            # 死亡阈值（soft 混合下界）
EXPOSURE_FLOOR = 0.4    # defense 降仓地板


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def half_return(eq_path: Path) -> float | None:
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return float(eq["equity"].iloc[-1] / eq["equity"].iloc[SPLIT - 1] - 1) if SPLIT < len(eq) else None


def decompose_cost(eq: pd.DataFrame) -> dict:
    turnover = eq["turnover"].sum()
    total_cost = eq["cost"].sum()
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {"impact_cost": float(impact), "impact_share": float(impact / total_cost) if total_cost > 0 else 0.0}


def _variant(name, m, dec, eq_path):
    return {
        "variant": name, "total_return": m.get("total_return"),
        "annualized_return": m.get("annualized_return"), "max_drawdown": m.get("max_drawdown"),
        "sharpe_like": m.get("sharpe_like"), "second_half_return": half_return(eq_path),
        "avg_gross_exposure": m.get("avg_gross_exposure"), "avg_turnover": m.get("avg_turnover"),
        "avg_positions_count": m.get("avg_positions_count"), "final_equity": m.get("final_equity"),
        "impact_cost": dec["impact_cost"], "impact_share": dec["impact_share"],
    }


def seg_stats(net: pd.Series):
    return {"total_return": float(np.prod(1 + net) - 1), "sharpe": float(sharpe_like(net)) if len(net) > 5 else float("nan")}


def main() -> None:
    RUN_LOG.write_text("", encoding="utf-8")
    log("=== P10e 启动 ===")
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    symbols = list(close.columns)
    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    market_exposure = load_market_exposure(None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08, below_ma_exposure=0.60, crash_exposure=0.0)

    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= LIQ_THR).fillna(False)
    liquid_mask_aligned = liquid_mask.reindex(index=close.index, columns=close.columns).fillna(False)
    feat_arrays = {k: fr.values for k, fr in features.items()}
    label_arr = label.values

    # 复用 P10c 精确打分管线
    log("构建 linear + mlp 分（精确复刻 P8b 语义）...")
    linear_score, mlp_score = build_linear_mlp_scores(
        features, label, symbols, feat_arrays, label_arr, liquid_mask_aligned
    )

    # regime 信号：MLP 分数滚动 60 日 rank-IC（因果）
    mlp_ic = daily_ic({"mlp": mlp_score}, label)["mlp"]
    trailing = mlp_ic.rolling(IC_WIN, min_periods=IC_MIN).mean()
    alpha_dead = (trailing < THR_DEAD).fillna(False)
    adv = ((THR_HI - trailing) / (THR_HI - THR_LO)).clip(0, 1).fillna(1.0)  # 1=全线性(防御), 0=全mlp

    # regime 分段统计（验证识别 W4）
    n = len(close.index)
    edges = [0, n // 4, n // 2, 3 * n // 4, n]
    log("regime 信号（alpha_dead 占比）按 W1–W4：")
    for k in range(4):
        seg = alpha_dead.iloc[edges[k]:edges[k + 1]]
        log(f"  W{k+1}: dead% = {seg.mean():.2f}  (trailing IC 均值={trailing.iloc[edges[k]:edges[k+1]].mean():.4f})")
    log(f"  全程 dead% = {alpha_dead.mean():.2f}")

    # 构建门控分数
    switch_score = mlp_score.copy()
    switch_score[alpha_dead] = linear_score[alpha_dead]   # dead 日用线性（防御）
    soft_score = adv.values[:, None] * linear_score.values + (1 - adv.values[:, None]) * mlp_score.values
    soft_score = pd.DataFrame(soft_score, index=label.index, columns=symbols)
    custom_exposure = market_exposure.where(~alpha_dead, EXPOSURE_FLOOR)  # defense 降仓

    # 载入 P10c 基线
    p10c = json.loads(P10C_METRICS.read_text(encoding="utf-8")) if P10C_METRICS.exists() else {}
    p10c_sweep = {r["aum"]: {v["variant"]: v for v in r["variants"]} for r in p10c.get("aum_sweep", [])}

    from train_next_open_rank_model import run_walk_forward
    report = {
        "method": "regime-gated MLP via trailing-60d rank-IC of mlp score; switch to defensive linear + optional exposure cut in dead-alpha regime",
        "regime_params": {"ic_window": IC_WIN, "thr_dead": THR_DEAD, "thr_hi": THR_HI, "thr_lo": THR_LO, "exposure_floor": EXPOSURE_FLOOR},
        "regime_dead_pct": {"W1": float(alpha_dead.iloc[edges[0]:edges[1]].mean()), "W2": float(alpha_dead.iloc[edges[1]:edges[2]].mean()), "W3": float(alpha_dead.iloc[edges[2]:edges[3]].mean()), "W4": float(alpha_dead.iloc[edges[3]:edges[4]].mean()), "overall": float(alpha_dead.mean())},
        "aum_sweep": [],
    }

    for aum in AUMS:
        log(f"\n=== AUM={aum:,.0f} ===")
        row = {"aum": aum, "variants": [], "by_window": {}}
        specs = [
            ("p10e_switch", switch_score, market_exposure),
            ("p10e_defense", switch_score, custom_exposure),
            ("p10e_soft", soft_score, market_exposure),
        ]
        eqs = {}
        for key, score_df, expo in specs:
            ic_run = daily_ic({key: score_df}, label)
            eq, _, _ = run_walk_forward(
                close=close, open_px=open_px, features={key: score_df}, label=label, ic=ic_run,
                train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
                rebalance_frequency=REBALANCE, max_position_weight=0.04, leverage=1.0,
                commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
                impact_model="sqrt", impact_ref_participation=IMPACT_REF,
                max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=expo,
                initial_capital=aum, max_training_horizon=MTH, feature_directions=None,
                amount=amount, feature_selection=None, max_daily_amount_participation=None,
                liquid_mask=None,
            )
            m = calculate_walk_forward_metrics(eq, aum)
            eq.to_csv(OUT / f"equity_aum_{int(aum)}_{key}.csv", index=False, encoding="utf-8")
            dec = decompose_cost(eq)
            eqs[key] = eq
            row["variants"].append(_variant(key, m, dec, OUT / f"equity_aum_{int(aum)}_{key}.csv"))
            log(f"  {key:14s} total={m.get('total_return'):+.4f} sharpe={m.get('sharpe_like'):.3f} "
                f"dd={m.get('max_drawdown'):.3f} impact_share={dec['impact_share']:.2f}")
            # OOS 分段
            net = (eq["gross_return"] - eq["cost"]).reset_index(drop=True)
            w = {f"W{k+1}": seg_stats(net.iloc[edges[k]:edges[k + 1]]) for k in range(4)}
            row["by_window"][key] = w
        # 基线对照
        base = p10c_sweep.get(aum, {})
        for bk in ["incumbent", "mlp_only", "ensemble_ew"]:
            if bk in base:
                b = base[bk]
                log(f"  [基线] {bk:12s} total={b['total_return']:+.4f} sharpe={b['sharpe_like']:.3f}")
        report["aum_sweep"].append(row)
        (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总 OOS（W1–W4）对比
    log("\n=== OOS 分段 Sharpe（W1→W4）===")
    for aum in AUMS:
        log(f"--- AUM={aum:,.0f} ---")
        r = next(x for x in report["aum_sweep"] if x["aum"] == aum)
        keys = ["incumbent", "mlp_only", "ensemble_ew", "p10e_switch", "p10e_defense", "p10e_soft"]
        hdr = "variant".ljust(14) + "".join(f"{f'W{k+1}':>12}" for k in range(4))
        log(hdr)
        for kk in keys:
            if kk in r["by_window"]:
                ws = r["by_window"][kk]
                log("  " + kk.ljust(14) + "".join(f"{ws[f'W{k+1}']['sharpe']:>12.3f}" for k in range(4)))
            elif kk in p10c_sweep.get(aum, {}):
                # baseline 用其权益曲线重算分段
                eq = pd.read_csv(OUT.parent / "p10c_ensemble" / f"equity_aum_{int(aum)}_{kk}.csv", parse_dates=["date"])
                net = (eq["gross_return"] - eq["cost"]).reset_index(drop=True)
                ws = {f"W{k+1}": seg_stats(net.iloc[edges[k]:edges[k + 1]]) for k in range(4)}
                log("  " + kk.ljust(14) + "".join(f"{ws[f'W{k+1}']['sharpe']:>12.3f}" for k in range(4)))

    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=== P10e 完成 ===")


if __name__ == "__main__":
    main()
