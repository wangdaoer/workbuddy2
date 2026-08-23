"""P10e causal-verification: rule out the 1-period look-ahead leak in the regime signal.

The non-causal regime signal trailing[t] = mean(mlp_ic[t-59:t]) uses label[t], which is the
return from t-open to t+1-open and is NOT known at t-close (when the trade weight is decided).
That is a 1-period look-ahead. This script recomputes everything with a CAUSAL signal:
    trailing_c[t] = mean(mlp_ic[t-60:t-1])   (shift IC by 1 before rolling)
so at decision time t we only use IC info observed through t-1.

Also saves linear/mlp scores to npz so the (7-min) build runs once and gated experiments are cheap.
Runs both non-causal (nc) and causal (c) versions of switch/defense/soft at 4 AUM and prints a
corrected W1-W4 decomposition (trading-calendar aligned) so we can see if the W4 defense survives.
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
    daily_ic, load_market_exposure, run_walk_forward,
)
from p10c_ensemble import (
    select_positive, build_linear_mlp_scores, MLP, MIN_LIVE_SYMBOLS as _M,
    TRAIN_DAYS, MTH, LIQ_LOOKBACK, THR as LIQ_THR, RETRAIN,
)
from score_cache import load_or_build  # 内容寻址缓存守卫（根治换面板形状错）

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
OUT = HERE / "outputs" / "p10e_regime_gated"
SCORES_NPZ = OUT / "linear_mlp_scores.npz"
RUN_LOG = OUT / "run_causal.log"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
TOP_N = 20
REBALANCE = 1
AUMS = [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]
EDGES = [0, 220, 440, 660, 880]
IC_WIN = 60
IC_MIN = 40
THR_DEAD = 0.0
THR_HI = 0.03
THR_LO = 0.0
EXPOSURE_FLOOR = 0.4


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def seg_stats(net):
    net = np.asarray(net, float)
    if len(net) < 6 or net.std() == 0:
        return float("nan"), float(np.prod(1 + net) - 1)
    return float(net.mean() / net.std() * np.sqrt(252)), float(np.prod(1 + net) - 1)


def build_gated(linear_score, mlp_score, label, symbols, causal: bool):
    mlp_ic = daily_ic({"mlp": mlp_score}, label)["mlp"]
    if causal:
        # P0-3 残留修复 (2026-08-23): label[t]=open[t+2]/open[t+1]-1 在 t+2 开盘才成熟,
        # 因果 trailing 须用 shift(2) (as-of d 只用 IC<=d-2); 原 shift(1) 泄漏 1 天.
        trailing = mlp_ic.shift(2).rolling(IC_WIN, min_periods=IC_MIN).mean()
    else:
        trailing = mlp_ic.rolling(IC_WIN, min_periods=IC_MIN).mean()
    alpha_dead = (trailing < THR_DEAD).fillna(False)
    adv = ((THR_HI - trailing) / (THR_HI - THR_LO)).clip(0, 1).fillna(1.0)
    switch = mlp_score.copy()
    switch[alpha_dead] = linear_score[alpha_dead]
    soft = adv.values[:, None] * linear_score.values + (1 - adv.values[:, None]) * mlp_score.values
    soft = pd.DataFrame(soft, index=label.index, columns=symbols)
    custom_exposure = None  # set by caller with market_exposure
    return {"alpha_dead": alpha_dead, "trailing": trailing, "switch": switch,
            "soft": soft, "adv": adv}


def main():
    RUN_LOG.write_text("", encoding="utf-8")
    log("=== P10e causal verification 启动 ===")
    t0 = time.time()
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

    # 内容寻址缓存守卫：面板指纹变更自动失效重建（同 production_soft_score）。
    def _build_scores():
        log("构建 linear + mlp 分...")
        return build_linear_mlp_scores(features, label, symbols, feat_arrays, label_arr, liquid_mask_aligned)
    linear_score, mlp_score = load_or_build(
        SCORES_NPZ, label.index, symbols, _build_scores, panel_csv=PANEL, use_cache=True)
    log(f"分已存至 {SCORES_NPZ}")

    report = {"causal_compare": [], "dead_pct": {}}
    for causal in [False, True]:
        tag = "causal" if causal else "noncausal"
        log(f"\n=== regime 信号: {tag} ===")
        g = build_gated(linear_score, mlp_score, label, symbols, causal)
        # dead% on trading days only
        tr_start = TRAIN_DAYS + MTH
        dead_trade = g["alpha_dead"].iloc[tr_start:]
        edges_t = [0, len(dead_trade)//4, len(dead_trade)//2, 3*len(dead_trade)//4, len(dead_trade)]
        dead_w = {f"W{k+1}": float(dead_trade.iloc[edges_t[k]:edges_t[k+1]].mean()) for k in range(4)}
        log(f"  dead% (交易日) W1-W4 = {dead_w}")
        report["dead_pct"][tag] = dead_w

        custom_exposure = market_exposure.where(~g["alpha_dead"], EXPOSURE_FLOOR)
        specs = [
            (f"p10e_switch_{tag}", g["switch"], market_exposure),
            (f"p10e_defense_{tag}", g["switch"], custom_exposure),
            (f"p10e_soft_{tag}", g["soft"], market_exposure),
        ]
        for aum in AUMS:
            log(f"  AUM={aum:,.0f}")
            row = {"aum": aum, "causal": causal, "variants": [], "by_window": {}}
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
                row["variants"].append({"variant": key, "total_return": m.get("total_return"),
                                        "sharpe_like": m.get("sharpe_like"), "max_drawdown": m.get("max_drawdown")})
                log(f"    {key:18s} total={m.get('total_return'):+.4f} sharpe={m.get('sharpe_like'):.3f} dd={m.get('max_drawdown'):.3f}")
                net = (eq["gross_return"] - eq["cost"]).reset_index(drop=True)
                # corrected edges: reindex net onto full panel date axis
                s = pd.Series(0.0, index=close.index)
                s.loc[eq["date"].values] = net.values
                ws = {}
                for k in range(4):
                    seg = s.iloc[EDGES[k]:EDGES[k+1]].values
                    seg = seg[~np.isnan(seg)]
                    if k == 0 or len(seg) < 6 or np.allclose(seg, 0):
                        ws[f"W{k+1}"] = {"sharpe": float("nan"), "ret": float(np.prod(1+seg)-1)}
                    else:
                        sh, rt = seg_stats(seg)
                        ws[f"W{k+1}"] = {"sharpe": sh, "ret": rt}
                row["by_window"][key] = ws
            report["causal_compare"].append(row)
            (OUT / "metrics_causal.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary table
    log("\n=== W4 Sharpe: noncausal vs causal (the look-ahead test) ===")
    log("variant".ljust(20) + "".join(f"AUM={int(a):>11,}" for a in AUMS))
    for key in ["p10e_switch", "p10e_defense", "p10e_soft"]:
        for tag in ["noncausal", "causal"]:
            kk = f"{key}_{tag}"
            line = f"{kk:20s}"
            for aum in AUMS:
                r = next(x for x in report["causal_compare"] if x["aum"] == aum and x["causal"] == (tag == "causal"))
                w4 = r["by_window"][kk]["W4"]["sharpe"]
                line += f"{w4:>12.3f}"
            log(line)
    (OUT / "metrics_causal.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"=== 完成 ({time.time()-t0:.0f}s) ===")


if __name__ == "__main__":
    main()
