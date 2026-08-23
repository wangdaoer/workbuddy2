"""P10b：纪律化选择 —— 用「IC 稳定性门槛 + 去冗余聚类」替换朴素 IC>0。

P10 证明：朴素 IC>0 选择对因子增殖无防御，冗余副本（reversal_20↔reversal_5 相关1.00 等）
被一并收编，稀释稳健信号、抬高换手，模型塌缩。本实验测试：若加入选择纪律，扩张集能否
(a) 救回（≥原始16）甚至 (b) 超越原始16。

两组对照（均在 34 因子扩张集上）：
  V1 stability      : 全34因子 + 纪律化选择（IC>0 占比≥min_frac 且 t-stat≥min_t）
  V2 stability+dedup: 先按 IC 时间序列相关性聚类去冗余（每簇留最高IC代表），再纪律化选择

基线（同进程/同数据）：
  BASE original16   : 从存盘 CSV 重算（P10 已落盘）
  BASE naive34      : P10 结果（0.175 / -0.172），直接引入对比

每完成一步重写 metrics.json，防长任务超时丢进度。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import next_open_return_label
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import (
    calculate_walk_forward_metrics,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    run_walk_forward,
    build_features,
)
from factor_expansion import build_features_expanded

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
P10 = HERE / "outputs" / "p10_factor_expansion"
OUT = HERE / "outputs" / "p10b_disciplined_selection"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
MTH = 1
LIQ_LOOKBACK = 252
THR = 1e8
RETRAIN = 20
TOP_N = 20
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
AUMS = [100_000_000.0, 1_000_000_000.0]   # 关键两档
MIN_FRAC = 0.60          # IC>0 占比门槛
MIN_T = 1.50             # IC t-stat 门槛
CORR_THR = 0.70          # 去冗余聚类阈值


def select_disciplined(training_ic: pd.DataFrame) -> list[str]:
    """纪律化选择：均值 IC>0 且 IC>0 占比≥MIN_FRAC 且 t-stat≥MIN_T。"""
    m = training_ic.mean()
    frac_pos = (training_ic > 0).mean()
    n = len(training_ic)
    sd = training_ic.std()
    sd = sd.replace(0, np.nan)
    t = m / (sd / np.sqrt(n))
    out = []
    for f in training_ic.columns:
        if m[f] > 0 and frac_pos.get(f, 0) >= MIN_FRAC and (t.get(f, 0) or 0) >= MIN_T:
            out.append(f)
    return out


def cluster_factors(ic: pd.DataFrame, corr_thr: float = CORR_THR):
    """按 IC 时间序列相关性聚类去冗余；每簇留均值 IC 最高的因子作代表。"""
    fcorr = ic.corr().abs().fillna(0)
    mean_ic = ic.mean().sort_values(ascending=False)
    assigned = set()
    clusters = []  # [representative, [members]]
    for f in mean_ic.index:
        if f in assigned:
            continue
        placed = False
        for cl in clusters:
            if fcorr.loc[f, cl[0]] >= corr_thr:
                cl[1].append(f); assigned.add(f); placed = True; break
        if not placed:
            clusters.append([f, [f]]); assigned.add(f)
    kept = [cl[0] for cl in clusters]
    return kept, clusters


def decompose_cost(eq: pd.DataFrame) -> dict:
    turnover = float(eq["turnover"].sum())
    total_cost = float(eq["cost"].sum())
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {"impact_cost": impact,
            "impact_share": impact / total_cost if total_cost > 0 else 0.0}


def main() -> None:
    t0 = time.time()
    payload = {
        "method": "P10b disciplined selection (IC stability + de-redundancy) vs naive expansion",
        "params": {"MIN_FRAC": MIN_FRAC, "MIN_T": MIN_T, "CORR_THR": CORR_THR},
    }

    # 基线：原始16（重算存盘CSV）
    base_rows = []
    for aum in [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]:
        p = P10 / f"equity_original_aum_{int(aum)}.csv"
        if p.exists():
            eq = pd.read_csv(p)
            m = calculate_walk_forward_metrics(eq, aum)
            base_rows.append({"aum": aum, "sharpe_like": m.get("sharpe_like"),
                              "total_return": m.get("total_return"),
                              "annualized_return": m.get("annualized_return"),
                              "max_drawdown": m.get("max_drawdown"),
                              "avg_turnover": m.get("avg_turnover")})
    payload["base_original16"] = base_rows
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[base] original16 recomputed:", {r['aum']: round(r['sharpe_like'],3) for r in base_rows})

    # 面板 + 因子
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= THR).fillna(False)
    feats_exp = build_features_expanded(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    ic_exp = daily_ic(feats_exp, label)
    me = load_market_exposure(None, close.index, ma_window=120,
                               risk_off_drawdown_20d=-0.08, below_ma_exposure=0.60, crash_exposure=0.0)

    # 去冗余聚类
    kept, clusters = cluster_factors(ic_exp, CORR_THR)
    payload["cluster"] = {
        "n_total": len(feats_exp), "n_clusters": len(clusters),
        "kept_representatives": kept,
        "clusters": {c[0]: c[1] for c in clusters},
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cluster] {len(feats_exp)}因子 → {len(clusters)}簇, 保留 {len(kept)} 代表")
    for c in clusters:
        print(f"   簇代表 {c[0]:22s} <- {c[1]}")

    feats_reduced = {k: feats_exp[k] for k in kept}

    def sweep(tag, feats, ic, selection_name):
        rows = []
        for aum in AUMS:
            print(f"\n##### {tag} AUM={aum:,.0f} #####")
            eq, _, _ = run_walk_forward(
                close=close, open_px=open_px, features=feats, label=label, ic=ic,
                train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
                rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
                commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS, impact_model="sqrt",
                impact_ref_participation=IMPACT_REF, max_buy_open_gap=0.06, limit_buffer=0.995,
                market_exposure=me, initial_capital=aum, max_training_horizon=MTH,
                feature_directions=None, amount=amount,
                feature_selection=select_disciplined,
                max_daily_amount_participation=None, liquid_mask=liquid_mask,
            )
            m = calculate_walk_forward_metrics(eq, aum)
            dec = decompose_cost(eq)
            eq.to_csv(OUT / f"equity_{tag}_aum_{int(aum)}.csv", index=False, encoding="utf-8")
            rows.append({"aum": aum, "sharpe_like": m.get("sharpe_like"),
                         "total_return": m.get("total_return"),
                         "annualized_return": m.get("annualized_return"),
                         "max_drawdown": m.get("max_drawdown"),
                         "avg_turnover": m.get("avg_turnover"),
                         "impact_cost_share": dec["impact_share"]})
            print(f"  sharpe={m.get('sharpe_like'):.3f} total={m.get('total_return'):.4f} "
                  f"dd={m.get('max_drawdown'):.3f} turnover={m.get('avg_turnover'):.3f}")
            payload.setdefault(tag, []).append(rows[-1])
            (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[save] metrics.json updated: {tag} aum={aum}")
        return rows

    sweep("V1_stability", feats_exp, ic_exp, "select_disciplined")
    sweep("V2_stability_dedup", feats_reduced, daily_ic(feats_reduced, label), "select_disciplined")

    payload["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== P10b 完成，用时 {payload['elapsed_sec']}s ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
