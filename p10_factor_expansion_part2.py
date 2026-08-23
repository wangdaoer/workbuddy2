"""P10 part2：复用已存的原始集权益曲线，只补跑扩张集关键 AUM。

- 原始集 4 AUM 已在首次运行中落盘（equity_original_aum_*.csv），这里直接重算指标，省 ~80min。
- 扩张集只跑关键两档：1e8（实证容量 headline）、1e9（冲击临界点）。
- 每完成一步即重写 metrics.json，防止长任务超时丢失进度。
- 计算新因子全样本均值 IC，给出增量贡献证据。
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
OUT = HERE / "outputs" / "p10_factor_expansion"
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
AUMS = [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]
EXP_AUMS = [100_000_000.0, 1_000_000_000.0]   # 只补跑关键两档


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def decompose_cost(eq: pd.DataFrame) -> dict:
    turnover = float(eq["turnover"].sum())
    total_cost = float(eq["cost"].sum())
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {"impact_cost": impact,
            "impact_share": impact / total_cost if total_cost > 0 else 0.0}


def recompute_original() -> list[dict]:
    rows = []
    for aum in AUMS:
        p = OUT / f"equity_original_aum_{int(aum)}.csv"
        if not p.exists():
            print(f"[orig] 缺 {p.name}，跳过"); continue
        eq = pd.read_csv(p)
        m = calculate_walk_forward_metrics(eq, aum)
        dec = decompose_cost(eq)
        rows.append({"aum": aum, "total_return": m.get("total_return"),
                     "annualized_return": m.get("annualized_return"),
                     "max_drawdown": m.get("max_drawdown"),
                     "sharpe_like": m.get("sharpe_like"),
                     "avg_turnover": m.get("avg_turnover"),
                     "avg_gross_exposure": m.get("avg_gross_exposure"),
                     "impact_cost_share": dec["impact_share"]})
        print(f"[orig] AUM={aum:,.0f} sharpe={m.get('sharpe_like'):.3f} "
              f"total={m.get('total_return'):.4f} (recomputed from saved CSV)")
    return rows


def main() -> None:
    t0 = time.time()
    payload = {"method": "P10 part2: orig recomputed from saved CSV + expanded key AUMs",
               "n_factors_original": None, "n_factors_expanded": None}

    # 1) 原始集：从存盘 CSV 重算
    payload["original"] = recompute_original()
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[save] metrics.json written with original (recomputed)")

    # 2) 面板 + 因子 + label + ic
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= THR).fillna(False)
    print(f"面板 {close.shape[0]}日×{close.shape[1]}标的; 流动性达标均值 {liquid_mask.sum(axis=1).mean():.0f}")

    feats_orig = build_features(close, open_px, high, low, amount)
    feats_exp = build_features_expanded(close, open_px, high, low, amount)
    payload["n_factors_original"] = len(feats_orig)
    payload["n_factors_expanded"] = len(feats_exp)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    ic_orig = daily_ic(feats_orig, label)
    ic_exp = daily_ic(feats_exp, label)
    me = load_market_exposure(None, close.index, ma_window=120,
                               risk_off_drawdown_20d=-0.08, below_ma_exposure=0.60, crash_exposure=0.0)

    new_names = [f for f in feats_exp if f not in feats_orig]
    new_ic = ic_exp.mean()[new_names].sort_values(ascending=False)
    payload["new_factor_mean_ic"] = {k: round(float(v), 4) for k, v in new_ic.items()}
    payload["new_factors_positive_ic"] = int((new_ic > 0).sum())
    print(f"\n=== 新因子均值 IC（{payload['new_factors_positive_ic']}/{len(new_names)} 为正）===")
    for k, v in new_ic.items():
        print(f"  {k:28s} {v:+.4f} {'✓' if v > 0 else '✗'}")
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[save] metrics.json updated with new-factor IC")

    # 3) 扩张集：只跑关键 AUM
    payload.setdefault("expanded", [])
    for aum in EXP_AUMS:
        print(f"\n##### EXPANDED AUM={aum:,.0f} #####")
        eq, _, _ = run_walk_forward(
            close=close, open_px=open_px, features=feats_exp, label=label, ic=ic_exp,
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
            rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
            commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS, impact_model="sqrt",
            impact_ref_participation=IMPACT_REF, max_buy_open_gap=0.06, limit_buffer=0.995,
            market_exposure=me, initial_capital=aum, max_training_horizon=MTH,
            feature_directions=None, amount=amount, feature_selection=select_positive,
            max_daily_amount_participation=None, liquid_mask=liquid_mask,
        )
        m = calculate_walk_forward_metrics(eq, aum)
        dec = decompose_cost(eq)
        eq.to_csv(OUT / f"equity_expanded_aum_{int(aum)}.csv", index=False, encoding="utf-8")
        payload["expanded"].append({"aum": aum, "total_return": m.get("total_return"),
                                    "annualized_return": m.get("annualized_return"),
                                    "max_drawdown": m.get("max_drawdown"),
                                    "sharpe_like": m.get("sharpe_like"),
                                    "avg_turnover": m.get("avg_turnover"),
                                    "avg_gross_exposure": m.get("avg_gross_exposure"),
                                    "impact_cost_share": dec["impact_share"]})
        print(f"  sharpe={m.get('sharpe_like'):.3f} total={m.get('total_return'):.4f} "
              f"dd={m.get('max_drawdown'):.3f} impact_share={dec['impact_share']:.3f}")
        (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[save] metrics.json updated with expanded", aum)

    payload["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== P10 part2 完成，用时 {payload['elapsed_sec']}s ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
