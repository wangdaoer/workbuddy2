"""P10：因子库扩张实验（对比 P8b 冠军因子集 vs 扩张因子集）。

同数据（data_panel.csv）、同窗口、同参数下，分别用「原始 16 因子」与
「扩张因子集（16 + 13 基础 + 5 复合 = 34）」跑 walk-forward，对比 Sharpe /
总收益 / 冲击成本，并分析哪些新因子提供正 IC（自适应选择后的净贡献）。

为严格对照，original 与 expanded 在**同一进程、同一数据、同一窗口**内跑，
避免数据/窗口差异污染比较。另加载 p8b 既有 metrics.json 做交叉校验。
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
)
from factor_expansion import build_features_expanded

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
P8B = HERE / "outputs" / "p8b_dynamic_liquidity" / "metrics.json"
OUT = HERE / "outputs" / "p10_factor_expansion"
OUT.mkdir(parents=True, exist_ok=True)

# —— 与 p8b 完全一致的核心参数 ——
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


def load_panel():
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= THR).fillna(False)
    return close, open_px, high, low, amount, liquid_mask


def main() -> None:
    t0 = time.time()
    close, open_px, high, low, amount, liquid_mask = load_panel()
    print(f"面板: {close.shape[0]} 日 × {close.shape[1]} 标的; "
          f"平均每日流动性达标 {liquid_mask.sum(axis=1).mean():.0f}")

    # 两套因子
    from train_next_open_rank_model import build_features
    feats_orig = build_features(close, open_px, high, low, amount)
    feats_exp = build_features_expanded(close, open_px, high, low, amount)
    print(f"原始因子数={len(feats_orig)}  扩张因子数={len(feats_exp)}")

    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    ic_orig = daily_ic(feats_orig, label)
    ic_exp = daily_ic(feats_exp, label)
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    # 因子 IC 分析（全样本均值 IC）
    mean_ic_orig = ic_orig.mean()
    mean_ic_exp = ic_exp.mean()
    new_factor_names = [f for f in feats_exp if f not in feats_orig]
    new_ic = mean_ic_exp[new_factor_names].sort_values(ascending=False)
    print("\n=== 新因子均值 IC（全样本）===")
    for f, v in new_ic.items():
        print(f"  {f:28s} IC={v:+.4f}  {'✓' if v > 0 else '✗'}")

    results = {"original": [], "expanded": []}
    for tag, feats, ic in [("original", feats_orig, ic_orig), ("expanded", feats_exp, ic_exp)]:
        for aum in AUMS:
            print(f"\n##### {tag} AUM={aum:,.0f} #####")
            eq, _, _ = run_walk_forward(
                close=close, open_px=open_px, features=feats, label=label, ic=ic,
                train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
                rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
                commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
                impact_model="sqrt", impact_ref_participation=IMPACT_REF,
                max_buy_open_gap=0.06, limit_buffer=0.995,
                market_exposure=market_exposure, initial_capital=aum,
                max_training_horizon=MTH, feature_directions=None, amount=amount,
                feature_selection=select_positive, max_daily_amount_participation=None,
                liquid_mask=liquid_mask,
            )
            m = calculate_walk_forward_metrics(eq, aum)
            dec = decompose_cost(eq)
            eq.to_csv(OUT / f"equity_{tag}_aum_{int(aum)}.csv", index=False, encoding="utf-8")
            results[tag].append({
                "aum": aum,
                "total_return": m.get("total_return"),
                "annualized_return": m.get("annualized_return"),
                "max_drawdown": m.get("max_drawdown"),
                "sharpe_like": m.get("sharpe_like"),
                "avg_turnover": m.get("avg_turnover"),
                "avg_gross_exposure": m.get("avg_gross_exposure"),
                "impact_cost_share": dec["impact_share"],
            })
            print(f"  total={m.get('total_return'):.4f} sharpe={m.get('sharpe_like'):.3f} "
                  f"dd={m.get('max_drawdown'):.3f} impact_share={dec['impact_share']:.3f}")

    # 交叉校验：original 1e8 应≈ p8b 既有
    p8b = json.loads(Path(P8B).read_text(encoding="utf-8"))
    p8b_1e8 = {r["aum"]: r for r in p8b["aum_sweep"]}.get(1e8)
    orig_1e8 = {r["aum"]: r for r in results["original"]}.get(1e8)
    if p8b_1e8 and orig_1e8:
        print(f"\n[校验] p8b Sharpe(1e8)={p8b_1e8['sharpe_like']:.3f} "
              f"vs 本脚本 original(1e8)={orig_1e8['sharpe_like']:.3f} "
              f"Δ={orig_1e8['sharpe_like']-p8b_1e8['sharpe_like']:+.3f}")

    payload = {
        "method": "P10 factor expansion vs P8b champion, same data/window/params",
        "n_factors_original": len(feats_orig),
        "n_factors_expanded": len(feats_exp),
        "new_factor_mean_ic": {k: round(float(v), 4) for k, v in new_ic.items()},
        "new_factors_positive_ic": int((new_ic > 0).sum()),
        "cross_check_p8b_orig_1e8": {
            "p8b_sharpe": p8b_1e8["sharpe_like"] if p8b_1e8 else None,
            "orig_sharpe": orig_1e8["sharpe_like"] if orig_1e8 else None,
        },
        "original": results["original"],
        "expanded": results["expanded"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== P10 完成，用时 {payload['elapsed_sec']}s ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
