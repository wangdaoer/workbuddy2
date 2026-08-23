"""P9-5：训练港股通横截面选股信号（复用 P8b walk-forward 机器）。

目标：把第 9 节里"流动性加权港股通指数 beta"（Sharpe 1.03）替换为
**训练出的港股通横截面选股 alpha**，并以其真实 walk-forward Sharpe / 容量
替换第 9 节的"流动性池容量估算 4,327 亿"。

方法学与 P8b（A 股冠军）完全一致，仅数据源换成港股通归一化面板：
  - 因子：build_features（动量/反转/突破/流动性/日内等 14 个横截面 rank 因子）
  - 标签：next_open_return_label（次日开盘收益）
  - 因子选择：成熟窗口 IC 为正的因子（自适应，消除固定子集前视）
  - 打分：IC 加权横截面排序，top_n=20
  - 成本：sqrt 参与度冲击（impact_bps=0.7, ref=1%）+ 佣金 1bp
  - 流动性：滚动 median 日成交额 ≥ THR 的动态 universe 筛选
  - 容量：不硬约束，AUM 扫���看冲击成本占比上升 → 反推容量上限

数据源：external_data/daily-market-data-tdx/hk_connect/normalized/*.csv
（646 个交易日 × 2486 标的，与 A 股同 schema）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import next_open_return_label
from run_backtest import pivot_prices, prepare_prices
from train_next_open_rank_model import (
    build_features,
    calculate_walk_forward_metrics,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    run_walk_forward,
)

HERE = Path(__file__).resolve().parent
NORM = HERE / "external_data" / "daily-market-data-tdx" / "hk_connect" / "normalized"
OUT = HERE / "outputs" / "p9_5_hk_connect_signal"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
MTH = 1
LIQ_LOOKBACK = 126          # 港股通历史短，用 126 日滚动（min_periods=60）保证测试窗充足
THR = 1e8                   # HKD/日，≈ A 股 1e8 CNY 阈值同量级 → 227 只流动性 universe
RETRAIN = 20
TOP_N = 20
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
AUMS = [1e8, 1e9, 1e10, 5e10, 1e11, 5e11]


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def decompose_cost(eq: pd.DataFrame) -> dict:
    turnover = float(eq["turnover"].sum())
    total_cost = float(eq["cost"].sum())
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {
        "total_cost": total_cost,
        "commission_cost": commission,
        "impact_cost": impact,
        "impact_share": impact / total_cost if total_cost > 0 else 0.0,
    }


def ic_summary(features: dict[str, pd.DataFrame], label: pd.DataFrame) -> list[dict]:
    ic = daily_ic(features, label)
    mean_ic = ic.mean().sort_values(ascending=False)
    return [
        {"factor": k, "mean_ic": round(float(mean_ic[k]), 4), "share_pos": round(float((ic[k] > 0).mean()), 3)}
        for k in mean_ic.index
    ]


def main() -> None:
    files = sorted(NORM.glob("ths_hk_connect_*.csv"))
    print(f"拼接港股通归一化面板：{len(files)} 个日文件 …")
    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw = prepare_prices(long, None, None, strict_validation=False)

    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    print(f"面板：{close.shape[0]} 交易日 × {close.shape[1]} 标的")

    # 动态流动性 mask（滚动 median 日成交额 ≥ THR，仅用历史，无未来泄漏）
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=60).median()
    liquid_mask = (roll_med >= THR).fillna(False)
    print(f"动态流动性 mask: THR={THR:.0e} HKD, 平均每日达标标的数={liquid_mask.sum(axis=1).mean():.0f}")

    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    ic = daily_ic(features, label)
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )
    top_factors = ic_summary(features, label)
    print("Top 因子（样本内 mean IC）：")
    for r in top_factors[:6]:
        print(f"  {r['factor']:32s} IC={r['mean_ic']:+.4f} pos={r['share_pos']:.2f}")

    rows = []
    for aum in AUMS:
        print(f"\n=== P9-5 港股通 walk-forward, AUM={aum:,.0f}, sqrt-impact ===")
        eq, _, _ = run_walk_forward(
            close=close, open_px=open_px, features=features, label=label, ic=ic,
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
        eq.to_csv(OUT / f"equity_curve_aum_{int(aum)}.csv", index=False, encoding="utf-8")
        dec = decompose_cost(eq)
        rows.append({
            "aum": aum,
            "total_return": m.get("total_return"),
            "annualized_return": m.get("annualized_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_like": m.get("sharpe_like"),
            "avg_turnover": m.get("avg_turnover"),
            "avg_gross_exposure": m.get("avg_gross_exposure"),
            "avg_positions_count": m.get("avg_positions_count"),
            "impact_cost": dec["impact_cost"],
            "impact_cost_share": dec["impact_share"],
        })
        print(f"  ann={m.get('annualized_return'):.4f} sharpe={m.get('sharpe_like'):.3f} "
              f"dd={m.get('max_drawdown'):.3f} turnover={m.get('avg_turnover'):.3f} "
              f"impact_share={dec['impact_share']:.2f}")

    # 容量上限：冲击成本占比越过 50% 的 AUM（与 P8b 同口径）
    cap_row = None
    for r in rows:
        if r["impact_cost_share"] >= 0.5:
            cap_row = r
            break
    capacity_ceiling = cap_row["aum"] if cap_row else None

    report = {
        "data_source": "ggtday.zip (real HK connect, normalized panel 646 days x 2486 symbols)",
        "method": "P8b champion walk-forward: IC-weighted cross-sectional rank, top_n=20, sqrt impact, dynamic liquidity screen (rolling-126d median daily amount >= 1e8 HKD)",
        "threshold_hkd": THR,
        "liq_lookback": LIQ_LOOKBACK,
        "avg_daily_liquid_symbols": float(liquid_mask.sum(axis=1).mean()),
        "top_factors_insample": top_factors,
        "aum_sweep": rows,
        "capacity_ceiling_aum": capacity_ceiling,
        "note": "capacity_ceiling = AUM where impact-cost share >= 50% (P8b 同口径); None = 未到 50% 即在扫描上限内",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P9-5 港股通横截面信号 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
