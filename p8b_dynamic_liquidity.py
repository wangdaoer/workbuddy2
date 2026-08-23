"""P8b：动态流动性筛选（逐 retrain 用 trailing median 重筛 universe）。

P8 用「建仓前固定 universe」验证流动性筛选抬升容量。本阶段做更贴近实盘的
**动态流动性筛选**：在 run_walk_forward 内，每个 retrain 点用截至当日的
滚动 252 日 median 日成交额 ≥ THR 重筛可交易 universe，因子选择/打分/选股
只在当期流动性标的内进行。退市/降流动性票被剔除，新晋流动性票被纳入。

实现：在全 universe 上建 features/ic，传入 liquid_mask（dates×symbols 布尔）；
run_walk_forward 在 retrain 块把 training_ic 限制到 live_cols，下游自然只在
live_cols 内打分选股。复用 P3 冠军 + P7 sqrt 冲击（无硬上限）。

扫 AUM ∈ {1e6,1e8,5e8,1e9}，与 P8-fixed 直接对比。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from execution_rules import next_open_return_label
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import (
    build_features,
    calculate_walk_forward_metrics,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    run_walk_forward,
)

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
P8 = HERE / "outputs" / "p8_liquid_universe" / "metrics.json"
OUT = HERE / "outputs" / "p8b_dynamic_liquidity"
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
SPLIT = 440
AUMS = [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def half_return(eq_path: Path, split: int = SPLIT) -> float | None:
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if split >= len(eq):
        return None
    return float(eq["equity"].iloc[-1] / eq["equity"].iloc[split - 1] - 1)


def decompose_cost(eq: pd.DataFrame) -> dict:
    turnover = eq["turnover"].sum()
    total_cost = eq["cost"].sum()
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {
        "total_cost": float(total_cost), "commission_cost": float(commission),
        "impact_cost": float(impact),
        "impact_share": float(impact / total_cost) if total_cost > 0 else 0.0,
    }


def main() -> None:
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    # 动态流动性 mask：滚动 252 日 trailing median 日成交额 ≥ THR（无未来泄漏，仅用历史）
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= THR).fillna(False)
    print(f"动态流动性 mask: THR={THR:.0e}, 平均每日达标标的数={liquid_mask.sum(axis=1).mean():.0f}")

    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    ic = daily_ic(features, label)
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    p8 = json.loads(Path(P8).read_text(encoding="utf-8"))
    p8_sweep = {r["aum"]: r for r in p8["aum_sweep"]}

    rows = []
    for aum in AUMS:
        print(f"\n=== P8b: dynamic liquidity, AUM={aum:,.0f}, sqrt-impact (no hard cap) ===")
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
        eq_path = OUT / f"equity_curve_aum_{int(aum)}.csv"
        eq.to_csv(eq_path, index=False, encoding="utf-8")
        dec = decompose_cost(eq)
        avg_gross = float(eq["gross_exposure"].mean())
        p8r = p8_sweep.get(aum)
        rows.append({
            "aum": aum,
            "total_return": m.get("total_return"),
            "annualized_return": m.get("annualized_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_like": m.get("sharpe_like"),
            "second_half_return": half_return(eq_path),
            "avg_gross_exposure": avg_gross,
            "impact_cost": dec["impact_cost"], "impact_cost_share": dec["impact_share"],
            "p8_fixed_return": (p8r["total_return"] if p8r else None),
            "return_vs_p8_fixed": (m.get("total_return", 0) - p8r["total_return"] if p8r else None),
        })
        print(f"  total={m.get('total_return'):.4f} sharpe={m.get('sharpe_like'):.3f} "
              f"dd={m.get('max_drawdown'):.3f} avg_gross={avg_gross:.3f} "
              f"impact_share={dec['impact_share']:.2f} vs_P8fixed={rows[-1]['return_vs_p8_fixed']}")

    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "P3 champion + sqrt impact + DYNAMIC liquidity screening (rolling-252d median daily amount >= 1e8 at each retrain)",
        "threshold": THR, "liq_lookback": LIQ_LOOKBACK,
        "avg_daily_liquid_symbols": float(liquid_mask.sum(axis=1).mean()),
        "aum_sweep": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P8b dynamic liquidity (vs P8 fixed universe) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
