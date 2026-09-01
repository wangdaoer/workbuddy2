"""P8：流动性 universe（只交易高流动性标的）以抬升容量上限。

P7 结论：全 universe 下 P3 在 500M 因冲击成本转亏（−7.08%）。机制是策略会挑中
低流动性小票，其日成交额相对大 AUM 太小 → 参与度高 → sqrt 冲击爆炸。

本阶段验证「扩 universe 到流动性子集（大盘/高成交，作为跨市场扩展的 within-data 代理）」
能否抬升容量上限：
- 用建仓前一年 trailing median 日成交额 ≥ THR 定义流动性 universe（纯历史，无未来泄漏）；
- 在受限 universe 上跑 P3 冠军 + sqrt 冲击（无硬上限），扫 AUM ∈ {1e6,1e8,5e8,1e9}；
- 对比 P7 全 universe 同 AUM，看净收益是否在大 AUM 仍为正（容量上限抬升）。

THR=1e8（median 日成交 ≥1亿，约 2382 只，大盘/中大盘子集）。
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
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
P7 = HERE / "outputs" / "p7_real_impact_cost" / "metrics.json"
OUT = HERE / "outputs" / "p8_liquid_universe"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
MTH = 1
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

    # 流动性 universe：建仓前一年 trailing median 日成交额 ≥ THR（无未来泄漏）
    first_signal_idx = TRAIN_DAYS + MTH
    hist_amount = amount.iloc[:first_signal_idx]
    med_hist = hist_amount.median(axis=0)
    liquid = med_hist[med_hist >= THR].index.tolist()
    print(f"流动性 universe（THR={THR:.0e}）: {len(liquid)} 只 / 共 {close.shape[1]} 只")
    close_l = close[liquid]; open_l = open_px[liquid]; high_l = high[liquid]
    low_l = low[liquid]; amount_l = amount[liquid]

    features = build_features(close_l, open_l, high_l, low_l, amount_l)
    label = next_open_return_label(open_l, max_abs_daily_return=MAX_ABS)
    ic = daily_ic(features, label)
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    p7 = json.loads(Path(P7).read_text(encoding="utf-8"))
    p7_sweep = {r["aum"]: r for r in p7["aum_sweep"]}

    rows = []
    for aum in AUMS:
        print(f"\n=== P8: liquid universe, AUM={aum:,.0f}, sqrt-impact (no hard cap) ===")
        eq, _, _ = run_walk_forward(
            close=close_l, open_px=open_l, features=features, label=label, ic=ic,
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
            rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
            commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
            impact_model="sqrt", impact_ref_participation=IMPACT_REF,
            max_buy_open_gap=0.06, limit_buffer=0.995,
            market_exposure=market_exposure, initial_capital=aum,
            max_training_horizon=MTH, feature_directions=None, amount=amount_l,
            feature_selection=select_positive, max_daily_amount_participation=None,
        )
        m = calculate_walk_forward_metrics(eq, aum)
        eq_path = OUT / f"equity_curve_aum_{int(aum)}.csv"
        eq.to_csv(eq_path, index=False, encoding="utf-8")
        dec = decompose_cost(eq)
        avg_gross = float(eq["gross_exposure"].mean())
        p7r = p7_sweep.get(aum)
        rows.append({
            "aum": aum, "universe_size": len(liquid),
            "total_return": m.get("total_return"),
            "annualized_return": m.get("annualized_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_like": m.get("sharpe_like"),
            "second_half_return": half_return(eq_path),
            "avg_gross_exposure": avg_gross,
            "impact_cost": dec["impact_cost"], "impact_cost_share": dec["impact_share"],
            "p7_full_universe_return": (p7r["total_return"] if p7r else None),
            "return_vs_p7_full": (m.get("total_return", 0) - p7r["total_return"] if p7r else None),
        })
        print(f"  total={m.get('total_return'):.4f} sharpe={m.get('sharpe_like'):.3f} "
              f"dd={m.get('max_drawdown'):.3f} avg_gross={avg_gross:.3f} "
              f"impact_share={dec['impact_share']:.2f} vs_P7_full={rows[-1]['return_vs_p7_full']}")

    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "P3 champion + sqrt impact, restricted to liquid universe (trailing-median daily amount >= 1e8 at inception)",
        "threshold": THR, "universe_size": len(liquid), "total_universe": close.shape[1],
        "aum_sweep": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P8 liquid universe (vs P7 full universe) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
