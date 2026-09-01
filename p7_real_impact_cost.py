"""P7：真实冲击成本模型（square-root impact，Almgren/Kyle 风格）。

P6-scale 用「硬成交量上限 + flat 冲击」暴露了 under-deployment（大 AUM 建不起仓），
但 flat 冲击未对「参与越高成本越大」定价。真实市场里你可以成交任意量，
只是参与度高时冲击成本非线性放大。

本脚本在 P3 冠军配置上：
- 关闭硬容量上限（max_daily_amount_participation=None），让策略真实部署资本；
- 开启 impact_model="sqrt"：impact_s = impact_bps × sqrt(participation_s / ref)，
  participation_s = |Δ权重_s| × equity / 当日成交额_s；
- 在 impact_bps=0.7（ref=1% 参与）基准下扫 AUM ∈ {1e6, 1e7, 1e8, 5e8}；
- 从权益曲线的 cost/turnover 列分解 冲击成本 vs 佣金成本；
- 对比 P6-scale（硬上限 + flat 冲击）：真实成本下大 AUM 的衰减是否更陡/更缓。

复用预计算 features/ic/label/market_exposure（与 P3/P6 逐位一致）。
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
P6S = HERE / "outputs" / "p6b_capacity_at_scale" / "metrics.json"
OUT = HERE / "outputs" / "p7_real_impact_cost"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
SPLIT = 440
AUMS = [1_000_000.0, 10_000_000.0, 100_000_000.0, 500_000_000.0]


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def half_return(eq_path: Path, split: int = SPLIT) -> float | None:
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if split >= len(eq):
        return None
    return float(eq["equity"].iloc[-1] / eq["equity"].iloc[split - 1] - 1)


def decompose_cost(eq: pd.DataFrame) -> dict:
    """从权益曲线分解 总/佣金/冲击 成本（累计，组合单位）。"""
    turnover = eq["turnover"].sum()
    total_cost = eq["cost"].sum()
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {
        "total_cost": float(total_cost),
        "commission_cost": float(commission),
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
    symbols = list(close.columns)

    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    ic = daily_ic(features, label)
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    p6s = json.loads(Path(P6S).read_text(encoding="utf-8"))
    p6s_baseline = p6s["baseline_1e6_P6"]
    p6s_sweep = {r["aum"]: r for r in p6s["aum_sweep"]}

    rows = []
    for aum in AUMS:
        print(f"\n=== P7: AUM={aum:,.0f}, sqrt-impact (no hard cap) ===")
        eq, _, _ = run_walk_forward(
            close=close, open_px=open_px, features=features, label=label, ic=ic,
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
            rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
            commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
            impact_model="sqrt", impact_ref_participation=IMPACT_REF,
            max_buy_open_gap=0.06, limit_buffer=0.995,
            market_exposure=market_exposure, initial_capital=aum,
            max_training_horizon=1, feature_directions=None, amount=amount,
            feature_selection=select_positive,
            max_daily_amount_participation=None,
        )
        m = calculate_walk_forward_metrics(eq, aum)
        eq_path = OUT / f"equity_curve_aum_{int(aum)}.csv"
        eq.to_csv(eq_path, index=False, encoding="utf-8")
        dec = decompose_cost(eq)
        avg_gross = float(eq["gross_exposure"].mean())
        # 对比 P6-scale 同 AUM（硬上限+flat冲击）
        p6s_row = p6s_sweep.get(aum)
        rows.append({
            "aum": aum,
            "total_return": m.get("total_return"),
            "annualized_return": m.get("annualized_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_like": m.get("sharpe_like"),
            "final_equity": m.get("final_equity"),
            "second_half_return": half_return(eq_path),
            "avg_gross_exposure": avg_gross,
            "total_cost": dec["total_cost"],
            "commission_cost": dec["commission_cost"],
            "impact_cost": dec["impact_cost"],
            "impact_cost_share": dec["impact_share"],
            "p6s_total_return": (p6s_row["total_return"] if p6s_row else None),
            "return_vs_p6s": (m.get("total_return", 0) - p6s_row["total_return"] if p6s_row else None),
        })
        print(f"  total={m.get('total_return'):.4f} sharpe={m.get('sharpe_like'):.3f} "
              f"dd={m.get('max_drawdown'):.3f} avg_gross={avg_gross:.3f} "
              f"impact_cost={dec['impact_cost']:.2f} impact_share={dec['impact_share']:.2f}")

    baseline = {
        "aum": 1_000_000.0,
        "total_return": p6s_baseline["total_return"],
        "annualized_return": p6s_baseline["annualized_return"],
        "max_drawdown": p6s_baseline["max_drawdown"],
        "sharpe_like": p6s_baseline["sharpe_like"],
        "second_half_return": p6s_baseline["second_half_return"],
        "avg_gross_exposure": None,
        "total_cost": None, "commission_cost": None, "impact_cost": None, "impact_cost_share": None,
        "p6s_total_return": p6s_baseline["total_return"],
        "return_vs_p6s": 0.0,
    }
    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "P3 champion + square-root impact model (no hard capacity cap), AUM sweep",
        "impact_spec": {
            "model": "sqrt",
            "impact_bps_ref": IMPACT_BPS,
            "ref_participation": IMPACT_REF,
            "commission_bps": COMMISSION_BPS,
            "participation_s": "|delta_weight_s| * equity / daily_amount_s",
            "impact_s": "impact_bps * sqrt(participation_s / ref_participation)",
        },
        "baseline_1e6_P6scale": baseline,
        "aum_sweep": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P7 real impact cost (P3 champion, sqrt-impact, no hard cap) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
