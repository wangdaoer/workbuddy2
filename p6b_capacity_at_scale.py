"""P6-scale：机构级 AUM 容量测试（在已验证的 1% 参与度下扫规模）。

P6 在 100 万权益下证明 P3 冠军不受容量约束。机构实盘是 1000 万~数亿规模，
此时 capacity_weight = trailing-median 日成交额 × 0.01 / equity 随 equity 放大而收紧，
单票日调仓权重上限骤降——可能「建不起仓 / 无法充分部署」。

本脚本在 participation=0.01（P6 验证的标准保守档）下扫 AUM ∈ {1e7, 1e8, 5e8}，
复用预计算 features/ic/label/market_exposure，记录：总收益 / 夏普 / 回撤 /
累计被拒买权重 / 命中 session 数 / 平均毛暴露（看是否 under-deployment）。
1e6 基线直接引用 P6 metrics.json。
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
P6 = HERE / "outputs" / "p6_capacity_calibration" / "metrics.json"
OUT = HERE / "outputs" / "p6b_capacity_at_scale"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
PARTICIPATION = 0.01
SPLIT = 440
AUMS = [10_000_000.0, 100_000_000.0, 500_000_000.0]


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def half_return(eq_path: Path, split: int = SPLIT) -> float | None:
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if split >= len(eq):
        return None
    return float(eq["equity"].iloc[-1] / eq["equity"].iloc[split - 1] - 1)


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

    p6 = json.loads(Path(P6).read_text(encoding="utf-8"))
    p6_01 = next(r for r in p6["capacity_sweep"] if r["participation"] == PARTICIPATION)

    rows = []
    for aum in AUMS:
        print(f"\n=== P6-scale: AUM={aum:,.0f}, participation={PARTICIPATION} ===")
        eq, _, _ = run_walk_forward(
            close=close, open_px=open_px, features=features, label=label, ic=ic,
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
            rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
            commission_bps=1.0, impact_bps=0.7, max_buy_open_gap=0.06, limit_buffer=0.995,
            market_exposure=market_exposure, initial_capital=aum,
            max_training_horizon=1, feature_directions=None, amount=amount,
            feature_selection=select_positive,
            max_daily_amount_participation=PARTICIPATION,
        )
        m = calculate_walk_forward_metrics(eq, aum)
        eq_path = OUT / f"equity_curve_aum_{int(aum)}.csv"
        eq.to_csv(eq_path, index=False, encoding="utf-8")
        cap_blocked = float(eq["capacity_blocked_buy_weight"].sum())
        cap_sessions = int((eq["capacity_limited_symbols"] > 0).sum())
        avg_gross = float(eq["gross_exposure"].mean())
        rows.append({
            "aum": aum,
            "participation": PARTICIPATION,
            "total_return": m.get("total_return"),
            "annualized_return": m.get("annualized_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_like": m.get("sharpe_like"),
            "final_equity": m.get("final_equity"),
            "second_half_return": half_return(eq_path),
            "capacity_blocked_buy_weight_total": cap_blocked,
            "capacity_limited_sessions": cap_sessions,
            "avg_gross_exposure": avg_gross,
            "return_vs_1e6": m.get("total_return", 0) - p6_01["total_return"],
        })
        print(f"  total={m.get('total_return'):.4f} sharpe={m.get('sharpe_like'):.3f} "
              f"dd={m.get('max_drawdown'):.3f} cap_blocked_w={cap_blocked:.2f} "
              f"cap_sessions={cap_sessions} avg_gross={avg_gross:.3f}")

    baseline = {
        "aum": 1_000_000.0,
        "participation": PARTICIPATION,
        "total_return": p6_01["total_return"],
        "annualized_return": p6_01["annualized_return"],
        "max_drawdown": p6_01["max_drawdown"],
        "sharpe_like": p6_01["sharpe_like"],
        "final_equity": p6_01["final_equity"] * 1.0,
        "second_half_return": p6_01["second_half_return"],
        "capacity_blocked_buy_weight_total": p6_01["capacity_blocked_buy_weight_total"],
        "capacity_limited_sessions": p6_01["capacity_limited_sessions"],
        "avg_gross_exposure": None,
        "return_vs_1e6": 0.0,
    }
    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "P3 champion + capacity participation=0.01, AUM sweep",
        "note": "capacity_weight = trailing-median daily amount * 0.01 / equity; scales with 1/AUM",
        "baseline_1e6_P6": baseline,
        "aum_sweep": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P6-scale AUM sweep (P3 champion, participation=0.01) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
