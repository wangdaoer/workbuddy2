"""P6：成本/容量真实校准（在 P3 冠军线性配置上开启真实可成交约束）。

P1–P5 全部基于「无容量上限」的合成回测假设。真实 A 股里小票/低流动性标的
根本无法按横截面排序的「理想权重」成交。本脚本在 P3 冠军配置
（自适应正 IC 选择 + 线性 IC 加权）之上开启 `max_daily_amount_participation`，
把单票日调仓权重变化 clip 在其 trailing-median 日成交额的 participation 倍以内，
检验 P3 的 +11.9% 在真实可成交规模下是否仍成立。

扫描 participation ∈ {0.02, 0.01, 0.005, 0.002}（标准 A 股保守假设约 1%=0.01），
复用预计算的 features/ic/label/market_exposure 以省时。无容量基线 = P3 metrics.json。
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
P3 = HERE / "outputs" / "p3_adaptive_selection" / "metrics.json"
OUT = HERE / "outputs" / "p6_capacity_calibration"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
INIT_CAP = 1_000_000.0
SPLIT = 440
PARTICIPATIONS = [0.02, 0.01, 0.005, 0.002]


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

    p3 = json.loads(Path(P3).read_text(encoding="utf-8"))["adaptive"]

    rows = []
    for part in PARTICIPATIONS:
        print(f"\n=== P6: P3 champion + capacity participation={part} ===")
        eq, _, _ = run_walk_forward(
            close=close, open_px=open_px, features=features, label=label, ic=ic,
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
            rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
            commission_bps=1.0, impact_bps=0.7, max_buy_open_gap=0.06, limit_buffer=0.995,
            market_exposure=market_exposure, initial_capital=INIT_CAP,
            max_training_horizon=1, feature_directions=None, amount=amount,
            feature_selection=select_positive,
            max_daily_amount_participation=part,
        )
        m = calculate_walk_forward_metrics(eq, INIT_CAP)
        eq_path = OUT / f"equity_curve_p6_p{part}.csv"
        eq.to_csv(eq_path, index=False, encoding="utf-8")
        cap_blocked = float(eq["capacity_blocked_buy_weight"].sum())
        cap_sessions = int((eq["capacity_limited_symbols"] > 0).sum())
        rows.append({
            "participation": part,
            "total_return": m.get("total_return"),
            "annualized_return": m.get("annualized_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_like": m.get("sharpe_like"),
            "final_equity": m.get("final_equity"),
            "second_half_return": half_return(eq_path),
            "capacity_blocked_buy_weight_total": cap_blocked,
            "capacity_limited_sessions": cap_sessions,
            "return_vs_p3": m.get("total_return", 0) - p3["total_return"],
        })
        print(f"  total={m.get('total_return'):.4f} sharpe={m.get('sharpe_like'):.3f} "
              f"dd={m.get('max_drawdown'):.3f} cap_blocked_w={cap_blocked:.3f} cap_sessions={cap_sessions}")

    baseline = {
        "participation": None,
        "total_return": p3["total_return"],
        "annualized_return": p3["annualized_return"],
        "max_drawdown": p3["max_drawdown"],
        "sharpe_like": p3["sharpe_like"],
        "final_equity": p3["final_equity"],
        "second_half_return": p3["second_half_return"],
        "capacity_blocked_buy_weight_total": 0.0,
        "capacity_limited_sessions": 0,
        "return_vs_p3": 0.0,
    }
    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "P3 champion (adaptive positive-IC linear) + max_daily_amount_participation capacity constraint",
        "note": "capacity_weight = trailing-median daily amount * participation / equity; per-symbol daily weight change clipped to +/- capacity_weight",
        "baseline_no_capacity_P3": baseline,
        "capacity_sweep": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P6 capacity calibration (P3 champion) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
