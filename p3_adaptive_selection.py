"""P3：自适应因子选择（walk-forward 每个 retrain 点用成熟窗口 IC 重选有效因子）。

动机（P2 诚实结论）：固定前半窗子集虽有样本外改进，但后半窗口边缘衰减（-7.2%）。
本脚本让因子选择在**每个 retrain 点**基于当地成熟窗口 IC 重做，从构造上消除
look-ahead，预期样本外更稳。

复用 run_walk_forward 的 feature_selection 回调（默认 None 时行为与原 incumbent 完全一致）。
对比三方：incumbent(16) / P2固定子集(9) / P3自适应。
"""

from __future__ import annotations

from pathlib import Path

import json
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
INC_METRICS = HERE / "outputs" / "p1_real" / "next_open_rank_model" / "metrics.json"
P2_METRICS = HERE / "outputs" / "p2_factor_analysis" / "metrics.json"
OUT = HERE / "outputs" / "p3_adaptive_selection"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
INIT_CAP = 1_000_000.0
SPLIT = 440  # 后半窗口（样本外）切分点


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    """成熟窗口 IC 为正的因子才保留。"""
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def half_metrics(eq_path: Path, split: int = SPLIT) -> dict:
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    s = eq["equity"]
    if split >= len(eq):
        return {"first": None, "second": None}
    first = s.iloc[split - 1] / s.iloc[0] - 1
    second = s.iloc[-1] / s.iloc[split - 1] - 1
    return {"first": float(first), "second": float(second)}


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
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    eq, _, _ = run_walk_forward(
        close=close, open_px=open_px, features=features, label=label, ic=daily_ic(features, label),
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
        rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
        commission_bps=1.0, impact_bps=0.7, max_buy_open_gap=0.06, limit_buffer=0.995,
        market_exposure=market_exposure, initial_capital=INIT_CAP,
        max_training_horizon=1, feature_directions=None, amount=amount,
        feature_selection=select_positive,
    )
    metrics = calculate_walk_forward_metrics(eq, INIT_CAP)
    eq.to_csv(OUT / "equity_curve_adaptive.csv", index=False, encoding="utf-8")

    incumbent = json.loads(Path(INC_METRICS).read_text(encoding="utf-8"))
    p2 = json.loads(Path(P2_METRICS).read_text(encoding="utf-8"))

    h = half_metrics(OUT / "equity_curve_adaptive.csv")
    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "adaptive feature selection per retrain (mature-window positive IC)",
        "adaptive": {
            "total_return": metrics.get("total_return"),
            "annualized_return": metrics.get("annualized_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe_like": metrics.get("sharpe_like"),
            "final_equity": metrics.get("final_equity"),
            "first_half_return": h["first"],
            "second_half_return": h["second"],
        },
        "vs_incumbent": {
            "return_lift": (metrics.get("total_return", 0) - incumbent.get("total_return", 0)),
            "sharpe_lift": (metrics.get("sharpe_like", 0) - incumbent.get("sharpe_like", 0)),
            "second_half_lift": (h["second"] - half_metrics(Path(INC_METRICS).with_name("equity_curve.csv")).get("second", 0)),
        },
        "vs_p2_fixed_tuned": {
            "return_lift": (metrics.get("total_return", 0) - p2["tuned_model"]["total_return"]),
            "sharpe_lift": (metrics.get("sharpe_like", 0) - p2["tuned_model"]["sharpe_like"]),
            "second_half_lift": (h["second"] - half_metrics(Path(P2_METRICS).with_name("equity_curve_tuned.csv")).get("second", 0)),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== P3 自适应因子选择 vs incumbent / P2固定子集（均基于清洁真实数据）===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
