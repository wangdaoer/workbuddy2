"""P5：regime 门控加权（在 P3 自适应选择之上）。

P2 regime IC 表洞察：牛市里 intraday_return/close_position 翻正、熊市里反转类主导。
本脚本让线性 P3 模型在**每个 retrain 点**用「当前 regime 在成熟窗口内的 IC 子样本」
选因子+加权——牛/震荡/熊下各自使用有效因子符号，把 P2 发现工程化。

对比：P3 自适应线性（无 regime 门控） / P4 MLP on 选定因子 / 本脚本 P5 regime 门控。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
P3 = HERE / "outputs" / "p3_adaptive_selection" / "metrics.json"
P4 = HERE / "outputs" / "p4_mlp_on_selected" / "metrics.json"
OUT = HERE / "outputs" / "p5_regime_gated"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
INIT_CAP = 1_000_000.0
ROLL = 120
SPLIT = 440


def regime_label(close: pd.DataFrame) -> pd.Series:
    ew = close.mean(axis=1)
    mom = ew.pct_change(ROLL, fill_method=None)
    q = mom.quantile([0.33, 0.67])
    return pd.cut(mom, bins=[-np.inf, q[0.33], q[0.67], np.inf],
                  labels=["bear", "sideways", "bull"])


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
    regime = regime_label(close)
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    print("=== P5：regime 门控加权 + 自适应选择（线性）===")
    eq, _, _ = run_walk_forward(
        close=close, open_px=open_px, features=features, label=label, ic=daily_ic(features, label),
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
        rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
        commission_bps=1.0, impact_bps=0.7, max_buy_open_gap=0.06, limit_buffer=0.995,
        market_exposure=market_exposure, initial_capital=INIT_CAP,
        max_training_horizon=1, feature_directions=None, amount=amount,
        feature_selection=select_positive, regime=regime,
    )
    metrics = calculate_walk_forward_metrics(eq, INIT_CAP)
    eq.to_csv(OUT / "equity_curve_p5.csv", index=False, encoding="utf-8")

    p3 = json.loads(Path(P3).read_text(encoding="utf-8"))["adaptive"]
    p4 = json.loads(Path(P4).read_text(encoding="utf-8"))["mlp_selected"]

    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "regime-gated weighting (per-retrain IC in current-regime subsample) + adaptive positive-IC selection",
        "p5": {
            "total_return": metrics.get("total_return"),
            "annualized_return": metrics.get("annualized_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe_like": metrics.get("sharpe_like"),
            "final_equity": metrics.get("final_equity"),
            "second_half_return": half_return(OUT / "equity_curve_p5.csv"),
        },
        "vs_p3_adaptive_linear": {
            "return_lift": metrics.get("total_return", 0) - p3["total_return"],
            "sharpe_lift": metrics.get("sharpe_like", 0) - p3["sharpe_like"],
            "second_half_lift": (half_return(OUT / "equity_curve_p5.csv") or 0) - (p3["second_half_return"] or 0),
        },
        "vs_p4_mlp_selected": {
            "return_lift": metrics.get("total_return", 0) - p4["total_return"],
            "sharpe_lift": metrics.get("sharpe_like", 0) - p4["sharpe_like"],
            "second_half_lift": (half_return(OUT / "equity_curve_p5.csv") or 0) - (p4["second_half_return"] or 0),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P5 (regime 门控) vs P3 自适应线性 / P4 MLP on 选定 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
