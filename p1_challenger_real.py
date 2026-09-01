"""P1 挑战者（真实 A 股数据版）：用 hsjday.zip 接入的真实历史数据，验证
「非线性模型能否比线性 incumbent 多捕获信号」这一提升框架。

复用 p1_challenger 的 MLP / walk_forward_scores / build_features 等实现（保证
因子集、执行约束、成本与 incumbent 完全一致），仅把数据来源与基线目录指向真实数据。

与合成补充数据不同，真实 A 股 2022-12 → 2026-07 是漫长熊/震荡市，因子 IC 普遍
接近 0。本脚本如实报告挑战者 rank-IC 与 incumbent 综合 rank-IC 的**相对**与
**绝对**强度，不夸大「提升」。
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
    build_multi_horizon_ic,
    calculate_walk_forward_metrics,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    run_walk_forward,
)
# 复用已验证的 MLP 与挑战者 walk-forward 打分逻辑（含 target 标准化修复）
from p1_challenger import MLP, walk_forward_scores

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
INC_DIR = HERE / "outputs" / "p1_real" / "next_open_rank_model"
OUT = HERE / "outputs" / "p1_real" / "challenger"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
INIT_CAP = 1_000_000.0
MAX_TRAIN_HORIZON = 1

# 经济意义阈值：rank-IC 绝对值低于该值视为「无实际信号」
MEANINGFUL_IC = 0.02


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

    print("=== 真实数据 P1 挑战者：walk-forward 训练 MLP 并生成分数 ===")
    score = walk_forward_scores(features, label, symbols)

    ic_challenger = daily_ic({"mlp": score}, label)["mlp"]
    mean_rank_ic = float(ic_challenger.mean())
    print(f"挑战者平均 rank-IC = {mean_rank_ic:.4f}")

    ic = ic_challenger.to_frame("mlp")
    equity, weights, trades = run_walk_forward(
        close=close, open_px=open_px, features={"mlp": score}, label=label, ic=ic,
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
        rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
        commission_bps=1.0, impact_bps=0.7, max_buy_open_gap=0.06, limit_buffer=0.995,
        market_exposure=market_exposure, initial_capital=INIT_CAP,
        max_training_horizon=MAX_TRAIN_HORIZON, feature_directions=None, amount=amount,
    )
    metrics = calculate_walk_forward_metrics(equity, INIT_CAP)

    # 真实 incumbent 基线指标
    incumbent = json.loads((INC_DIR / "metrics.json").read_text(encoding="utf-8")) if (INC_DIR / "metrics.json").exists() else {}

    # 公平对比：用 incumbent 滚动因子权重重建线性综合分 IC
    inc_comp_ic = None
    wpath = INC_DIR / "rolling_feature_weights.csv"
    if wpath.exists():
        w = pd.read_csv(wpath, parse_dates=["date"]).set_index("date")
        w = w.reindex(label.index).ffill()
        comp = pd.DataFrame(0.0, index=label.index, columns=symbols)
        for f in w.columns:
            if f in features:
                contrib = features[f].reindex(index=label.index, columns=symbols).mul(
                    w[f].reindex(label.index).values, axis=0
                )
                comp = comp.add(contrib, fill_value=0.0)
        ic_inc = daily_ic({"inc": comp}, label)["inc"]
        inc_comp_ic = float(ic_inc.dropna().mean()) if ic_inc.notna().any() else None

    signal_lift = (mean_rank_ic - inc_comp_ic) if inc_comp_ic is not None else None

    # 诚实判定：相对提升 + 绝对信号强度双重门槛
    if inc_comp_ic is not None and mean_rank_ic > inc_comp_ic and mean_rank_ic >= MEANINGFUL_IC:
        verdict = "CHALLENGER_WINS"
    elif inc_comp_ic is not None and mean_rank_ic > inc_comp_ic:
        verdict = "CHALLENGER_WINS_ON_SIGNAL"
    elif inc_comp_ic is not None and abs(mean_rank_ic) < MEANINGFUL_IC and abs(inc_comp_ic) < MEANINGFUL_IC:
        verdict = "INCONCLUSIVE_LOW_SIGNAL"
    else:
        verdict = "INCONCLUSIVE"

    risk_note = (
        "真实 A 股 2022-12→2026-07 为漫长熊/震荡市，线性因子 IC 普遍≈0；"
        "挑战者即便在相对意义上有提升，绝对 rank-IC 仍远低于经济可用阈值，"
        "不构成可实盘信号。这正是对合成数据的诚实对照。"
        if verdict in ("CHALLENGER_WINS_ON_SIGNAL", "INCONCLUSIVE_LOW_SIGNAL")
        else ""
    )

    comparison = {
        "data_source": "hsjday.zip (real A-share, TDX .day)",
        "panel": str(PANEL),
        "date_start": str(close.index[0].date()) if hasattr(close.index[0], "date") else str(close.index[0]),
        "date_end": str(close.index[-1].date()) if hasattr(close.index[-1], "date") else str(close.index[-1]),
        "n_symbols": len(symbols),
        "n_days": int(close.shape[0]),
        "challenger": {
            "mean_rank_ic": mean_rank_ic,
            "sharpe_like": metrics.get("sharpe_like"),
            "max_drawdown": metrics.get("max_drawdown"),
            "total_return": metrics.get("total_return"),
            "annualized_return": metrics.get("annualized_return"),
            "trade_days": metrics.get("trade_days"),
            "final_equity": metrics.get("final_equity"),
        },
        "incumbent": {
            "mean_rank_ic_composite": inc_comp_ic,
            "sharpe_like": incumbent.get("sharpe_like"),
            "max_drawdown": incumbent.get("max_drawdown"),
            "total_return": incumbent.get("total_return"),
            "annualized_return": incumbent.get("annualized_return"),
            "trade_days": incumbent.get("trade_days"),
            "final_equity": incumbent.get("final_equity"),
        },
        "signal_lift": signal_lift,
        "return_lift": (metrics.get("total_return", 0) - incumbent.get("total_return", 0)),
        "sharpe_lift": (metrics.get("sharpe_like", 0) - incumbent.get("sharpe_like", 0)),
        "meaningful_ic_threshold": MEANINGFUL_IC,
        "verdict": verdict,
        "risk_note": risk_note,
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    equity.to_csv(OUT / "equity_curve.csv", index=False, encoding="utf-8")
    print("\n=== 真实数据 P1 对比（challenger vs incumbent，均基于真实 A 股）===")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
