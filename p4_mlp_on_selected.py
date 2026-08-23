"""P4：把 P3 自适应选出的有效因子集喂回 P1 的 MLP 挑战者。

问题：非线性在「已清洗因子」上能否比 P3 自适应线性进一步增益？

做法（与 P3 同一选择逻辑，但作用于 MLP 输入）：
- 在 MLP 每个 retrain 点，用成熟窗口 IC 选出正 IC 因子子集；
- 仅用该子集训练 MLP、对下一区间打分（单一 meta-feature）；
- 把 MLP 分数注入 run_walk_forward（执行/约束/成本与 incumbent 一致）。

对比三方（均基于清洁真实面板）：
  A) P1 全因子 MLP（outputs/p1_real/challenger）
  B) P3 自适应线性（outputs/p3_adaptive_selection）
  C) 本脚本：P3选定因子 + MLP（非线性 on 已清洗因子）
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
from p1_challenger import MLP

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
P1_MLP = HERE / "outputs" / "p1_real" / "challenger" / "comparison.json"
P3 = HERE / "outputs" / "p3_adaptive_selection" / "metrics.json"
OUT = HERE / "outputs" / "p4_mlp_on_selected"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
INIT_CAP = 1_000_000.0
MAX_TRAIN_HORIZON = 1
SPLIT = 440


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def walk_forward_scores_selected(features, label, symbols, ic, select_fn):
    """与 P1 walk_forward_scores 同构，但在每个 retrain 点仅用 select_fn 选出的有效因子。"""
    dates = list(label.index)
    fnames = list(features)
    score = pd.DataFrame(np.nan, index=label.index, columns=symbols)
    n = len(dates)
    first = TRAIN_DAYS + MAX_TRAIN_HORIZON
    for i in range(first, n - 2, RETRAIN):
        start = i - TRAIN_DAYS - MAX_TRAIN_HORIZON
        end_train = i - MAX_TRAIN_HORIZON
        sel = select_fn(ic.iloc[start:end_train])
        if len(sel) < 3:
            sel = fnames
        Xs, ys = [], []
        for t in range(start, end_train):
            row = np.column_stack([features[f].iloc[t].reindex(symbols).values for f in sel])
            yt = label.iloc[t].reindex(symbols).values
            mask = ~np.isnan(row).any(axis=1) & ~np.isnan(yt)
            if mask.sum() < 50:
                continue
            Xs.append(row[mask])
            ys.append(yt[mask])
        if len(Xs) < 1:
            continue
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        mlp = MLP().fit(X, y)
        j_end = min(i + RETRAIN, n - 2)
        for j in range(i, j_end):
            row = np.column_stack([features[f].iloc[j].reindex(symbols).values for f in sel])
            pred = np.full(len(symbols), np.nan)
            mask = ~np.isnan(row).any(axis=1)
            if mask.sum() > 0:
                pred[mask] = mlp.predict(row[mask])
            score.iloc[j] = pred
        print(f"  retrain@{dates[i]} 因子数={len(sel)}")
    return score


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
    ic = daily_ic(features, label)  # 16 因子 × 880 日
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    print("=== P4：P3选定因子 + MLP（非线性 on 已清洗因子）===")
    score = walk_forward_scores_selected(features, label, symbols, ic, select_positive)

    ic_challenger = daily_ic({"mlp": score}, label)["mlp"]
    mean_rank_ic = float(ic_challenger.mean())
    print(f"MLP(选定因子) 平均 rank-IC = {mean_rank_ic:.4f}")

    equity, _, _ = run_walk_forward(
        close=close, open_px=open_px, features={"mlp": score}, label=label, ic=ic_challenger.to_frame("mlp"),
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
        rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
        commission_bps=1.0, impact_bps=0.7, max_buy_open_gap=0.06, limit_buffer=0.995,
        market_exposure=market_exposure, initial_capital=INIT_CAP,
        max_training_horizon=MAX_TRAIN_HORIZON, feature_directions=None, amount=amount,
    )
    metrics = calculate_walk_forward_metrics(equity, INIT_CAP)
    equity.to_csv(OUT / "equity_curve_p4.csv", index=False, encoding="utf-8")

    p1 = json.loads(Path(P1_MLP).read_text(encoding="utf-8"))
    p3 = json.loads(Path(P3).read_text(encoding="utf-8"))

    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "P3 adaptive factor selection fed into P1 MLP (nonlinear on cleaned factors)",
        "mlp_selected": {
            "mean_rank_ic": mean_rank_ic,
            "total_return": metrics.get("total_return"),
            "annualized_return": metrics.get("annualized_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe_like": metrics.get("sharpe_like"),
            "final_equity": metrics.get("final_equity"),
            "second_half_return": half_return(OUT / "equity_curve_p4.csv"),
        },
        "vs_p1_allfactor_mlp": {
            "ic_lift": mean_rank_ic - p1["challenger"]["mean_rank_ic"],
            "return_lift": metrics.get("total_return", 0) - p1["challenger"]["total_return"],
            "sharpe_lift": metrics.get("sharpe_like", 0) - p1["challenger"]["sharpe_like"],
        },
        "vs_p3_adaptive_linear": {
            "ic_note": "MLP 单一 meta-feature IC=%.4f vs P3 线性多因子组合（无单一 IC 数）" % mean_rank_ic,
            "return_lift": metrics.get("total_return", 0) - p3["adaptive"]["total_return"],
            "sharpe_lift": metrics.get("sharpe_like", 0) - p3["adaptive"]["sharpe_like"],
            "second_half_lift": (half_return(OUT / "equity_curve_p4.csv") or 0) - (p3["adaptive"]["second_half_return"] or 0),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P4 (MLP on P3选定因子) vs P1全因子MLP / P3自适应线性 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
