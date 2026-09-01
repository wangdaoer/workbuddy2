"""P2：基于真实 A 股数据的因子衰减 / regime 分析与调优。

复用模型自有 machinery（build_features / daily_ic / run_walk_forward /
load_market_exposure），在**清洁真实面板**上：

1. 因子衰减分析：每个因子的 120 日滚动 rank-IC，及其衰减趋势（末段 - 首段）。
2. regime 分析：按等权市场指数的 60 日动能把交易日分为 牛 / 震荡 / 熊，
   计算各因子在不同 regime 下的平均 rank-IC，识别「在反转市有效」的因子。
3. 真实数据调优子集：用前半窗口（前 440 日）选出正 IC 因子组成有效子集，
   固定子集后 walk-forward 训练，与 incumbent（全 15 因子）对比——
   检验「丢弃失效因子、聚焦有效因子」能否在真实数据上提升。

所有 IC 均为 Spearman rank-IC（与 P0/P1 完全一致口径）。
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
INC_METRICS = HERE / "outputs" / "p1_real" / "next_open_rank_model" / "metrics.json"
OUT = HERE / "outputs" / "p2_factor_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
RETRAIN = 20
TOP_N = 20
INIT_CAP = 1_000_000.0
ROLL = 120          # 滚动 IC 窗口
SEL_DAYS = 440      # 前半窗口用于因子选择（避免对后半做 look-ahead）


def regime_label(close: pd.DataFrame) -> pd.Series:
    """用等权市场指数的 60 日动能定义 regime：牛 / 震荡 / 熊。"""
    ew = close.mean(axis=1)                       # 等权指数（价格水平）
    mom60 = ew.pct_change(ROLL, fill_method=None)  # 60 日动能
    q = mom60.quantile([0.33, 0.67])
    return pd.cut(mom60, bins=[-np.inf, q[0.33], q[0.67], np.inf],
                  labels=["bear", "sideways", "bull"])


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
    fnames = list(features)

    # —— 1) 每日 rank-IC（所有因子）——
    ic = daily_ic({f: features[f] for f in fnames}, label)

    # 2) 滚动 IC 与衰减趋势
    roll = ic.rolling(ROLL).mean()
    first_blk = roll.iloc[ROLL:2 * ROLL].mean()      # 首段（约第120-240日）
    last_blk = roll.iloc[-ROLL:].mean()              # 末段（最近120日）
    decay = last_blk - first_blk
    full_mean = ic.mean()
    rolling_summary = pd.DataFrame({
        "full_mean_ic": full_mean,
        "first120_240_mean_ic": first_blk,
        "last120_mean_ic": last_blk,
        "decay_trend": decay,
    }).sort_values("full_mean_ic", ascending=False)
    rolling_summary.to_csv(OUT / "rolling_ic.csv", encoding="utf-8")

    # 3) regime IC
    reg = regime_label(close)
    reg_df = reg.reindex(label.index)
    regime_rows = []
    for r in ["bull", "sideways", "bear"]:
        mask = (reg_df == r).values
        if mask.sum() < 30:
            continue
        row = {"regime": r, "n_days": int(mask.sum())}
        for f in fnames:
            x = ic[f].where(mask)
            row[f] = float(x.dropna().mean())
        regime_rows.append(row)
    regime_ic = pd.DataFrame(regime_rows).set_index("regime")
    regime_ic.to_csv(OUT / "regime_ic.csv", encoding="utf-8")

    # —— 4) 真实数据调优子集：前半窗口正 IC 因子 ——
    sel_mask = ic.index < ic.index[SEL_DAYS]
    sel_ic = ic.loc[sel_mask].mean().sort_values(ascending=False)
    effective = [f for f in sel_ic.index if sel_ic[f] > 0]
    dead = [f for f in fnames if f not in effective]
    print(f"[因子选择] 前半窗口({SEL_DAYS}日) 正 IC 有效因子 {len(effective)} 个：")
    print("  " + ", ".join(effective))
    print(f"[因子选择] 失效/负 IC 因子 {len(dead)} 个：")
    print("  " + ", ".join(dead))

    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    # 调优模型：固定有效子集，walk-forward（IC 加权该子集）
    tuned_features = {f: features[f] for f in effective}
    ic_tuned = ic[effective]
    eq_tuned, w_tuned, _ = run_walk_forward(
        close=close, open_px=open_px, features=tuned_features, label=label, ic=ic_tuned,
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
        rebalance_frequency=1, max_position_weight=0.04, leverage=1.0,
        commission_bps=1.0, impact_bps=0.7, max_buy_open_gap=0.06, limit_buffer=0.995,
        market_exposure=market_exposure, initial_capital=INIT_CAP,
        max_training_horizon=1, feature_directions=None, amount=amount,
    )
    metrics_tuned = calculate_walk_forward_metrics(eq_tuned, INIT_CAP)
    # 调优模型综合分 IC（用有效子集 IC 重建）
    comp_tuned = pd.DataFrame(0.0, index=label.index, columns=symbols)
    for f in effective:
        comp_tuned = comp_tuned.add(features[f].reindex(index=label.index, columns=symbols).mul(
            ic_tuned[f].reindex(label.index).values, axis=0), fill_value=0.0)
    ic_comp_tuned = float(daily_ic({"tuned": comp_tuned}, label)["tuned"].dropna().mean())

    incumbent = json.loads(Path(INC_METRICS).read_text(encoding="utf-8")) if Path(INC_METRICS).exists() else {}

    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "n_symbols": len(symbols),
        "n_days": int(close.shape[0]),
        "selection_window_days": SEL_DAYS,
        "effective_factors": effective,
        "dead_factors": dead,
        "tuned_model": {
            "n_factors": len(effective),
            "composite_rank_ic": ic_comp_tuned,
            "total_return": metrics_tuned.get("total_return"),
            "annualized_return": metrics_tuned.get("annualized_return"),
            "max_drawdown": metrics_tuned.get("max_drawdown"),
            "sharpe_like": metrics_tuned.get("sharpe_like"),
            "final_equity": metrics_tuned.get("final_equity"),
        },
        "incumbent": {
            "n_factors": len(fnames),
            "composite_rank_ic": incumbent.get("mean_rank_ic_composite"),
            "total_return": incumbent.get("total_return"),
            "annualized_return": incumbent.get("annualized_return"),
            "max_drawdown": incumbent.get("max_drawdown"),
            "sharpe_like": incumbent.get("sharpe_like"),
            "final_equity": incumbent.get("final_equity"),
        },
        "lift": {
            "composite_rank_ic": (ic_comp_tuned - (incumbent.get("mean_rank_ic_composite") or 0)),
            "total_return": (metrics_tuned.get("total_return", 0) - incumbent.get("total_return", 0)),
            "sharpe_like": (metrics_tuned.get("sharpe_like", 0) - incumbent.get("sharpe_like", 0)),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    eq_tuned.to_csv(OUT / "equity_curve_tuned.csv", index=False, encoding="utf-8")
    print("\n=== P2 调优子集模型 vs incumbent（均基于清洁真实数据）===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
