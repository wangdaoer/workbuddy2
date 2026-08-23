"""Generate latest candidate table from the walk-forward rank model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generate_strong_pullback_candidates import load_latest_weights
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import build_features, clean_matrix


CN_COLUMNS = {
    "symbol": "股票代码",
    "rank": "模型排名",
    "selected": "是否入选",
    "target_weight": "建议目标权重",
    "score": "模型分数",
    "trend_state": "趋势状态",
    "close": "收盘价",
    "return_5d": "5日收益",
    "return_20d": "20日收益",
    "return_60d": "60日收益",
    "return_120d": "120日收益",
    "distance_ma20": "相对MA20",
    "distance_ma60": "相对MA60",
    "distance_ma120": "相对MA120",
    "breakout_20d": "20日突破幅度",
    "intraday_return": "日内收益",
    "close_position": "收盘位置",
    "avg_amount_20d": "20日成交额中位数",
    "momentum_5": "动量5日因子",
    "momentum_20": "动量20日因子",
    "momentum_60": "动量60日因子",
    "reversal_5": "反转5日因子",
    "breakout_20": "突破20日因子",
    "distance_ma20_factor": "MA20距离因子",
    "volatility_20": "波动20日因子",
    "liquidity_20": "流动性20日因子",
    "intraday_return_factor": "日内收益因子",
    "close_position_factor": "收盘位置因子",
    "strong_pullback_20_5": "20日强势回调因子",
    "strong_pullback_60_5": "60日强势回调因子",
    "breakout_pullback_20_5": "突破回调因子",
    "anti_chase_intraday": "非追高因子",
    "liquid_pullback": "流动性回调因子",
}


def latest_metrics(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    idx = close.index.get_loc(date)
    c = close.iloc[idx]
    o = open_px.iloc[idx]
    h = high.iloc[idx]
    l = low.iloc[idx]
    ma20 = close.rolling(20).mean().iloc[idx]
    ma60 = close.rolling(60).mean().iloc[idx]
    ma120 = close.rolling(120).mean().iloc[idx]
    out = pd.DataFrame(index=close.columns)
    out["close"] = c
    out["return_5d"] = close.pct_change(5, fill_method=None).iloc[idx]
    out["return_20d"] = close.pct_change(20, fill_method=None).iloc[idx]
    out["return_60d"] = close.pct_change(60, fill_method=None).iloc[idx]
    out["return_120d"] = close.pct_change(120, fill_method=None).iloc[idx]
    out["distance_ma20"] = c / (ma20 + 1e-12) - 1.0
    out["distance_ma60"] = c / (ma60 + 1e-12) - 1.0
    out["distance_ma120"] = c / (ma120 + 1e-12) - 1.0
    out["breakout_20d"] = c / (high.rolling(20).max().shift(1).iloc[idx] + 1e-12) - 1.0
    out["intraday_return"] = c / (o + 1e-12) - 1.0
    out["close_position"] = (c - l) / ((h - l).replace(0, np.nan) + 1e-12)
    out["avg_amount_20d"] = amount.replace(0, np.nan).rolling(20).median().iloc[idx]
    return out


def load_trend_state(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["symbol", "trend_state"])
    frame = pd.read_csv(path, dtype={"symbol": str})
    if "symbol" not in frame or "trend_state" not in frame:
        return pd.DataFrame(columns=["symbol", "trend_state"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    return frame[["symbol", "trend_state"]].drop_duplicates("symbol", keep="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latest rank-model candidates.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="outputs/high_return_v2/rank_model_candidates_latest.csv")
    parser.add_argument("--trend-state", default=None)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--top-n", type=int, default=120)
    parser.add_argument("--selected-n", type=int, default=40)
    parser.add_argument("--max-position-weight", type=float, default=0.029)
    parser.add_argument("--leverage", type=float, default=0.93)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--allowed-trend-states", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    asof = pd.Timestamp(args.asof_date) if args.asof_date else close.index.max()
    if asof not in close.index:
        asof = close.index[close.index <= asof].max()

    weights = load_latest_weights(Path(args.weights))
    features = build_features(close, open_px, high, low, amount)
    score = None
    for name, weight in weights.items():
        if name not in features:
            continue
        weighted = features[name].loc[asof].fillna(0.0) * weight
        score = weighted if score is None else score + weighted
    if score is None:
        raise ValueError("No matching features between weights and feature builder.")

    feature_snapshot = pd.DataFrame({name: frame.loc[asof] for name, frame in features.items()})
    feature_snapshot = feature_snapshot.rename(
        columns={
            "distance_ma20": "distance_ma20_factor",
            "intraday_return": "intraday_return_factor",
            "close_position": "close_position_factor",
        }
    )
    metrics = latest_metrics(close, open_px, high, low, amount, asof)
    out = feature_snapshot.join(metrics)
    out["score"] = score
    out = out.dropna(subset=["score", "close"]).sort_values("score", ascending=False)
    out = out.reset_index().rename(columns={"index": "symbol"})
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)

    trend = load_trend_state(args.trend_state)
    if not trend.empty:
        out = out.merge(trend, on="symbol", how="left")
    else:
        out["trend_state"] = ""
    if args.allowed_trend_states:
        allowed = {x.strip() for x in args.allowed_trend_states.split(",") if x.strip()}
        out = out[out["trend_state"].isin(allowed)]

    out = out.head(args.top_n).reset_index(drop=True)
    out.insert(1, "rank", range(1, len(out) + 1))
    out["selected"] = out["rank"].le(args.selected_n)
    selected_weight = min(args.max_position_weight, args.leverage / max(args.selected_n, 1))
    out["target_weight"] = np.where(out["selected"], selected_weight, 0.0)

    order = [
        "symbol",
        "rank",
        "selected",
        "target_weight",
        "score",
        "trend_state",
        "close",
        "return_5d",
        "return_20d",
        "return_60d",
        "return_120d",
        "distance_ma20",
        "distance_ma60",
        "distance_ma120",
        "breakout_20d",
        "intraday_return",
        "close_position",
        "avg_amount_20d",
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "reversal_5",
        "breakout_20",
        "distance_ma20_factor",
        "volatility_20",
        "liquidity_20",
        "intraday_return_factor",
        "close_position_factor",
        "strong_pullback_20_5",
        "strong_pullback_60_5",
        "breakout_pullback_20_5",
        "anti_chase_intraday",
        "liquid_pullback",
    ]
    out = out[[c for c in order if c in out.columns]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    cn_output = output.with_name(output.stem + "_cn" + output.suffix)
    out.rename(columns=CN_COLUMNS).to_csv(cn_output, index=False, encoding="utf-8-sig")
    print(f"Generated {len(out)} rank candidates for {asof.strftime('%Y-%m-%d')}: {output}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
