"""Generate latest strong-stock pullback candidates from the trained rank model."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import build_features, clean_matrix


ST_NAME_COLUMNS = ("stock_name", "name", "股票名称", "股票简称", "名称")


def st_name_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in ST_NAME_COLUMNS:
        if column not in frame:
            continue
        names = frame[column].fillna("").astype(str).str.strip().str.upper().str.replace("＊", "*", regex=False)
        mask = mask | names.map(lambda text: bool(re.match(r"^(?:\*?ST|S\*ST)", text)))
    return mask


def load_latest_weights(path: Path) -> pd.Series:
    weights = pd.read_csv(path)
    if weights.empty:
        raise ValueError(f"Empty weights file: {path}")
    row = weights.iloc[-1].drop(labels=["date"], errors="ignore")
    return pd.to_numeric(row, errors="coerce").fillna(0.0)


def latest_raw_metrics(close: pd.DataFrame, open_px: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, amount: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    idx = close.index.get_loc(date)
    c = close.iloc[idx]
    o = open_px.iloc[idx]
    h = high.iloc[idx]
    l = low.iloc[idx]
    prior_close_5 = close.shift(5).iloc[idx]
    prior_close_25 = close.shift(25).iloc[idx]
    prior_close_65 = close.shift(65).iloc[idx]
    ma20 = close.rolling(20).mean().iloc[idx]
    ma60 = close.rolling(60).mean().iloc[idx]
    out = pd.DataFrame(index=close.columns)
    out["close"] = c
    out["return_5d"] = close.pct_change(5, fill_method=None).iloc[idx]
    out["return_20d"] = close.pct_change(20, fill_method=None).iloc[idx]
    out["return_60d"] = close.pct_change(60, fill_method=None).iloc[idx]
    out["pullback_5d"] = -out["return_5d"]
    out["prior_return_20_before_pullback"] = prior_close_5 / (prior_close_25 + 1e-12) - 1.0
    out["prior_return_60_before_pullback"] = prior_close_5 / (prior_close_65 + 1e-12) - 1.0
    out["raw_distance_ma20"] = c / (ma20 + 1e-12) - 1.0
    out["raw_distance_ma60"] = c / (ma60 + 1e-12) - 1.0
    out["raw_intraday_return"] = c / (o + 1e-12) - 1.0
    out["raw_close_position"] = (c - l) / ((h - l).replace(0, np.nan) + 1e-12)
    out["breakout_20d"] = c / (high.rolling(20).max().shift(1).iloc[idx] + 1e-12) - 1.0
    out["avg_amount_20d"] = amount.replace(0, np.nan).rolling(20).median().iloc[idx]
    return out


def filter_strong_pullback_candidates(
    candidates: pd.DataFrame,
    *,
    min_close: float = 2.0,
    min_avg_amount_20d: float = 30_000_000.0,
    min_pullback_5d: float = 0.03,
    max_pullback_5d: float = 0.18,
    min_prior_return_20: float = 0.08,
    min_prior_return_60: float = 0.18,
    min_return_20d: float = -0.12,
    min_return_60d: float = 0.0,
    min_distance_ma60: float = -0.10,
    max_intraday_return: float = 0.05,
) -> pd.DataFrame:
    out = candidates.copy()
    numeric_columns = [
        "close",
        "return_5d",
        "return_20d",
        "return_60d",
        "pullback_5d",
        "prior_return_20_before_pullback",
        "prior_return_60_before_pullback",
        "raw_distance_ma60",
        "raw_intraday_return",
        "avg_amount_20d",
    ]
    if "raw_distance_ma60" not in out and "distance_ma60" in out:
        out["raw_distance_ma60"] = out["distance_ma60"]

    for col in numeric_columns:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    pullback = out["pullback_5d"].where(out["pullback_5d"].notna(), -out["return_5d"])
    prior_strength = (
        out["prior_return_20_before_pullback"].ge(min_prior_return_20)
        | out["prior_return_60_before_pullback"].ge(min_prior_return_60)
    )
    trend_intact = (
        out["return_20d"].ge(min_return_20d)
        & out["return_60d"].ge(min_return_60d)
        & out["raw_distance_ma60"].ge(min_distance_ma60)
    )
    executable = (
        out["close"].ge(min_close)
        & out["avg_amount_20d"].ge(min_avg_amount_20d)
        & out["raw_intraday_return"].le(max_intraday_return)
    )
    pullback_ok = pullback.ge(min_pullback_5d) & pullback.le(max_pullback_5d)
    tradable_name = ~st_name_mask(out)
    return out.loc[tradable_name & prior_strength & trend_intact & executable & pullback_ok].copy()


def reason(row: pd.Series) -> str:
    tags = []
    if row.get("breakout_pullback_20_5", 0) >= 0.8:
        tags.append("突破后回调")
    if row.get("strong_pullback_20_5", 0) >= 0.8:
        tags.append("20日强势回调")
    if row.get("strong_pullback_60_5", 0) >= 0.8:
        tags.append("60日强势回调")
    if row.get("anti_chase_intraday", 0) >= 0.7:
        tags.append("非追高")
    if row.get("reversal_5", 0) >= 0.7:
        tags.append("短线回撤")
    return "；".join(tags) if tags else "综合排名"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strong-stock pullback candidates.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="strong_pullback_candidates_latest.csv")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--min-close", type=float, default=2.0)
    parser.add_argument("--min-avg-amount-20d", type=float, default=30_000_000.0)
    parser.add_argument("--min-pullback-5d", type=float, default=0.03)
    parser.add_argument("--max-pullback-5d", type=float, default=0.18)
    parser.add_argument("--min-prior-return-20", type=float, default=0.08)
    parser.add_argument("--min-prior-return-60", type=float, default=0.18)
    parser.add_argument("--min-return-20d", type=float, default=-0.12)
    parser.add_argument("--min-return-60d", type=float, default=0.0)
    parser.add_argument("--min-distance-ma60", type=float, default=-0.10)
    parser.add_argument("--max-intraday-return", type=float, default=0.05)
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
        row = features[name].loc[asof].fillna(0.0) * weight
        score = row if score is None else score + row
    if score is None:
        raise ValueError("No matching features between weights and feature builder.")

    feature_snapshot = pd.DataFrame({name: frame.loc[asof] for name, frame in features.items()})
    metrics = latest_raw_metrics(close, open_px, high, low, amount, asof)
    out = feature_snapshot.join(metrics)
    out["score"] = score
    out = out.dropna(subset=["score", "close"]).sort_values("score", ascending=False)
    out = filter_strong_pullback_candidates(
        out,
        min_close=args.min_close,
        min_avg_amount_20d=args.min_avg_amount_20d,
        min_pullback_5d=args.min_pullback_5d,
        max_pullback_5d=args.max_pullback_5d,
        min_prior_return_20=args.min_prior_return_20,
        min_prior_return_60=args.min_prior_return_60,
        min_return_20d=args.min_return_20d,
        min_return_60d=args.min_return_60d,
        min_distance_ma60=args.min_distance_ma60,
        max_intraday_return=args.max_intraday_return,
    )
    out = out.sort_values("score", ascending=False).head(args.top_n).reset_index().rename(columns={"index": "symbol"})
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["reason"] = out.apply(reason, axis=1)
    keep = [
        "symbol",
        "score",
        "reason",
        "close",
        "return_5d",
        "return_20d",
        "return_60d",
        "pullback_5d",
        "prior_return_20_before_pullback",
        "prior_return_60_before_pullback",
        "raw_distance_ma20",
        "raw_distance_ma60",
        "breakout_20d",
        "raw_intraday_return",
        "raw_close_position",
        "avg_amount_20d",
        "breakout_pullback_20_5",
        "strong_pullback_20_5",
        "strong_pullback_60_5",
        "anti_chase_intraday",
        "reversal_5",
    ]
    out = out[[c for c in keep if c in out.columns]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Generated {len(out)} candidates for {asof.strftime('%Y-%m-%d')}: {output}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
