"""Real-time trend state classification from price, volume, and lifeline rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix


TREND_STATES = (
    "数据不足",
    "未启动",
    "起爆观察",
    "趋势确认",
    "生命线健康",
    "回调可观察",
    "生命线预警",
    "趋势破坏",
)


def _safe_float(value: object) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _row_score(row: dict[str, object]) -> float:
    state_score = {
        "起爆观察": 1.0,
        "回调可观察": 0.9,
        "生命线健康": 0.75,
        "趋势确认": 0.6,
        "生命线预警": 0.25,
        "未启动": 0.0,
        "趋势破坏": -1.0,
        "数据不足": -2.0,
    }.get(str(row.get("trend_state")), 0.0)
    ret60 = _safe_float(row.get("return_60d"))
    ret120 = _safe_float(row.get("return_120d"))
    amount_ratio = _safe_float(row.get("amount_ratio"))
    drawdown = abs(min(_safe_float(row.get("drawdown_20d")), 0.0))
    return float(
        state_score
        + np.nan_to_num(ret60, nan=0.0) * 0.6
        + np.nan_to_num(ret120, nan=0.0) * 0.4
        + min(np.nan_to_num(amount_ratio, nan=0.0), 5.0) * 0.03
        - drawdown * 0.5
    )


def choose_realtime_lifeline(
    close: pd.Series,
    asof_date: pd.Timestamp | None = None,
    *,
    windows: tuple[int, ...] = (20, 30, 60, 120),
    lookback: int = 120,
    breach_buffer: float = 0.03,
    max_breach_days: int = 3,
) -> dict[str, object]:
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if close.empty:
        return {
            "lifeline_ma": None,
            "lifeline_value": np.nan,
            "distance_to_lifeline": np.nan,
            "lifeline_breach_days_5": 0,
            "lifeline_breach_days_20": 0,
        }

    asof = pd.Timestamp(asof_date) if asof_date is not None else close.index[-1]
    history = close.loc[:asof].dropna()
    if history.empty:
        return {
            "lifeline_ma": None,
            "lifeline_value": np.nan,
            "distance_to_lifeline": np.nan,
            "lifeline_breach_days_5": 0,
            "lifeline_breach_days_20": 0,
        }

    candidates: list[dict[str, object]] = []
    for window in windows:
        line = history.rolling(window, min_periods=max(5, window // 2)).mean()
        segment = pd.concat(
            [
                history.tail(lookback).rename("close"),
                line.reindex(history.tail(lookback).index).rename("line"),
            ],
            axis=1,
        ).dropna()
        if segment.empty:
            continue
        distance = segment["close"] / (segment["line"] + 1e-12) - 1.0
        breach = distance.lt(-breach_buffer)
        current_line = float(line.iloc[-1])
        current_distance = float(history.iloc[-1] / (current_line + 1e-12) - 1.0)
        candidates.append(
            {
                "lifeline_ma": window,
                "lifeline_value": current_line,
                "distance_to_lifeline": current_distance,
                "breach_days_lookback": int(breach.sum()),
                "min_distance_lookback": float(distance.min()),
                "avg_distance_lookback": float(distance.mean()),
                "line": line,
            }
        )

    if not candidates:
        return {
            "lifeline_ma": None,
            "lifeline_value": np.nan,
            "distance_to_lifeline": np.nan,
            "lifeline_breach_days_5": 0,
            "lifeline_breach_days_20": 0,
        }

    valid = [
        row
        for row in candidates
        if int(row["breach_days_lookback"]) <= max_breach_days
        and float(row["distance_to_lifeline"]) >= -breach_buffer
    ]
    if valid:
        chosen = sorted(valid, key=lambda row: int(row["lifeline_ma"]))[0]
    else:
        chosen = sorted(
            candidates,
            key=lambda row: (
                int(row["breach_days_lookback"]),
                -float(row["min_distance_lookback"]),
                int(row["lifeline_ma"]),
            ),
        )[0]

    line = chosen.pop("line")
    recent = history.tail(20)
    recent_line = line.reindex(recent.index)
    recent_distance = recent / (recent_line + 1e-12) - 1.0
    chosen["lifeline_breach_days_5"] = int(recent_distance.tail(5).lt(-breach_buffer).sum())
    chosen["lifeline_breach_days_20"] = int(recent_distance.lt(-breach_buffer).sum())
    return chosen


def classify_trend_state(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    amount: pd.Series,
    asof_date: pd.Timestamp | None = None,
    *,
    breakout_window: int = 60,
    amount_window: int = 20,
    amount_multiplier: float = 1.5,
    breakout_buffer: float = 0.01,
    breach_buffer: float = 0.03,
) -> dict[str, object]:
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if close.empty:
        return {"trend_state": "数据不足"}

    asof = pd.Timestamp(asof_date) if asof_date is not None else close.index[-1]
    close = close.loc[:asof].dropna()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    low = pd.to_numeric(low, errors="coerce").reindex(close.index)
    amount = pd.to_numeric(amount, errors="coerce").reindex(close.index)
    if len(close) < max(80, breakout_window + 20):
        return {
            "asof_date": asof.strftime("%Y-%m-%d"),
            "trend_state": "数据不足",
            "close": float(close.iloc[-1]) if not close.empty else np.nan,
        }

    c = float(close.iloc[-1])
    ma20 = close.rolling(20, min_periods=20).mean()
    ma30 = close.rolling(30, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=40).mean()
    ma120 = close.rolling(120, min_periods=60).mean()
    prev_high = high.rolling(breakout_window, min_periods=max(20, breakout_window // 2)).max().shift(1)
    amount_median = amount.replace(0, np.nan).rolling(
        amount_window, min_periods=max(5, amount_window // 2)
    ).median().shift(1)

    prev_high_now = _safe_float(prev_high.iloc[-1])
    amount_ratio = _safe_float(amount.iloc[-1] / (amount_median.iloc[-1] + 1e-12))
    ma20_now = _safe_float(ma20.iloc[-1])
    ma30_now = _safe_float(ma30.iloc[-1])
    ma60_now = _safe_float(ma60.iloc[-1])
    ma120_now = _safe_float(ma120.iloc[-1])
    high_20 = _safe_float(high.rolling(20, min_periods=5).max().iloc[-1])
    high_120 = _safe_float(high.rolling(120, min_periods=60).max().iloc[-1])
    low_120 = _safe_float(low.rolling(120, min_periods=60).min().iloc[-1])
    drawdown_20d = c / (high_20 + 1e-12) - 1.0 if high_20 > 0 else np.nan
    trend_range_120d = high_120 / (low_120 + 1e-12) - 1.0 if high_120 > 0 and low_120 > 0 else np.nan
    return_20d = close.pct_change(20, fill_method=None).iloc[-1]
    return_60d = close.pct_change(60, fill_method=None).iloc[-1]
    return_120d = close.pct_change(120, fill_method=None).iloc[-1]
    ma60_slope_20d = ma60.iloc[-1] / (ma60.iloc[-21] + 1e-12) - 1.0 if len(ma60.dropna()) > 21 else np.nan
    ma120_slope_20d = ma120.iloc[-1] / (ma120.iloc[-21] + 1e-12) - 1.0 if len(ma120.dropna()) > 21 else np.nan

    is_ignition = bool(
        np.isfinite(prev_high_now)
        and c > prev_high_now * (1.0 + breakout_buffer)
        and amount_ratio >= amount_multiplier
        and np.isfinite(ma60_now)
        and c > ma60_now
        and np.isfinite(ma20_now)
        and ma20_now >= ma60_now * 0.98
    )
    trend_confirmed = bool(
        (
            np.isfinite(return_60d)
            and return_60d >= 0.25
            and np.isfinite(ma60_now)
            and c > ma60_now
            and np.nan_to_num(ma60_slope_20d, nan=0.0) > 0
        )
        or (
            np.isfinite(return_120d)
            and return_120d >= 0.50
            and np.isfinite(ma120_now)
            and c > ma120_now
            and np.nan_to_num(ma120_slope_20d, nan=0.0) > 0
        )
        or (
            np.isfinite(trend_range_120d)
            and trend_range_120d >= 0.45
            and np.isfinite(ma120_now)
            and np.nan_to_num(ma60_slope_20d, nan=0.0) > 0
        )
    )

    lifeline = choose_realtime_lifeline(
        close,
        asof,
        breach_buffer=breach_buffer,
        max_breach_days=3,
    )
    distance_to_lifeline = _safe_float(lifeline.get("distance_to_lifeline"))
    breach_days_5 = int(lifeline.get("lifeline_breach_days_5") or 0)
    breach_days_20 = int(lifeline.get("lifeline_breach_days_20") or 0)

    pullback_ok = bool(
        trend_confirmed
        and np.isfinite(drawdown_20d)
        and -0.18 <= drawdown_20d <= -0.04
        and np.isfinite(distance_to_lifeline)
        and distance_to_lifeline >= -breach_buffer
    )
    warning = bool(trend_confirmed and breach_days_5 in (1, 2))
    broken = bool(trend_confirmed and (breach_days_5 >= 3 or distance_to_lifeline < -0.08))

    if broken:
        state = "趋势破坏"
    elif warning:
        state = "生命线预警"
    elif pullback_ok:
        state = "回调可观察"
    elif is_ignition:
        state = "起爆观察"
    elif trend_confirmed and np.isfinite(distance_to_lifeline) and distance_to_lifeline >= 0:
        state = "生命线健康"
    elif trend_confirmed:
        state = "趋势确认"
    else:
        state = "未启动"

    row: dict[str, object] = {
        "asof_date": asof.strftime("%Y-%m-%d"),
        "trend_state": state,
        "close": c,
        "is_ignition": is_ignition,
        "trend_confirmed": trend_confirmed,
        "return_20d": _safe_float(return_20d),
        "return_60d": _safe_float(return_60d),
        "return_120d": _safe_float(return_120d),
        "drawdown_20d": _safe_float(drawdown_20d),
        "trend_range_120d": _safe_float(trend_range_120d),
        "breakout_pct": c / (prev_high_now + 1e-12) - 1.0 if prev_high_now > 0 else np.nan,
        "amount_ratio": amount_ratio,
        "ma20": ma20_now,
        "ma30": ma30_now,
        "ma60": ma60_now,
        "ma120": ma120_now,
        "ma60_slope_20d": _safe_float(ma60_slope_20d),
        "ma120_slope_20d": _safe_float(ma120_slope_20d),
        **lifeline,
    }
    row["trend_state_score"] = _row_score(row)
    return row


def build_trend_state_table(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    asof_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if asof_date is None:
        asof = close.index.max()
    else:
        asof = pd.Timestamp(asof_date)
        if asof not in close.index:
            asof = close.index[close.index <= asof].max()

    rows = []
    for symbol in close.columns:
        row = classify_trend_state(close[symbol], high[symbol], low[symbol], amount[symbol], asof)
        row["symbol"] = str(symbol).zfill(6)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["trend_state_score", "return_120d", "return_60d"], ascending=[False, False, False])
    return out


def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "symbol": "股票代码",
            "asof_date": "日期",
            "trend_state": "趋势状态",
            "trend_state_score": "趋势状态分",
            "close": "收盘价",
            "is_ignition": "是否起爆",
            "trend_confirmed": "趋势确认",
            "return_20d": "20日收益",
            "return_60d": "60日收益",
            "return_120d": "120日收益",
            "drawdown_20d": "20日高点回撤",
            "breakout_pct": "突破幅度",
            "amount_ratio": "成交额放大倍数",
            "ma20": "MA20",
            "ma30": "MA30",
            "ma60": "MA60",
            "ma120": "MA120",
            "ma60_slope_20d": "MA60的20日斜率",
            "ma120_slope_20d": "MA120的20日斜率",
            "lifeline_ma": "生命线均线",
            "lifeline_value": "生命线价格",
            "distance_to_lifeline": "相对生命线距离",
            "lifeline_breach_days_5": "近5日跌破生命线天数",
            "lifeline_breach_days_20": "近20日跌破生命线天数",
            "breach_days_lookback": "回看期跌破生命线天数",
            "min_distance_lookback": "回看期相对生命线最小距离",
            "avg_distance_lookback": "回看期相对生命线平均距离",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real-time trend state table.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/trend_state_latest")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    table = build_trend_state_table(close, high, low, amount, args.asof_date)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "trend_state.csv", index=False, encoding="utf-8-sig")
    cn = to_chinese_columns(table)
    cn.to_csv(output_dir / "trend_state_cn.csv", index=False, encoding="utf-8-sig")
    actionable_states = ["起爆观察", "回调可观察", "生命线健康"]
    actionable = cn[cn["趋势状态"].isin(actionable_states)].head(args.top_n)
    actionable.to_csv(output_dir / "trend_state_actionable_cn.csv", index=False, encoding="utf-8-sig")

    summary = {
        "asof_date": str(table["asof_date"].iloc[0]) if not table.empty and "asof_date" in table else None,
        "total_symbols": int(len(table)),
        "state_counts": table["trend_state"].value_counts().to_dict() if not table.empty else {},
        "actionable_count": int(table["trend_state"].isin(actionable_states).sum()) if not table.empty else 0,
    }
    (output_dir / "summary.json").write_text(
        pd.Series(summary).to_json(force_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary)
    if not actionable.empty:
        print(actionable.head(30).to_string(index=False))
    print(f"Trend state outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
