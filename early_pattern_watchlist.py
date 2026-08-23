"""Scan early observation patterns similar to the user's chart examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_daily_personal_overlay_report import load_name_map
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix


CN_COLUMNS = {
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "pattern_type": "观察模式",
    "pattern_score": "观察评分",
    "pattern_reason": "观察理由",
    "close": "收盘价",
    "return_5d": "5日收益",
    "return_20d": "20日收益",
    "return_60d": "60日收益",
    "return_120d": "120日收益",
    "range_15d": "15日振幅",
    "drawdown_20d": "20日高点回撤",
    "distance_ma20": "相对MA20",
    "distance_ma60": "相对MA60",
    "breakout_20d": "20日突破幅度",
    "amount_ratio": "成交额放大倍数",
    "close_position": "收盘位置",
    "hidden_accumulation_score": "隐性吸筹分",
    "up_day_amount_persistence": "上涨日放量连续性",
    "amount_trend_ratio_20d": "20日成交额趋势比",
    "single_day_amount_spike": "单日成交额尖峰",
}
CN_COLUMNS.update(
    {
        "hidden_accumulation_quality": "隐性吸筹质量",
        "hidden_accumulation_trade_watch": "高质量观察",
    }
)

HIDDEN_TRADE_WATCH_RET20_MIN = 0.08
HIDDEN_TRADE_WATCH_RET20_MAX = 0.15
HIDDEN_TRADE_WATCH_DRAWDOWN20_MIN = -0.08
HIDDEN_TRADE_WATCH_DISTANCE_MA20_MIN = 0.0
HIDDEN_TRADE_WATCH_DISTANCE_MA20_MAX = 0.06
HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MIN = 1.0
HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MAX = 1.6


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def is_high_quality_hidden_accumulation(metrics: dict[str, object]) -> bool:
    ret20 = _safe_float(metrics.get("return_20d"))
    drawdown20 = _safe_float(metrics.get("drawdown_20d"))
    distance_ma20 = _safe_float(metrics.get("distance_ma20"))
    amount_ratio = _safe_float(metrics.get("amount_ratio"))
    return (
        HIDDEN_TRADE_WATCH_RET20_MIN <= ret20 <= HIDDEN_TRADE_WATCH_RET20_MAX
        and drawdown20 >= HIDDEN_TRADE_WATCH_DRAWDOWN20_MIN
        and HIDDEN_TRADE_WATCH_DISTANCE_MA20_MIN <= distance_ma20 <= HIDDEN_TRADE_WATCH_DISTANCE_MA20_MAX
        and HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MIN <= amount_ratio <= HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MAX
    )


def _pct_change(series: pd.Series, window: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= window:
        return np.nan
    return _safe_float(values.iloc[-1] / (values.iloc[-window - 1] + 1e-12) - 1.0)


def _ma(series: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=max(5, window // 2)).mean()


def detect_pattern_row(
    close: pd.Series,
    open_px: pd.Series,
    high: pd.Series,
    low: pd.Series,
    amount: pd.Series,
    asof_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if asof_date is not None:
        close = close.loc[: pd.Timestamp(asof_date)]
    if len(close) < 80:
        return {"pattern_type": ""}

    open_px = pd.to_numeric(open_px, errors="coerce").reindex(close.index).ffill()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index).ffill()
    low = pd.to_numeric(low, errors="coerce").reindex(close.index).ffill()
    amount = pd.to_numeric(amount, errors="coerce").reindex(close.index)

    c = _safe_float(close.iloc[-1])
    o = _safe_float(open_px.iloc[-1])
    h = _safe_float(high.iloc[-1])
    l = _safe_float(low.iloc[-1])
    ma20 = _ma(close, 20)
    ma30 = _ma(close, 30)
    ma60 = _ma(close, 60)
    ma120 = _ma(close, 120)
    ma20_now = _safe_float(ma20.iloc[-1])
    ma30_now = _safe_float(ma30.iloc[-1])
    ma60_now = _safe_float(ma60.iloc[-1])
    ma120_now = _safe_float(ma120.iloc[-1])
    ma20_prev = _safe_float(ma20.iloc[-6]) if len(ma20) > 6 else np.nan
    ma60_prev = _safe_float(ma60.iloc[-21]) if len(ma60) > 21 else np.nan

    ret5 = _pct_change(close, 5)
    ret20 = _pct_change(close, 20)
    ret60 = _pct_change(close, 60)
    ret120 = _pct_change(close, 120)
    high_15 = _safe_float(high.tail(15).max())
    low_15 = _safe_float(low.tail(15).min())
    high_20 = _safe_float(high.tail(20).max())
    high_60 = _safe_float(high.tail(60).max())
    prior_high_20 = _safe_float(high.rolling(20, min_periods=10).max().shift(1).iloc[-1])
    amount_base = _safe_float(amount.tail(21).iloc[:-1].replace(0, np.nan).median())
    amount_ratio = _safe_float(amount.iloc[-1] / (amount_base + 1e-12)) if amount_base > 0 else np.nan
    amount_20 = amount.tail(20).replace(0, np.nan)
    amount_first_half = _safe_float(amount_20.iloc[:10].median())
    amount_second_half = _safe_float(amount_20.iloc[10:].median())
    amount_trend_ratio_20d = (
        amount_second_half / (amount_first_half + 1e-12)
        if amount_first_half > 0 and amount_second_half > 0
        else np.nan
    )
    single_day_amount_spike = (
        _safe_float(amount_20.max() / (amount_20.median() + 1e-12))
        if _safe_float(amount_20.median()) > 0
        else np.nan
    )
    range15 = high_15 / (low_15 + 1e-12) - 1.0 if high_15 > 0 and low_15 > 0 else np.nan
    drawdown20 = c / (high_20 + 1e-12) - 1.0 if high_20 > 0 else np.nan
    breakout20 = c / (prior_high_20 + 1e-12) - 1.0 if prior_high_20 > 0 else np.nan
    distance_ma20 = c / (ma20_now + 1e-12) - 1.0 if ma20_now > 0 else np.nan
    distance_ma60 = c / (ma60_now + 1e-12) - 1.0 if ma60_now > 0 else np.nan
    close_position = (c - l) / (h - l + 1e-12) if h > l else np.nan
    ma60_slope20 = ma60_now / (ma60_prev + 1e-12) - 1.0 if ma60_now > 0 and ma60_prev > 0 else np.nan
    ma20_slope5 = ma20_now / (ma20_prev + 1e-12) - 1.0 if ma20_now > 0 and ma20_prev > 0 else np.nan

    ma120_ok = not np.isfinite(ma120_now) or ma60_now >= ma120_now * 0.92
    trend_intact = (
        c >= ma60_now * 0.94
        and ma20_now >= ma60_now * 0.95
        and ma120_ok
        and np.nan_to_num(ma60_slope20, nan=0.0) >= -0.02
    )
    prior_strength = (
        np.nan_to_num(ret60, nan=0.0) >= 0.30
        or np.nan_to_num(ret120, nan=0.0) >= 0.50
        or c / (close.iloc[-70] + 1e-12) - 1.0 >= 0.30
    )
    platform = (
        prior_strength
        and trend_intact
        and np.nan_to_num(range15, nan=9.0) <= 0.24
        and np.nan_to_num(drawdown20, nan=-9.0) >= -0.18
        and c >= high_20 * 0.90
        and np.nan_to_num(ret5, nan=0.0) <= 0.20
        and np.nan_to_num(distance_ma20, nan=0.0) <= 0.16
    )

    earlier_high = _safe_float(high.iloc[-85:-20].max()) if len(high) >= 85 else _safe_float(high.iloc[:-20].max())
    correction_low = _safe_float(low.iloc[-65:-5].min()) if len(low) >= 70 else np.nan
    correction_depth = correction_low / (earlier_high + 1e-12) - 1.0 if earlier_high > 0 and correction_low > 0 else np.nan
    rebuilt = (
        c >= ma20_now * 0.99
        and c >= ma60_now * 0.97
        and np.nan_to_num(ma20_slope5, nan=0.0) > 0
        and np.nan_to_num(ret20, nan=0.0) >= 0.05
    )
    renewed_breakout = (
        np.nan_to_num(breakout20, nan=-9.0) >= -0.03
        or (high_60 > 0 and c >= high_60 * 0.96)
    )
    not_overheated = (
        np.nan_to_num(ret5, nan=0.0) <= 0.35
        and np.nan_to_num(range15, nan=9.0) <= 0.90
        and np.nan_to_num(distance_ma20, nan=0.0) <= 0.55
        and np.nan_to_num(c / (o + 1e-12) - 1.0 if o > 0 else 0.0, nan=0.0) <= 0.13
    )
    second_wave = (
        trend_intact
        and rebuilt
        and np.nan_to_num(correction_depth, nan=0.0) <= -0.14
        and renewed_breakout
        and np.nan_to_num(amount_ratio, nan=1.0) >= 1.15
        and not_overheated
    )

    daily_return = close.pct_change(fill_method=None)
    recent_amount_median = amount.rolling(20, min_periods=10).median()
    recent = pd.DataFrame(
        {
            "ret": daily_return,
            "amount": amount,
            "amount_median": recent_amount_median,
        }
    ).tail(15)
    up_days = recent["ret"].gt(0)
    up_day_amount_persistence = (
        float(recent.loc[up_days, "amount"].gt(recent.loc[up_days, "amount_median"]).mean())
        if int(up_days.sum()) >= 5
        else np.nan
    )
    last_10_lift = _safe_float(close.iloc[-1] / (close.iloc[-11] + 1e-12) - 1.0) if len(close) > 11 else np.nan
    max_recent_daily_return = _safe_float(daily_return.tail(10).max())
    hidden_accumulation = (
        trend_intact
        and np.nan_to_num(ret20, nan=0.0) >= -0.03
        and 0.02 <= np.nan_to_num(last_10_lift, nan=-9.0) <= 0.22
        and np.nan_to_num(drawdown20, nan=-9.0) >= -0.13
        and np.nan_to_num(distance_ma20, nan=0.0) <= 0.18
        and np.nan_to_num(amount_trend_ratio_20d, nan=0.0) >= 1.12
        and np.nan_to_num(up_day_amount_persistence, nan=0.0) >= 0.55
        and np.nan_to_num(single_day_amount_spike, nan=9.0) <= 2.80
        and np.nan_to_num(max_recent_daily_return, nan=0.0) <= 0.10
        and np.nan_to_num(amount_ratio, nan=1.0) <= 2.40
    )
    hidden_accumulation_raw_score = (
        0.7
        + max(np.nan_to_num(last_10_lift, nan=0.0), 0.0) * 2.0
        + max(np.nan_to_num(amount_trend_ratio_20d, nan=1.0) - 1.0, 0.0) * 0.8
        + max(np.nan_to_num(up_day_amount_persistence, nan=0.0) - 0.5, 0.0)
        + max(np.nan_to_num(distance_ma60, nan=0.0), 0.0) * 0.2
    )
    hidden_accumulation_score = min(float(hidden_accumulation_raw_score), 1.85)

    base = {
        "close": c,
        "return_5d": ret5,
        "return_20d": ret20,
        "return_60d": ret60,
        "return_120d": ret120,
        "range_15d": range15,
        "drawdown_20d": drawdown20,
        "distance_ma20": distance_ma20,
        "distance_ma60": distance_ma60,
        "breakout_20d": breakout20,
        "amount_ratio": amount_ratio,
        "close_position": close_position,
        "correction_depth": correction_depth,
        "intraday_return": c / (o + 1e-12) - 1.0 if o > 0 else np.nan,
        "hidden_accumulation_score": float(hidden_accumulation_score),
        "up_day_amount_persistence": up_day_amount_persistence,
        "amount_trend_ratio_20d": amount_trend_ratio_20d,
        "single_day_amount_spike": single_day_amount_spike,
    }
    high_quality_hidden = is_high_quality_hidden_accumulation(base)
    base["hidden_accumulation_quality"] = "高质量观察" if high_quality_hidden else "普通观察"
    base["hidden_accumulation_trade_watch"] = bool(high_quality_hidden)

    if second_wave:
        score = (
            1.0
            + min(np.nan_to_num(ret20, nan=0.0), 0.8)
            + min(np.nan_to_num(amount_ratio, nan=1.0) - 1.0, 2.0) * 0.2
            + max(np.nan_to_num(breakout20, nan=0.0), 0.0)
        )
        return {
            **base,
            "pattern_type": "趋势二波启动",
            "pattern_score": float(score),
            "pattern_reason": "前期上升后完成调整，重新放量站上均线系统并接近/突破前高",
        }
    if hidden_accumulation:
        return {
            **base,
            "pattern_type": "隐性吸筹观察",
            "pattern_score": float(hidden_accumulation_score),
            "pattern_reason": "连续温和放量且上涨日成交额更活跃，价格缓慢抬升但未出现单日爆量追高",
        }
    if platform:
        score = (
            0.8
            + min(np.nan_to_num(ret60, nan=0.0), 1.0) * 0.4
            + max(0.0, 0.24 - np.nan_to_num(range15, nan=0.24))
            + max(0.0, np.nan_to_num(distance_ma60, nan=0.0)) * 0.2
        )
        return {
            **base,
            "pattern_type": "强势平台蓄势",
            "pattern_score": float(score),
            "pattern_reason": "上涨后横向消化，回撤较浅，短中期均线托住，接近平台上沿",
        }
    return {**base, "pattern_type": "", "pattern_score": 0.0, "pattern_reason": ""}


def scan_early_patterns(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    asof_date: pd.Timestamp | None = None,
    top_n: int = 80,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in close.columns:
        decision = detect_pattern_row(
            close[symbol],
            open_px[symbol],
            high[symbol],
            low[symbol],
            amount[symbol],
            asof_date,
        )
        if decision.get("pattern_type"):
            rows.append({"symbol": str(symbol).zfill(6), **decision})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["pattern_score", "return_60d"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def write_report(output: Path, table: pd.DataFrame) -> Path:
    report = output.with_suffix(".md")
    lines = [
        "# 提前观察形态扫描",
        "",
        "这张表只用于提前观察分析，不直接触发买入或加仓。",
        "",
        f"- 候选数量: {len(table)}",
    ]
    if not table.empty and "pattern_type" in table:
        lines.append(f"- 模式分布: {json.dumps(table['pattern_type'].value_counts().to_dict(), ensure_ascii=False)}")
        show_cols = [
            "symbol",
            "stock_name",
            "pattern_type",
            "pattern_score",
            "pattern_reason",
            "close",
            "return_20d",
            "return_60d",
            "range_15d",
            "drawdown_20d",
            "amount_ratio",
            "hidden_accumulation_score",
            "up_day_amount_persistence",
            "amount_trend_ratio_20d",
            "single_day_amount_spike",
        ]
        lines.extend(["", table[[c for c in show_cols if c in table.columns]].head(50).to_markdown(index=False)])
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate early pattern watchlist.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/high_return_v2/early_pattern_watchlist_latest.csv")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--names-source", default=None)
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
    out = scan_early_patterns(close, open_px, high, low, amount, asof, args.top_n)
    if args.names_source:
        names = load_name_map(Path(args.names_source))
        out.insert(1, "stock_name", out["symbol"].map(names).fillna("") if not out.empty else "")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    cn_output = output.with_name(output.stem + "_cn" + output.suffix)
    out.rename(columns=CN_COLUMNS).to_csv(cn_output, index=False, encoding="utf-8-sig")
    report = write_report(output, out)
    print(f"Generated {len(out)} early pattern rows for {asof.strftime('%Y-%m-%d')}: {output}")
    print(f"Report: {report}")
    if not out.empty:
        print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
