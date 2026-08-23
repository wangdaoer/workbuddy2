"""Evaluate forward returns after early observation pattern signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from early_pattern_watchlist import (
    HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MAX,
    HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MIN,
    HIDDEN_TRADE_WATCH_DISTANCE_MA20_MAX,
    HIDDEN_TRADE_WATCH_DISTANCE_MA20_MIN,
    HIDDEN_TRADE_WATCH_DRAWDOWN20_MIN,
    HIDDEN_TRADE_WATCH_RET20_MAX,
    HIDDEN_TRADE_WATCH_RET20_MIN,
    scan_early_patterns,
)
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix


HIDDEN_ACCUMULATION_PATTERN = "隐性吸筹观察"

Scanner = Callable[
    [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, int],
    pd.DataFrame,
]


def clean_symbol(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else text


def _date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_return(exit_open: object, entry_open: object) -> float:
    try:
        exit_value = float(exit_open)
        entry_value = float(entry_open)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(exit_value) or not np.isfinite(entry_value) or entry_value <= 0:
        return np.nan
    return exit_value / entry_value - 1.0


def attach_forward_returns(
    candidates: pd.DataFrame,
    open_px: pd.DataFrame,
    asof_date: pd.Timestamp,
    *,
    horizons: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    horizons = tuple(int(h) for h in horizons)
    if candidates.empty:
        out = candidates.copy()
        for horizon in horizons:
            out[f"exit_date_{horizon}d"] = []
            out[f"exit_open_{horizon}d"] = []
            out[f"forward_return_{horizon}d"] = []
        return out

    open_px = open_px.sort_index()
    dates = pd.Index(pd.to_datetime(open_px.index)).sort_values()
    asof = pd.Timestamp(asof_date)
    asof_pos = int(dates.searchsorted(asof, side="right")) - 1
    entry_pos = asof_pos + 1

    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        symbol = clean_symbol(row.get("symbol", ""))
        record = row.to_dict()
        record["symbol"] = symbol
        record["asof_date"] = _date_text(asof)

        entry_open = np.nan
        entry_date = None
        if symbol in open_px.columns and 0 <= entry_pos < len(dates):
            entry_date = dates[entry_pos]
            entry_open = open_px.at[entry_date, symbol]
        record["entry_date"] = _date_text(entry_date)
        record["entry_open"] = float(entry_open) if pd.notna(entry_open) else np.nan

        for horizon in horizons:
            exit_pos = entry_pos + horizon
            exit_date = dates[exit_pos] if symbol in open_px.columns and 0 <= exit_pos < len(dates) else None
            exit_open = open_px.at[exit_date, symbol] if exit_date is not None else np.nan
            record[f"exit_date_{horizon}d"] = _date_text(exit_date)
            record[f"exit_open_{horizon}d"] = float(exit_open) if pd.notna(exit_open) else np.nan
            record[f"forward_return_{horizon}d"] = _safe_return(exit_open, entry_open)
        rows.append(record)

    return pd.DataFrame(rows)


def evaluate_pattern_forward_returns(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    *,
    asof_dates: Iterable[pd.Timestamp],
    horizons: Iterable[int] = (1, 3, 5),
    pattern_types: tuple[str, ...] | None = (HIDDEN_ACCUMULATION_PATTERN,),
    top_n: int = 80,
    scanner: Scanner = scan_early_patterns,
) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for asof_date in asof_dates:
        asof = pd.Timestamp(asof_date)
        candidates = scanner(close, open_px, high, low, amount, asof, top_n)
        if candidates.empty:
            continue
        if pattern_types is not None and "pattern_type" in candidates.columns:
            candidates = candidates[candidates["pattern_type"].isin(pattern_types)].copy()
        if candidates.empty:
            continue
        samples.append(attach_forward_returns(candidates, open_px, asof, horizons=horizons))
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


def scan_hidden_accumulation_history(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    *,
    asof_dates: Iterable[pd.Timestamp],
    top_n: int = 80,
) -> pd.DataFrame:
    close = close.apply(pd.to_numeric, errors="coerce").sort_index()
    open_px = open_px.apply(pd.to_numeric, errors="coerce").reindex_like(close)
    high = high.apply(pd.to_numeric, errors="coerce").reindex_like(close)
    low = low.apply(pd.to_numeric, errors="coerce").reindex_like(close)
    amount = amount.apply(pd.to_numeric, errors="coerce").reindex_like(close)
    amount_nonzero = amount.replace(0, np.nan)

    ma20 = close.rolling(20, min_periods=10).mean()
    ma60 = close.rolling(60, min_periods=30).mean()
    ma120 = close.rolling(120, min_periods=60).mean()
    ma60_slope20 = ma60 / (ma60.shift(20) + 1e-12) - 1.0
    ret20 = close.pct_change(20, fill_method=None)
    ret60 = close.pct_change(60, fill_method=None)
    high20 = high.rolling(20, min_periods=10).max()
    drawdown20 = close / (high20 + 1e-12) - 1.0
    distance_ma20 = close / (ma20 + 1e-12) - 1.0
    distance_ma60 = close / (ma60 + 1e-12) - 1.0
    amount_base = amount_nonzero.shift(1).rolling(20, min_periods=10).median()
    amount_ratio = amount / (amount_base + 1e-12)
    amount_first_half = amount_nonzero.shift(10).rolling(10, min_periods=5).median()
    amount_second_half = amount_nonzero.rolling(10, min_periods=5).median()
    amount_trend_ratio_20d = amount_second_half / (amount_first_half + 1e-12)
    amount_median_20d = amount_nonzero.rolling(20, min_periods=10).median()
    single_day_amount_spike = amount_nonzero.rolling(20, min_periods=10).max() / (amount_median_20d + 1e-12)

    daily_return = close.pct_change(fill_method=None)
    recent_amount_median = amount_nonzero.rolling(20, min_periods=10).median()
    up_days = daily_return.gt(0)
    active_up_days = up_days & amount.gt(recent_amount_median)
    up_count = up_days.astype(float).rolling(15, min_periods=1).sum()
    active_up_count = active_up_days.astype(float).rolling(15, min_periods=1).sum()
    up_day_amount_persistence = (active_up_count / (up_count + 1e-12)).where(up_count.ge(5))
    last_10_lift = close / (close.shift(10) + 1e-12) - 1.0
    max_recent_daily_return = daily_return.rolling(10, min_periods=5).max()

    ma120_ok = ma120.isna() | ma60.ge(ma120 * 0.92)
    trend_intact = (
        close.ge(ma60 * 0.94)
        & ma20.ge(ma60 * 0.95)
        & ma120_ok
        & ma60_slope20.fillna(0.0).ge(-0.02)
    )
    hidden = (
        trend_intact
        & ret20.fillna(0.0).ge(-0.03)
        & last_10_lift.fillna(-9.0).ge(0.02)
        & last_10_lift.fillna(-9.0).le(0.22)
        & drawdown20.fillna(-9.0).ge(-0.13)
        & distance_ma20.fillna(0.0).le(0.18)
        & amount_trend_ratio_20d.fillna(0.0).ge(1.12)
        & up_day_amount_persistence.fillna(0.0).ge(0.55)
        & single_day_amount_spike.fillna(9.0).le(2.80)
        & max_recent_daily_return.fillna(0.0).le(0.10)
        & amount_ratio.fillna(1.0).le(2.40)
    )
    hidden_score = (
        0.7
        + last_10_lift.clip(lower=0.0).fillna(0.0) * 2.0
        + (amount_trend_ratio_20d - 1.0).clip(lower=0.0).fillna(0.0) * 0.8
        + (up_day_amount_persistence - 0.5).clip(lower=0.0).fillna(0.0)
        + distance_ma60.clip(lower=0.0).fillna(0.0) * 0.2
    ).clip(upper=1.85)
    trade_watch = (
        ret20.ge(HIDDEN_TRADE_WATCH_RET20_MIN)
        & ret20.le(HIDDEN_TRADE_WATCH_RET20_MAX)
        & drawdown20.ge(HIDDEN_TRADE_WATCH_DRAWDOWN20_MIN)
        & distance_ma20.ge(HIDDEN_TRADE_WATCH_DISTANCE_MA20_MIN)
        & distance_ma20.le(HIDDEN_TRADE_WATCH_DISTANCE_MA20_MAX)
        & amount_ratio.ge(HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MIN)
        & amount_ratio.le(HIDDEN_TRADE_WATCH_AMOUNT_RATIO_MAX)
    )

    feature_frames = {
        "close": close,
        "return_20d": ret20,
        "return_60d": ret60,
        "drawdown_20d": drawdown20,
        "distance_ma20": distance_ma20,
        "distance_ma60": distance_ma60,
        "amount_ratio": amount_ratio,
        "hidden_accumulation_score": hidden_score,
        "up_day_amount_persistence": up_day_amount_persistence,
        "amount_trend_ratio_20d": amount_trend_ratio_20d,
        "single_day_amount_spike": single_day_amount_spike,
    }

    date_index = pd.Index(pd.to_datetime(close.index)).sort_values()
    rows: list[pd.DataFrame] = []
    for requested_asof in asof_dates:
        pos = int(date_index.searchsorted(pd.Timestamp(requested_asof), side="right")) - 1
        if pos < 0:
            continue
        asof = date_index[pos]
        mask = hidden.loc[asof].fillna(False)
        if not bool(mask.any()):
            continue
        symbols = mask[mask].index.map(clean_symbol)
        table = pd.DataFrame({"symbol": symbols})
        table["asof_date"] = _date_text(asof)
        table["pattern_type"] = HIDDEN_ACCUMULATION_PATTERN
        table["pattern_score"] = hidden_score.loc[asof, mask.index[mask]].to_numpy(dtype=float)
        table["pattern_reason"] = "连续温和放量且上涨日成交额更活跃，价格缓慢抬升但未出现单日爆量追高"
        trade_values = trade_watch.loc[asof, mask.index[mask]].fillna(False).to_numpy(dtype=bool)
        table["hidden_accumulation_trade_watch"] = trade_values
        table["hidden_accumulation_quality"] = np.where(trade_values, "高质量观察", "普通观察")
        for name, frame in feature_frames.items():
            table[name] = frame.loc[asof, mask.index[mask]].to_numpy(dtype=float)
        table = table.sort_values(["pattern_score", "return_60d"], ascending=[False, False]).head(top_n)
        rows.append(table.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_forward_returns(samples: pd.DataFrame, *, horizons: Iterable[int] = (1, 3, 5)) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "pattern_type",
                "horizon_days",
                "sample_count",
                "avg_return",
                "median_return",
                "win_rate",
                "p25_return",
                "p75_return",
                "avg_pattern_score",
            ]
        )

    rows: list[dict[str, object]] = []
    for pattern_type, group in samples.groupby("pattern_type", dropna=False):
        for horizon in horizons:
            col = f"forward_return_{int(horizon)}d"
            values = pd.to_numeric(group.get(col), errors="coerce").dropna()
            rows.append(
                {
                    "pattern_type": pattern_type,
                    "horizon_days": int(horizon),
                    "sample_count": int(len(values)),
                    "avg_return": float(values.mean()) if len(values) else np.nan,
                    "median_return": float(values.median()) if len(values) else np.nan,
                    "win_rate": float(values.gt(0).mean()) if len(values) else np.nan,
                    "p25_return": float(values.quantile(0.25)) if len(values) else np.nan,
                    "p75_return": float(values.quantile(0.75)) if len(values) else np.nan,
                    "avg_pattern_score": float(pd.to_numeric(group.get("pattern_score"), errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["pattern_type", "horizon_days"]).reset_index(drop=True)


def _format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["avg_return", "median_return", "win_rate", "p25_return", "p75_return"]:
        if col in out:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2%}")
    if "avg_pattern_score" in out:
        out["avg_pattern_score"] = out["avg_pattern_score"].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return out


def write_forward_return_report(samples: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, token: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"early_pattern_forward_returns_{token}.csv"
    summary_path = output_dir / f"early_pattern_forward_summary_{token}.csv"
    trade_watch_path = output_dir / f"early_pattern_forward_trade_watch_{token}.csv"
    report_path = output_dir / f"early_pattern_forward_report_{token}.md"

    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trade_watch = (
        samples[samples["hidden_accumulation_trade_watch"].fillna(False).astype(bool)].copy()
        if "hidden_accumulation_trade_watch" in samples
        else pd.DataFrame()
    )
    trade_watch.to_csv(trade_watch_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 提前观察形态历史收益检验",
        "",
        "口径：收盘后识别形态，下一交易日开盘买入，持有指定交易日后按开盘价退出；该报告只用于研究观察，不是自动交易指令。",
        "",
        f"- 样本数: {len(samples)}",
    ]
    if not samples.empty and "asof_date" in samples:
        lines.append(f"- 样本区间: {samples['asof_date'].min()} 至 {samples['asof_date'].max()}")
    if not summary.empty:
        lines.extend(["", "## 汇总", "", _format_percent_columns(summary).to_markdown(index=False)])
    if not trade_watch.empty:
        trade_watch_summary = summarize_forward_returns(
            trade_watch,
            horizons=sorted(
                int(col.removeprefix("forward_return_").removesuffix("d"))
                for col in trade_watch.columns
                if col.startswith("forward_return_") and col.endswith("d")
            ),
        )
        lines.extend(
            [
                "",
                "## 高质量观察子集",
                "",
                "规则：20日涨幅8%-15%，20日回撤不深于8%，相对MA20在0%-6%，当日成交额为前期中位数的1.0-1.6倍。",
                "",
                _format_percent_columns(trade_watch_summary).to_markdown(index=False),
            ]
        )
    if not samples.empty:
        show_cols = [
            "asof_date",
            "symbol",
            "stock_name",
            "pattern_type",
            "pattern_score",
            "entry_date",
            "forward_return_1d",
            "forward_return_3d",
            "forward_return_5d",
        ]
        display = samples[[c for c in show_cols if c in samples.columns]].head(80).copy()
        for col in [c for c in display.columns if c.startswith("forward_return_")]:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2%}")
        lines.extend(["", "## 样本预览", "", display.to_markdown(index=False)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_asof_dates(
    dates: pd.Index,
    *,
    start: str | None = None,
    end: str | None = None,
    step: int = 5,
    warmup: int = 120,
) -> list[pd.Timestamp]:
    idx = pd.Index(pd.to_datetime(dates)).sort_values()
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    if warmup > 0:
        full_idx = pd.Index(pd.to_datetime(dates)).sort_values()
        idx = idx[full_idx.searchsorted(idx) >= warmup]
    return [pd.Timestamp(value) for value in idx[:: max(int(step), 1)]]


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons:
        raise argparse.ArgumentTypeError("At least one horizon is required.")
    if any(h <= 0 for h in horizons):
        raise argparse.ArgumentTypeError("Horizons must be positive integers.")
    return horizons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate early pattern forward returns.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/early_pattern_forward_test_latest")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 3, 5))
    parser.add_argument("--pattern-type", default=HIDDEN_ACCUMULATION_PATTERN)
    parser.add_argument("--all-patterns", action="store_true")
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    asof_dates = sample_asof_dates(close.index, start=args.start, end=args.end, step=args.step)
    pattern_types = None if args.all_patterns else (args.pattern_type,)
    if pattern_types == (HIDDEN_ACCUMULATION_PATTERN,):
        hidden_candidates = scan_hidden_accumulation_history(
            close,
            open_px,
            high,
            low,
            amount,
            asof_dates=asof_dates,
            top_n=args.top_n,
        )
        samples = (
            pd.concat(
                [
                    attach_forward_returns(group, open_px, pd.Timestamp(asof), horizons=args.horizons)
                    for asof, group in hidden_candidates.groupby("asof_date", sort=True)
                ],
                ignore_index=True,
            )
            if not hidden_candidates.empty
            else pd.DataFrame()
        )
    else:
        samples = evaluate_pattern_forward_returns(
            close,
            open_px,
            high,
            low,
            amount,
            asof_dates=asof_dates,
            horizons=args.horizons,
            pattern_types=pattern_types,
            top_n=args.top_n,
        )
    summary = summarize_forward_returns(samples, horizons=args.horizons)
    token = args.token or (pd.Timestamp(args.end).strftime("%Y%m%d") if args.end else pd.Timestamp(close.index.max()).strftime("%Y%m%d"))
    output_dir = Path(args.output_dir)
    write_forward_return_report(samples, summary, output_dir, token)
    print(summary.to_string(index=False) if not summary.empty else "No forward-return samples.")
    print(f"Forward-return report saved to: {output_dir}")


if __name__ == "__main__":
    main()
