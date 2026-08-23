"""Track high-quality hidden accumulation watchlist outcomes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluate_early_pattern_forward_returns import attach_forward_returns
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix


CN_COLUMNS = {
    "watch_date": "观察日期",
    "asof_date": "信号日期",
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "pattern_type": "观察模式",
    "hidden_accumulation_quality": "隐性吸筹质量",
    "hidden_accumulation_trade_watch": "高质量观察",
    "pattern_score": "观察评分",
    "return_20d": "20日收益",
    "drawdown_20d": "20日高点回撤",
    "distance_ma20": "相对MA20",
    "amount_ratio": "成交额放大倍数",
    "entry_date": "跟踪买入日",
    "entry_open": "跟踪买入开盘价",
    "exit_date_1d": "1日退出日",
    "exit_open_1d": "1日退出开盘价",
    "forward_return_1d": "1日跟踪收益",
    "exit_date_3d": "3日退出日",
    "exit_open_3d": "3日退出开盘价",
    "forward_return_3d": "3日跟踪收益",
    "exit_date_5d": "5日退出日",
    "exit_open_5d": "5日退出开盘价",
    "forward_return_5d": "5日跟踪收益",
    "completed_horizons": "已完成周期",
    "pending_horizons": "待完成周期",
    "tracking_status": "跟踪状态",
}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "高质量观察"}


def _watch_date_from_path(path: Path) -> str | None:
    match = re.search(r"early_pattern_watchlist_(20\d{6})\.csv$", path.name)
    if not match:
        return None
    token = match.group(1)
    return f"{token[:4]}-{token[4:6]}-{token[6:]}"


def _read_watchlist_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, dtype={"symbol": str}, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, dtype={"symbol": str})


def load_high_quality_watchlists(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        watch_date = _watch_date_from_path(Path(path))
        if watch_date is None or not Path(path).exists():
            continue
        table = _read_watchlist_csv(Path(path))
        if "hidden_accumulation_trade_watch" not in table.columns:
            continue
        mask = table["hidden_accumulation_trade_watch"].map(_truthy)
        selected = table.loc[mask].copy()
        if selected.empty:
            continue
        selected["symbol"] = selected["symbol"].astype(str).str.zfill(6)
        selected.insert(0, "watch_date", watch_date)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["watch_date", "symbol"], keep="last")
    sort_cols = [col for col in ["watch_date", "pattern_score", "symbol"] if col in out.columns]
    ascending = [True, False, True][: len(sort_cols)]
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _status_for_row(row: pd.Series, horizons: tuple[int, ...]) -> tuple[str, str, str]:
    if not str(row.get("entry_date", "") or "").strip():
        return "", ",".join(str(h) for h in horizons), "pending_entry"
    completed = [
        str(h)
        for h in horizons
        if pd.notna(pd.to_numeric(row.get(f"forward_return_{h}d"), errors="coerce"))
    ]
    pending = [str(h) for h in horizons if str(h) not in completed]
    if not pending:
        status = "complete"
    elif completed:
        status = "partial"
    else:
        status = "pending_exit"
    return ",".join(completed), ",".join(pending), status


def build_tracking_table(
    watchlist: pd.DataFrame,
    open_px: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    horizons = tuple(int(h) for h in horizons)
    if watchlist.empty:
        return pd.DataFrame()

    tracked: list[pd.DataFrame] = []
    for watch_date, group in watchlist.groupby("watch_date", sort=True):
        tracked.append(attach_forward_returns(group.copy(), open_px, pd.Timestamp(watch_date), horizons=horizons))
    out = pd.concat(tracked, ignore_index=True) if tracked else pd.DataFrame()
    if out.empty:
        return out

    statuses = out.apply(lambda row: _status_for_row(row, horizons), axis=1)
    out["completed_horizons"] = [item[0] for item in statuses]
    out["pending_horizons"] = [item[1] for item in statuses]
    out["tracking_status"] = [item[2] for item in statuses]
    sort_cols = [col for col in ["watch_date", "tracking_status", "symbol"] if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _format_report_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for col in [c for c in display.columns if c.startswith("forward_return_")]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2%}")
    for col in ["return_20d", "drawdown_20d", "distance_ma20"]:
        if col in display:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2%}")
    if "amount_ratio" in display:
        display["amount_ratio"] = display["amount_ratio"].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    return display


def write_tracking_outputs(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    table.rename(columns=CN_COLUMNS).to_csv(
        output.with_name(f"{output.stem}_cn{output.suffix}"),
        index=False,
        encoding="utf-8-sig",
    )

    status_counts = table["tracking_status"].value_counts().to_dict() if "tracking_status" in table else {}
    lines = [
        "# 高质量隐性吸筹跟踪",
        "",
        "口径：收盘后进入观察，下一交易日开盘作为跟踪入口，按1/3/5个交易日后的开盘价计算结果。该表只用于研究复盘，不是自动交易指令。",
        "",
        f"- 跟踪样本数: {len(table)}",
        f"- 跟踪状态: {status_counts}",
    ]
    if not table.empty:
        show_cols = [
            "watch_date",
            "symbol",
            "stock_name",
            "tracking_status",
            "entry_date",
            "forward_return_1d",
            "forward_return_3d",
            "forward_return_5d",
            "pending_horizons",
        ]
        display = _format_report_table(table[[c for c in show_cols if c in table.columns]].head(80))
        lines.extend(["", display.to_markdown(index=False)])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons or any(h <= 0 for h in horizons):
        raise argparse.ArgumentTypeError("Horizons must be positive integers.")
    return horizons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track high-quality hidden accumulation watch outcomes.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--watchlist-dir", default="outputs/high_return_v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 3, 5))
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    open_px = clean_matrix(pivot_prices(raw, "open"), args.max_abs_daily_return)
    watchlist_dir = Path(args.watchlist_dir)
    watchlists = sorted(watchlist_dir.glob("early_pattern_watchlist_20*.csv"))
    high_quality = load_high_quality_watchlists(watchlists)
    tracked = build_tracking_table(high_quality, open_px, horizons=args.horizons)
    output = Path(args.output)
    write_tracking_outputs(tracked, output)
    print(f"Tracked high-quality hidden accumulation rows: {len(tracked)}")
    if not tracked.empty:
        print(tracked[["watch_date", "symbol", "stock_name", "tracking_status", "pending_horizons"]].to_string(index=False))
    print(f"Tracking outputs saved to: {output}")


if __name__ == "__main__":
    main()
