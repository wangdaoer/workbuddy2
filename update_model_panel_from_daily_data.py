"""Merge the historical model panel with normalized daily market exports."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from panel_io import read_panel, write_panel_atomic
from ths_daily_data import normalize_daily_market_file, parse_number, parse_percent_ratio
from trading_calendar import validate_trading_sessions
from workspace_paths import daily_data_root


DEFAULT_BASE_PANEL = Path("data/base_panel.csv")
DEFAULT_DAILY_DIR = daily_data_root() / "ths_exports" / "normalized"
DEFAULT_OUTPUT = Path("data_panel_history_main_chinext_20220101_latest.csv")
DEFAULT_BENCHMARK = daily_data_root() / "benchmarks" / "510300.csv"
DEFAULT_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605")
ZERO_PLACEHOLDER_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
OPTIONAL_POINT_IN_TIME_COLUMNS = (
    "main_net_inflow",
    "main_net_volume_ratio",
)


@dataclass(frozen=True)
class DailySummary:
    date: str
    file: str
    rows: int
    symbols: int
    amount_source: str
    raw_positive_ratio: float
    money_flow_source: str = "unavailable"
    money_flow_ratio_source: str = "unavailable"


def clean_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else ""


def resolve_base_panel(path: Path) -> Path:
    if path.exists() or path != DEFAULT_BASE_PANEL:
        return path
    prefix = "data_panel_history_main_chinext_20220101_"
    panels = sorted(
        candidate
        for suffix in ("csv", "parquet")
        for candidate in Path.cwd().glob(f"{prefix}*.{suffix}")
        if len(candidate.stem.removeprefix(prefix)) == 8
        and candidate.stem.removeprefix(prefix).isdigit()
    )
    return panels[-1] if panels else path


def drop_all_zero_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ZERO_PLACEHOLDER_COLUMNS if column in df.columns]
    if len(columns) != len(ZERO_PLACEHOLDER_COLUMNS):
        return df
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    mask = values.notna().all(axis=1) & values.eq(0.0).all(axis=1)
    return df.loc[~mask].copy()


def allowed_code(code: str, prefixes: tuple[str, ...]) -> bool:
    return bool(code) and any(code.startswith(prefix) for prefix in prefixes)


def _numeric(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype(float)


def _estimate_amount(market_cap: pd.Series, turnover_rate: pd.Series) -> pd.Series:
    amount = pd.to_numeric(market_cap, errors="coerce") * pd.to_numeric(turnover_rate, errors="coerce") / 100.0
    return amount.where(amount.gt(0))


def _finalize(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].map(clean_code)
    frame = frame[frame["symbol"].map(lambda x: allowed_code(x, prefixes))]
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in OPTIONAL_POINT_IN_TIME_COLUMNS:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["amount"] = frame["amount"].where(frame["amount"].gt(0))
    frame["volume"] = frame["volume"].where(frame["volume"].gt(0), frame["amount"] / frame["close"])
    frame = frame.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
    frame = frame[
        frame["open"].gt(0)
        & frame["high"].gt(0)
        & frame["low"].gt(0)
        & frame["close"].gt(0)
        & frame["volume"].gt(0)
        & frame["amount"].gt(0)
    ]
    columns = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    columns.extend(column for column in OPTIONAL_POINT_IN_TIME_COLUMNS if column in frame)
    return frame[columns]


def read_daily_csv(path: Path, trade_date: str, prefixes: tuple[str, ...]) -> tuple[pd.DataFrame, DailySummary]:
    frame, sources = normalize_daily_market_file(path, trade_date)
    tradable = frame["close"].gt(0)
    raw_amount = frame["amount"]
    raw_positive_ratio = float(raw_amount[tradable].gt(0).mean()) if tradable.any() else 0.0
    raw_volume_ratio = float(frame["volume"][tradable].gt(0).mean()) if tradable.any() else 0.0
    raw_amount_usable = (
        raw_positive_ratio >= 0.90
        and raw_volume_ratio >= 0.80
        and float(raw_amount[raw_amount.gt(0)].median() or 0.0) >= 1_000_000.0
    )
    amount_source = "raw_amount"
    if not raw_amount_usable:
        frame["amount"] = _estimate_amount(frame["market_cap"], frame["turnover_rate"])
        frame["volume"] = frame["amount"] / frame["close"]
        amount_source = "market_cap_x_turnover_rate"

    out = _finalize(frame, prefixes)
    summary = DailySummary(
        date=trade_date,
        file=str(path),
        rows=int(len(out)),
        symbols=int(out["symbol"].nunique()),
        amount_source=amount_source,
        raw_positive_ratio=raw_positive_ratio,
        money_flow_source=sources.money_flow,
        money_flow_ratio_source=sources.money_flow_ratio,
    )
    return out, summary


def read_daily_xls(path: Path, trade_date: str, prefixes: tuple[str, ...]) -> tuple[pd.DataFrame, DailySummary]:
    frame, sources = normalize_daily_market_file(path, trade_date)
    tradable = frame["close"].gt(0)
    raw_positive_ratio = float(frame["amount"][tradable].gt(0).mean()) if tradable.any() else 0.0
    amount_source = "raw_amount"
    if raw_positive_ratio < 0.90:
        frame["amount"] = _estimate_amount(frame["market_cap"], frame["turnover_rate"])
        amount_source = "market_cap_x_turnover_rate"
    frame["volume"] = frame["volume"].where(frame["volume"].gt(0), frame["amount"] / frame["close"])

    out = _finalize(frame, prefixes)
    summary = DailySummary(
        date=trade_date,
        file=str(path),
        rows=int(len(out)),
        symbols=int(out["symbol"].nunique()),
        amount_source=amount_source,
        raw_positive_ratio=raw_positive_ratio,
        money_flow_source=(
            sources.money_flow if frame["main_net_inflow"].notna().any() else "unavailable"
        ),
        money_flow_ratio_source=(
            sources.money_flow_ratio
            if frame["main_net_volume_ratio"].notna().any()
            else "unavailable"
        ),
    )
    return out, summary


def date_from_name(path: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot infer trade date from filename: {path}")
    return match.group(1)


def select_daily_files(daily_dir: Path, start_date: str, end_date: str | None) -> list[Path]:
    candidates = sorted(daily_dir.glob("ths_hs_a_share_*.csv")) + sorted(
        daily_dir.glob("ths_hs_a_share_*.xls")
    )
    by_date: dict[str, list[Path]] = {}
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else None
    for path in candidates:
        if not re.fullmatch(
            r"ths_hs_a_share_\d{4}-\d{2}-\d{2}\.(?:csv|xls)", path.name
        ):
            continue
        trade_date = pd.Timestamp(date_from_name(path))
        if trade_date < start or (end is not None and trade_date > end):
            continue
        by_date.setdefault(trade_date.strftime("%Y-%m-%d"), []).append(path)

    selected = []
    for trade_date, files in sorted(by_date.items()):
        files = sorted(files, key=lambda p: 0 if p.suffix.lower() == ".csv" else 1)
        selected.append(files[0])
    return selected


def load_daily_panel(
    daily_dir: Path,
    start_date: str,
    end_date: str | None,
    prefixes: tuple[str, ...],
) -> tuple[pd.DataFrame, list[DailySummary]]:
    frames = []
    summaries = []
    for path in select_daily_files(daily_dir, start_date, end_date):
        trade_date = date_from_name(path)
        if path.suffix.lower() == ".csv":
            frame, summary = read_daily_csv(path, trade_date, prefixes)
            xls_path = path.with_suffix(".xls")
            if xls_path.exists():
                xls_frame, xls_summary = read_daily_xls(xls_path, trade_date, prefixes)
                point_in_time_columns = [
                    column
                    for column in OPTIONAL_POINT_IN_TIME_COLUMNS
                    if column in xls_frame and xls_frame[column].notna().any()
                ]
                if point_in_time_columns:
                    flow = xls_frame[["date", "symbol", *point_in_time_columns]]
                    frame = frame.drop(columns=point_in_time_columns, errors="ignore").merge(
                        flow,
                        on=["date", "symbol"],
                        how="left",
                        validate="one_to_one",
                    )
                    summary = replace(
                        summary,
                        money_flow_source=f"xls:{xls_summary.money_flow_source}",
                        money_flow_ratio_source=(
                            f"xls:{xls_summary.money_flow_ratio_source}"
                        ),
                    )
        else:
            frame, summary = read_daily_xls(path, trade_date, prefixes)
        if not frame.empty:
            frames.append(frame)
        summaries.append(summary)
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                *OPTIONAL_POINT_IN_TIME_COLUMNS,
            ]
        ), summaries
    return pd.concat(frames, ignore_index=True), summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the model panel with daily normalized exports.")
    parser.add_argument("--base-panel", default=str(DEFAULT_BASE_PANEL))
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--daily-start", default="2000-01-01")
    parser.add_argument("--daily-end", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = resolve_base_panel(Path(args.base_panel))
    daily_dir = Path(args.daily_dir)
    output = Path(args.output)
    prefixes = tuple(x.strip() for x in args.prefixes.split(",") if x.strip())

    base = drop_all_zero_placeholders(read_panel(base_path, parse_dates=["date"]))
    base["symbol"] = base["symbol"].map(clean_code)
    base_cut = base[base["date"] < pd.Timestamp(args.daily_start)].copy()
    daily, summaries = load_daily_panel(daily_dir, args.daily_start, args.daily_end, prefixes)
    daily = drop_all_zero_placeholders(daily)
    if daily.empty:
        raise ValueError(f"No daily rows loaded from {daily_dir}")

    out = pd.concat([base_cut, daily], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].map(clean_code)
    out = out.dropna(subset=["date", "symbol"])
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    calendar_audit = validate_trading_sessions(
        out["date"], Path(args.benchmark), context="updated model panel"
    )
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_panel_atomic(out, output)

    print(
        f"Updated panel rows={len(out)} days={out['date'].nunique()} "
        f"symbols={out['symbol'].nunique()} latest={out['date'].max()} "
        f"calendar_sessions={calendar_audit['panel_session_count']} path={output}"
    )
    for summary in summaries:
        print(
            f"{summary.date} rows={summary.rows} symbols={summary.symbols} "
            f"amount_source={summary.amount_source} raw_positive_ratio={summary.raw_positive_ratio:.3f} "
            f"money_flow_source={summary.money_flow_source} "
            f"money_flow_ratio_source={summary.money_flow_ratio_source} "
            f"file={Path(summary.file).name}"
        )


if __name__ == "__main__":
    main()
