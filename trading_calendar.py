"""Fail-closed trading-session validation using the refreshed benchmark calendar."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_session_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    dates = pd.to_datetime(pd.Index(values), errors="raise").normalize()
    if dates.isna().any():
        raise ValueError("trading-session dates cannot be missing")
    return pd.DatetimeIndex(dates.unique()).sort_values()


def load_benchmark_sessions(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        raise FileNotFoundError(f"benchmark calendar does not exist: {path}")
    frame = pd.read_csv(path, usecols=["date"])
    sessions = normalize_session_dates(frame["date"])
    if sessions.empty:
        raise ValueError("benchmark calendar is empty")
    return sessions


def trading_session_audit(
    panel_dates: Iterable[object],
    benchmark_path: Path,
) -> dict[str, object]:
    panel_sessions = normalize_session_dates(panel_dates)
    if panel_sessions.empty:
        raise ValueError("panel trading calendar is empty")
    benchmark_sessions = load_benchmark_sessions(benchmark_path)
    expected = benchmark_sessions[
        (benchmark_sessions >= panel_sessions.min())
        & (benchmark_sessions <= panel_sessions.max())
    ]
    extra = panel_sessions[~panel_sessions.isin(expected)]
    missing = expected[~expected.isin(panel_sessions)]
    return {
        "panel_first_date": panel_sessions.min().strftime("%Y-%m-%d"),
        "panel_last_date": panel_sessions.max().strftime("%Y-%m-%d"),
        "panel_session_count": int(len(panel_sessions)),
        "benchmark_first_date": benchmark_sessions.min().strftime("%Y-%m-%d"),
        "benchmark_last_date": benchmark_sessions.max().strftime("%Y-%m-%d"),
        "expected_session_count": int(len(expected)),
        "invalid_panel_dates": [value.strftime("%Y-%m-%d") for value in extra],
        "missing_panel_dates": [value.strftime("%Y-%m-%d") for value in missing],
        "passed": bool(len(extra) == 0 and len(missing) == 0),
    }


def validate_trading_sessions(
    panel_dates: Iterable[object],
    benchmark_path: Path,
    *,
    context: str,
) -> dict[str, object]:
    audit = trading_session_audit(panel_dates, benchmark_path)
    if not audit["passed"]:
        raise ValueError(
            f"{context} trading-calendar mismatch: "
            f"invalid_panel_dates={audit['invalid_panel_dates']} "
            f"missing_panel_dates={audit['missing_panel_dates']}"
        )
    return audit
