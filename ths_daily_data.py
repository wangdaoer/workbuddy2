"""Shared point-in-time parser for normalized and THS daily market exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NORMALIZED_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "market_cap",
    "main_net_inflow",
    "main_net_volume_ratio",
)


@dataclass(frozen=True)
class DailyMarketSources:
    date: str
    volume: str
    amount: str
    money_flow: str
    money_flow_ratio: str


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text in {"--", "-", "nan", "None"}:
        return np.nan
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    if text.endswith("亿"):
        multiplier = 100_000_000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10_000.0
        text = text[:-1]
    text = text.replace(",", "").replace("↑", "").replace("↓", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) * multiplier if match else np.nan


def parse_percent_ratio(value: object) -> float:
    """Convert a percentage-point field such as ``0.77`` to ``0.0077``."""

    parsed = parse_number(value)
    return parsed / 100.0 if np.isfinite(parsed) else np.nan


def _read_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        frame = pd.read_csv(path, sep="\t", dtype=str, encoding="gb18030")
    elif suffix == ".xlsx":
        frame = pd.read_excel(path, dtype=str)
    elif suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported daily market file: {path}")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> tuple[pd.Series, str]:
    for name in names:
        if name in frame.columns:
            return frame[name], name
    return pd.Series(np.nan, index=frame.index, dtype=float), "unavailable"


def _numbers(frame: pd.DataFrame, names: tuple[str, ...]) -> tuple[pd.Series, str]:
    values, source = _first_column(frame, names)
    return values.map(parse_number).astype(float), source


def normalize_daily_market_file(
    path: Path,
    trade_date: str,
) -> tuple[pd.DataFrame, DailyMarketSources]:
    """Return one source-independent daily schema without applying usability filters."""

    raw = _read_raw(path)
    date_values, date_source = _first_column(raw, ("trade_date", "date"))
    if date_source == "unavailable":
        dates = pd.Series(pd.Timestamp(trade_date), index=raw.index)
        date_source = "filename"
    else:
        dates = pd.to_datetime(date_values, errors="coerce").fillna(pd.Timestamp(trade_date))

    symbols, _ = _first_column(raw, ("security_code", "raw_security_code", "symbol", "代码"))
    close, _ = _numbers(raw, ("close", "close_price", "现价"))
    open_values, open_source = _numbers(raw, ("open", "open_price", "开盘"))
    high, high_source = _numbers(raw, ("high", "high_price", "最高"))
    low, low_source = _numbers(raw, ("low", "low_price", "最低"))
    if open_source == "unavailable":
        open_values = close.copy()
    if high_source == "unavailable":
        high = close.copy()
    if low_source == "unavailable":
        low = close.copy()

    volume, volume_source = _numbers(raw, ("volume", "总手", "总成交量", "成交量"))
    amount, amount_source = _numbers(raw, ("amount", "turnover", "总金额", "成交额"))
    turnover_rate, _ = _numbers(raw, ("turnover_rate", "换手"))
    market_cap, _ = _numbers(raw, ("market_cap", "总市值"))
    main_net_inflow, flow_source = _numbers(
        raw,
        ("main_net_inflow", "net_money_flow", "net_mf_amount", "money_flow", "大单净额"),
    )
    ratio_values, ratio_source = _first_column(
        raw,
        ("main_net_volume_ratio", "main_net_volume_ratio_pct", "主力净量"),
    )
    ratio_parser = parse_number if ratio_source == "main_net_volume_ratio" else parse_percent_ratio
    main_net_volume_ratio = ratio_values.map(ratio_parser).astype(float)

    normalized = pd.DataFrame(
        {
            "date": dates,
            "symbol": symbols,
            "open": open_values,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "turnover_rate": turnover_rate,
            "market_cap": market_cap,
            "main_net_inflow": main_net_inflow,
            "main_net_volume_ratio": main_net_volume_ratio,
        }
    )
    sources = DailyMarketSources(
        date=date_source,
        volume=volume_source,
        amount=amount_source,
        money_flow=flow_source,
        money_flow_ratio=ratio_source,
    )
    normalized = normalized.loc[:, NORMALIZED_COLUMNS]
    normalized.attrs["daily_market_sources"] = {
        "date": sources.date,
        "volume": sources.volume,
        "amount": sources.amount,
        "money_flow": sources.money_flow,
        "money_flow_ratio": sources.money_flow_ratio,
    }
    return normalized, sources
