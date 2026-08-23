"""TongDaXin .day archive reader.

The files in TDX day archives are fixed-width binary records.  We keep this
reader separate from the main daily_prices table because Shanghai/Shenzhen
codes can collide when represented as six digits only.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import pandas as pd


TDX_DAY_RECORD = struct.Struct("<iiiiifII")
TDX_DAY_COLUMNS = ["market", "symbol", "date", "open", "high", "low", "close", "amount", "volume", "asset_type", "source"]


@dataclass(frozen=True)
class TdxArchiveSpec:
    path: Path
    market: str
    asset_type: str


def parse_tdx_day_bytes(data: bytes, symbol: str, market: str, asset_type: str, source: str) -> pd.DataFrame:
    if len(data) % TDX_DAY_RECORD.size != 0:
        raise ValueError(f"TDX .day data length is not divisible by {TDX_DAY_RECORD.size}: {source}")

    rows = []
    for offset in range(0, len(data), TDX_DAY_RECORD.size):
        date_int, open_i, high_i, low_i, close_i, amount, volume, _reserved = TDX_DAY_RECORD.unpack_from(data, offset)
        try:
            trade_date = date(date_int // 10_000, date_int // 100 % 100, date_int % 100).isoformat()
        except ValueError:
            continue
        rows.append(
            {
                "market": market,
                "symbol": symbol,
                "date": trade_date,
                "open": open_i / 100.0,
                "high": high_i / 100.0,
                "low": low_i / 100.0,
                "close": close_i / 100.0,
                "amount": float(amount),
                "volume": float(volume),
                "asset_type": asset_type,
                "source": source,
            }
        )
    return pd.DataFrame(rows, columns=TDX_DAY_COLUMNS)


def infer_tdx_archive_specs(root: str | Path) -> list[TdxArchiveSpec]:
    root_path = Path(root)
    specs: list[TdxArchiveSpec] = []
    patterns = {
        "shlday.zip": ("SH", "stock"),
        "szlday.zip": ("SZ", "stock"),
        "shzsday.zip": ("SH", "index"),
        "szzsday.zip": ("SZ", "index"),
    }
    for path in sorted(root_path.rglob("*.zip")):
        lowered = path.name.lower()
        if lowered in patterns:
            market, asset_type = patterns[lowered]
            specs.append(TdxArchiveSpec(path=path, market=market, asset_type=asset_type))
    return specs


def read_tdx_day_archive(spec: TdxArchiveSpec, symbol_filter: set[str] | None = None) -> pd.DataFrame:
    frames = list(iter_tdx_day_archive_frames(spec, symbol_filter=symbol_filter))
    if not frames:
        return pd.DataFrame(columns=TDX_DAY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def iter_tdx_day_archive_frames(spec: TdxArchiveSpec, symbol_filter: set[str] | None = None) -> Iterable[pd.DataFrame]:
    for _member, frame in iter_tdx_day_archive_member_frames(spec, symbol_filter=symbol_filter):
        yield frame


def iter_tdx_day_archive_member_frames(
    spec: TdxArchiveSpec,
    symbol_filter: set[str] | None = None,
) -> Iterable[tuple[str, pd.DataFrame]]:
    with ZipFile(spec.path) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".day"):
                continue
            name = Path(member).stem.lower()
            if len(name) < 8:
                continue
            market = name[:2].upper()
            symbol = name[2:8]
            if symbol_filter is not None and symbol not in symbol_filter:
                continue
            if market != spec.market:
                market = spec.market
            yield (
                member,
                parse_tdx_day_bytes(
                    archive.read(member),
                    symbol=symbol,
                    market=market,
                    asset_type=spec.asset_type,
                    source=f"{spec.path}!{member}",
                ),
            )


def read_tdx_archives(specs: Iterable[TdxArchiveSpec]) -> pd.DataFrame:
    frames = [read_tdx_day_archive(spec) for spec in specs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
