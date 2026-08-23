from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


def configured_path(env_name: str, relative_default: str | Path) -> Path:
    value = os.environ.get(env_name, "").strip()
    return Path(value).expanduser() if value else Path(relative_default)


def is_absolute_path(path: str | Path) -> bool:
    text = os.fspath(path)
    return (
        Path(text).is_absolute()
        or PurePosixPath(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
    )


def resolve_workspace_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if is_absolute_path(candidate) else root / candidate


def daily_data_root() -> Path:
    return configured_path("QUANT_DATA_ROOT", "external_data/daily-market-data")


def fallback_data_root() -> Path:
    return configured_path("QUANT_FALLBACK_ROOT", "external_data/exchange-data-ingest")


def stock_data_root() -> Path:
    return configured_path("QUANT_STOCK_DATA_ROOT", "external_data/stock-data")


def tdx_data_root() -> Path:
    return configured_path("QUANT_TDX_ROOT", "external_data/tdx")


def dashboard_root() -> Path:
    return configured_path("QUANT_DASHBOARD_ROOT", "external_data/stock-analysis-dashboard")


def personal_trades_file() -> Path:
    return configured_path("QUANT_PERSONAL_TRADES_FILE", "inputs/personal_trades.xls")
