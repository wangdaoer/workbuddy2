"""Build a consolidated long-form OHLC panel for the high-risk backtest."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from workspace_paths import daily_data_root, fallback_data_root


DAILY_DATA_ROOT = daily_data_root()
PRIMARY_SNAPSHOTS = (
    DAILY_DATA_ROOT / "ths_exports" / "normalized",
    DAILY_DATA_ROOT / "snapshots",
)
SECONDARY_SNAPSHOTS = (fallback_data_root() / "data" / "normalized",)

THS_FILE_RE = re.compile(r"ths_hs_a_share_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)
SNAPSHOT_RE = re.compile(r"snapshot_all_(\d{8})\.csv$", re.IGNORECASE)
MARKET_SNAPSHOT_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_market_snapshot\.csv$", re.IGNORECASE)


def _read_panel_csv(path: Path) -> pd.DataFrame:
    # symbol 一律按字符串读取：真实 6 位代码（如 '000001'）若被推断为 int 会丢前导零、
    # 跨市场可能碰撞；保持零填充字符串对模型（symbol 为不透明 ID）无损，且利于研究追溯。
    df = pd.read_csv(path, dtype={"symbol": str})
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    cols = set(df.columns.str.lower())
    if "trade_date" in cols and "security_code" in cols:
        # exchange snapshot format
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["trade_date"], errors="coerce"),
                "symbol": df["security_code"].astype(str).str.strip(),
                "open": pd.to_numeric(df.get("open_price"), errors="coerce"),
                "high": pd.to_numeric(df.get("high_price"), errors="coerce"),
                "low": pd.to_numeric(df.get("low_price"), errors="coerce"),
                "close": pd.to_numeric(df.get("close_price"), errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
            }
        )
        return out

    # THS normalized csv
    if (
        "trade_date" in cols
        and "prefixed_security_code" in cols
        and "open_price" in cols
        and "high_price" in cols
        and "low_price" in cols
        and "close_price" in cols
    ):
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["trade_date"], errors="coerce"),
                "symbol": df["prefixed_security_code"].astype(str).str.strip(),
                "open": pd.to_numeric(df["open_price"], errors="coerce"),
                "high": pd.to_numeric(df["high_price"], errors="coerce"),
                "low": pd.to_numeric(df["low_price"], errors="coerce"),
                "close": pd.to_numeric(df["close_price"], errors="coerce"),
                "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
            }
        )
        return out

    if {"date", "open", "high", "low", "close"}.issubset(cols):
        return (
            df.rename(
                columns={
                    "symbol": "symbol",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                }
            )
            .loc[:, ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
            .assign(
                date=lambda x: pd.to_datetime(x["date"], errors="coerce"),
                symbol=lambda x: x["symbol"].astype(str).str.strip(),
                volume=pd.to_numeric(df.get("volume"), errors="coerce"),
                amount=pd.to_numeric(df.get("amount"), errors="coerce"),
            )
        )

    raise ValueError(f"Unsupported schema in {path}: columns={list(df.columns)}")


def _collect_csvs(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.glob("*.csv") if p.is_file())


def _append_if_match(path: Path, files: list[Path]) -> None:
    if (
        THS_FILE_RE.search(path.name)
        or SNAPSHOT_RE.search(path.name)
        or MARKET_SNAPSHOT_RE.search(path.name)
    ):
        files.append(path)


def collect_panel_files(data_dirs: tuple[Path, ...], start_date: Optional[str] = None, end_date: Optional[str] = None) -> list[Path]:
    selected: list[Path] = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for path in _collect_csvs(data_dir):
            _append_if_match(path, selected)

    if not selected:
        return []

    selected = sorted(set(selected), key=lambda p: _file_date_key(p))
    if start_date:
        selected = [p for p in selected if _file_date_key(p) >= start_date.replace("-", "")]
    if end_date:
        selected = [p for p in selected if _file_date_key(p) <= end_date.replace("-", "")]

    return selected


def _file_date_key(path: Path) -> str:
    match = THS_FILE_RE.search(path.name)
    if match:
        return match.group(1).replace("-", "")
    match = SNAPSHOT_RE.search(path.name)
    if match:
        return match.group(1)
    match = MARKET_SNAPSHOT_RE.search(path.name)
    if match:
        return match.group(1).replace("-", "")
    return ""


def build_panel(
    data_dirs: tuple[Path, ...],
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    files = collect_panel_files(data_dirs, start_date=start_date, end_date=end_date)
    if not files:
        raise FileNotFoundError("No compatible snapshot files found in configured directories.")

    frames = []
    for f in files:
        try:
            frames.append(_read_panel_csv(f))
        except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
            warnings.warn(f"Skipping unreadable snapshot {f}: {exc}", RuntimeWarning, stacklevel=2)
            continue

    if not frames:
        raise ValueError("All selected snapshot files failed to parse.")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "symbol", "close"]).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    return out[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified backtest panel from local snapshot sources.")
    parser.add_argument("--start-date", default=None, help="Optional lower date bound, e.g. 2026-06-15.")
    parser.add_argument("--end-date", default=None, help="Optional upper date bound, e.g. 2026-06-26.")
    parser.add_argument("--output", default="data_panel.csv", help="Output CSV path.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If primary sources are empty, fall back to secondary snapshot directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = PRIMARY_SNAPSHOTS
    try:
        panel = build_panel(sources, args.start_date, args.end_date)
    except FileNotFoundError:
        if not args.allow_fallback:
            raise
        panel = build_panel(SECONDARY_SNAPSHOTS, args.start_date, args.end_date)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False, encoding="utf-8")
    print(f"Built panel: rows={len(panel)}, days={panel['date'].nunique()}, symbols={panel['symbol'].nunique()}, path={output}")


if __name__ == "__main__":
    main()
