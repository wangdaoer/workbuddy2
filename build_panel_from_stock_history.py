"""Build long-form OHLCV panel from selected stock history CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from workspace_paths import stock_data_root


DEFAULT_STOCK_DIR = stock_data_root() / "stocks"
DEFAULT_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605")


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols)
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=usecols, encoding="gbk")


def _clean_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)[-6:]


def _allowed_code(code: str, prefixes: tuple[str, ...]) -> bool:
    return any(code.startswith(prefix) for prefix in prefixes)


def load_symbols(path: Path | None, stock_dir: Path, prefixes: tuple[str, ...], limit: int | None) -> list[str]:
    if path is None:
        symbols = [
            _clean_code(p.stem)
            for p in sorted(stock_dir.glob("*.csv"))
            if _allowed_code(_clean_code(p.stem), prefixes)
        ]
        return symbols[:limit] if limit else symbols

    df = _read_csv(path)
    if "symbol" in df:
        raw = df["symbol"]
    elif "code" in df:
        raw = df["code"]
    else:
        raise ValueError(f"Universe file must contain symbol or code column: {path}")
    symbols = [_clean_code(x) for x in raw.dropna().tolist()]
    seen: set[str] = set()
    out = []
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    out = [x for x in out if _allowed_code(x, prefixes)]
    return out[:limit] if limit else out


def load_one(path: Path, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    df = _read_csv(path, usecols=lambda c: c in cols)
    if df.empty or "date" not in df or "close" not in df:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open" not in df:
        df["open"] = df["close"]
    if "high" not in df:
        df["high"] = df["close"]
    if "low" not in df:
        df["low"] = df["close"]
    if "volume" not in df:
        df["volume"] = np.nan
    if "amount" not in df:
        df["amount"] = np.nan
    df["amount"] = df["amount"].fillna(df["volume"] * df["close"])

    df = df.dropna(subset=["date", "close"]).copy()
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    df["symbol"] = symbol
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build backtest panel from selected stock histories.")
    parser.add_argument("--stock-dir", default=str(DEFAULT_STOCK_DIR))
    parser.add_argument("--universe", default=None)
    parser.add_argument("--output", default="data_panel_history.csv")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_dir = Path(args.stock_dir)
    prefixes = tuple(x.strip() for x in args.prefixes.split(",") if x.strip())
    universe = Path(args.universe) if args.universe else None
    symbols = load_symbols(universe, stock_dir, prefixes, args.limit)
    frames = []
    missing = []
    for symbol in symbols:
        path = stock_dir / f"{symbol}.csv"
        if not path.exists():
            missing.append(symbol)
            continue
        frame = load_one(path, symbol, args.start_date, args.end_date)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No panel rows built.")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8")
    print(
        f"Built panel: rows={len(out)}, days={out['date'].nunique()}, "
        f"symbols={out['symbol'].nunique()}, missing={len(missing)}, path={output}"
    )


if __name__ == "__main__":
    main()
