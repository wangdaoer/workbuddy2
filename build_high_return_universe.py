"""Build a high-return candidate universe from local stock history CSVs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from workspace_paths import stock_data_root


DEFAULT_STOCK_DIR = stock_data_root() / "stocks"
DEFAULT_MCAP = stock_data_root() / "stock_market_cap_yi.csv"
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


def _allowed_code(code: str, prefixes: Iterable[str]) -> bool:
    return any(code.startswith(prefix) for prefix in prefixes)


def _return_at(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return np.nan
    base = close.iloc[-periods - 1]
    if not np.isfinite(base) or base <= 0:
        return np.nan
    return float(close.iloc[-1] / base - 1.0)


def _max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    peak = close.cummax()
    dd = close / peak - 1.0
    return float(dd.min())


def score_one_file(path: Path, asof: pd.Timestamp | None, min_history_days: int) -> dict[str, object] | None:
    code = _clean_code(path.stem)
    cols = ["date", "code", "name", "open", "high", "low", "close", "volume", "amount"]
    try:
        df = _read_csv(path, usecols=lambda c: c in cols)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        warnings.warn(f"Skipping unreadable stock history {path}: {exc}", RuntimeWarning, stacklevel=2)
        return None
    if df.empty or "date" not in df or "close" not in df:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if asof is not None:
        df = df[df["date"] <= asof]
    if len(df) < min_history_days:
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df else close
    amount = df["amount"].astype(float) if "amount" in df else pd.Series(np.nan, index=df.index)
    ret = close.pct_change(fill_method=None)

    prev_high_20 = high.shift(1).tail(20).max()
    prev_high_60 = high.shift(1).tail(60).max()
    latest_close = float(close.iloc[-1])
    breakout_20 = latest_close / prev_high_20 - 1.0 if prev_high_20 and prev_high_20 > 0 else np.nan
    breakout_60 = latest_close / prev_high_60 - 1.0 if prev_high_60 and prev_high_60 > 0 else np.nan

    name = ""
    if "name" in df and not df["name"].dropna().empty:
        name = str(df["name"].dropna().iloc[-1])

    return {
        "symbol": code,
        "name": name,
        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "history_days": int(len(df)),
        "last_close": latest_close,
        "return_5d": _return_at(close, 5),
        "return_10d": _return_at(close, 10),
        "return_20d": _return_at(close, 20),
        "return_60d": _return_at(close, 60),
        "breakout_20d": breakout_20,
        "breakout_60d": breakout_60,
        "volatility_20d": float(ret.tail(20).std(ddof=0)),
        "max_drawdown_60d": _max_drawdown(close.tail(60)),
        "avg_amount_20d": float(amount.replace(0, np.nan).tail(20).median()),
        "source_path": str(path),
    }


def add_market_cap(df: pd.DataFrame, market_cap_path: Path | None) -> pd.DataFrame:
    if market_cap_path is None or not market_cap_path.exists():
        return df
    mcap = _read_csv(market_cap_path)
    if "code" not in mcap:
        return df
    mcap = mcap.copy()
    mcap["symbol"] = mcap["code"].map(_clean_code)
    keep = ["symbol"]
    for col in ["market_cap_yi", "float_market_cap_yi", "latest_price", "snapshot_date"]:
        if col in mcap:
            keep.append(col)
    return df.merge(mcap[keep].drop_duplicates("symbol", keep="last"), on="symbol", how="left")


def rank_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    safe_cols = [
        "return_5d",
        "return_10d",
        "return_20d",
        "breakout_20d",
        "avg_amount_20d",
        "volatility_20d",
    ]
    for col in safe_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_rank"] = out[col].rank(pct=True)

    dd_abs = pd.to_numeric(out["max_drawdown_60d"], errors="coerce").abs()
    dd_penalty = dd_abs.rank(pct=True)
    out["high_return_score"] = (
        out["return_20d_rank"].fillna(0.0) * 0.30
        + out["return_10d_rank"].fillna(0.0) * 0.20
        + out["return_5d_rank"].fillna(0.0) * 0.15
        + out["breakout_20d_rank"].fillna(0.0) * 0.15
        + out["avg_amount_20d_rank"].fillna(0.0) * 0.12
        + out["volatility_20d_rank"].fillna(0.0) * 0.08
        - dd_penalty.fillna(0.0) * 0.10
    )
    return out.sort_values("high_return_score", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-return universe from stock history CSVs.")
    parser.add_argument("--stock-dir", default=str(DEFAULT_STOCK_DIR))
    parser.add_argument("--market-cap", default=str(DEFAULT_MCAP))
    parser.add_argument("--output", default="high_return_universe.csv")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--max-symbols", type=int, default=500)
    parser.add_argument("--min-history-days", type=int, default=180)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_dir = Path(args.stock_dir)
    if not stock_dir.exists():
        raise FileNotFoundError(f"stock-dir not found: {stock_dir}")

    prefixes = tuple(x.strip() for x in args.prefixes.split(",") if x.strip())
    asof = pd.Timestamp(args.asof_date) if args.asof_date else None
    rows = []
    for path in sorted(stock_dir.glob("*.csv")):
        code = _clean_code(path.stem)
        if not _allowed_code(code, prefixes):
            continue
        row = score_one_file(path, asof=asof, min_history_days=args.min_history_days)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No eligible symbols found.")

    out = pd.DataFrame(rows)
    out = add_market_cap(out, Path(args.market_cap) if args.market_cap else None)
    out = rank_score(out)
    out = out.head(args.max_symbols)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(
        f"Built universe: rows={len(out)}, "
        f"date_min={out['last_date'].min()}, date_max={out['last_date'].max()}, path={output}"
    )


if __name__ == "__main__":
    main()
