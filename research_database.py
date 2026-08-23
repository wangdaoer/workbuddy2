"""Local SQLite store for A-share research data.

This module is deliberately research-only: it stores observations and prices,
but contains no broker, order, or account functionality.
"""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
TDX_PRICE_COLUMNS = ["market", "symbol", "date", "open", "high", "low", "close", "volume", "amount", "asset_type", "source"]
SH_INDEX_SYMBOLS = {"000001", "000002", "000003", "000016", "000300", "000905", "000906", "000852", "000688"}
SZ_INDEX_PREFIXES = ("399",)
A_SHARE_SYMBOL_PATTERN = re.compile(r"^(?:SH|SZ)?(\d{1,6})(?:\.(?:SH|SZ))?$", re.IGNORECASE)
A_SHARE_STOCK_PREFIXES = {
    "SH": ("600", "601", "603", "605", "688", "689"),
    "SZ": ("000", "001", "002", "003", "300", "301"),
}


def normalize_a_share_symbols(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    digits = text.str.extract(A_SHARE_SYMBOL_PATTERN, expand=False)
    return digits.str.zfill(6)


def normalize_a_share_symbol(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\.0$", "", str(value).strip())
    match = A_SHARE_SYMBOL_PATTERN.fullmatch(text)
    return match.group(1).zfill(6) if match else None


def is_a_share_stock_symbol(market: object, symbol: object) -> bool:
    market_code = str(market).upper()
    symbol_code = normalize_a_share_symbol(symbol)
    prefixes = A_SHARE_STOCK_PREFIXES.get(market_code, ())
    return symbol_code is not None and symbol_code.startswith(prefixes)


def classify_tdx_asset_type(market: str, symbol: str, raw_asset_type: str) -> str:
    market = str(market).upper()
    symbol = str(symbol).zfill(6)
    raw_asset_type = str(raw_asset_type)
    if market == "SH" and symbol in SH_INDEX_SYMBOLS:
        return "index"
    if market == "SZ" and symbol.startswith(SZ_INDEX_PREFIXES):
        return "index"
    return raw_asset_type


class ResearchDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS instruments (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS daily_prices (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, amount REAL,
                    source TEXT,
                    PRIMARY KEY (symbol, date)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
                CREATE TABLE IF NOT EXISTS tdx_daily_prices (
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, amount REAL,
                    asset_type TEXT NOT NULL,
                    source TEXT,
                    PRIMARY KEY (market, symbol, date, asset_type)
                );
                CREATE INDEX IF NOT EXISTS idx_tdx_daily_prices_date ON tdx_daily_prices(date);
                CREATE INDEX IF NOT EXISTS idx_tdx_daily_prices_symbol ON tdx_daily_prices(market, symbol);
                CREATE TABLE IF NOT EXISTS tdx_imported_files (
                    archive TEXT NOT NULL,
                    member TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (archive, member)
                );
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    date TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(date, kind, source, row_key)
                );
                CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(date);
                """
            )
            self._ensure_observations_v2(conn)

    def _ensure_observations_v2(self, conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(observations)").fetchall()}
        if "row_key" in columns:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DROP TABLE IF EXISTS observations_v2_new")
            conn.execute(
                """CREATE TABLE observations_v2_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                row_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(date, kind, source, row_key)
                )"""
            )
            conn.execute(
                """INSERT INTO observations_v2_new
                (id, symbol, date, kind, source, row_key, payload)
                SELECT id, symbol, date, kind, source,
                   printf('%08d', ROW_NUMBER() OVER (
                       PARTITION BY date, kind, source ORDER BY id
                   ) - 1),
                   payload
                FROM observations"""
            )
            source_rows = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            migrated_rows = conn.execute("SELECT COUNT(*) FROM observations_v2_new").fetchone()[0]
            if source_rows != migrated_rows:
                raise RuntimeError(
                    f"Observation migration row mismatch: source={source_rows}, migrated={migrated_rows}"
                )
            conn.execute("DROP INDEX IF EXISTS idx_observations_date")
            conn.execute("DROP TABLE observations")
            conn.execute("ALTER TABLE observations_v2_new RENAME TO observations")
            conn.execute("CREATE INDEX idx_observations_date ON observations(date)")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def query(self, sql: str, params: Iterable[object] = ()) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=tuple(params))

    def query_tdx_history(
        self,
        sql: str,
        tdx_db: str | Path,
        params: Iterable[object] = (),
    ) -> pd.DataFrame:
        tdx_path = Path(tdx_db)
        if not tdx_path.exists():
            raise FileNotFoundError(f"TDX history database not found: {tdx_path}")
        with self.connect() as conn:
            conn.execute("ATTACH DATABASE ? AS tdx_history", (str(tdx_path),))
            try:
                result = pd.read_sql_query(sql, conn, params=tuple(params))
            finally:
                conn.execute("DETACH DATABASE tdx_history")
            return result

    def query_tdx_history_normalized(
        self,
        tdx_db: str | Path,
        symbol: str | None = None,
        symbols: Iterable[str] | None = None,
        market: str | None = None,
        asset_type: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        where = []
        params: list[object] = []
        if symbol is not None:
            where.append("symbol = ?")
            params.append(str(symbol).zfill(6))
        if symbols is not None:
            cleaned_symbols = sorted({str(item).zfill(6) for item in symbols})
            if cleaned_symbols:
                where.append(f"symbol IN ({','.join(['?'] * len(cleaned_symbols))})")
                params.extend(cleaned_symbols)
        if market is not None:
            where.append("market = ?")
            params.append(str(market).upper())
        if start is not None:
            where.append("date >= ?")
            params.append(start)
        if end is not None:
            where.append("date <= ?")
            params.append(end)
        sql = """
            SELECT market, symbol, date, open, high, low, close, volume, amount,
                   asset_type AS raw_asset_type, source
            FROM tdx_history.tdx_daily_prices
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY market, symbol, date"
        result = self.query_tdx_history(sql, tdx_db, params=params)
        if result.empty:
            result["asset_type"] = []
            return result
        result["asset_type"] = [
            classify_tdx_asset_type(row.market, row.symbol, row.raw_asset_type)
            for row in result.itertuples(index=False)
        ]
        if asset_type is not None:
            result = result[result["asset_type"] == asset_type].reset_index(drop=True)
            if asset_type == "stock" and not result.empty:
                valid_stock = [
                    is_a_share_stock_symbol(row.market, row.symbol)
                    for row in result.itertuples(index=False)
                ]
                result = result.loc[valid_stock].reset_index(drop=True)
        result["_asset_type_rank"] = (result["raw_asset_type"] != result["asset_type"]).astype(int)
        result = (
            result.sort_values(["market", "symbol", "date", "asset_type", "_asset_type_rank", "source"])
            .drop_duplicates(["market", "symbol", "date", "asset_type"], keep="first")
            .drop(columns=["_asset_type_rank"])
            .reset_index(drop=True)
        )
        columns = ["market", "symbol", "date", "open", "high", "low", "close", "volume", "amount", "asset_type", "raw_asset_type", "source"]
        return result[columns]

    def _price_records(self, frame: pd.DataFrame, source: str) -> list[tuple]:
        missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"price data missing columns: {missing}")
        rows = frame[PRICE_COLUMNS].copy()
        rows["symbol"] = normalize_a_share_symbols(rows["symbol"])
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        rows = rows.dropna(subset=["symbol", "date"]).drop_duplicates(["symbol", "date"])
        rows["source"] = source
        return list(rows.itertuples(index=False, name=None))

    def import_prices_into(
        self,
        conn: sqlite3.Connection,
        frame: pd.DataFrame,
        source: str = "",
        update_existing: bool = False,
    ) -> int:
        records = self._price_records(frame, source)
        before = conn.total_changes
        if update_existing:
            conn.executemany(
                """INSERT INTO daily_prices
                (symbol,date,open,high,low,close,volume,amount,source)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol,date) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume, amount=excluded.amount,
                    source=excluded.source
                WHERE daily_prices.open IS NOT excluded.open
                   OR daily_prices.high IS NOT excluded.high
                   OR daily_prices.low IS NOT excluded.low
                   OR daily_prices.close IS NOT excluded.close
                   OR daily_prices.volume IS NOT excluded.volume
                   OR daily_prices.amount IS NOT excluded.amount
                   OR daily_prices.source IS NOT excluded.source""",
                records,
            )
        else:
            conn.executemany(
                """INSERT OR IGNORE INTO daily_prices
                (symbol,date,open,high,low,close,volume,amount,source)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                records,
            )
        return conn.total_changes - before

    def import_prices(self, frame: pd.DataFrame, source: str = "", update_existing: bool = False) -> int:
        with self.connect() as conn:
            return self.import_prices_into(conn, frame, source, update_existing)

    def import_tdx_prices(self, frame: pd.DataFrame) -> int:
        records = self._tdx_price_records(frame)
        with self.connect() as conn:
            return self._insert_tdx_price_records(conn, records)

    def import_tdx_price_frames(self, frames: Iterable[pd.DataFrame], batch_size: int = 50_000) -> int:
        total = 0
        batch: list[tuple] = []
        with self.connect() as conn:
            for frame in frames:
                batch.extend(self._tdx_price_records(frame))
                if len(batch) >= batch_size:
                    total += self._insert_tdx_price_records(conn, batch)
                    conn.commit()
                    batch = []
            if batch:
                total += self._insert_tdx_price_records(conn, batch)
                conn.commit()
        return total

    def import_tdx_member_frames(
        self,
        archive: str,
        member_frames: Iterable[tuple[str, pd.DataFrame]],
        batch_size: int = 50_000,
    ) -> dict[str, int]:
        inserted = 0
        read_rows = 0
        skipped_files = 0
        imported_files = 0
        price_batch: list[tuple] = []
        marker_batch: list[tuple[str, str, int]] = []
        with self.connect() as conn:
            done = {
                row[0]
                for row in conn.execute(
                    "SELECT member FROM tdx_imported_files WHERE archive = ?",
                    (archive,),
                ).fetchall()
            }
            for member, frame in member_frames:
                if member in done:
                    skipped_files += 1
                    continue
                read_rows += len(frame)
                price_batch.extend(self._tdx_price_records(frame))
                marker_batch.append((archive, member, len(frame)))
                imported_files += 1
                if len(price_batch) >= batch_size:
                    inserted += self._insert_tdx_price_records(conn, price_batch)
                    self._insert_tdx_file_markers(conn, marker_batch)
                    conn.commit()
                    price_batch = []
                    marker_batch = []
            if price_batch or marker_batch:
                inserted += self._insert_tdx_price_records(conn, price_batch)
                self._insert_tdx_file_markers(conn, marker_batch)
                conn.commit()
        return {
            "read_rows": read_rows,
            "inserted_rows": inserted,
            "skipped_files": skipped_files,
            "imported_files": imported_files,
        }

    def _tdx_price_records(self, frame: pd.DataFrame) -> list[tuple]:
        if frame.empty:
            return []
        missing = [column for column in TDX_PRICE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"TDX price data missing columns: {missing}")
        rows = frame[TDX_PRICE_COLUMNS].copy()
        rows["market"] = rows["market"].astype(str).str.upper()
        rows["symbol"] = normalize_a_share_symbols(rows["symbol"])
        rows["asset_type"] = rows["asset_type"].astype(str)
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        rows = rows.dropna(subset=["market", "symbol", "date", "asset_type"]).drop_duplicates(
            ["market", "symbol", "date", "asset_type"]
        )
        return list(rows.itertuples(index=False, name=None))

    def _insert_tdx_price_records(self, conn: sqlite3.Connection, records: Iterable[tuple]) -> int:
        before = conn.total_changes
        conn.executemany(
            """INSERT OR IGNORE INTO tdx_daily_prices
            (market,symbol,date,open,high,low,close,volume,amount,asset_type,source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            records,
        )
        return conn.total_changes - before

    def has_tdx_imported_file(self, archive: str, member: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM tdx_imported_files WHERE archive = ? AND member = ? LIMIT 1",
                (archive, member),
            ).fetchone()
            return row is not None

    def mark_tdx_imported_file(self, archive: str, member: str, rows: int) -> None:
        with self.connect() as conn:
            self._insert_tdx_file_markers(conn, [(archive, member, int(rows))])

    def _insert_tdx_file_markers(self, conn: sqlite3.Connection, records: Iterable[tuple[str, str, int]]) -> None:
        conn.executemany(
            """INSERT OR REPLACE INTO tdx_imported_files
            (archive, member, rows) VALUES (?, ?, ?)""",
            records,
        )

    def _observation_records(self, frame: pd.DataFrame, kind: str, source: str) -> list[tuple]:
        if "date" not in frame.columns:
            raise ValueError("observation data must contain date")
        records: list[tuple] = []
        duplicate_counts: dict[tuple[str, str], int] = {}
        for row in frame.to_dict("records"):
            date = pd.to_datetime(row.pop("date"), errors="coerce")
            if pd.isna(date):
                continue
            symbol = normalize_a_share_symbol(row.pop("symbol", None))
            date_text = date.strftime("%Y-%m-%d")
            payload = json.dumps(row, ensure_ascii=False, default=str, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(f"{symbol or ''}\0{payload}".encode("utf-8")).hexdigest()
            duplicate_key = (date_text, digest)
            occurrence = duplicate_counts.get(duplicate_key, 0)
            duplicate_counts[duplicate_key] = occurrence + 1
            row_key = digest if occurrence == 0 else f"{digest}:{occurrence}"
            records.append((symbol, date_text, kind, source, row_key, payload))
        return records

    def import_observations_into(
        self,
        conn: sqlite3.Connection,
        frame: pd.DataFrame,
        kind: str,
        source: str,
    ) -> int:
        records = self._observation_records(frame, kind, source)
        existing_rows = conn.execute(
            "SELECT date, row_key, symbol, payload FROM observations WHERE kind = ? AND source = ?",
            (kind, source),
        ).fetchall()
        existing = {(row[0], row[1]): (row[2], row[3]) for row in existing_rows}
        desired = {(row[1], row[4]): (row[0], row[5]) for row in records}
        if existing == desired:
            return 0
        conn.execute("DELETE FROM observations WHERE kind = ? AND source = ?", (kind, source))
        conn.executemany(
            """INSERT INTO observations (symbol,date,kind,source,row_key,payload)
            VALUES (?,?,?,?,?,?)""",
            records,
        )
        return max(len(existing), len(desired))

    def import_observations(self, frame: pd.DataFrame, kind: str, source: str) -> int:
        with self.connect() as conn:
            return self.import_observations_into(conn, frame, kind, source)
