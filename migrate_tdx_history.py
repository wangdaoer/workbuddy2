from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

from research_database import ResearchDatabase


def _validate_source_and_target(source_db: str | Path, target_db: str | Path) -> tuple[Path, Path]:
    source_path = Path(source_db)
    target_path = Path(target_db)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source_path}")
    if source_path.resolve() == target_path.resolve():
        raise ValueError("Source and target databases must be different files")
    return source_path, target_path


def migrate_tdx_history(
    source_db: str | Path,
    target_db: str | Path,
    *,
    delete_source: bool = False,
    vacuum: bool = True,
) -> dict[str, int]:
    source_path, target_path = _validate_source_and_target(source_db, target_db)
    ResearchDatabase(target_path)

    with closing(sqlite3.connect(source_path)) as conn:
        conn.execute("ATTACH DATABASE ? AS target", (str(target_path),))
        price_rows = conn.execute("SELECT COUNT(*) FROM tdx_daily_prices").fetchone()[0]
        file_rows = conn.execute("SELECT COUNT(*) FROM tdx_imported_files").fetchone()[0]
        conn.execute(
            """
            INSERT OR REPLACE INTO target.tdx_daily_prices
            (market, symbol, date, open, high, low, close, volume, amount, asset_type, source)
            SELECT market, symbol, date, open, high, low, close, volume, amount, asset_type, source
            FROM main.tdx_daily_prices
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO target.tdx_imported_files
            (archive, member, rows, imported_at)
            SELECT archive, member, rows, imported_at
            FROM main.tdx_imported_files
            """
        )

        missing_prices = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT market, symbol, date, open, high, low, close, volume, amount, asset_type, source
                FROM main.tdx_daily_prices
                EXCEPT
                SELECT market, symbol, date, open, high, low, close, volume, amount, asset_type, source
                FROM target.tdx_daily_prices
            )
            """
        ).fetchone()[0]
        missing_files = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT archive, member, rows, imported_at FROM main.tdx_imported_files
                EXCEPT
                SELECT archive, member, rows, imported_at FROM target.tdx_imported_files
            )
            """
        ).fetchone()[0]
        if missing_prices or missing_files:
            raise RuntimeError(
                f"TDX migration verification failed: missing_prices={missing_prices}, missing_files={missing_files}"
            )

        if delete_source:
            conn.execute("DELETE FROM main.tdx_daily_prices")
            conn.execute("DELETE FROM main.tdx_imported_files")
        conn.commit()
        conn.execute("DETACH DATABASE target")

    if delete_source and vacuum:
        with closing(sqlite3.connect(source_path)) as conn:
            conn.execute("VACUUM")

    return {"price_rows": int(price_rows), "file_rows": int(file_rows)}


def rebuild_main_without_tdx(
    source_db: str | Path,
    rebuilt_db: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    source_path, rebuilt_path = _validate_source_and_target(source_db, rebuilt_db)
    if rebuilt_path.exists():
        if not overwrite:
            raise FileExistsError(f"Rebuilt database already exists: {rebuilt_path}")
        rebuilt_path.unlink()
    ResearchDatabase(rebuilt_path)

    with closing(sqlite3.connect(source_path)) as conn:
        conn.execute("ATTACH DATABASE ? AS rebuilt", (str(rebuilt_path),))
        instrument_rows = conn.execute("SELECT COUNT(*) FROM main.instruments").fetchone()[0]
        daily_price_rows = conn.execute("SELECT COUNT(*) FROM main.daily_prices").fetchone()[0]
        observation_rows = conn.execute("SELECT COUNT(*) FROM main.observations").fetchone()[0]
        conn.execute(
            """
            INSERT OR REPLACE INTO rebuilt.instruments
            (symbol, name, market, updated_at)
            SELECT symbol, name, market, updated_at
            FROM main.instruments
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO rebuilt.daily_prices
            (symbol, date, open, high, low, close, volume, amount, source)
            SELECT symbol, date, open, high, low, close, volume, amount, source
            FROM main.daily_prices
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO rebuilt.observations
            (id, symbol, date, kind, source, row_key, payload)
            SELECT id, symbol, date, kind, source, row_key, payload
            FROM main.observations
            """
        )
        rebuilt_observation_rows = conn.execute("SELECT COUNT(*) FROM rebuilt.observations").fetchone()[0]
        if rebuilt_observation_rows != observation_rows:
            raise RuntimeError(
                "Rebuilt observation row count mismatch: "
                f"source={observation_rows}, rebuilt={rebuilt_observation_rows}"
            )
        conn.commit()
        conn.execute("DETACH DATABASE rebuilt")

    return {
        "daily_price_rows": int(daily_price_rows),
        "observation_rows": int(observation_rows),
        "instrument_rows": int(instrument_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy TDX history tables into a separate SQLite database.")
    parser.add_argument("--source-db", default="data/research.sqlite3")
    parser.add_argument("--target-db", default="data/tdx_history.sqlite3")
    parser.add_argument("--rebuilt-main-db")
    parser.add_argument("--delete-source", action="store_true", help="Delete copied TDX rows from the source after verification.")
    parser.add_argument("--overwrite-rebuilt", action="store_true")
    parser.add_argument("--no-vacuum", action="store_true")
    args = parser.parse_args()
    result = migrate_tdx_history(
        args.source_db,
        args.target_db,
        delete_source=args.delete_source,
        vacuum=not args.no_vacuum,
    )
    print(result)
    if args.rebuilt_main_db:
        print(rebuild_main_without_tdx(args.source_db, args.rebuilt_main_db, overwrite=args.overwrite_rebuilt))


if __name__ == "__main__":
    main()
