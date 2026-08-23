from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_research_database import (
    normalize_asof_date,
    prepare_daily_price_frame,
    prepare_observation_frame,
    select_supported_observation_files,
)
from research_database import ResearchDatabase, normalize_a_share_symbols
from update_model_panel_from_daily_data import parse_number, parse_percent_ratio
from workspace_paths import daily_data_root


PRICE_USABLE_COLUMNS = ("open", "high", "low", "close")
FACTOR_USABLE_COLUMNS = (
    "volume",
    "amount",
    "main_net_inflow",
    "main_net_volume_ratio",
)
STATUS_SCHEMA_VERSION = 3


def _daily_candidates(daily_dir: Path, asof_date: str) -> list[Path]:
    return [
        daily_dir / f"ths_hs_a_share_{asof_date}{suffix}"
        for suffix in (".xls", ".xlsx", ".csv")
    ]


def _load_daily_frame(daily_dir: Path, asof_date: str) -> tuple[Path, pd.DataFrame]:
    errors: list[str] = []
    for path in _daily_candidates(daily_dir, asof_date):
        if not path.exists():
            continue
        try:
            frame = prepare_daily_price_frame(path)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if frame is not None and not frame.empty:
            return path, frame
        errors.append(f"{path.name}: symbol/date/close could not be resolved")
    detail = "; ".join(errors) if errors else "no matching files"
    raise FileNotFoundError(f"No usable daily market file for {asof_date}: {detail}")


def _parsed_dates(frame: pd.DataFrame, label: str) -> pd.Series:
    parsed = pd.to_datetime(frame["date"].astype(str), errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid dates")
    return parsed


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[column].map(parse_number).astype(float)


def _first_numeric_series(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    percent_points: bool = False,
) -> tuple[pd.Series, str]:
    for column in columns:
        if column not in frame.columns:
            continue
        parser = parse_percent_ratio if percent_points and column != "main_net_volume_ratio" else parse_number
        values = frame[column].map(parser).astype(float)
        if values.notna().any():
            return values, column
    return pd.Series(np.nan, index=frame.index, dtype=float), "unavailable"


def _price_usable_mask(frame: pd.DataFrame) -> pd.Series:
    symbols = normalize_a_share_symbols(frame["symbol"])
    price_values = pd.DataFrame(
        {column: _numeric_series(frame, column) for column in PRICE_USABLE_COLUMNS}
    )
    return symbols.notna() & np.isfinite(price_values).all(axis=1) & price_values.gt(0).all(axis=1)


def _normalize_database_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in PRICE_USABLE_COLUMNS:
        normalized[column] = _numeric_series(frame, column)
    amount = _numeric_series(frame, "amount")
    market_cap, _ = _first_numeric_series(frame, ("market_cap", "总市值"))
    turnover_rate, _ = _first_numeric_series(frame, ("turnover_rate", "换手"))
    estimated_amount = market_cap * turnover_rate / 100.0
    normalized["amount"] = amount.where(
        amount.gt(0), estimated_amount.where(estimated_amount.gt(0))
    )
    volume = _numeric_series(frame, "volume")
    normalized["volume"] = volume.where(
        volume.gt(0),
        (normalized["amount"] / normalized["close"]).where(normalized["close"].gt(0)),
    )
    return normalized


def _daily_freshness_evidence(
    frame: pd.DataFrame,
    parsed_dates: pd.Series,
) -> dict[str, object]:
    source_provenance = frame.attrs.get("daily_market_sources", {})
    symbols = normalize_a_share_symbols(frame["symbol"])
    valid_symbol = symbols.notna()
    normalized_dates = parsed_dates.dt.strftime("%Y-%m-%d")
    identity = pd.DataFrame({"date": normalized_dates, "symbol": symbols})
    duplicate_rows = int(identity.loc[valid_symbol].duplicated(["date", "symbol"], keep=False).sum())

    price_values = pd.DataFrame(
        {column: _numeric_series(frame, column) for column in PRICE_USABLE_COLUMNS}
    )
    price_usable = _price_usable_mask(frame)

    volume = _numeric_series(frame, "volume")
    amount = _numeric_series(frame, "amount")
    market_cap, market_cap_source = _first_numeric_series(frame, ("market_cap", "总市值"))
    turnover_rate, turnover_source = _first_numeric_series(frame, ("turnover_rate", "换手"))
    estimated_amount = market_cap * turnover_rate / 100.0
    amount = amount.where(amount.gt(0), estimated_amount.where(estimated_amount.gt(0)))
    close = price_values["close"]
    volume = volume.where(volume.gt(0), (amount / close).where(close.gt(0)))
    main_net_inflow, main_net_inflow_source = _first_numeric_series(
        frame,
        ("main_net_inflow", "大单净额"),
    )
    main_net_volume_ratio, main_net_volume_ratio_source = _first_numeric_series(
        frame,
        ("main_net_volume_ratio", "main_net_volume_ratio_pct", "主力净量"),
        percent_points=True,
    )
    factor_masks = pd.DataFrame(
        {
            "volume": np.isfinite(volume) & volume.gt(0),
            "amount": np.isfinite(amount) & amount.gt(0),
            "main_net_inflow": np.isfinite(main_net_inflow),
            "main_net_volume_ratio": np.isfinite(main_net_volume_ratio),
        }
    ).mul(price_usable, axis=0)
    factor_usable = factor_masks.any(axis=1)

    expected_symbols = int(symbols.loc[valid_symbol].nunique())
    price_usable_rows = int(identity.loc[price_usable].drop_duplicates().shape[0])
    factor_usable_rows = int(identity.loc[factor_usable].drop_duplicates().shape[0])
    factor_column_rows = {
        column: int(identity.loc[factor_masks[column]].drop_duplicates().shape[0])
        for column in FACTOR_USABLE_COLUMNS
    }
    price_denominator = expected_symbols or 1
    factor_denominator = price_usable_rows or 1
    factor_sources = {
        "volume": "source_or_amount_div_close" if factor_masks["volume"].any() else "unavailable",
        "amount": (
            "source_or_market_cap_x_turnover_rate"
            if factor_masks["amount"].any()
            else "unavailable"
        ),
        "main_net_inflow": source_provenance.get("money_flow", main_net_inflow_source),
        "main_net_volume_ratio": source_provenance.get(
            "money_flow_ratio", main_net_volume_ratio_source
        ),
        "market_cap": market_cap_source,
        "turnover_rate": turnover_source,
    }
    source_latest = normalized_dates.max() if not normalized_dates.empty else None
    price_latest = normalized_dates.loc[price_usable].max() if price_usable.any() else None
    factor_latest = normalized_dates.loc[factor_usable].max() if factor_usable.any() else None
    return {
        "source_latest_date": source_latest,
        "price_usable_latest_date": price_latest,
        "factor_usable_latest_date": factor_latest,
        "source_unique_symbols": expected_symbols,
        "duplicate_date_symbol_rows": duplicate_rows,
        "price_usable_rows": price_usable_rows,
        "price_unusable_rows": expected_symbols - price_usable_rows,
        "price_unusable_symbols": sorted(
            identity.loc[valid_symbol & ~price_usable, "symbol"].dropna().unique().tolist()
        ),
        "price_usable_ratio": price_usable_rows / price_denominator,
        "factor_usable_rows": factor_usable_rows,
        "factor_usable_ratio": factor_usable_rows / factor_denominator,
        "factor_usable_columns": [
            column for column, rows in factor_column_rows.items() if rows > 0
        ],
        "factor_column_rows": factor_column_rows,
        "factor_column_coverage": {
            column: rows / factor_denominator for column, rows in factor_column_rows.items()
        },
        "factor_sources": factor_sources,
    }


def write_status_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def canonical_source_path(path: Path) -> str:
    return str(Path(path).resolve())


def remove_observation_source_aliases(
    conn,
    observation_paths: list[Path],
) -> int:
    canonical_by_kind = {
        path.stem: canonical_source_path(path) for path in observation_paths
    }
    aliases = conn.execute("SELECT DISTINCT kind, source FROM observations").fetchall()
    before = conn.total_changes
    for kind, source in aliases:
        canonical = canonical_by_kind.get(kind)
        if canonical is None or source == canonical:
            continue
        try:
            resolved_source = canonical_source_path(Path(source))
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved_source == canonical:
            conn.execute(
                "DELETE FROM observations WHERE kind = ? AND source = ?",
                (kind, source),
            )
    return int(conn.total_changes - before)


def sync_daily_research_database(
    db_path: Path,
    daily_dir: Path,
    output_dir: Path,
    asof_date: str,
) -> dict[str, object]:
    token = normalize_asof_date(asof_date)
    iso_date = pd.to_datetime(token, format="%Y%m%d").strftime("%Y-%m-%d")
    daily_path, daily_frame = _load_daily_frame(daily_dir, iso_date)
    daily_dates = _parsed_dates(daily_frame, str(daily_path)).dt.strftime("%Y-%m-%d")
    if set(daily_dates.unique()) != {iso_date}:
        raise ValueError(
            f"Daily market file must contain only {iso_date}: found={sorted(daily_dates.unique())}"
        )
    daily_frame = daily_frame.copy()
    daily_frame["date"] = daily_dates
    daily_frame = _normalize_database_price_columns(daily_frame)
    freshness = _daily_freshness_evidence(
        daily_frame,
        pd.to_datetime(daily_frame["date"], errors="raise"),
    )
    if freshness["duplicate_date_symbol_rows"]:
        raise ValueError(
            "Daily market file contains duplicate date-symbol rows: "
            f"rows={freshness['duplicate_date_symbol_rows']}"
        )
    if freshness["price_usable_latest_date"] != iso_date:
        raise RuntimeError(
            f"Daily market OHLC is not usable for {iso_date}: "
            f"price_usable_latest_date={freshness['price_usable_latest_date']}"
        )
    daily_source_rows = int(len(daily_frame))
    daily_frame = daily_frame.loc[_price_usable_mask(daily_frame)].copy()

    observation_paths = select_supported_observation_files(output_dir, token)
    if not observation_paths:
        raise FileNotFoundError(f"No dated observation CSV files found for {iso_date} in {output_dir}")

    observation_frames: list[tuple[Path, pd.DataFrame]] = []
    for path in observation_paths:
        frame = prepare_observation_frame(path, token)
        observation_frames.append((path, frame))

    db = ResearchDatabase(db_path)
    expected_symbols = int(normalize_a_share_symbols(daily_frame["symbol"]).dropna().nunique())
    import_symbols = sorted(
        normalize_a_share_symbols(daily_frame["symbol"]).dropna().unique().tolist()
    )
    observation_read_rows = sum(len(frame) for _, frame in observation_frames)
    with db.connect() as conn:
        conn.execute("CREATE TEMP TABLE current_daily_symbols (symbol TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO current_daily_symbols(symbol) VALUES (?)",
            ((symbol,) for symbol in import_symbols),
        )
        before_cleanup = conn.total_changes
        conn.execute(
            """DELETE FROM daily_prices
               WHERE date = ?
                 AND symbol NOT IN (SELECT symbol FROM current_daily_symbols)""",
            (iso_date,),
        )
        daily_removed_unusable = int(conn.total_changes - before_cleanup)
        conn.execute("DROP TABLE current_daily_symbols")
        daily_changed = int(
            db.import_prices_into(
                conn,
                daily_frame,
                canonical_source_path(daily_path),
                update_existing=True,
            )
        )
        observation_removed_source_alias_rows = remove_observation_source_aliases(
            conn, [path for path, _frame in observation_frames]
        )
        observation_changed = 0
        for path, frame in observation_frames:
            observation_changed += int(
                db.import_observations_into(
                    conn,
                    frame,
                    path.stem,
                    canonical_source_path(path),
                )
            )

        coverage_row = conn.execute(
            """SELECT COUNT(*) AS total_rows, COUNT(DISTINCT symbol) AS symbols,
                      MIN(date) AS first_date, MAX(date) AS latest_date,
                      SUM(CASE WHEN date = ? THEN 1 ELSE 0 END) AS asof_rows
               FROM daily_prices""",
            (iso_date,),
        ).fetchone()
        observation_total = int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
        asof_rows = int(coverage_row[4] or 0)
        if asof_rows != expected_symbols:
            raise RuntimeError(
                f"Daily database coverage mismatch for {iso_date}: "
                f"rows={asof_rows}, expected_symbols={expected_symbols}"
            )

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "success",
        "asof_date": iso_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "database": canonical_source_path(db_path),
        "daily_source": canonical_source_path(daily_path),
        "daily_read_rows": daily_source_rows,
        "daily_import_rows": int(len(daily_frame)),
        **freshness,
        "daily_expected_symbols": expected_symbols,
        "daily_changed_rows": daily_changed,
        "daily_removed_unusable_rows": daily_removed_unusable,
        "observation_files": [canonical_source_path(path) for path in observation_paths],
        "observation_file_count": len(observation_paths),
        "observation_read_rows": int(observation_read_rows),
        "observation_removed_source_alias_rows": observation_removed_source_alias_rows,
        "observation_changed_rows": int(observation_changed),
        "database_daily_rows": int(coverage_row[0]),
        "database_symbols": int(coverage_row[1]),
        "database_first_date": coverage_row[2],
        "database_latest_date": coverage_row[3],
        "database_asof_rows": asof_rows,
        "database_observation_rows": observation_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally sync today's prices and model observations to SQLite.")
    parser.add_argument("--db", default="data/research.sqlite3")
    parser.add_argument(
        "--daily-dir", default=str(daily_data_root() / "ths_exports" / "normalized")
    )
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--status-output", required=True)
    args = parser.parse_args()

    status_path = Path(args.status_output)
    try:
        status = sync_daily_research_database(
            Path(args.db),
            Path(args.daily_dir),
            Path(args.output_dir),
            args.asof_date,
        )
    except Exception as exc:
        failure = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "status": "failed",
            "asof_date": args.asof_date,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "database": args.db,
            "message": f"{type(exc).__name__}: {exc}",
        }
        write_status_atomic(status_path, failure)
        raise
    write_status_atomic(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
