from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from panel_io import read_panel
from research_database import ResearchDatabase, normalize_a_share_symbols
from tdx_day_source import infer_tdx_archive_specs, iter_tdx_day_archive_member_frames
from ths_daily_data import normalize_daily_market_file
from workspace_paths import daily_data_root, tdx_data_root


DATE_TOKEN = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2}|20\d{6})(?!\d)")
OBSERVATION_PREFIXES = (
    "daily_personal_overlay_changes_",
    "daily_personal_overlay_selected_",
    "early_pattern_watchlist_",
    "hidden_accumulation_trade_watch_tracking_",
    "main_net_volume_shadow_",
    "institutional_accumulation_shadow_",
    "institutional_accumulation_tracking_",
    "czsc_structure_shadow_",
    "merged_model_decision_table_",
    "merged_priority_watchlist_",
    "merged_state_pattern_scan_",
    "rank_model_candidates_trend_gated_",
    "strategy_family_forward_",
    "strategy_family_health_",
    "strategy_arena_",
    "trend_ignition_score_forward_",
)


def extract_date_tokens(path: Path) -> list[str]:
    tokens = []
    for raw_token in DATE_TOKEN.findall(path.stem):
        token = raw_token.replace("-", "")
        if not pd.isna(pd.to_datetime(token, format="%Y%m%d", errors="coerce")):
            tokens.append(token)
    return tokens


def infer_file_date(path: Path) -> str | None:
    tokens = extract_date_tokens(path)
    if not tokens:
        return None
    return pd.to_datetime(max(tokens), format="%Y%m%d").strftime("%Y-%m-%d")


def normalize_asof_date(value: str) -> str:
    if re.fullmatch(r"(?:20\d{6}|20\d{2}-\d{2}-\d{2})", value) is None:
        raise ValueError(f"Invalid as-of date: {value!r}; expected YYYYMMDD or YYYY-MM-DD")
    token = value.replace("-", "")
    if pd.isna(pd.to_datetime(token, format="%Y%m%d", errors="coerce")):
        raise ValueError(f"Invalid as-of date: {value!r}; expected YYYYMMDD or YYYY-MM-DD")
    return token


def discover_latest_panel(root: Path = Path(".")) -> Path | None:
    candidates = [
        path
        for suffix in ("csv", "parquet", "pq")
        for path in root.glob(f"data_panel_history_main_chinext_*.{suffix}")
    ]
    dated = [(max(tokens), path) for path in candidates if (tokens := extract_date_tokens(path))]
    return max(dated, default=(None, None), key=lambda item: item[0])[1]


def select_observation_files(output_dir: Path, asof_date: str | None = None) -> list[Path]:
    candidates = sorted(output_dir.glob("*.csv"))
    dated = [(path, tokens) for path in candidates if (tokens := extract_date_tokens(path))]
    selected_date = normalize_asof_date(asof_date) if asof_date else None
    if not dated:
        return []
    selected_date = selected_date or max(max(tokens) for _, tokens in dated)
    return [path for path, tokens in dated if selected_date in tokens]


def select_supported_observation_files(
    output_dir: Path,
    asof_date: str | None = None,
) -> list[Path]:
    return [
        path
        for path in select_observation_files(output_dir, asof_date)
        if path.name.startswith(OBSERVATION_PREFIXES) and not path.stem.endswith("_cn")
    ]


def unresolved_observation_files(output_dir: Path) -> list[Path]:
    return [path for path in sorted(output_dir.glob("*.csv")) if not extract_date_tokens(path)]


def read_table(path: Path) -> pd.DataFrame:
    # THS exports can be tab-delimited text despite an .xls suffix.
    if path.suffix.lower() == ".xls":
        return pd.read_csv(path, sep="\t", dtype=str, encoding="gb18030")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, low_memory=False)


def prepare_daily_price_frame(path: Path) -> pd.DataFrame | None:
    inferred_date = infer_file_date(path)
    frame, sources = normalize_daily_market_file(path, inferred_date or "1970-01-01")
    if inferred_date is None and sources.date == "filename":
        return None
    return frame


def prepare_observation_frame(path: Path, asof_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "date" not in frame.columns:
        inferred_date = infer_file_date(path)
        if inferred_date:
            frame["date"] = inferred_date
    if "date" not in frame.columns:
        raise ValueError(f"Observation file has no resolvable date: {path}")

    parsed_dates = pd.to_datetime(frame["date"].astype(str), errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"Observation file contains invalid dates: {path}")
    asof_token = normalize_asof_date(asof_date)
    asof_timestamp = pd.to_datetime(asof_token, format="%Y%m%d")
    if (parsed_dates > asof_timestamp).any():
        raise ValueError(f"Observation file contains dates after {asof_timestamp:%Y-%m-%d}: {path}")

    frame = frame.copy()
    frame["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update the local A-share research SQLite database.")
    parser.add_argument("--db", default="data/research.sqlite3")
    parser.add_argument("--panel", help="Panel CSV. Defaults to the latest dated panel in the current directory.")
    parser.add_argument("--skip-panel", action="store_true")
    parser.add_argument(
        "--daily-dir", default=str(daily_data_root() / "ths_exports" / "normalized")
    )
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--asof-date", help="Observation date in YYYYMMDD or YYYY-MM-DD. Defaults to the latest dated output.")
    parser.add_argument("--tdx-root", default=str(tdx_data_root()))
    parser.add_argument("--skip-tdx", action="store_true")
    parser.add_argument("--tdx-symbols-from-panel", action="store_true")
    args = parser.parse_args()

    db = ResearchDatabase(args.db)
    panel = Path(args.panel) if args.panel else discover_latest_panel()
    if panel is None:
        panel = Path("__panel_not_found__")
        if not args.skip_panel:
            print("prices panel: no dated panel found; skipped", flush=True)
    if panel.exists() and not args.skip_panel:
        frame = read_panel(panel, low_memory=False)
        price_cols = {"symbol", "date", "open", "high", "low", "close", "volume", "amount"}
        if price_cols.issubset(frame.columns):
            print(f"prices panel: {db.import_prices(frame, str(panel))}")

    daily_dir = Path(args.daily_dir)
    daily_paths = (
        sorted(daily_dir.glob("ths_hs_a_share_*.xls"))
        + sorted(daily_dir.glob("ths_hs_a_share_*.xlsx"))
        + sorted(daily_dir.glob("ths_hs_a_share_*.csv"))
    )
    for path in daily_paths:
        frame = prepare_daily_price_frame(path)
        if frame is not None:
            print(f"prices {path.name}: {db.import_prices(frame, str(path))}")
        else:
            print(f"prices {path.name}: skipped; symbol/date/close could not be resolved", flush=True)

    output_dir = Path(args.output_dir)
    for path in unresolved_observation_files(output_dir):
        print(f"observations {path.name}: skipped; filename date could not be resolved", flush=True)
    observation_paths = select_supported_observation_files(output_dir, args.asof_date)
    selected_asof = (
        normalize_asof_date(args.asof_date)
        if args.asof_date
        else max((max(extract_date_tokens(path)) for path in observation_paths), default=None)
    )
    for path in observation_paths:
        frame = prepare_observation_frame(path, selected_asof)
        print(
            f"observations {path.name}: "
            f"{db.import_observations(frame, path.stem, str(path.resolve()))}"
        )

    tdx_symbol_filter = None
    if args.tdx_symbols_from_panel and panel.exists():
        symbol_frame = pd.read_csv(panel, usecols=["symbol"], low_memory=False)
        tdx_symbol_filter = set(normalize_a_share_symbols(symbol_frame["symbol"]).dropna())
        print(f"tdx symbol filter: {len(tdx_symbol_filter)} symbols", flush=True)

    if not args.skip_tdx:
        tdx_root = Path(args.tdx_root)
        if tdx_root.exists():
            for spec in infer_tdx_archive_specs(tdx_root):
                archive_key = str(spec.path)
                result = db.import_tdx_member_frames(
                    archive_key,
                    iter_tdx_day_archive_member_frames(spec, symbol_filter=tdx_symbol_filter),
                )
                print(
                    f"tdx {spec.path.name}: read={result['read_rows']} inserted={result['inserted_rows']} "
                    f"imported_files={result['imported_files']} skipped_files={result['skipped_files']}",
                    flush=True,
                )

    print(db.query("SELECT COUNT(*) AS rows, MIN(date) AS first_date, MAX(date) AS last_date FROM daily_prices").to_dict("records")[0])
    print(db.query("SELECT COUNT(*) AS rows, MIN(date) AS first_date, MAX(date) AS last_date FROM tdx_daily_prices").to_dict("records")[0])
    print(db.query("SELECT kind, COUNT(*) AS rows FROM observations GROUP BY kind ORDER BY rows DESC").to_dict("records"))


if __name__ == "__main__":
    main()
