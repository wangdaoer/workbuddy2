"""Export a small, sanitized web snapshot from daily research artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SNAPSHOT_SCHEMA_VERSION = 2
DEFAULT_STALE_AFTER_DAYS = 3
RUN_CARD_PATTERN = re.compile(r"daily_run_card_(\d{8})\.json$")
SYMBOL_PATTERN = re.compile(r"^[0-9]{6}$")
STOCK_NAME_PATTERN = re.compile(r"^[\u3400-\u9fffA-Za-z0-9*·（）()＋+&.\-]{1,30}$")
STRATEGY_FAMILY_LABELS = {
    "trend_momentum": "趋势动量",
    "strong_pullback": "强势回调二波",
    "hidden_accumulation": "隐性吸筹",
}
ALLOWED_STRATEGY_FAMILIES = set(STRATEGY_FAMILY_LABELS)
ALLOWED_PRIORITY_BUCKETS = {
    "model_focus",
    "action_focus",
    "risk_watch",
    "pattern_watch",
    "review_later",
}
ALLOWED_WARNING_CODES = {
    "artifact_missing:stability_report",
    "data_stale",
    "run_failed",
}
SENSITIVE_PUBLIC_TEXT_PATTERN = re.compile(
    r"(?:"
    r"(?:^|[\s\"'=])[a-z]:"
    r"|[a-z][a-z0-9+.-]*://"
    r"|(?:^|[^a-z0-9])[^\s/\\]+\.(?:py|js|ps1|sh|bat|cmd|sqlite3?|db|csv)(?:[^a-z0-9]|$)"
    r"|target[\s_.-]*(?:weight|leverage)"
    r"|personal[\s_-]*(?:action|position|trade)"
    r"|(?:^|[^a-z0-9])(?:buy|sell|execute|broker|purchase|allocation|position)(?:[^a-z0-9]|$)"
    r"|(?:place[\s_-]*order|order[\s_-]*(?:now|id|quantity|side))"
    r"|(?:^|[^a-z0-9])(?:py|python3?|node|deno|rscript|git|powershell|pwsh|cmd(?:\.exe)?|bash|zsh|curl|wget|sqlite3)(?=\s|$)"
    r"|--[a-z0-9_-]+"
    r"|(?:api[\s_-]*key|access[\s_-]*token|client[\s_-]*secret|password|bearer)"
    r"|(?:^|[^a-z0-9])(?:sk[-_](?:proj[-_]|live[-_])?|gh[pousr]_|vercel_blob_rw_|xox[baprs]-|AKIA[0-9A-Z]{8,})"
    r"|(?:买入|卖出|下单|仓位|持仓|加仓|减仓|清仓|建仓|止损|平仓|目标权重|交易指令|执行交易)"
    r")",
    re.IGNORECASE,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_latest_sources(source_dir: Path) -> tuple[Path, Path]:
    """Return the newest run card that has a matching non-localized watchlist."""
    source_dir = source_dir.expanduser().resolve()
    candidates: list[tuple[str, Path]] = []
    for path in source_dir.glob("daily_run_card_*.json"):
        match = RUN_CARD_PATTERN.match(path.name)
        if match:
            candidates.append((match.group(1), path))

    for token, run_card in sorted(candidates, reverse=True):
        watchlist = source_dir / f"merged_priority_watchlist_{token}.csv"
        if watchlist.is_file():
            return run_card, watchlist

    raise FileNotFoundError(
        f"No matching daily run card and priority watchlist found in {source_dir}"
    )


def build_research_snapshot(
    run_card_path: Path,
    watchlist_path: Path,
    *,
    published_at: datetime | None = None,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
) -> dict[str, Any]:
    """Build a web-safe snapshot and validate it against both source artifacts."""
    if stale_after_days < 0:
        raise ValueError("stale_after_days must be non-negative")

    run_card_path = run_card_path.expanduser().resolve()
    watchlist_path = watchlist_path.expanduser().resolve()
    run_card = _load_json_object(run_card_path)
    rows = _read_watchlist(watchlist_path)

    asof_date = _parse_asof_date(run_card.get("asof_date"))
    _validate_source_tokens(run_card_path, watchlist_path, asof_date)
    published = _normalized_datetime(published_at or datetime.now(timezone.utc))
    age_days = (published.date() - asof_date).days
    if age_days < 0:
        raise ValueError("published_at cannot be earlier than asof_date")

    run_status = "success" if run_card.get("run_status") == "success" else "failed"
    freshness_status = "fresh" if age_days <= stale_after_days else "stale"
    if run_status != "success":
        freshness_status = "failed"

    verification = run_card.get("verification")
    verification = verification if isinstance(verification, dict) else {}
    expected_rows = _optional_int(verification.get("priority_rows"))
    if expected_rows is not None and expected_rows != len(rows):
        raise ValueError(
            "Priority watchlist row count does not match the daily run card: "
            f"card={expected_rows}, csv={len(rows)}"
        )

    top10 = [_safe_watchlist_row(row) for row in rows[:10]]
    _validate_top10(run_card.get("top10"), top10)

    warnings = _safe_warning_codes(run_card.get("warnings"))
    if freshness_status == "stale" and "data_stale" not in warnings:
        warnings.append("data_stale")
    if run_status != "success" and "run_failed" not in warnings:
        warnings.append("run_failed")

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "project": "lianghua",
        "research_only": True,
        "trade_instruction": False,
        "asof_date": asof_date.isoformat(),
        "generated_at": _safe_timestamp(run_card.get("generated_at")),
        "published_at": _format_datetime(published),
        "run_status": run_status,
        "freshness": {
            "status": freshness_status,
            "age_days": age_days,
            "stale_after_days": stale_after_days,
        },
        "summary": {
            "priority_rows": len(rows),
            "selected_rows": _optional_int(verification.get("selected_rows")),
            "change_rows": _optional_int(verification.get("change_rows")),
            "model_decision_rows": _optional_int(
                verification.get("model_decision_rows")
            ),
            "early_pattern_rows": _optional_int(
                verification.get("early_pattern_rows")
            ),
        },
        "coverage": {
            "database_latest_date": _optional_text(
                verification.get("research_database_latest_date")
            ),
            "database_asof_rows": _optional_int(
                verification.get("research_database_asof_rows")
            ),
            "database_daily_rows": _optional_int(
                verification.get("research_database_daily_rows")
            ),
            "database_observation_rows": _optional_int(
                verification.get("research_database_observation_rows")
            ),
            "benchmark_latest_date": _optional_text(
                verification.get("benchmark_latest_date")
            ),
            "benchmark_source_agreement": _optional_bool(
                verification.get("benchmark_source_agreement")
            ),
        },
        "quality": {
            "missing_stock_names": _optional_int(
                verification.get("missing_stock_names")
            ),
            "warnings": warnings,
        },
        "watchlist": {
            "bucket_counts": _safe_count_map(
                verification.get("priority_bucket_counts"),
                allowed_keys=ALLOWED_PRIORITY_BUCKETS,
            ),
            "strategy_family_counts": _safe_count_map(
                verification.get("priority_strategy_family_counts"),
                allowed_keys=ALLOWED_STRATEGY_FAMILIES,
            ),
            "top10": top10,
        },
        "source_integrity": {
            "run_card_sha256": file_sha256(run_card_path),
            "watchlist_sha256": file_sha256(watchlist_path),
        },
    }


def write_research_snapshot(snapshot: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_watchlist(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "stock_name", "strategy_family", "priority_bucket"}
        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Priority watchlist missing columns: {missing}")
        return list(reader)


def _safe_watchlist_row(row: dict[str, str]) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "").strip()
    stock_name = str(row.get("stock_name") or "").strip()
    strategy_family = str(row.get("strategy_family") or "").strip()
    priority_bucket = str(row.get("priority_bucket") or "").strip()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("Watchlist symbol is not approved for publication")
    if not STOCK_NAME_PATTERN.fullmatch(stock_name):
        raise ValueError("Watchlist stock_name is not approved for publication")
    if _contains_sensitive_public_text(stock_name):
        raise ValueError("Watchlist stock_name is not safe for publication")
    if strategy_family not in ALLOWED_STRATEGY_FAMILIES:
        raise ValueError("Watchlist strategy_family is not approved for publication")
    if priority_bucket not in ALLOWED_PRIORITY_BUCKETS:
        raise ValueError("Watchlist priority_bucket is not approved for publication")
    return {
        "symbol": symbol,
        "stock_name": stock_name,
        "strategy_family": strategy_family,
        "strategy_family_cn": STRATEGY_FAMILY_LABELS[strategy_family],
        "priority_bucket": priority_bucket,
        "priority_score": _optional_float(row.get("priority_score")),
    }


def _validate_top10(raw_top10: Any, exported_top10: list[dict[str, Any]]) -> None:
    if not isinstance(raw_top10, list) or not raw_top10:
        return
    expected_symbols = [
        str(row.get("symbol") or "").strip()
        for row in raw_top10[:10]
        if isinstance(row, dict)
    ]
    exported_symbols = [str(row.get("symbol") or "") for row in exported_top10]
    if expected_symbols != exported_symbols:
        raise ValueError(
            "Priority watchlist top rows do not match the daily run card top10"
        )


def _validate_source_tokens(
    run_card_path: Path, watchlist_path: Path, asof_date: date
) -> None:
    token = asof_date.strftime("%Y%m%d")
    run_match = RUN_CARD_PATTERN.match(run_card_path.name)
    if run_match and run_match.group(1) != token:
        raise ValueError("Daily run card filename date does not match asof_date")
    watchlist_match = re.search(r"merged_priority_watchlist_(\d{8})\.csv$", watchlist_path.name)
    if watchlist_match and watchlist_match.group(1) != token:
        raise ValueError("Priority watchlist filename date does not match asof_date")


def _parse_asof_date(value: Any) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid asof_date: {value!r}") from exc


def _normalized_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return _normalized_datetime(datetime.fromisoformat(normalized))
    except ValueError as exc:
        raise ValueError(f"Invalid published_at: {value!r}") from exc


def _format_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_timestamp(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return _format_datetime(_parse_datetime(str(value)))
    except ValueError:
        return None


def _safe_warning_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    warnings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text not in ALLOWED_WARNING_CODES:
            continue
        warnings.append(text)
    return warnings


def _safe_count_map(value: Any, *, allowed_keys: set[str]) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        name = str(key).strip()
        number = _optional_int(count)
        if name not in allowed_keys:
            raise ValueError("Snapshot count label is not approved for publication")
        if number is not None:
            result[name] = number
    return result


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _contains_sensitive_public_text(text: str) -> bool:
    if any(character in text for character in ("\\", "\n", "\r")):
        return True
    if any(ord(character) < 32 for character in text):
        return True
    return SENSITIVE_PUBLIC_TEXT_PATTERN.search(text) is not None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a sanitized web snapshot from daily research artifacts."
    )
    parser.add_argument("--source-dir", default=os.environ.get("QUANT_RESEARCH_OUTPUT_DIR"))
    parser.add_argument("--run-card")
    parser.add_argument("--watchlist")
    parser.add_argument("--output", required=True)
    parser.add_argument("--published-at")
    parser.add_argument("--stale-after-days", type=int, default=DEFAULT_STALE_AFTER_DAYS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.run_card) != bool(args.watchlist):
        raise SystemExit("--run-card and --watchlist must be provided together")
    if args.run_card:
        run_card_path = Path(args.run_card)
        watchlist_path = Path(args.watchlist)
    elif args.source_dir:
        run_card_path, watchlist_path = discover_latest_sources(Path(args.source_dir))
    else:
        raise SystemExit(
            "Provide --source-dir, or set QUANT_RESEARCH_OUTPUT_DIR, or provide both "
            "--run-card and --watchlist"
        )

    snapshot = build_research_snapshot(
        run_card_path,
        watchlist_path,
        published_at=_parse_datetime(args.published_at) if args.published_at else None,
        stale_after_days=args.stale_after_days,
    )
    output_path = write_research_snapshot(snapshot, Path(args.output))
    print(
        json.dumps(
            {
                "status": "ok",
                "asof_date": snapshot["asof_date"],
                "run_status": snapshot["run_status"],
                "freshness": snapshot["freshness"]["status"],
                "priority_rows": snapshot["summary"]["priority_rows"],
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
