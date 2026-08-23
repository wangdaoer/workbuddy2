from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from workspace_paths import dashboard_root


DEFAULT_OUTPUT_ROOT = Path("outputs/high_return_v2")
DEFAULT_DASHBOARD_OUTPUT = dashboard_root() / "data" / "quant-model3-latest.json"
LOCAL_PATH_RE = re.compile(r"[A-Za-z]:\\[^\"'\n\r]+")


def date_token(asof_date: str) -> str:
    return asof_date.replace("-", "")


def latest_asof_date(output_root: Path) -> str:
    candidates = sorted(output_root.glob("merged_priority_watchlist_*.csv"), key=lambda item: item.stat().st_mtime)
    candidates = [
        item
        for item in candidates
        if not item.stem.endswith("_cn") and re.search(r"merged_priority_watchlist_\d{8}", item.stem)
    ]
    if not candidates:
        raise FileNotFoundError(f"No merged_priority_watchlist_YYYYMMDD.csv found under {output_root}")
    match = re.search(r"(\d{8})", candidates[-1].stem)
    if not match:
        raise ValueError(f"Cannot infer date from {candidates[-1]}")
    token = match.group(1)
    return f"{token[:4]}-{token[4:6]}-{token[6:8]}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def latest_state_record(state_log: Path, asof_date: str) -> dict[str, Any]:
    if not state_log.exists():
        return {}
    records: list[dict[str, Any]] = []
    for line in state_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("asof_date") == asof_date:
            records.append(record)
    return records[-1] if records else {}


def read_state_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in state record: {path}")
    return value


def sanitize_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, str):
        if LOCAL_PATH_RE.search(value):
            return LOCAL_PATH_RE.sub("[local-path]", value)
        return value
    return value


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def number_or_none(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def pct_or_none(value: Any) -> float | None:
    number = number_or_none(value)
    if number is None:
        return None
    return round(number * 100.0, 4)


def bool_value(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y", "是"}


def bucket_label(bucket: str) -> str:
    return {
        "action_focus": "行动关注",
        "model_focus": "模型关注",
        "pattern_watch": "形态观察",
        "risk_watch": "风险观察",
        "review_later": "稍后复盘",
    }.get(bucket, bucket or "未分层")


def shape_priority_row(row: dict[str, str]) -> dict[str, Any]:
    bucket = clean_text(row.get("priority_bucket"))
    shadow_signal = clean_text(row.get("shadow_account_signal")) or "neutral"
    shadow_label = clean_text(row.get("shadow_account_signal_cn")) or {
        "risk": "风险提示",
        "prefer": "偏好确认",
        "neutral": "无额外提示",
    }.get(shadow_signal, shadow_signal)
    shaped = {
        "code": clean_text(row.get("symbol") or row.get("code")),
        "name": clean_text(row.get("stock_name") or row.get("name")),
        "bucket": bucket,
        "bucketLabel": bucket_label(bucket),
        "priorityScore": number_or_none(row.get("priority_score")),
        "rank": number_or_none(row.get("personal_rank")),
        "selected": bool_value(row.get("personal_selected")),
        "targetWeightPct": pct_or_none(row.get("personal_target_weight")),
        "action": clean_text(row.get("personal_action")),
        "actionLabel": clean_text(row.get("personal_action_cn")),
        "reason": clean_text(row.get("personal_reasons_cn") or row.get("pattern_reason")),
        "trendState": clean_text(row.get("trend_state")),
        "patternType": clean_text(row.get("pattern_type")),
        "patternScore": number_or_none(row.get("pattern_score")),
        "riskFlags": clean_text(row.get("risk_flags")),
        "close": number_or_none(row.get("close")),
        "return5dPct": pct_or_none(row.get("return_5d")),
        "return20dPct": pct_or_none(row.get("return_20d")),
        "return60dPct": pct_or_none(row.get("return_60d")),
        "closePositionPct": pct_or_none(row.get("close_position")),
        "shadowAccount": {
            "signal": shadow_signal,
            "label": shadow_label,
            "score": number_or_none(row.get("shadow_account_signal_score")),
            "notes": clean_text(row.get("shadow_account_notes")),
        },
    }
    if shaped["shadowAccount"]["score"] is None:
        shaped["shadowAccount"].pop("score")
    if not shaped["shadowAccount"]["notes"]:
        shaped["shadowAccount"].pop("notes")
    return {key: value for key, value in shaped.items() if value not in (None, "")}


def compact_artifacts(output_root: Path, asof_date: str) -> dict[str, str]:
    token = date_token(asof_date)
    names = {
        "priorityWatchlist": f"merged_priority_watchlist_{token}.csv",
        "priorityWatchlistCn": f"merged_priority_watchlist_{token}_cn.csv",
        "modelDecision": f"merged_model_decision_table_{token}.csv",
        "earlyWatchlist": f"early_pattern_watchlist_{token}.csv",
        "report": f"merged_daily_outputs_{token}.md",
    }
    return {key: value for key, value in names.items() if (output_root / value).exists()}


def _validate_current_inputs(
    asof_date: str,
    paths: dict[str, Path],
    state_record: dict[str, Any],
    row_counts: dict[str, int],
) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required MarketLens input missing: {', '.join(missing)}")
    if state_record.get("asof_date") != asof_date:
        raise ValueError(
            "MarketLens state date mismatch: "
            f"expected {asof_date}, got {state_record.get('asof_date')!r}"
        )
    if state_record.get("run_status") != "success":
        raise ValueError(
            "MarketLens publication requires a successful current run state, got "
            f"{state_record.get('run_status')!r}"
        )

    verification = state_record.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("MarketLens state record is missing verification details")
    expected_counts = {
        "priority_rows": row_counts["priority"],
        "early_pattern_rows": row_counts["early"],
        "model_decision_rows": row_counts["model"],
    }
    for key, actual in expected_counts.items():
        if verification.get(key) != actual:
            raise ValueError(
                f"MarketLens state count mismatch for {key}: "
                f"state={verification.get(key)!r}, actual={actual}"
            )

    artifacts = state_record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("MarketLens state record is missing artifact paths")
    artifact_keys = {
        "priority": "priority",
        "early": "early_watchlist",
        "model": "model_decision",
    }
    for input_key, artifact_key in artifact_keys.items():
        recorded = artifacts.get(artifact_key)
        if not recorded:
            raise ValueError(f"MarketLens state record is missing artifact: {artifact_key}")
        recorded_path = Path(str(recorded)).resolve(strict=False)
        expected_path = paths[input_key].resolve(strict=False)
        if recorded_path != expected_path:
            raise ValueError(
                f"MarketLens artifact mismatch for {artifact_key}: "
                f"state={recorded_path}, expected={expected_path}"
            )


def build_feed(
    asof_date: str,
    output_root: Path,
    state_log: Path,
    *,
    state_record: dict[str, Any] | None = None,
    require_complete_inputs: bool = False,
) -> dict[str, Any]:
    token = date_token(asof_date)
    priority_path = output_root / f"merged_priority_watchlist_{token}.csv"
    early_path = output_root / f"early_pattern_watchlist_{token}.csv"
    model_path = output_root / f"merged_model_decision_table_{token}.csv"
    priority_rows_raw = read_csv_rows(priority_path)
    early_rows = read_csv_rows(early_path)
    model_rows = read_csv_rows(model_path)
    current_state = dict(state_record) if state_record is not None else latest_state_record(
        state_log, asof_date
    )
    if require_complete_inputs:
        _validate_current_inputs(
            asof_date,
            {"priority": priority_path, "early": early_path, "model": model_path},
            current_state,
            {
                "priority": len(priority_rows_raw),
                "early": len(early_rows),
                "model": len(model_rows),
            },
        )
    priority_rows = [shape_priority_row(row) for row in priority_rows_raw]
    bucket_counts = dict(Counter(row.get("bucket", "未分层") for row in priority_rows))
    shadow_account_counts = dict(
        Counter(
            row.get("shadowAccount", {}).get("signal", "neutral")
            for row in priority_rows
        )
    )
    verification = {
        "tests": current_state.get("verification", {}).get("tests", "unknown"),
        "priorityRows": len(priority_rows),
        "earlyPatternRows": len(early_rows),
        "modelDecisionRows": len(model_rows),
        "missingNames": sum(1 for row in priority_rows if not row.get("name")),
    }
    feed = {
        "schemaVersion": 1,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "asofDate": asof_date,
        "source": {
            "model": "high_risk_quant_model3",
            "purpose": "MarketLens website research snapshot",
            "publicPathsOnly": True,
        },
        "runType": current_state.get("run_type", "unknown"),
        "verification": verification,
        "bucketCounts": bucket_counts,
        "shadowAccountCounts": shadow_account_counts,
        "buckets": [
            {"bucket": bucket, "label": bucket_label(bucket), "count": count}
            for bucket, count in bucket_counts.items()
        ],
        "top10": priority_rows[:10],
        "priorityRows": priority_rows,
        "artifacts": compact_artifacts(output_root, asof_date),
    }
    return sanitize_public_value(feed)


def export_feed(
    asof_date: str,
    output_root: Path,
    state_log: Path,
    output_path: Path,
    dashboard_output: Path | None = None,
    *,
    state_record: dict[str, Any] | None = None,
    require_complete_inputs: bool = False,
) -> dict[str, Any]:
    feed = build_feed(
        asof_date,
        output_root,
        state_log,
        state_record=state_record,
        require_complete_inputs=require_complete_inputs,
    )
    text = json.dumps(feed, ensure_ascii=False, indent=2) + "\n"
    _write_text_atomic(output_path, text)
    if dashboard_output:
        if dashboard_output.resolve(strict=False) != output_path.resolve(strict=False):
            _write_text_atomic(dashboard_output, text)
    return feed


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Model3 snapshot for the MarketLens website.")
    parser.add_argument("--asof-date", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--state-log", default="daily_run_state.jsonl")
    parser.add_argument("--state-record", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--dashboard-output", default=str(DEFAULT_DASHBOARD_OUTPUT))
    parser.add_argument("--require-complete-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    asof_date = args.asof_date or latest_asof_date(output_root)
    output_path = Path(args.output) if args.output else output_root / "marketlens_model3_latest.json"
    dashboard_output = Path(args.dashboard_output) if args.dashboard_output else None
    state_record = read_state_record(Path(args.state_record)) if args.state_record else None
    feed = export_feed(
        asof_date,
        output_root,
        Path(args.state_log),
        output_path,
        dashboard_output,
        state_record=state_record,
        require_complete_inputs=args.require_complete_inputs,
    )
    print(f"Exported Model3 feed: {output_path}")
    if dashboard_output:
        print(f"Dashboard copy: {dashboard_output}")
    print(f"Rows: {feed['verification']['priorityRows']} / asof: {feed['asofDate']}")


if __name__ == "__main__":
    main()
