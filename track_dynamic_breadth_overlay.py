"""Forward-only observer for the preregistered dynamic breadth overlay.

This module compares the incumbent `core_rank` equity curve with the
challenger `core_breadth_guard` equity curve, but only on shared dates on or
after the preregistered observation start date. Historical performance before
the forward window is preserved as background context only and never enters the
promotion gate statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "dynamic_breadth_overlay_preregistration.json"

LEDGER_FIELDS = (
    "registration_id",
    "date",
    "recorded_at",
    "source_latest_date",
    "observation_day_index",
    "incumbent_id",
    "challenger_id",
    "incumbent_equity",
    "challenger_equity",
    "incumbent_daily_return",
    "challenger_daily_return",
    "excess_return",
    "incumbent_observation_nav",
    "challenger_observation_nav",
    "incumbent_observation_drawdown",
    "challenger_observation_drawdown",
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "registration_id",
    "status",
    "frozen_at",
    "observation_start_date",
    "target_valid_trade_days",
    "incumbent_model_id",
    "challenger_model_id",
    "input_contract",
    "gates",
    "research_only",
    "promotion_allowed",
    "automatic_model_change",
    "notes",
}
INPUT_CONTRACT_KEYS = {
    "required_columns",
    "latest_date_must_match",
    "reject_duplicate_dates",
}
GATE_KEYS = {
    "cumulative_excess_return_gt",
    "min_positive_excess_day_ratio",
    "max_drawdown_worsening",
}


@dataclass(frozen=True)
class Preregistration:
    registration_id: str
    frozen_at: str
    observation_start_date: str
    target_valid_trade_days: int
    incumbent_model_id: str
    challenger_model_id: str
    cumulative_excess_return_gt: float
    min_positive_excess_day_ratio: float
    max_drawdown_worsening: float
    research_only: bool
    promotion_allowed: bool
    automatic_model_change: bool
    config_sha256: str


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _require_exact_keys(mapping: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(mapping.keys())
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unknown = sorted(actual.difference(expected))
        raise ValueError(f"{label} keys mismatch: missing={missing}, unknown={unknown}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preregistration(path: Path) -> Preregistration:
    raw = _load_json_object(path)
    _require_exact_keys(raw, TOP_LEVEL_KEYS, label="dynamic breadth preregistration")

    input_contract = raw["input_contract"]
    if not isinstance(input_contract, dict):
        raise ValueError("input_contract must be an object")
    _require_exact_keys(input_contract, INPUT_CONTRACT_KEYS, label="input_contract")

    gates = raw["gates"]
    if not isinstance(gates, dict):
        raise ValueError("gates must be an object")
    _require_exact_keys(gates, GATE_KEYS, label="gates")

    if int(raw["schema_version"]) != 1:
        raise ValueError("Unsupported dynamic breadth preregistration schema_version")
    if str(raw["status"]) != "preregistered_research_only":
        raise ValueError("Preregistration status must remain preregistered_research_only")
    if str(raw["registration_id"]) != "dynamic_breadth_overlay_strict_labels_20260722":
        raise ValueError("Preregistration registration_id must remain strict-label registration")
    if str(raw["frozen_at"]) != "2026-07-22":
        raise ValueError("Preregistration frozen_at must remain 2026-07-22")
    if str(raw["observation_start_date"]) != "2026-07-23":
        raise ValueError("Preregistration observation_start_date must remain 2026-07-23")
    if int(raw["target_valid_trade_days"]) != 60:
        raise ValueError("Preregistration target_valid_trade_days must remain 60")
    if str(raw["incumbent_model_id"]) != "core_rank":
        raise ValueError("Preregistration incumbent_model_id must remain core_rank")
    if str(raw["challenger_model_id"]) != "core_breadth_guard":
        raise ValueError("Preregistration challenger_model_id must remain core_breadth_guard")
    if list(input_contract["required_columns"]) != ["date", "equity"]:
        raise ValueError("Preregistration required_columns must remain ['date', 'equity']")
    if input_contract["latest_date_must_match"] is not True:
        raise ValueError("Preregistration latest_date_must_match must remain true")
    if input_contract["reject_duplicate_dates"] is not True:
        raise ValueError("Preregistration reject_duplicate_dates must remain true")
    if float(gates["cumulative_excess_return_gt"]) != 0.0:
        raise ValueError("Preregistration cumulative_excess_return_gt must remain 0.0")
    if float(gates["min_positive_excess_day_ratio"]) != 0.55:
        raise ValueError("Preregistration min_positive_excess_day_ratio must remain 0.55")
    if float(gates["max_drawdown_worsening"]) != 0.03:
        raise ValueError("Preregistration max_drawdown_worsening must remain 0.03")
    if raw["research_only"] is not True:
        raise ValueError("Preregistration research_only must remain true")
    if raw["promotion_allowed"] is not False:
        raise ValueError("Preregistration promotion_allowed must remain false")
    if raw["automatic_model_change"] is not False:
        raise ValueError("Preregistration automatic_model_change must remain false")

    return Preregistration(
        registration_id=str(raw["registration_id"]),
        frozen_at=str(raw["frozen_at"]),
        observation_start_date=str(raw["observation_start_date"]),
        target_valid_trade_days=int(raw["target_valid_trade_days"]),
        incumbent_model_id=str(raw["incumbent_model_id"]),
        challenger_model_id=str(raw["challenger_model_id"]),
        cumulative_excess_return_gt=float(gates["cumulative_excess_return_gt"]),
        min_positive_excess_day_ratio=float(gates["min_positive_excess_day_ratio"]),
        max_drawdown_worsening=float(gates["max_drawdown_worsening"]),
        research_only=True,
        promotion_allowed=False,
        automatic_model_change=False,
        config_sha256=_sha256_file(path),
    )


def load_equity_curve(path: Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"date", "equity"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} equity curve is missing columns {missing}: {path}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError(f"{label} equity curve contains invalid dates: {path}")
    if out["date"].duplicated().any():
        raise ValueError(f"{label} equity curve contains duplicate dates: {path}")
    if out["equity"].isna().any() or out["equity"].le(0.0).any():
        raise ValueError(f"{label} equity curve contains non-positive equity: {path}")

    out = out.sort_values("date").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError(f"{label} equity curve requires at least two rows: {path}")

    out["daily_return"] = out["equity"].pct_change(fill_method=None)
    return out[["date", "equity", "daily_return"]]


def _iso_date(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _serialize_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_float(value: str) -> float:
    return float(value.strip())


def _read_existing_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _product(returns: list[float]) -> float:
    nav = 1.0
    for value in returns:
        nav *= 1.0 + value
    return nav - 1.0


def _max_drawdown(returns: list[float]) -> float:
    nav = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in returns:
        nav *= 1.0 + value
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1.0)
    return drawdown


def build_forward_rows(
    incumbent: pd.DataFrame,
    challenger: pd.DataFrame,
    preregistration: Preregistration,
    *,
    source_latest_date: str,
) -> list[dict[str, object]]:
    start = pd.Timestamp(preregistration.observation_start_date).normalize()
    incumbent_by_date = incumbent.set_index("date")
    challenger_by_date = challenger.set_index("date")
    common_dates = sorted(date for date in incumbent_by_date.index.intersection(challenger_by_date.index) if date >= start)
    recorded_at = datetime.now().isoformat(timespec="seconds")

    incumbent_nav = 1.0
    challenger_nav = 1.0
    incumbent_peak = 1.0
    challenger_peak = 1.0
    rows: list[dict[str, object]] = []

    for observation_day_index, date in enumerate(common_dates, start=1):
        incumbent_daily = incumbent_by_date.at[date, "daily_return"]
        challenger_daily = challenger_by_date.at[date, "daily_return"]
        if pd.isna(incumbent_daily):
            raise ValueError(
                f"{preregistration.incumbent_model_id} daily return is unavailable on {_iso_date(date)}"
            )
        if pd.isna(challenger_daily):
            raise ValueError(
                f"{preregistration.challenger_model_id} daily return is unavailable on {_iso_date(date)}"
            )

        incumbent_daily = float(incumbent_daily)
        challenger_daily = float(challenger_daily)
        incumbent_nav *= 1.0 + incumbent_daily
        challenger_nav *= 1.0 + challenger_daily
        incumbent_peak = max(incumbent_peak, incumbent_nav)
        challenger_peak = max(challenger_peak, challenger_nav)

        rows.append(
            {
                "registration_id": preregistration.registration_id,
                "date": _iso_date(date),
                "recorded_at": recorded_at,
                "source_latest_date": source_latest_date,
                "observation_day_index": observation_day_index,
                "incumbent_id": preregistration.incumbent_model_id,
                "challenger_id": preregistration.challenger_model_id,
                "incumbent_equity": float(incumbent_by_date.at[date, "equity"]),
                "challenger_equity": float(challenger_by_date.at[date, "equity"]),
                "incumbent_daily_return": incumbent_daily,
                "challenger_daily_return": challenger_daily,
                "excess_return": challenger_daily - incumbent_daily,
                "incumbent_observation_nav": incumbent_nav,
                "challenger_observation_nav": challenger_nav,
                "incumbent_observation_drawdown": incumbent_nav / incumbent_peak - 1.0,
                "challenger_observation_drawdown": challenger_nav / challenger_peak - 1.0,
            }
        )
    return rows


def _merge_ledger_rows(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    replacement_keys = {
        (str(row["registration_id"]), str(row["date"]))
        for row in new_rows
    }
    preserved = [
        row
        for row in existing_rows
        if (str(row.get("registration_id", "")), str(row.get("date", ""))) not in replacement_keys
    ]
    serialized_new = [
        {field: _serialize_scalar(row.get(field)) for field in LEDGER_FIELDS}
        for row in new_rows
    ]
    return sorted(
        [*preserved, *serialized_new],
        key=lambda row: (str(row.get("registration_id", "")), str(row.get("date", ""))),
    )


def _background_context(
    incumbent: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    observation_start_date: str,
) -> dict[str, object]:
    start = pd.Timestamp(observation_start_date).normalize()
    common_dates = incumbent["date"].tolist()
    challenger_dates = set(challenger["date"].tolist())
    common_pre_start = [date for date in common_dates if date in challenger_dates and date < start]
    common_forward = [date for date in common_dates if date in challenger_dates and date >= start]
    return {
        "incumbent_first_date": _iso_date(pd.Timestamp(incumbent["date"].min())),
        "challenger_first_date": _iso_date(pd.Timestamp(challenger["date"].min())),
        "common_background_days_before_start": len(common_pre_start),
        "common_forward_days_available_from_inputs": len(common_forward),
        "note": f"Dates before {observation_start_date} are background only and excluded from forward statistics.",
    }


def summarize_observations(
    ledger_rows: list[dict[str, str]],
    preregistration: Preregistration,
    *,
    source_latest_date: str,
    background: Mapping[str, object],
) -> dict[str, object]:
    observed_rows = sorted(
        [row for row in ledger_rows if row.get("registration_id") == preregistration.registration_id],
        key=lambda row: str(row["date"]),
    )
    observation_count = len(observed_rows)
    remaining_days = max(preregistration.target_valid_trade_days - observation_count, 0)
    incumbent_returns = [_parse_float(row["incumbent_daily_return"]) for row in observed_rows]
    challenger_returns = [_parse_float(row["challenger_daily_return"]) for row in observed_rows]
    excess_returns = [_parse_float(row["excess_return"]) for row in observed_rows]

    incumbent_cumulative_return = _product(incumbent_returns) if incumbent_returns else None
    challenger_cumulative_return = _product(challenger_returns) if challenger_returns else None
    cumulative_excess_return = (
        challenger_cumulative_return - incumbent_cumulative_return
        if incumbent_cumulative_return is not None and challenger_cumulative_return is not None
        else None
    )
    positive_excess_day_ratio = (
        sum(value > 0.0 for value in excess_returns) / len(excess_returns)
        if excess_returns
        else None
    )
    incumbent_max_drawdown = _max_drawdown(incumbent_returns) if incumbent_returns else None
    challenger_max_drawdown = _max_drawdown(challenger_returns) if challenger_returns else None
    max_drawdown_worsening = (
        abs(challenger_max_drawdown) - abs(incumbent_max_drawdown)
        if incumbent_max_drawdown is not None and challenger_max_drawdown is not None
        else None
    )

    gate_evaluation_allowed = observation_count >= preregistration.target_valid_trade_days
    if gate_evaluation_allowed:
        cumulative_gate = bool(
            cumulative_excess_return is not None
            and cumulative_excess_return > preregistration.cumulative_excess_return_gt
        )
        hit_rate_gate = bool(
            positive_excess_day_ratio is not None
            and positive_excess_day_ratio >= preregistration.min_positive_excess_day_ratio
        )
        drawdown_gate = bool(
            max_drawdown_worsening is not None
            and max_drawdown_worsening <= preregistration.max_drawdown_worsening
        )
        all_gates_pass = cumulative_gate and hit_rate_gate and drawdown_gate
        status = "manual_review_ready" if all_gates_pass else "manual_review_not_ready"
    else:
        cumulative_gate = None
        hit_rate_gate = None
        drawdown_gate = None
        all_gates_pass = None
        status = "collecting"

    latest_observation_date = observed_rows[-1]["date"] if observed_rows else None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "registration_id": preregistration.registration_id,
        "config_sha256": preregistration.config_sha256,
        "status": status,
        "frozen_at": preregistration.frozen_at,
        "observation_start_date": preregistration.observation_start_date,
        "source_latest_date": source_latest_date,
        "latest_observation_date": latest_observation_date,
        "target_valid_trade_days": preregistration.target_valid_trade_days,
        "valid_observation_count": observation_count,
        "remaining_observation_days": remaining_days,
        "gate_evaluation_allowed": gate_evaluation_allowed,
        "observed_incumbent_cumulative_return": incumbent_cumulative_return,
        "observed_challenger_cumulative_return": challenger_cumulative_return,
        "cumulative_excess_return": cumulative_excess_return,
        "positive_excess_day_ratio": positive_excess_day_ratio,
        "observed_incumbent_max_drawdown": incumbent_max_drawdown,
        "observed_challenger_max_drawdown": challenger_max_drawdown,
        "max_drawdown_worsening": max_drawdown_worsening,
        "manual_review_gates": {
            "cumulative_excess_return_positive": cumulative_gate,
            "positive_excess_day_ratio_met": hit_rate_gate,
            "challenger_drawdown_within_tolerance": drawdown_gate,
            "all_gates_pass": all_gates_pass,
        },
        "research_only": preregistration.research_only,
        "promotion_allowed": preregistration.promotion_allowed,
        "automatic_model_change": preregistration.automatic_model_change,
        "provisional_only": not gate_evaluation_allowed,
        "background": dict(background),
    }


def render_report(summary: Mapping[str, object]) -> str:
    gates = summary.get("manual_review_gates") or {}
    background = summary.get("background") or {}
    lines = [
        "# Dynamic Breadth Overlay Forward Observer",
        "",
        f"- Registration ID: `{summary['registration_id']}`",
        f"- Frozen at: `{summary['frozen_at']}`",
        f"- Observation start: `{summary['observation_start_date']}`",
        f"- Shared source latest date: `{summary['source_latest_date']}`",
        f"- Status: `{summary['status']}`",
        f"- Valid observation days: `{summary['valid_observation_count']}/{summary['target_valid_trade_days']}`",
        f"- Gate evaluation allowed: `{summary['gate_evaluation_allowed']}`",
        f"- Research only: `{summary['research_only']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Automatic model change: `{summary['automatic_model_change']}`",
        f"- Provisional only: `{summary['provisional_only']}`",
        "",
        "## Forward Window Metrics",
        "",
        f"- Incumbent cumulative return: `{summary['observed_incumbent_cumulative_return']}`",
        f"- Challenger cumulative return: `{summary['observed_challenger_cumulative_return']}`",
        f"- Cumulative excess return: `{summary['cumulative_excess_return']}`",
        f"- Positive excess day ratio: `{summary['positive_excess_day_ratio']}`",
        f"- Incumbent max drawdown: `{summary['observed_incumbent_max_drawdown']}`",
        f"- Challenger max drawdown: `{summary['observed_challenger_max_drawdown']}`",
        f"- Max drawdown worsening: `{summary['max_drawdown_worsening']}`",
        "",
        "## Manual Review Gates",
        "",
        f"- Cumulative excess return positive: `{gates.get('cumulative_excess_return_positive')}`",
        f"- Positive excess day ratio met: `{gates.get('positive_excess_day_ratio_met')}`",
        f"- Challenger drawdown within tolerance: `{gates.get('challenger_drawdown_within_tolerance')}`",
        f"- All gates pass: `{gates.get('all_gates_pass')}`",
        "",
        "## Background Only",
        "",
        f"- Incumbent first date: `{background.get('incumbent_first_date')}`",
        f"- Challenger first date: `{background.get('challenger_first_date')}`",
        f"- Shared background days before start: `{background.get('common_background_days_before_start')}`",
        f"- Shared forward days available from inputs: `{background.get('common_forward_days_available_from_inputs')}`",
        f"- Note: `{background.get('note')}`",
        "",
        "Historical dates before the forward window remain descriptive context only and are excluded from promotion statistics.",
        "",
    ]
    return "\n".join(lines)


def _csv_text(rows: list[dict[str, str]]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _stage_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_publish_text(files: Mapping[Path, str]) -> None:
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    existed: dict[Path, bool] = {}
    try:
        for path, content in files.items():
            staged[path] = _stage_text(path, content)
        for path in files:
            existed[path] = path.exists()
            if path.exists():
                backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.bak")
                path.replace(backup)
                backups[path] = backup
            staged[path].replace(path)
    except Exception:
        for path, backup in backups.items():
            if path.exists():
                path.unlink(missing_ok=True)
            backup.replace(path)
        for path, did_exist in existed.items():
            if not did_exist and path.exists():
                path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def run_overlay(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(getattr(args, "config", DEFAULT_CONFIG)).resolve()
    preregistration = load_preregistration(config_path)
    incumbent_path = Path(args.incumbent_equity).resolve()
    challenger_path = Path(args.challenger_equity).resolve()
    ledger_path = Path(args.ledger).resolve()
    summary_path = Path(args.summary).resolve()
    report_path = Path(args.report).resolve()

    incumbent = load_equity_curve(incumbent_path, label=preregistration.incumbent_model_id)
    challenger = load_equity_curve(challenger_path, label=preregistration.challenger_model_id)
    incumbent_latest = pd.Timestamp(incumbent["date"].max()).normalize()
    challenger_latest = pd.Timestamp(challenger["date"].max()).normalize()
    if incumbent_latest != challenger_latest:
        raise ValueError(
            "incumbent and challenger equity curve latest dates must match exactly"
        )
    source_latest_date = _iso_date(incumbent_latest)

    background = _background_context(
        incumbent,
        challenger,
        observation_start_date=preregistration.observation_start_date,
    )
    new_rows = build_forward_rows(
        incumbent,
        challenger,
        preregistration,
        source_latest_date=source_latest_date,
    )
    existing_rows = _read_existing_ledger(ledger_path)
    ledger_rows = _merge_ledger_rows(existing_rows, new_rows)
    summary = summarize_observations(
        ledger_rows,
        preregistration,
        source_latest_date=source_latest_date,
        background=background,
    )
    report = render_report(summary)
    _atomic_publish_text(
        {
            ledger_path: _csv_text(ledger_rows),
            summary_path: json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            report_path: report,
        }
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track the preregistered dynamic breadth overlay on forward-only shared dates."
    )
    parser.add_argument("--incumbent-equity", required=True)
    parser.add_argument("--challenger-equity", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_overlay(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
