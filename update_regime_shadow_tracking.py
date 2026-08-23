"""Persist a durable regime shadow tracking ledger and readiness summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Mapping


LEDGER_FIELDS = (
    "asof_date",
    "recorded_at",
    "benchmark_fresh",
    "benchmark_last_date",
    "observation_valid",
    "decision",
    "risk_regime",
    "target_leverage",
    "baseline_daily_return",
    "dynamic_daily_return",
    "daily_return_delta",
    "baseline_equity",
    "dynamic_equity",
    "baseline_turnover",
    "dynamic_turnover",
    "baseline_cost",
    "dynamic_cost",
    "baseline_gross_exposure",
    "dynamic_gross_exposure",
    "full_history_return_delta",
    "full_history_drawdown_delta",
)

RETURN_TOLERANCE = 1e-6


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _required_float(row: Mapping[str, str], key: str, *, label: str) -> float:
    raw = str(row.get(key, "")).strip()
    if not raw:
        raise ValueError(f"{label} missing {key}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} invalid {key}: {raw}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} invalid {key}: {raw}")
    return value


def _optional_float(row: Mapping[str, str], key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Invalid {key}: {raw}")
    return value


def _read_equity_curve(path: Path, *, label: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"{label} equity curve must contain at least two rows")
    return rows


def _parse_latest_observation(rows: list[dict[str, str]], *, label: str) -> dict[str, object]:
    previous = rows[-2]
    latest = rows[-1]
    latest_date = str(latest.get("date", "")).strip()
    if not latest_date:
        raise ValueError(f"{label} latest row missing date")
    previous_equity = _required_float(previous, "equity", label=f"{label} previous row")
    latest_equity = _required_float(latest, "equity", label=f"{label} latest row")
    if previous_equity == 0.0:
        raise ValueError(f"{label} previous equity must be non-zero")
    daily_return = latest_equity / previous_equity - 1.0
    gross_return = _optional_float(latest, "gross_return")
    cost = _optional_float(latest, "cost")
    if gross_return is not None and cost is not None:
        expected_return = gross_return - cost
        if abs(daily_return - expected_return) > RETURN_TOLERANCE:
            raise ValueError(
                f"{label} derived daily return mismatch: {daily_return:.10f} vs {expected_return:.10f}"
            )
    return {
        "date": latest_date,
        "equity": latest_equity,
        "daily_return": daily_return,
        "turnover": _optional_float(latest, "turnover"),
        "cost": cost,
        "gross_exposure": _optional_float(latest, "gross_exposure"),
        "risk_regime": str(latest.get("risk_regime", "")).strip() or None,
        "target_leverage": _optional_float(latest, "target_leverage"),
    }


def _serialize_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    result = float(text)
    if not math.isfinite(result):
        raise ValueError(f"Invalid ledger float: {value}")
    return result


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


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _summary_from_rows(rows: list[dict[str, str]], target_days: int) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: row["asof_date"])
    valid_rows = [row for row in ordered if _parse_bool(row.get("observation_valid", "false"))]
    invalid_rows = [row for row in ordered if not _parse_bool(row.get("observation_valid", "false"))]
    baseline_returns = [_parse_float(row["baseline_daily_return"]) or 0.0 for row in valid_rows]
    dynamic_returns = [_parse_float(row["dynamic_daily_return"]) or 0.0 for row in valid_rows]
    baseline_turnover = [
        value
        for value in (_parse_float(row["baseline_turnover"]) for row in valid_rows)
        if value is not None
    ]
    dynamic_turnover = [
        value
        for value in (_parse_float(row["dynamic_turnover"]) for row in valid_rows)
        if value is not None
    ]
    baseline_exposure = [
        value
        for value in (_parse_float(row["baseline_gross_exposure"]) for row in valid_rows)
        if value is not None
    ]
    dynamic_exposure = [
        value
        for value in (_parse_float(row["dynamic_gross_exposure"]) for row in valid_rows)
        if value is not None
    ]
    baseline_cumulative = _product(baseline_returns)
    dynamic_cumulative = _product(dynamic_returns)
    baseline_drawdown = _max_drawdown(baseline_returns)
    dynamic_drawdown = _max_drawdown(dynamic_returns)
    baseline_avg_turnover = _average(baseline_turnover)
    dynamic_avg_turnover = _average(dynamic_turnover)
    baseline_avg_exposure = _average(baseline_exposure)
    dynamic_avg_exposure = _average(dynamic_exposure)
    remaining_days = max(target_days - len(valid_rows), 0)
    latest = ordered[-1] if ordered else None

    turnover_ratio: float | None
    turnover_ratio_gate: bool
    turnover_ratio_note: str
    if baseline_avg_turnover is None or dynamic_avg_turnover is None:
        turnover_ratio = None
        turnover_ratio_gate = False
        turnover_ratio_note = "missing_turnover"
    elif baseline_avg_turnover <= 0.0:
        turnover_ratio = None
        turnover_ratio_gate = False
        turnover_ratio_note = "baseline_turnover_zero_or_negative"
    else:
        turnover_ratio = dynamic_avg_turnover / baseline_avg_turnover
        turnover_ratio_gate = turnover_ratio <= 1.5
        turnover_ratio_note = "ok" if turnover_ratio_gate else "too_high"

    gate_results = {
        "no_invalid_observations": len(invalid_rows) == 0,
        "dynamic_outperforms_baseline": dynamic_cumulative > baseline_cumulative,
        "drawdown_not_materially_worse": dynamic_drawdown >= baseline_drawdown - 0.02,
        "turnover_ratio_within_limit": turnover_ratio_gate,
        "turnover_ratio": turnover_ratio,
        "turnover_ratio_note": turnover_ratio_note,
    }
    if len(valid_rows) < target_days:
        status = "collecting"
    elif all(
        (
            gate_results["no_invalid_observations"],
            gate_results["dynamic_outperforms_baseline"],
            gate_results["drawdown_not_materially_worse"],
            gate_results["turnover_ratio_within_limit"],
        )
    ):
        status = "manual_review_ready"
    else:
        status = "manual_review_not_ready"

    latest_risk_state = None
    if latest is not None:
        latest_risk_state = {
            "risk_regime": latest.get("risk_regime") or None,
            "target_leverage": _parse_float(latest.get("target_leverage", "")),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "latest_asof_date": latest.get("asof_date") if latest else None,
        "valid_observation_count": len(valid_rows),
        "invalid_observation_count": len(invalid_rows),
        "target_days": target_days,
        "remaining_days": remaining_days,
        "observed_cumulative_baseline_return": baseline_cumulative,
        "observed_cumulative_dynamic_return": dynamic_cumulative,
        "cumulative_return_delta": dynamic_cumulative - baseline_cumulative,
        "observed_baseline_max_drawdown": baseline_drawdown,
        "observed_dynamic_max_drawdown": dynamic_drawdown,
        "average_baseline_turnover": baseline_avg_turnover,
        "average_dynamic_turnover": dynamic_avg_turnover,
        "average_baseline_gross_exposure": baseline_avg_exposure,
        "average_dynamic_gross_exposure": dynamic_avg_exposure,
        "latest_benchmark_fresh": _parse_bool(latest.get("benchmark_fresh", "false")) if latest else None,
        "latest_benchmark_last_date": latest.get("benchmark_last_date") if latest else None,
        "latest_risk_state": latest_risk_state,
        "latest_decision": latest.get("decision") if latest else None,
        "manual_review_gates": gate_results,
        "automatic_promotion": False,
    }


def _render_report(summary: Mapping[str, object], latest_row: Mapping[str, str] | None) -> str:
    latest_risk_state = summary.get("latest_risk_state") or {}
    if not isinstance(latest_risk_state, Mapping):
        latest_risk_state = {}
    gate_results = summary.get("manual_review_gates") or {}
    if not isinstance(gate_results, Mapping):
        gate_results = {}
    lines = [
        "# Regime Shadow Tracking",
        "",
        f"- Status: `{summary['status']}`",
        f"- Valid observations: `{summary['valid_observation_count']}/{summary['target_days']}`",
        f"- Invalid observations: `{summary['invalid_observation_count']}`",
        f"- Remaining days: `{summary['remaining_days']}`",
        f"- Automatic promotion: `{summary['automatic_promotion']}`",
        "",
        "## Observed Shadow Performance",
        "",
        f"- Baseline cumulative return: `{summary['observed_cumulative_baseline_return']:.6f}`",
        f"- Dynamic cumulative return: `{summary['observed_cumulative_dynamic_return']:.6f}`",
        f"- Cumulative return delta: `{summary['cumulative_return_delta']:.6f}`",
        f"- Baseline max drawdown: `{summary['observed_baseline_max_drawdown']:.6f}`",
        f"- Dynamic max drawdown: `{summary['observed_dynamic_max_drawdown']:.6f}`",
        "",
        "## Latest Observation",
        "",
        f"- Benchmark fresh: `{summary['latest_benchmark_fresh']}`",
        f"- Risk regime: `{latest_risk_state.get('risk_regime')}`",
        f"- Target leverage: `{latest_risk_state.get('target_leverage')}`",
    ]
    if latest_row is not None:
        lines.extend(
            [
                f"- Latest asof date: `{latest_row['asof_date']}`",
                f"- Latest decision: `{latest_row['decision']}`",
                f"- Latest daily return delta: `{latest_row['daily_return_delta']}`",
                f"- Full-history return delta: `{latest_row['full_history_return_delta']}`",
                f"- Full-history drawdown delta: `{latest_row['full_history_drawdown_delta']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Manual Review Gates",
            "",
            f"- No invalid observations: `{gate_results.get('no_invalid_observations')}`",
            f"- Dynamic outperforms baseline: `{gate_results.get('dynamic_outperforms_baseline')}`",
            f"- Drawdown within tolerance: `{gate_results.get('drawdown_not_materially_worse')}`",
            f"- Turnover ratio within limit: `{gate_results.get('turnover_ratio_within_limit')}`",
            f"- Turnover ratio note: `{gate_results.get('turnover_ratio_note')}`",
            "",
            "This tracking ledger is observation-only and never promotes or changes the production strategy.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_text(rows: list[dict[str, str]]) -> str:
    from io import StringIO

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


def run_tracking(args: argparse.Namespace) -> dict[str, object]:
    comparison_dir = Path(args.comparison_dir).resolve()
    ledger_path = Path(args.ledger).resolve()
    summary_path = Path(args.summary).resolve()
    report_path = Path(args.report).resolve()
    target_days = int(args.target_days)
    comparison = _load_json_object(comparison_dir / "comparison.json")
    asof_date = str(comparison.get("asof") or comparison.get("asof_date") or "").strip()
    if not asof_date:
        raise ValueError("comparison.json missing asof")
    baseline_rows = _read_equity_curve(comparison_dir / "baseline" / "equity_curve.csv", label="baseline")
    dynamic_rows = _read_equity_curve(comparison_dir / "dynamic" / "equity_curve.csv", label="dynamic")
    baseline = _parse_latest_observation(baseline_rows, label="baseline")
    dynamic = _parse_latest_observation(dynamic_rows, label="dynamic")
    if baseline["date"] != asof_date or dynamic["date"] != asof_date or baseline["date"] != dynamic["date"]:
        raise ValueError(
            "comparison.json asof and latest baseline/dynamic equity dates must match exactly"
        )

    benchmark_last_date = str(comparison.get("benchmark_last_date") or "").strip() or asof_date
    raw_benchmark_fresh = comparison.get("benchmark_fresh") is True
    benchmark_fresh = raw_benchmark_fresh and benchmark_last_date == asof_date
    latest_dynamic_state = comparison.get("latest_dynamic_state")
    if not isinstance(latest_dynamic_state, dict):
        latest_dynamic_state = {}
    row = {
        "asof_date": asof_date,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_fresh": benchmark_fresh,
        "benchmark_last_date": benchmark_last_date,
        "observation_valid": benchmark_fresh,
        "decision": str(comparison.get("decision") or ""),
        "risk_regime": str(latest_dynamic_state.get("risk_regime") or dynamic.get("risk_regime") or ""),
        "target_leverage": latest_dynamic_state.get("target_leverage", dynamic.get("target_leverage")),
        "baseline_daily_return": baseline["daily_return"],
        "dynamic_daily_return": dynamic["daily_return"],
        "daily_return_delta": float(dynamic["daily_return"]) - float(baseline["daily_return"]),
        "baseline_equity": baseline["equity"],
        "dynamic_equity": dynamic["equity"],
        "baseline_turnover": baseline["turnover"],
        "dynamic_turnover": dynamic["turnover"],
        "baseline_cost": baseline["cost"],
        "dynamic_cost": dynamic["cost"],
        "baseline_gross_exposure": baseline["gross_exposure"],
        "dynamic_gross_exposure": dynamic["gross_exposure"],
        "full_history_return_delta": (
            float((comparison.get("delta") or {}).get("total_return", 0.0))
            if isinstance(comparison.get("delta"), dict)
            else 0.0
        ),
        "full_history_drawdown_delta": (
            float((comparison.get("delta") or {}).get("max_drawdown", 0.0))
            if isinstance(comparison.get("delta"), dict)
            else 0.0
        ),
    }

    existing_rows = _read_existing_ledger(ledger_path)
    preserved_rows = [prior for prior in existing_rows if prior.get("asof_date") != asof_date]
    serialized_row = {field: _serialize_scalar(row.get(field)) for field in LEDGER_FIELDS}
    ledger_rows = sorted([*preserved_rows, serialized_row], key=lambda item: item["asof_date"])
    summary = _summary_from_rows(ledger_rows, target_days)
    report = _render_report(summary, ledger_rows[-1] if ledger_rows else None)
    _atomic_publish_text(
        {
            ledger_path: _csv_text(ledger_rows),
            summary_path: json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            report_path: report,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update durable regime shadow tracking outputs.")
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-days", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    summary = run_tracking(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
