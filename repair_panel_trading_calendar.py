"""Quarantine non-session panel rows and atomically write a clean panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from panel_io import read_panel, write_panel_atomic
from trading_calendar import trading_session_audit, validate_trading_sessions


def repair_panel(
    panel_path: Path,
    benchmark_path: Path,
    output_path: Path,
    quarantine_path: Path,
) -> dict[str, object]:
    panel = read_panel(panel_path, parse_dates=["date"])
    audit_before = trading_session_audit(panel["date"], benchmark_path)
    invalid_dates = pd.to_datetime(audit_before["invalid_panel_dates"])
    if audit_before["missing_panel_dates"]:
        raise ValueError(
            "calendar repair cannot synthesize missing sessions: "
            f"{audit_before['missing_panel_dates']}"
        )
    if invalid_dates.empty:
        raise ValueError("panel has no invalid trading dates to repair")

    invalid_mask = panel["date"].dt.normalize().isin(invalid_dates)
    quarantine = panel.loc[invalid_mask].copy()
    cleaned = panel.loc[~invalid_mask].copy()
    if quarantine.empty:
        raise RuntimeError("calendar audit found invalid dates but no rows matched")
    audit_after = validate_trading_sessions(
        cleaned["date"], benchmark_path, context="repaired panel"
    )

    write_panel_atomic(quarantine, quarantine_path)
    write_panel_atomic(cleaned, output_path)
    return {
        "schema_version": 1,
        "status": "repaired",
        "input_panel": str(panel_path),
        "output_panel": str(output_path),
        "quarantine_path": str(quarantine_path),
        "input_rows": int(len(panel)),
        "output_rows": int(len(cleaned)),
        "quarantined_rows": int(len(quarantine)),
        "quarantined_dates": audit_before["invalid_panel_dates"],
        "audit_before": audit_before,
        "audit_after": audit_after,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair panel trading calendar.")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine-output", required=True)
    parser.add_argument("--audit-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = repair_panel(
        Path(args.panel),
        Path(args.benchmark),
        Path(args.output),
        Path(args.quarantine_output),
    )
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
