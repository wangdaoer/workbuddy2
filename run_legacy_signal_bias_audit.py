from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from execution_rules import (
    ZERO_PLACEHOLDER_COLUMNS,
    drop_terminal_zero_placeholders,
    normalize_symbol,
)
from legacy_observation_factors import (
    MONEY_FLOW_COLUMNS,
    OBSERVATION_SCORE_INPUTS,
    audit_static_universe,
    compute_observation_factors,
)
from run_backtest import BacktestEngine, StrategyConfig, prepare_prices


REUSABLE_FACTOR_CANDIDATES = [
    "momentum_20",
    "momentum_60",
    "breakout_distance_20",
    "trend_acceleration",
    "liquidity_20",
    "liquidity_stability_20",
]
REJECTED_ASSUMPTIONS = [
    "static_full_period_winner_universe",
    "same_close_fill",
    "unrestricted_limit_fill",
    "leverage_as_alpha",
]
RAW_POINT_IN_TIME = "raw_point_in_time"
CASE_ROLES = {
    "dynamic_same_close": "legacy_bias_control",
    "dynamic_lag1_close": "intermediate_bias_control",
    "dynamic_lag1_next_open": "execution_transition",
    "dynamic_lag1_next_open_limits": "partial_strict",
    "dynamic_lag1_next_open_limits_gap3": "partial_strict",
    "dynamic_lag1_next_open_limits_gap3_cost2x": "partial_strict",
}
EXPECTED_CASE_IDS = tuple(CASE_ROLES)
CASE_ALLOWED_CHANGED_PATHS = {
    "dynamic_same_close": ["execution.model", "universe.selection_lag_days"],
    "dynamic_lag1_close": ["universe.selection_lag_days"],
    "dynamic_lag1_next_open": ["execution.model"],
    "dynamic_lag1_next_open_limits": [
        "execution.block_limit_down_sells",
        "execution.block_limit_up_buys",
    ],
    "dynamic_lag1_next_open_limits_gap3": ["execution.max_buy_open_gap"],
    "dynamic_lag1_next_open_limits_gap3_cost2x": [
        "cost.commission_bps",
        "cost.impact_bps",
    ],
}


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    changed_assumption: str
    config: dict[str, object]


def load_raw_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config: {path}")
    return loaded


def build_ablation_cases(base: dict) -> list[AuditCase]:
    control = copy.deepcopy(base)
    control.setdefault("universe", {})["selection_lag_days"] = 0
    control.setdefault("execution", {})["model"] = "close_to_close"

    specifications = [
        ("dynamic_same_close", "legacy_bias_control", None),
        (
            "dynamic_lag1_close",
            "universe_selection_lag_days_1",
            ("universe", "selection_lag_days", 1),
        ),
        (
            "dynamic_lag1_next_open",
            "next_open_execution",
            ("execution", "model", "next_open"),
        ),
        (
            "dynamic_lag1_next_open_limits",
            "limit_fill_constraints",
            (
                "execution",
                None,
                {
                    "block_limit_up_buys": True,
                    "block_limit_down_sells": True,
                },
            ),
        ),
        (
            "dynamic_lag1_next_open_limits_gap3",
            "max_buy_open_gap_3pct",
            ("execution", "max_buy_open_gap", 0.03),
        ),
        (
            "dynamic_lag1_next_open_limits_gap3_cost2x",
            "commission_and_impact_bps_2x",
            (
                "cost",
                None,
                {
                    "commission_bps": float(base["cost"]["commission_bps"]) * 2,
                    "impact_bps": float(base["cost"]["impact_bps"]) * 2,
                },
            ),
        ),
    ]

    cases: list[AuditCase] = []
    current = control
    for case_id, changed_assumption, mutation in specifications:
        config = copy.deepcopy(current)
        if mutation is not None:
            section, key, value = mutation
            target = config.setdefault(section, {})
            if key is None:
                target.update(value)
            else:
                target[key] = value
        if case_id != "dynamic_same_close":
            config.setdefault("universe", {})["selection_lag_days"] = 1
        cases.append(AuditCase(case_id, changed_assumption, config))
        current = config
    return cases


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stream_file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _git_state(repo_path: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable", True
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        dirty = True
    return revision, dirty


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


class _PublicationLock(AbstractContextManager["_PublicationLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._owned = False

    def __enter__(self) -> "_PublicationLock":
        payload = json.dumps({"pid": os.getpid()}).encode("utf-8")
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    lock_data = json.loads(self.path.read_text(encoding="utf-8"))
                    owner_pid = int(lock_data.get("pid", -1))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    owner_pid = -1
                if _process_exists(owner_pid):
                    raise RuntimeError(
                        f"audit publication is locked by process {owner_pid}"
                    )
                self.path.unlink(missing_ok=True)
                continue
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)
            self._owned = True
            return self
        raise RuntimeError("could not acquire audit publication lock")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False


def _copy_file_atomically(source: Path, destination: Path) -> None:
    temporary = destination.parent / f".{destination.name}.latest-{uuid.uuid4().hex}"
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _repair_current_compatibility_output(output_dir: Path) -> bool:
    pointer_path = output_dir.parent / f".{output_dir.name}.CURRENT.json"
    versions_dir = output_dir.parent / f".{output_dir.name}.versions"
    manifest_name = "legacy_signal_bias_audit_manifest.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_id = pointer["run_id"]
        if (
            not isinstance(run_id, str)
            or not run_id
            or Path(run_id).name != run_id
            or "/" in run_id
            or "\\" in run_id
        ):
            return False
        version_dir = versions_dir / run_id
        manifest_path = version_dir / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id:
            return False
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            return False

        expected: dict[str, dict[str, object]] = {}
        for name, identity in artifacts.items():
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name == manifest_name
                or not isinstance(identity, dict)
            ):
                return False
            source = version_dir / name
            if not source.is_file() or _stream_file_identity(source) != identity:
                return False
            expected[name] = identity
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, identity in sorted(expected.items()):
        destination = output_dir / name
        if not destination.is_file() or _stream_file_identity(destination) != identity:
            _copy_file_atomically(version_dir / name, destination)

    if (
        not (output_dir / manifest_name).is_file()
        or (output_dir / manifest_name).read_bytes() != manifest_path.read_bytes()
    ):
        _copy_file_atomically(manifest_path, output_dir / manifest_name)
    return True


def _publish_atomically(
    output_dir: Path,
    build_staging: Callable[[Path], None],
    *,
    run_id: str = "manual-publication",
    verify_inputs: Callable[[], None] | None = None,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.publish.lock"
    with _PublicationLock(lock_path):
        _repair_current_compatibility_output(output_dir)
        for abandoned in output_dir.parent.glob(f".{output_dir.name}.staging-*"):
            if abandoned.is_dir():
                shutil.rmtree(abandoned)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-",
                dir=output_dir.parent,
            )
        )
        pointer = output_dir.parent / f".{output_dir.name}.CURRENT.tmp"
        try:
            build_staging(staging_dir)
            if verify_inputs is not None:
                verify_inputs()

            output_dir.mkdir(parents=True, exist_ok=True)
            versions_dir = output_dir.parent / f".{output_dir.name}.versions"
            versions_dir.mkdir(exist_ok=True)
            version_dir = versions_dir / run_id
            if version_dir.exists():
                raise FileExistsError(f"immutable audit version exists: {version_dir}")
            os.replace(staging_dir, version_dir)

            names = [path.name for path in version_dir.iterdir() if path.is_file()]
            names.sort(key=lambda name: name.endswith("_manifest.json"))
            for name in names:
                _copy_file_atomically(version_dir / name, output_dir / name)

            pointer.write_text(
                json.dumps({"run_id": run_id}, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            os.replace(
                pointer,
                output_dir.parent / f".{output_dir.name}.CURRENT.json",
            )
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            for temporary in (
                output_dir.glob(".*.latest-*") if output_dir.exists() else ()
            ):
                temporary.unlink(missing_ok=True)
            pointer.unlink(missing_ok=True)


def _verify_input_identities(
    identities: dict[Path, dict[str, object]],
) -> None:
    changed = [
        str(path)
        for path, expected in identities.items()
        if _stream_file_identity(path) != expected
    ]
    if changed:
        raise RuntimeError(
            "input changed during audit: " + ", ".join(sorted(changed))
        )


def run_ablation_case(
    case: AuditCase, data: pd.DataFrame
) -> dict[str, object]:
    config = StrategyConfig.from_dict(copy.deepcopy(case.config))
    result = BacktestEngine(config).run(data.copy())
    metrics = result["metrics"]
    row = _case_metadata_row(case)
    row.update(
        {
            "total_return": float(metrics["total_return"]),
            "annualized_return": float(metrics["annualized_return"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "turnover": float(metrics["avg_turnover"]),
            "gross_exposure": float(metrics["avg_gross_exposure"]),
            "blocked_limit_up_buys": int(metrics["blocked_limit_up_buys"]),
            "blocked_limit_down_sells": int(
                metrics["blocked_limit_down_sells"]
            ),
            "blocked_open_gap_buys": int(metrics["blocked_open_gap_buys"]),
            "blocked_orders_total": int(metrics["blocked_orders_total"]),
            "final_equity": float(metrics["final_equity"]),
            "source_identity": "broad_data",
            "source_row_count": int(len(data)),
            "source_symbol_count": int(data["symbol"].astype(str).nunique()),
        }
    )
    return row


def build_audit_verdict(
    cases: pd.DataFrame, universe_audit: dict
) -> dict[str, object]:
    returns: dict[str, float] = {}
    case_ids: list[str] = []
    if not cases.empty and {"case_id", "total_return"}.issubset(cases.columns):
        case_ids = cases["case_id"].astype(str).tolist()
        numeric = cases[["case_id", "total_return"]].copy()
        numeric["total_return"] = pd.to_numeric(
            numeric["total_return"], errors="coerce"
        )
        numeric = numeric.dropna(subset=["total_return"])
        returns = {
            str(row.case_id): float(row.total_return)
            for row in numeric.itertuples(index=False)
        }
    case_sequence_valid = case_ids == list(EXPECTED_CASE_IDS)
    evaluated = case_sequence_valid and set(returns) == set(EXPECTED_CASE_IDS)
    universe_delta = None
    same_close_delta = None
    if evaluated:
        universe_delta = round(
            float(returns["dynamic_same_close"])
            - float(returns["dynamic_lag1_close"]),
            12,
        )
        same_close_delta = round(
            float(returns["dynamic_lag1_close"])
            - float(returns["dynamic_lag1_next_open"]),
            12,
        )

    lookthrough = bool(universe_audit.get("lookthrough_suspect", False))
    bias_confirmed = lookthrough or (
        evaluated
        and ((universe_delta or 0.0) > 0.0 or (same_close_delta or 0.0) > 0.0)
    )
    return {
        "research_only": True,
        "trade_instruction": False,
        "legacy_bias_control_case": "dynamic_same_close",
        "clean_reference_case": "dynamic_lag1_next_open_limits_gap3_cost2x",
        "clean_reference_role": "partial_strict",
        "strict_execution_supported": False,
        "backtests_evaluated": evaluated,
        "case_sequence_valid": case_sequence_valid,
        "static_universe_lookthrough_suspect": lookthrough,
        "universe_lag_return_delta": universe_delta,
        "same_close_return_delta": same_close_delta,
        "legacy_bias_risk": (
            "confirmed"
            if bias_confirmed
            else "not_confirmed"
            if evaluated
            else "not_evaluated"
        ),
    }


def _case_metadata_row(case: AuditCase) -> dict[str, object]:
    config = StrategyConfig.from_dict(copy.deepcopy(case.config))
    return {
        "case_id": case.case_id,
        "changed_assumption": case.changed_assumption,
        "case_role": CASE_ROLES[case.case_id],
        "legacy_bias_control": case.case_id == "dynamic_same_close",
        "selection_lag_days": config.universe_selection_lag_days,
        "execution_model": config.execution_model,
        "block_limit_up_buys": config.block_limit_up_buys,
        "block_limit_down_sells": config.block_limit_down_sells,
        "limit_buffer": config.limit_buffer,
        "max_buy_open_gap": config.max_buy_open_gap,
        "commission_bps": config.commission_bps,
        "impact_bps": config.impact_bps,
        "leverage": config.leverage,
        "allowed_changed_paths": json.dumps(
            CASE_ALLOWED_CHANGED_PATHS[case.case_id], separators=(",", ":")
        ),
        "full_config_fingerprint": _fingerprint(case.config),
        "source_identity": "broad_data",
        "source_row_count": None,
        "source_symbol_count": None,
        "strict_execution_supported": False,
        "signal_fingerprint": _fingerprint(case.config["signal"]),
        "portfolio_fingerprint": _fingerprint(case.config["portfolio"]),
        "risk_fingerprint": _fingerprint(case.config["risk"]),
        "total_return": None,
        "annualized_return": None,
        "max_drawdown": None,
        "turnover": None,
        "gross_exposure": None,
        "blocked_limit_up_buys": None,
        "blocked_limit_down_sells": None,
        "blocked_open_gap_buys": None,
        "blocked_orders_total": None,
        "final_equity": None,
        "delta_vs_previous": None,
        "delta_total_return_vs_previous": None,
        "research_only": True,
        "trade_instruction": False,
    }


def _manifest_case(case: AuditCase) -> dict[str, object]:
    config = StrategyConfig.from_dict(copy.deepcopy(case.config))
    return {
        "case_id": case.case_id,
        "changed_assumption": case.changed_assumption,
        "case_role": CASE_ROLES[case.case_id],
        "source_identity": "broad_data",
        "allowed_changed_paths": CASE_ALLOWED_CHANGED_PATHS[case.case_id],
        "full_config_fingerprint": _fingerprint(case.config),
        "parameters": {
            "selection_lag_days": config.universe_selection_lag_days,
            "execution_model": config.execution_model,
            "block_limit_up_buys": config.block_limit_up_buys,
            "block_limit_down_sells": config.block_limit_down_sells,
            "limit_buffer": config.limit_buffer,
            "max_buy_open_gap": config.max_buy_open_gap,
            "commission_bps": config.commission_bps,
            "impact_bps": config.impact_bps,
            "leverage": config.leverage,
        },
        "config": copy.deepcopy(case.config),
    }


def _amount_provenance(data: pd.DataFrame) -> str:
    if "amount" not in data.columns:
        return "unavailable"

    present = data["amount"].notna()
    if not present.any():
        return "unavailable"
    numeric = pd.to_numeric(data["amount"], errors="coerce")
    numeric_present = numeric[present]
    if (
        numeric_present.isna().any()
        or not np.isfinite(numeric_present.to_numpy(dtype=float)).all()
        or not numeric_present.gt(0.0).any()
    ):
        return "unavailable"

    marker_columns = ("amount_substituted", "amount_is_substituted")
    for column in marker_columns:
        if column in data.columns:
            values = data[column]
            normalized = values.astype(str).str.strip().str.lower()
            if normalized.isin({"true", "1", "yes", "y"}).any():
                return "substituted"

    if "amount_source" in data.columns:
        source = data["amount_source"].astype(str).str.lower()
        if source.str.contains(
            r"substitut|estimat|derived|volume.?[x*].?close", regex=True
        ).any():
            return "substituted"

    if {"amount", "volume", "close"}.issubset(data.columns):
        amount = pd.to_numeric(data["amount"], errors="coerce")
        derived = pd.to_numeric(data["volume"], errors="coerce") * pd.to_numeric(
            data["close"], errors="coerce"
        )
        comparable = amount.notna() & derived.notna()
        if comparable.any() and np.isclose(
            amount[comparable].to_numpy(dtype=float),
            derived[comparable].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-8,
        ).all():
            return "substituted"
    return "raw_point_in_time"


def _money_flow_provenance(data: pd.DataFrame) -> str:
    recognized = [column for column in MONEY_FLOW_COLUMNS if column in data]
    if not recognized:
        return "unavailable"

    generic_columns = [
        column
        for column in ("money_flow_provenance", "money_flow_source")
        if column in data
    ]
    per_column_provenance = {
        flow_column: [
            column
            for column in (
                f"{flow_column}_provenance",
                f"{flow_column}_source",
            )
            if column in data
        ]
        for flow_column in recognized
    }
    if not generic_columns and any(
        not columns for columns in per_column_provenance.values()
    ):
        return "unverified"
    provenance_columns = [*generic_columns]
    provenance_columns.extend(
        column
        for columns in per_column_provenance.values()
        for column in columns
    )
    if not provenance_columns:
        return "unverified"

    values: list[str] = []
    for column in provenance_columns:
        normalized = (
            data[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"[\s-]+", "_", regex=True)
        )
        if normalized.isna().any() or normalized.eq("").any():
            return "unverified"
        values.extend(normalized.tolist())

    unique = set(values)
    if len(unique) != 1:
        return "mixed"
    value = unique.pop()
    if value == RAW_POINT_IN_TIME:
        return RAW_POINT_IN_TIME
    if "substitut" in value:
        return "substituted"
    if "deriv" in value or "estimat" in value:
        return "derived"
    return "unverified"


def _clean_factor_input(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    amount_provenance = _amount_provenance(data)
    money_flow_provenance = _money_flow_provenance(data)
    clean = data.copy()
    if amount_provenance != RAW_POINT_IN_TIME:
        clean.drop(columns=["amount"], inplace=True, errors="ignore")
    if money_flow_provenance != RAW_POINT_IN_TIME:
        clean.drop(columns=list(MONEY_FLOW_COLUMNS), inplace=True, errors="ignore")
    return clean, amount_provenance, money_flow_provenance


def _prepare_broad_panel(
    data: pd.DataFrame,
    *,
    quarantine_interior_zero_tails: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    close = pd.to_numeric(data["close"], errors="coerce")
    input_zero_rows = int(close.eq(0.0).sum())
    try:
        clean = drop_terminal_zero_placeholders(data)
    except ValueError:
        if not quarantine_interior_zero_tails:
            raise
    else:
        return clean, {
            "policy": "strict_terminal_only",
            "input_zero_close_rows": input_zero_rows,
            "terminal_placeholder_rows_dropped": input_zero_rows,
            "interior_zero_rows": 0,
            "quarantined_symbol_count": 0,
            "quarantined_symbols": [],
            "quarantined_tail_rows": 0,
            "quarantine_details": [],
            "effective_rows": int(len(clean)),
            "effective_symbol_count": int(clean["symbol"].astype(str).nunique()),
        }

    zero_close = close.eq(0.0)
    if not set(ZERO_PLACEHOLDER_COLUMNS).issubset(data.columns):
        raise ValueError(
            "zero close rows may be quarantined only when OHLCV/amount "
            "companions prove all-zero placeholders"
        )
    companions = data.loc[zero_close, ZERO_PLACEHOLDER_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if companions.isna().any().any() or not companions.eq(0.0).all(axis=None):
        raise ValueError(
            "zero close rows may be quarantined only when OHLCV/amount "
            "companions prove all-zero placeholders"
        )

    symbols = data["symbol"].map(normalize_symbol)
    dates = pd.to_datetime(data["date"], errors="raise")
    check = pd.DataFrame(
        {"symbol": symbols, "date": dates, "zero_close": zero_close},
        index=data.index,
    ).sort_values(["symbol", "date"], kind="mergesort")

    rejected_indexes: set[object] = set()
    terminal_indexes: set[object] = set()
    interior_zero_rows = 0
    details: list[dict[str, object]] = []
    for symbol, group in check.groupby("symbol", sort=False):
        zero_positions = group["zero_close"].to_numpy().nonzero()[0]
        if not len(zero_positions):
            continue
        first_zero = int(zero_positions[0])
        tail = group.iloc[first_zero:]
        if tail["zero_close"].all():
            terminal_indexes.update(tail.index.tolist())
            continue

        rejected_indexes.update(tail.index.tolist())
        interior_count = int(tail["zero_close"].sum())
        interior_zero_rows += interior_count
        details.append(
            {
                "symbol": symbol,
                "first_zero_date": pd.Timestamp(tail.iloc[0]["date"]).date().isoformat(),
                "interior_zero_rows": interior_count,
                "quarantined_tail_rows": int(len(tail)),
            }
        )

    clean = data.drop(index=[*rejected_indexes, *terminal_indexes]).copy()
    clean = drop_terminal_zero_placeholders(clean)
    quarantined_symbols = sorted(detail["symbol"] for detail in details)
    return clean, {
        "policy": "quarantine_symbol_tail_after_interior_zero",
        "input_zero_close_rows": input_zero_rows,
        "terminal_placeholder_rows_dropped": int(len(terminal_indexes)),
        "interior_zero_rows": interior_zero_rows,
        "quarantined_symbol_count": int(len(quarantined_symbols)),
        "quarantined_symbols": quarantined_symbols,
        "quarantined_tail_rows": int(len(rejected_indexes)),
        "quarantine_details": details,
        "effective_rows": int(len(clean)),
        "effective_symbol_count": int(clean["symbol"].astype(str).nunique()),
    }


def _input_metadata(
    path: Path,
    data: pd.DataFrame,
    *,
    amount_provenance: str | None = None,
    money_flow_provenance: str | None = None,
    clean_money_flow_evidence: bool | None = None,
    file_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    dates = pd.to_datetime(data["date"], errors="coerce")
    file_identity = file_identity or _stream_file_identity(path)
    metadata: dict[str, object] = {
        "path": str(path.resolve()),
        **file_identity,
        "rows": int(len(data)),
        "symbols": int(data["symbol"].astype(str).nunique()),
        "date_min": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_max": dates.max().date().isoformat() if dates.notna().any() else None,
    }
    if amount_provenance is not None:
        metadata["amount_provenance"] = amount_provenance
        metadata["clean_liquidity_evidence"] = (
            amount_provenance == RAW_POINT_IN_TIME
        )
    if money_flow_provenance is not None:
        metadata["money_flow_provenance"] = money_flow_provenance
        metadata["clean_money_flow_evidence"] = bool(
            clean_money_flow_evidence
            if clean_money_flow_evidence is not None
            else money_flow_provenance == RAW_POINT_IN_TIME
        )
    return metadata


def _factor_dictionary(
    availability: dict[str, bool],
    amount_provenance: str,
    money_flow_provenance: str,
    factor_metadata: dict[str, object],
) -> dict[str, object]:
    descriptions = {
        "momentum_20": "20-session trailing close momentum",
        "momentum_60": "60-session trailing close momentum",
        "breakout_distance_20": "distance from the shifted prior 20-session high",
        "trend_acceleration": "trailing close trend acceleration",
        "volatility_20": "20-session trailing close-return volatility",
        "liquidity_20": "20-session trailing median point-in-time amount",
        "liquidity_stability_20": "stability of trailing point-in-time amount",
        "flow_persistence": "share of positive point-in-time money-flow sessions",
        "capacity_risk": "position notional divided by trailing point-in-time amount",
    }
    directions = {
        "momentum_20": "higher_supports_score",
        "momentum_60": "higher_supports_score",
        "breakout_distance_20": "higher_supports_score",
        "trend_acceleration": "higher_supports_score",
        "volatility_20": "higher_penalizes_score",
        "liquidity_20": "higher_supports_score",
        "liquidity_stability_20": "higher_supports_score",
        "flow_persistence": "higher_supports_score",
        "capacity_risk": "higher_penalizes_score",
    }
    sources = {
        "momentum_20": "broad_data.close",
        "momentum_60": "broad_data.close",
        "breakout_distance_20": "broad_data.close",
        "trend_acceleration": "broad_data.close",
        "volatility_20": "broad_data.close",
        "liquidity_20": "broad_data.amount",
        "liquidity_stability_20": "broad_data.amount",
        "flow_persistence": (
            f"broad_data.{factor_metadata['money_flow_column']}"
            if factor_metadata.get("money_flow_column")
            else "unavailable"
        ),
        "capacity_risk": "broad_data.position_notional,broad_data.amount",
    }
    fields: dict[str, dict[str, object]] = {}
    for name, description in descriptions.items():
        available = bool(availability.get(name, False))
        fields[name] = {
            "description": description,
            "direction": directions[name],
            "timing": "observation_close_t_trailing_sessions",
            "source": sources[name],
            "available": available,
            "score_participation": name in OBSERVATION_SCORE_INPUTS,
            "clean_factor_claim": available,
            "evaluation_only": False,
            "signal_lag_sessions": 1,
        }

    risk_availability = factor_metadata.get("risk_availability", {})
    for name, direction, timing, source, evaluation_only in (
        (
            "limit_up_risk",
            "true_means_buy_execution_risk",
            "observation_close_t",
            "broad_data.close,prior_close,point_in_time_limit_fields_or_board_default",
            False,
        ),
        (
            "limit_down_risk",
            "true_means_sell_execution_risk",
            "observation_close_t",
            "broad_data.close,prior_close,point_in_time_limit_fields_or_board_default",
            False,
        ),
        (
            "opening_gap_risk",
            "true_means_next_open_buy_gap_exceeds_threshold",
            "next_observed_session_open_evaluation_only",
            "broad_data.open_t_plus_1,broad_data.close_t",
            True,
        ),
    ):
        fields[name] = {
            "description": name.replace("_", " "),
            "direction": direction,
            "timing": timing,
            "source": source,
            "available": bool(risk_availability.get(name, False)),
            "score_participation": False,
            "clean_factor_claim": False,
            "evaluation_only": evaluation_only,
        }

    for name, description, source in (
        (
            "history_eligible",
            "required trailing price factors are available",
            "derived factor availability",
        ),
        (
            "liquidity_eligible",
            "point-in-time trailing amount is available",
            "liquidity_20 availability",
        ),
        (
            "score_eligible",
            "row is eligible for an observation score",
            "history_eligible",
        ),
    ):
        fields[name] = {
            "description": description,
            "direction": "true_means_eligible",
            "timing": "observation_close_t",
            "source": source,
            "available": True,
            "score_participation": False,
            "clean_factor_claim": False,
            "evaluation_only": False,
        }
    fields["observation_score"] = {
        "description": "date-level normalized rank score from the fixed score allowlist",
        "direction": "higher_is_stronger_observation",
        "timing": "observation_close_t",
        "source": "OBSERVATION_SCORE_INPUTS",
        "available": True,
        "score_participation": False,
        "clean_factor_claim": True,
        "evaluation_only": False,
        "score_output": True,
    }

    factors = {
        name: copy.deepcopy(fields[name])
        for name in [*REUSABLE_FACTOR_CANDIDATES, "flow_persistence", "capacity_risk"]
    }
    factors["forward_return"] = {
        "description": "future return used only to evaluate audit outcomes",
        "available": True,
        "clean_factor_claim": False,
        "evaluation_only": True,
        "score_participation": False,
    }
    factors["opening_gap"] = copy.deepcopy(fields["opening_gap_risk"])
    return {
        "research_only": True,
        "trade_instruction": False,
        "signal_lag_sessions": 1,
        "signal_timing": (
            "close_t_actionable_no_earlier_than_next_observed_trading_session"
        ),
        "amount_provenance": amount_provenance,
        "money_flow_provenance": money_flow_provenance,
        "clean_money_flow_evidence": bool(
            availability.get("flow_persistence", False)
        ),
        "limit_status_coverage": factor_metadata["limit_status_coverage"],
        "unsupported_limit_status": factor_metadata["unsupported_limit_status"],
        "opening_gap_threshold": factor_metadata["opening_gap_threshold"],
        "fields": fields,
        "factors": factors,
    }


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_markdown(
    verdict: dict[str, object],
    factor_availability: dict[str, bool],
    amount_provenance: str,
    money_flow_provenance: str,
    skipped_backtests: bool,
    factor_metadata: dict[str, object],
    factor_snapshot: dict[str, object],
    data_quality: dict[str, object],
) -> str:
    liquidity_status = (
        "可作为观察候选，使用原始时点金额。"
        if factor_availability.get("liquidity_20", False)
        else f"仅保留候选定义；当前金额来源为 {amount_provenance}，不形成清洁因子结论。"
    )
    backtest_status = "已跳过" if skipped_backtests else "已完成"
    unavailable_fields = []
    if not factor_availability.get("liquidity_20", False):
        unavailable_fields.append(
            f"- 金额/流动性：来源为 {amount_provenance}，不形成清洁因子结论。"
        )
    if not factor_availability.get("flow_persistence", False):
        unavailable_fields.append(
            f"- 资金流：来源为 {money_flow_provenance}，不形成清洁因子结论。"
        )
    if not unavailable_fields:
        unavailable_fields.append("- 无。")
    return (
        "# 旧版信号偏差审计\n\n"
        "research_only: true  \n"
        "trade_instruction: false  \n\n"
        "本报告仅用于研究审计，不构成任何交易指令。\n\n"
        "## 审计边界\n\n"
        "- `broad_data` 驱动全部观察因子与六个动态消融回测。\n"
        "- `selected_data` 仅用于静态全周期回看审计，不进入因子或动态回测。\n"
        f"- 数据质量策略：`{data_quality['policy']}`；隔离证券 {data_quality['quarantined_symbol_count']} 个、隔离尾部记录 {data_quality['quarantined_tail_rows']} 行，不插值、不补价。\n"
        "- `dynamic_same_close`: `legacy_bias_control`，不属于清洁证据。\n"
        "- `dynamic_lag1_close`: `intermediate_bias_control`，仍使用同收盘价。\n"
        "- `dynamic_lag1_next_open`: `execution_transition`，尚未启用严格成交限制。\n"
        "- `dynamic_lag1_next_open_limits`: `partial_strict`；其后代同属 `partial_strict`。\n"
        f"- 涨跌停覆盖：`{factor_metadata['limit_status_coverage']}`；ST 5% and other exceptional limits are unsupported without point-in-time source fields.\n"
        "- 除旧版控制外，其余案例均显式使用 `universe.selection_lag_days=1`。\n"
        "- 清洁观察因子统一使用 `signal_lag_sessions=1`。\n"
        f"- 最新因子仅含全局最大观察日 `{factor_snapshot['global_max_observation_date']}`；过期证券 {factor_snapshot['stale_symbol_count']} 个：{factor_snapshot['stale_symbols']}。\n"
        f"- 回测状态：{backtest_status}；偏差风险：{verdict['legacy_bias_risk']}。\n\n"
        "## 已验证证据\n\n"
        f"- 回测状态：{backtest_status}；偏差风险：{verdict['legacy_bias_risk']}。\n"
        "- 清洁观察因子统一使用 `signal_lag_sessions=1`，且不生成交易指令。\n"
        "- 前向收益与开盘跳空仅用于评估，不作为观察因子。\n\n"
        "## 解释\n\n"
        "- 动态滚动动量：20 与 60 个已观察交易时段的收盘动量。\n"
        "- 前高突破：相对移位后的前 20 个交易时段最高价。\n"
        "- 趋势加速度：仅使用历史收盘序列计算。\n"
        f"- 流动性：{liquidity_status}\n\n"
        "## 明确拒绝\n\n"
        "- 静态全周期赢家成员：存在全周期回看风险，不进入清洁因子。\n"
        "- 同收盘价成交：仅保留为旧版偏差控制。\n"
        "- 无限制涨跌停成交：不作为可复用假设。\n"
        "- 杠杆作为阿尔法：杠杆仅是风险暴露，不是信号来源。\n\n"
        "## 不可用字段\n\n"
        + "\n".join(unavailable_fields)
        + "\n"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legacy signal timing bias.")
    parser.add_argument("--broad-data", required=True)
    parser.add_argument("--selected-data", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-backtests", action="store_true")
    parser.add_argument(
        "--quarantine-interior-zero-tails",
        action="store_true",
        help=(
            "Conservatively reject each affected symbol from its first proven "
            "interior all-zero placeholder onward."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    broad_path = Path(args.broad_data)
    selected_path = Path(args.selected_data)
    base_path = Path(args.base_config)
    output_dir = Path(args.output_dir)

    input_identities = {
        path: _stream_file_identity(path)
        for path in (broad_path, selected_path, base_path)
    }

    broad_raw = pd.read_csv(broad_path)
    selected_raw = pd.read_csv(selected_path)
    broad_panel, data_quality = _prepare_broad_panel(
        broad_raw,
        quarantine_interior_zero_tails=args.quarantine_interior_zero_tails,
    )
    (
        clean_factor_data,
        amount_provenance,
        money_flow_provenance,
    ) = _clean_factor_input(broad_panel)
    factors, factor_metadata = compute_observation_factors(clean_factor_data)
    flow_available = bool(
        factor_metadata["factor_availability"].get("flow_persistence", False)
    ) and money_flow_provenance == RAW_POINT_IN_TIME
    factor_metadata["factor_availability"]["flow_persistence"] = flow_available
    factor_metadata["money_flow_provenance"] = money_flow_provenance
    factor_metadata["clean_money_flow_evidence"] = flow_available
    universe_audit = audit_static_universe(selected_raw, broad_panel)

    base = load_raw_yaml(base_path)
    cases = build_ablation_cases(base)
    protected = {
        section: {_fingerprint(case.config[section]) for case in cases}
        for section in ("signal", "portfolio", "risk")
    }
    if any(len(fingerprints) != 1 for fingerprints in protected.values()):
        raise ValueError("Ablation cases changed a protected strategy fingerprint")

    if args.skip_backtests:
        case_rows = []
        for case in cases:
            row = _case_metadata_row(case)
            row["source_row_count"] = int(len(broad_panel))
            row["source_symbol_count"] = int(
                broad_panel["symbol"].astype(str).nunique()
            )
            case_rows.append(row)
    else:
        backtest_data = prepare_prices(
            broad_panel,
            base.get("start_date"),
            base.get("end_date"),
        )
        case_rows = []
        previous_return: float | None = None
        for case in cases:
            row = run_ablation_case(case, backtest_data)
            if previous_return is not None:
                delta = round(float(row["total_return"]) - previous_return, 12)
                row["delta_vs_previous"] = delta
                row["delta_total_return_vs_previous"] = delta
            previous_return = float(row["total_return"])
            case_rows.append(row)

    for row in case_rows:
        row["limit_status_coverage"] = factor_metadata["limit_status_coverage"]
        row["strict_execution_supported"] = False

    cases_frame = pd.DataFrame(case_rows)
    verdict = build_audit_verdict(cases_frame, universe_audit)
    availability = {
        name: bool(value)
        for name, value in factor_metadata["factor_availability"].items()
    }
    reusable_factors = [
        name for name in REUSABLE_FACTOR_CANDIDATES if availability.get(name, False)
    ]

    history = factors.sort_values(["symbol", "date"], kind="stable")
    global_latest_date = pd.Timestamp(history["date"].max())
    latest = history.loc[history["date"].eq(global_latest_date)].copy()
    all_symbols = set(history["symbol"].astype(str))
    latest_symbols = set(latest["symbol"].astype(str))
    stale_symbols = sorted(all_symbols.difference(latest_symbols))
    factor_snapshot = {
        "global_max_observation_date": global_latest_date.date().isoformat(),
        "latest_row_count": int(len(latest)),
        "latest_symbol_count": int(len(latest_symbols)),
        "stale_symbol_count": int(len(stale_symbols)),
        "stale_symbols": stale_symbols,
    }
    selected_money_flow_provenance = _money_flow_provenance(selected_raw)
    selected_flow_available = selected_money_flow_provenance == RAW_POINT_IN_TIME and any(
        column in selected_raw
        and pd.to_numeric(selected_raw[column], errors="coerce").notna().any()
        for column in MONEY_FLOW_COLUMNS
    )
    inputs = {
        "broad_data": _input_metadata(
            broad_path,
            broad_raw,
            amount_provenance=amount_provenance,
            money_flow_provenance=money_flow_provenance,
            clean_money_flow_evidence=flow_available,
            file_identity=input_identities[broad_path],
        ),
        "selected_data": _input_metadata(
            selected_path,
            selected_raw,
            amount_provenance=_amount_provenance(selected_raw),
            money_flow_provenance=selected_money_flow_provenance,
            clean_money_flow_evidence=selected_flow_available,
            file_identity=input_identities[selected_path],
        ),
        "base_config": {
            "path": str(base_path.resolve()),
            **input_identities[base_path],
            "full_config_fingerprint": _fingerprint(base),
        },
    }
    inputs["broad_data"]["source_role"] = "factors_and_dynamic_ablations"
    inputs["selected_data"]["source_role"] = "static_audit_only"
    inputs["base_config"]["source_role"] = "configuration"
    generated = datetime.now(timezone.utc)
    generated_at = generated.isoformat(timespec="seconds")
    run_id = (
        "legacy-signal-bias-audit-"
        f"{generated.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    )
    manifest_id = run_id
    code_revision, code_dirty = _git_state(Path(__file__).resolve().parent)
    source_identity = {
        "factor_history": "broad_data",
        "dynamic_ablations": "broad_data",
        "static_audit_selected": "selected_data",
        "static_audit_comparison": "broad_data",
    }

    cases_frame = cases_frame.copy()
    cases_frame["run_id"] = run_id
    cases_frame["manifest_id"] = manifest_id
    history = history.copy()
    latest = latest.copy()
    for frame in (history, latest):
        frame["run_id"] = run_id
        frame["manifest_id"] = manifest_id
    for row in case_rows:
        row["run_id"] = run_id
        row["manifest_id"] = manifest_id

    audit_payload = {
        "run_id": run_id,
        "manifest_id": manifest_id,
        "research_only": True,
        "trade_instruction": False,
        "generated_at": generated_at,
        "inputs": inputs,
        "source_identity": source_identity,
        "factor_snapshot": factor_snapshot,
        "data_quality": data_quality,
        "universe_audit": universe_audit,
        "ablation_results": case_rows,
        "audit_verdict": verdict,
        "reusable_factors": reusable_factors,
        "rejected_assumptions": REJECTED_ASSUMPTIONS,
    }
    dictionary_payload = _factor_dictionary(
        availability,
        amount_provenance,
        money_flow_provenance,
        factor_metadata,
    )
    dictionary_payload["run_id"] = run_id
    dictionary_payload["manifest_id"] = manifest_id
    markdown = (
        f"run_id: {run_id}  \n"
        f"manifest_id: {manifest_id}  \n\n"
        + _build_markdown(
            verdict,
            availability,
            amount_provenance,
            money_flow_provenance,
            args.skip_backtests,
            factor_metadata,
            factor_snapshot,
            data_quality,
        )
    )
    manifest_base = {
        "run_id": run_id,
        "manifest_id": manifest_id,
        "schema_version": 1,
        "generated_at": generated_at,
        "code_revision": code_revision,
        "code_dirty": code_dirty,
        "research_only": True,
        "trade_instruction": False,
        "inputs": inputs,
        "base_config_fingerprint": _fingerprint(base),
        "source_identity": source_identity,
        "limit_status_coverage": factor_metadata["limit_status_coverage"],
        "data_quality": data_quality,
        "cases": [_manifest_case(case) for case in cases],
    }

    artifact_names = (
        "legacy_signal_bias_audit.csv",
        "observation_factor_history.csv",
        "observation_factors_latest.csv",
        "legacy_signal_bias_audit.json",
        "observation_factor_dictionary.json",
        "legacy_signal_bias_audit.md",
    )

    def build_staging(staging_dir: Path) -> None:
        cases_frame.to_csv(staging_dir / artifact_names[0], index=False)
        history.to_csv(staging_dir / artifact_names[1], index=False)
        latest.to_csv(staging_dir / artifact_names[2], index=False)
        _write_json(staging_dir / artifact_names[3], audit_payload)
        _write_json(staging_dir / artifact_names[4], dictionary_payload)
        (staging_dir / artifact_names[5]).write_text(markdown, encoding="utf-8")

        manifest = copy.deepcopy(manifest_base)
        manifest["artifacts"] = {
            name: _stream_file_identity(staging_dir / name)
            for name in artifact_names
        }
        _write_json(
            staging_dir / "legacy_signal_bias_audit_manifest.json",
            manifest,
        )

    _publish_atomically(
        output_dir,
        build_staging,
        run_id=run_id,
        verify_inputs=lambda: _verify_input_identities(input_identities),
    )


if __name__ == "__main__":
    main()
