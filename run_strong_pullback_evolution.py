from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from run_backtest import load_prices, pivot_prices
from run_strong_pullback_satellite import (
    build_regime_leverage_schedule,
    run_satellite_walk_forward,
)
from strategy_evolution_core import (
    _exclusive_file_lock,
    EvolutionState,
    FoldMetrics,
    PromotionPolicy,
    StateWriteExpectation,
    commit_evolution_state_transition,
    evaluate_candidate,
    fingerprint_payload,
    load_evolution_state,
    load_evolution_transition_journal,
    promote_to_shadow,
    reconcile_evolution_transition,
    rollback_shadow,
)
from strong_pullback_evolution import (
    CoreTestSegment,
    EvolutionConfig,
    TradingFold,
    assess_test_result,
    build_core_test_segments,
    build_selection_segments,
    build_group_candidates,
    calculate_fold_metrics,
    calculate_segment_metrics,
    choose_group_winner,
    evaluate_promotion,
    is_windows_reserved_device_name,
    load_evolution_config,
    run_strong_pullback_folds,
)
from train_next_open_rank_model import clean_matrix, load_market_exposure


REQUIRED_EVOLUTION_COLUMNS = {
    "date", "symbol", "open", "high", "low", "close", "volume", "amount",
}
FILTER_KEYS = (
    "min_close", "min_avg_amount_20d", "min_pullback_5d", "max_pullback_5d",
    "min_prior_return_20", "min_prior_return_60", "min_return_20d",
    "min_return_60d", "min_distance_ma60", "max_intraday_return",
)
TRIAL_METRIC_KEYS = (
    "total_return", "annualized_return", "max_drawdown", "sharpe_like", "avg_turnover",
    "avg_gross_exposure", "trade_days", "rolling_window_count", "negative_window_rate",
    "worst_rolling_return",
)
TRIAL_ARTIFACTS = {
    "equity_curve.csv": ("date", "equity", "gross_return", "cost", "turnover", "gross_exposure"),
    "rolling_feature_weights.csv": ("date",),
    "trade_audit.csv": ("signal_date",),
    "selected_candidates.csv": ("signal_date",),
}
STRATEGY_CODE_DEPENDENCIES = (
    "strategy_evolution_core.py",
    "strong_pullback_evolution.py",
    "run_strong_pullback_evolution.py",
    "run_strong_pullback_satellite.py",
    "train_next_open_rank_model.py",
    "run_backtest.py",
    "execution_rules.py",
    "generate_strong_pullback_candidates.py",
)


@dataclass(frozen=True)
class PriceBundle:
    close: pd.DataFrame
    open_px: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    amount: pd.DataFrame
    market_exposure: pd.Series
    benchmark_effective_date: pd.Timestamp | None = None
    benchmark_close: pd.Series | None = None


@dataclass(frozen=True)
class StrategyRun:
    equity: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    candidates: pd.DataFrame


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _peak_memory_bytes() -> int:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    else:
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return peak if sys.platform == "darwin" else peak * 1024
        except (ImportError, OSError, ValueError):
            pass
    return 0


def _file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_code_fingerprint(project_root: Path | None = None) -> str:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent
    return stable_hash({
        filename: _file_content_hash(root / filename)
        for filename in STRATEGY_CODE_DEPENDENCIES
    })


def _group_seed(random_seed: int, group_id: str) -> int:
    return random_seed ^ int(fingerprint_payload({"group_id": group_id})[:16], 16)


def _promotion_policy(config: EvolutionConfig) -> PromotionPolicy:
    core = config.evolution_core
    return PromotionPolicy(
        min_folds=core.min_folds,
        min_filled_trades_per_fold=core.min_filled_trades_per_fold,
        min_positive_fold_ratio=core.min_positive_fold_ratio,
        min_mean_return_improvement=core.min_mean_return_improvement,
        max_drawdown_floor=core.max_drawdown_floor,
        max_drawdown_worsening=core.max_drawdown_worsening,
        max_turnover_ratio=core.max_turnover_ratio,
        max_pnl_concentration=core.max_pnl_concentration,
    )


@dataclass(frozen=True)
class RunEvidence:
    data_path: str
    data_size: int
    data_mtime_ns: int
    data_content_hash: str
    benchmark_path: str | None
    benchmark_size: int | None
    benchmark_mtime_ns: int | None
    benchmark_content_hash: str | None
    config_path: str
    config_content_hash: str
    config_hash: str
    git_commit: str
    code_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))

    @property
    def data_fingerprint(self) -> str:
        return stable_hash({
            "data_content_hash": self.data_content_hash,
            "benchmark_content_hash": self.benchmark_content_hash,
            "config_hash": self.config_hash,
        })


@dataclass(frozen=True)
class EvolutionOutcome:
    run_id: str
    run_dir: Path
    champion_id: str
    champion_params: dict[str, object]
    test_status: str
    test_reason: str


def build_run_evidence(
    data_path: Path,
    config_path: Path,
    config: EvolutionConfig,
    benchmark_path: Path | None,
    git_commit: str,
) -> RunEvidence:
    stat = data_path.resolve().stat()
    benchmark_stat = benchmark_path.resolve().stat() if benchmark_path else None
    return RunEvidence(
        data_path=str(data_path.resolve()),
        data_size=int(stat.st_size),
        data_mtime_ns=int(stat.st_mtime_ns),
        data_content_hash=_file_content_hash(data_path),
        benchmark_path=str(benchmark_path.resolve()) if benchmark_path else None,
        benchmark_size=int(benchmark_stat.st_size) if benchmark_stat else None,
        benchmark_mtime_ns=int(benchmark_stat.st_mtime_ns) if benchmark_stat else None,
        benchmark_content_hash=_file_content_hash(benchmark_path) if benchmark_path else None,
        config_path=str(config_path.resolve()),
        config_content_hash=_file_content_hash(config_path),
        config_hash=stable_hash({
            "strategy": config.strategy,
            "evolution_core": asdict(config.evolution_core),
            "periods": asdict(config.periods),
            "baseline": config.baseline,
            "search_groups": [asdict(group) for group in config.search_groups],
            "selection": asdict(config.selection),
        }),
        git_commit=git_commit,
        code_fingerprint=build_code_fingerprint(),
    )


def resolved_config_dict(config: EvolutionConfig) -> dict[str, object]:
    return {
        "strategy": config.strategy,
        "evolution_core": asdict(config.evolution_core),
        "periods": {
            key: value.strftime("%Y-%m-%d") if value is not None else None
            for key, value in asdict(config.periods).items()
        },
        "baseline": dict(config.baseline),
        "search_groups": [
            {
                "id": group.group_id,
                "hypothesis_cn": group.hypothesis_cn,
                "candidates": [
                    {"id": candidate.candidate_id, "overrides": dict(candidate.overrides)}
                    for candidate in group.candidates
                ],
            }
            for group in config.search_groups
        ],
        "selection": asdict(config.selection),
    }


def can_resume_trial(
    trial_state: Mapping[str, object],
    evidence_fingerprint: str,
    params_hash: str,
    trial_id: str,
) -> bool:
    return (
        trial_state.get("status") == "completed"
        and trial_state.get("trial_id") == trial_id
        and trial_state.get("evidence_fingerprint") == evidence_fingerprint
        and trial_state.get("params_hash") == params_hash
    )


def _state_path_is_config(path: Path) -> bool:
    return any(part.lower() == "configs" for part in path.resolve().parts)


def _validate_runtime_state_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.parent.name.lower() != "evolution_state" or resolved.suffix.lower() != ".json":
        raise ValueError(
            "state_path must be a JSON file directly under a dedicated evolution_state directory"
        )
    return resolved


def _resolve_run_dir(output_root: Path, run_id: str) -> tuple[Path, Path]:
    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).is_absolute()
        or PureWindowsPath(run_id).is_absolute()
        or any(ord(character) < 32 or ord(character) == 127 for character in run_id)
        or any(character in '<>:"|?*' for character in run_id)
        or run_id.endswith(".")
        or is_windows_reserved_device_name(run_id)
    ):
        raise ValueError("run_id must be a safe single relative path component")
    resolved_root = Path(output_root).resolve()
    resolved_run_dir = (resolved_root / run_id).resolve()
    try:
        relative = resolved_run_dir.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("run_id resolves outside output_root") from exc
    if len(relative.parts) != 1 or relative.parts[0] in {"", ".", ".."}:
        raise ValueError("run_id must resolve to one child of output_root")
    if _state_path_is_config(resolved_run_dir):
        raise ValueError("run_dir must remain outside configs")
    return resolved_root, resolved_run_dir


def _resolve_trial_dir(run_dir: Path, trial_id: str) -> Path:
    if (
        not isinstance(trial_id, str)
        or not trial_id.strip()
        or trial_id in {".", ".."}
        or "/" in trial_id
        or "\\" in trial_id
        or Path(trial_id).is_absolute()
        or PureWindowsPath(trial_id).is_absolute()
        or any(ord(character) < 32 or ord(character) == 127 for character in trial_id)
        or any(character in '<>:"|?*' for character in trial_id)
        or trial_id.endswith(".")
        or is_windows_reserved_device_name(trial_id)
    ):
        raise ValueError("trial_id must be a safe single relative path component")
    trials_root = (Path(run_dir).resolve() / "trials").resolve()
    trial_dir = (trials_root / trial_id).resolve()
    try:
        relative = trial_dir.relative_to(trials_root)
    except ValueError as exc:
        raise ValueError("trial_id resolves outside trials") from exc
    if len(relative.parts) != 1 or relative.parts[0] in {"", ".", ".."}:
        raise ValueError("trial_id must resolve to one direct child of trials")
    return trial_dir


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    )


def publish_run_metadata(
    output_root: Path,
    manifest: Mapping[str, object],
) -> None:
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
    ):
        raise ValueError("manifest run_id must be a safe non-empty filename")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".evolution_metadata.lock"
    registry_path = root / "evolution_registry.jsonl"
    with _exclusive_file_lock(lock_path):
        records: list[dict[str, object]] = []
        if registry_path.exists():
            for line_number, line in enumerate(
                registry_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Corrupt evolution registry line {line_number}"
                    ) from exc
                if not isinstance(record, dict) or not isinstance(
                    record.get("run_id"), str
                ):
                    raise ValueError(
                        f"Invalid evolution registry line {line_number}"
                    )
                records.append(record)
        replacement = dict(manifest)
        updated: list[dict[str, object]] = []
        replaced = False
        seen_ids: set[str] = set()
        for record in records:
            record_id = str(record["run_id"])
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            if record_id == run_id:
                updated.append(replacement)
                replaced = True
            else:
                updated.append(record)
        if not replaced:
            updated.append(replacement)
        registry_content = "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in updated
        )
        _atomic_write_text(registry_path, registry_content)
        _atomic_write_json(root / "latest.json", replacement)


def reconcile_startup_transition(
    output_root: Path,
    state_path: Path,
    default_state: EvolutionState,
) -> None:
    before = load_evolution_transition_journal(state_path)
    if before is None:
        return
    resolved_root, prior_run_dir = _resolve_run_dir(output_root, before.run_id)
    reconciled = (
        reconcile_evolution_transition(state_path, default_state)
        if before.status == "pending"
        else before
    )
    if reconciled is None:
        return
    run_id = reconciled.run_id
    if run_id != before.run_id:
        raise ValueError("Reconciled transition journal changed run_id")
    manifest_path = prior_run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Could not reconcile interrupted run manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise ValueError("Interrupted run manifest does not match transition journal")
    if (
        manifest.get("status") not in {"running", "pending"}
        and manifest.get("promotion_persistence_status") != "pending"
    ):
        if (
            reconciled.status == "committed"
            and manifest.get("status") == "success"
            and manifest.get("promotion_persistence_status") == "committed"
            and manifest.get("global_state_changed") is True
        ):
            publish_run_metadata(resolved_root, manifest)
        return
    recovered_at = datetime.now(timezone.utc).isoformat()
    committed = reconciled.status == "committed"
    manifest.update({
        "status": "success" if committed else "failed",
        "promotion_persistence_status": reconciled.status,
        "global_state_changed": committed,
        "state_transition_recovered_at_utc": recovered_at,
        "state_transition_recovery_reason": reconciled.reason,
    })
    if committed:
        manifest.setdefault("completed_at_utc", recovered_at)
    else:
        manifest.setdefault("failed_at_utc", recovered_at)
        manifest.setdefault(
            "error", reconciled.reason or "state transition was not committed"
        )
    recovered_legacy_status = manifest.get("final_legacy_status")
    if not isinstance(recovered_legacy_status, str):
        recovered_legacy_status = None
    recovered_legacy_reasons_raw = manifest.get("final_legacy_reasons")
    recovered_legacy_reasons = (
        tuple(str(reason) for reason in recovered_legacy_reasons_raw)
        if isinstance(recovered_legacy_reasons_raw, list)
        else ()
    )
    _atomic_write_text(
        manifest_path.parent / "shadow_decision.md",
        _promotion_decision_markdown(
            run_id,
            str(manifest.get("champion_id") or "unknown"),
            manifest.get("selected_experiment_id")
            if isinstance(manifest.get("selected_experiment_id"), str)
            else None,
            reconciled.status,
            reconciled.reason or "state transition reconciled at startup",
            recovered_legacy_status,
            recovered_legacy_reasons,
        ),
    )
    _atomic_write_json(manifest_path, manifest)
    if committed:
        publish_run_metadata(resolved_root, manifest)


def _promotion_decision_markdown(
    run_id: str,
    champion_id: str,
    selected_experiment_id: str | None,
    persistence_status: str,
    reason: str,
    final_legacy_status: str | None = None,
    final_legacy_reasons: tuple[str, ...] = (),
) -> str:
    lines = [
        "# 影子决定",
        "",
        f"- 运行：`{run_id}`",
        f"- 候选：`{champion_id}`",
        f"- 实验：`{selected_experiment_id or 'none'}`",
        f"- 持久化状态：`{persistence_status}`",
        f"- 全局状态已改变：`{'true' if persistence_status == 'committed' else 'false'}`",
    ]
    if final_legacy_status is not None:
        lines.extend([
            f"- Final locked-candidate legacy status: `{final_legacy_status}`",
            "- Final locked-candidate legacy reasons: "
            + ("; ".join(final_legacy_reasons) if final_legacy_reasons else "none"),
        ])
    lines.extend([
        f"- 原因：{reason}",
        "",
        "正式 YAML 未修改，也未产生任何券商订单。",
        "",
    ])
    return "\n".join(lines)


def _state_records_promotion(
    state_path: Path,
    default_state: EvolutionState,
    run_id: str,
    selected_experiment_id: str | None,
) -> bool:
    try:
        journal = load_evolution_transition_journal(state_path)
    except (OSError, ValueError):
        return False
    if (
        journal is None
        or journal.run_id != run_id
        or journal.experiment_id != selected_experiment_id
        or not selected_experiment_id
        or not Path(state_path).exists()
    ):
        return False
    try:
        state = load_evolution_state(Path(state_path), default_state)
    except (OSError, ValueError):
        return False
    target_is_committed = (
        fingerprint_payload(state) == journal.target_state_fingerprint
        and state.last_completed_run_id == run_id
        and state.shadow_experiment_id == selected_experiment_id
    )
    if not target_is_committed:
        return False
    if journal.status == "pending":
        try:
            reconciled = reconcile_evolution_transition(state_path, default_state)
        except (OSError, ValueError):
            return True
        return reconciled is not None and reconciled.status == "committed"
    return journal.status == "committed"


def persist_evolution_outcome(
    run_dir: Path,
    snapshot: EvolutionState,
    candidate_scores: list[dict[str, object]],
    experiments: list[dict[str, object]],
    decision_markdown: str,
    dry_run: bool,
    promote_shadow: bool,
    state_path: Path,
    expected_previous_fingerprint: str | StateWriteExpectation | None,
) -> bool:
    run_dir = Path(run_dir)
    state_path = Path(state_path)
    if _state_path_is_config(state_path):
        raise ValueError("state_path must not be under configs")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evolution_state_snapshot.json").write_text(
        json.dumps(
            _json_safe(asdict(snapshot)), sort_keys=True, ensure_ascii=True,
            allow_nan=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    score_frame = pd.DataFrame(candidate_scores)
    if score_frame.empty:
        score_frame = pd.DataFrame(columns=(
            "group_id", "candidate_id", "parent_id", "experiment_id",
            "status", "failed_gates", "gates",
        ))
    score_frame.to_csv(
        run_dir / "candidate_scores.csv", index=False, encoding="utf-8-sig"
    )
    experiments_dir = run_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        experiment_id = experiment.get("experiment_id")
        if (
            not isinstance(experiment_id, str)
            or not experiment_id
            or Path(experiment_id).name != experiment_id
        ):
            raise ValueError("experiment_id must be a safe non-empty filename")
        (experiments_dir / f"{experiment_id}.json").write_text(
            json.dumps(
                _json_safe(experiment), sort_keys=True, ensure_ascii=True,
                allow_nan=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    _atomic_write_text(run_dir / "shadow_decision.md", decision_markdown)
    selected_experiment_id = snapshot.shadow_experiment_id
    eligible_selected_score = any(
        score.get("experiment_id") == selected_experiment_id
        and score.get("status") == "eligible_for_shadow"
        for score in candidate_scores
    )
    is_promotion = (
        snapshot.shadow_status == "shadow"
        and bool(selected_experiment_id)
        and eligible_selected_score
    )
    is_rollback = snapshot.shadow_status == "rolled_back"
    if dry_run or not promote_shadow or not (is_promotion or is_rollback):
        return False
    commit_evolution_state_transition(
        state_path,
        snapshot,
        operation="promote" if is_promotion else "rollback",
        run_id=snapshot.last_completed_run_id or snapshot.shadow_run_id or "unknown-run",
        experiment_id=selected_experiment_id,
        expected_previous_fingerprint=expected_previous_fingerprint,
    )
    return True


def _bundle_has_holdout_dates(bundle: PriceBundle, validation_end: pd.Timestamp) -> bool:
    members = {
        "close": bundle.close,
        "open_px": bundle.open_px,
        "high": bundle.high,
        "low": bundle.low,
        "amount": bundle.amount,
        "market_exposure": bundle.market_exposure,
    }
    if bundle.benchmark_close is not None:
        members["benchmark_close"] = bundle.benchmark_close
    return any(
        not member.index.empty
        and pd.Timestamp(member.index.max()) > validation_end
        for member in members.values()
    )


def _has_complete_cached_metrics(metrics: object) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    for segment in ("train", "validation"):
        values = metrics.get(segment)
        if not isinstance(values, Mapping):
            return False
        for key in TRIAL_METRIC_KEYS:
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False
    return True


def _has_complete_trial_artifacts(trial_dir: Path, run: StrategyRun) -> bool:
    frames = {
        "equity_curve.csv": run.equity,
        "rolling_feature_weights.csv": run.weights,
        "trade_audit.csv": run.trades,
        "selected_candidates.csv": run.candidates,
    }
    for filename, required_columns in TRIAL_ARTIFACTS.items():
        frame = frames[filename]
        if not (trial_dir / filename).exists():
            return False
        if not frame.empty and not set(required_columns).issubset(frame.columns):
            return False
    return not run.equity.empty


def validate_input_schema(path: Path) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_EVOLUTION_COLUMNS - columns
    if missing:
        raise ValueError(f"Missing evolution input columns: {sorted(missing)}")


def _csv_max_date(path: Path, label: str) -> pd.Timestamp:
    try:
        dates = pd.read_csv(path, usecols=["date"])["date"]
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not read {label} date coverage") from exc
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.empty or parsed.isna().any():
        raise ValueError(f"{label} date coverage is empty or invalid")
    return pd.Timestamp(parsed.max()).normalize()


def validate_state_mutation_freshness(
    data_path: Path,
    benchmark_path: Path | None,
    asof_date: pd.Timestamp,
) -> str:
    panel_max = _csv_max_date(data_path, "panel")
    selected_asof = pd.Timestamp(asof_date).normalize()
    if selected_asof != panel_max:
        raise ValueError(
            "asof_date must equal the selected panel maximum date for state mutation"
        )
    if benchmark_path is None:
        raise ValueError("benchmark must cover the panel maximum date for state mutation")
    benchmark_max = _csv_max_date(benchmark_path, "benchmark")
    if benchmark_max < panel_max:
        raise ValueError("benchmark must reach the panel maximum date for state mutation")
    return panel_max.strftime("%Y-%m-%d")


def validate_loaded_bundle_freshness(
    bundle: PriceBundle,
    asof_date: pd.Timestamp,
) -> str:
    selected_asof = pd.Timestamp(asof_date).normalize()
    effective_dates: dict[str, pd.Timestamp] = {}
    common_valid: pd.DataFrame | None = None
    common_index: pd.DatetimeIndex | None = None
    common_columns: pd.Index | None = None
    for name in ("close", "open_px", "high", "low", "amount"):
        frame = getattr(bundle, name)
        if frame.empty:
            raise ValueError(f"effective loaded {name} data is empty")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        finite_values = pd.DataFrame(
            np.isfinite(numeric.to_numpy(dtype=float)),
            index=frame.index,
            columns=frame.columns,
        )
        valid_rows = finite_values.any(axis=1)
        valid_index = pd.DatetimeIndex(frame.index[valid_rows]).normalize()
        if valid_index.empty:
            raise ValueError(f"effective loaded {name} data is empty")
        effective_dates[name] = pd.Timestamp(valid_index.max()).normalize()
        if common_valid is None:
            common_index = pd.DatetimeIndex(frame.index)
            common_columns = frame.columns
            common_valid = finite_values
        else:
            assert common_index is not None and common_columns is not None
            common_valid &= finite_values.reindex(
                index=common_index, columns=common_columns, fill_value=False
            )

    assert common_valid is not None and common_index is not None
    common_dates = common_index[common_valid.any(axis=1)]
    if common_dates.empty:
        raise ValueError("common effective loaded price observation is empty")
    common_effective_date = pd.Timestamp(common_dates.max()).normalize()
    if common_effective_date < selected_asof:
        raise ValueError(
            "common effective loaded price observation must reach asof_date"
        )
    effective_dates["common_price"] = common_effective_date

    exposure = bundle.market_exposure.dropna()
    if exposure.empty:
        raise ValueError("effective loaded benchmark series is empty")
    effective_dates["benchmark"] = (
        pd.Timestamp(bundle.benchmark_effective_date).normalize()
        if bundle.benchmark_effective_date is not None
        else pd.Timestamp(exposure.index.max()).normalize()
    )
    stale = {
        name: value.strftime("%Y-%m-%d")
        for name, value in effective_dates.items()
        if value < selected_asof
    }
    if stale:
        raise ValueError(
            "effective loaded data must reach asof_date; stale members: "
            + json.dumps(stale, sort_keys=True)
        )
    return min(effective_dates.values()).strftime("%Y-%m-%d")


def load_price_bundle(
    data_path: Path,
    end_date: pd.Timestamp,
    benchmark_path: Path | None,
    params: Mapping[str, object],
) -> PriceBundle:
    validate_input_schema(data_path)
    raw = load_prices(data_path, None, end_date.strftime("%Y-%m-%d"))
    max_abs_return = float(params["max_abs_daily_return"])
    close = clean_matrix(pivot_prices(raw, "close"), max_abs_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), max_abs_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), max_abs_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), max_abs_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    market_exposure = load_market_exposure(
        str(benchmark_path) if benchmark_path else None,
        close.index,
        ma_window=int(params["market_ma_window"]),
        risk_off_drawdown_20d=float(params["market_risk_off_drawdown_20d"]),
        below_ma_exposure=float(params["market_below_ma_exposure"]),
        crash_exposure=float(params["market_crash_exposure"]),
    )
    benchmark_effective_date = None
    benchmark_close = pd.Series(np.nan, index=close.index, dtype=float)
    if benchmark_path is not None:
        benchmark = pd.read_csv(benchmark_path, usecols=["date", "close"])
        benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
        benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
        benchmark = benchmark.dropna(subset=["date", "close"])
        benchmark = benchmark.loc[np.isfinite(benchmark["close"])]
        benchmark = benchmark.loc[benchmark["date"] <= end_date]
        if benchmark.empty:
            raise ValueError("benchmark has no finite effective close")
        benchmark_effective_date = pd.Timestamp(benchmark["date"].max()).normalize()
        benchmark_close = benchmark.set_index("date")["close"].reindex(close.index).ffill()
    if close.index.max() > end_date:
        raise AssertionError("Price bundle extends beyond requested end date")
    return PriceBundle(
        close, open_px, high, low, amount, market_exposure,
        benchmark_effective_date=benchmark_effective_date,
        benchmark_close=benchmark_close,
    )


def execute_strategy_trial(bundle: PriceBundle, params: Mapping[str, object]) -> StrategyRun:
    filter_kwargs = {key: float(params[key]) for key in FILTER_KEYS}
    regime_schedule = None
    if params["regime_strong_leverage"] is not None:
        benchmark_close = getattr(bundle, "benchmark_close", None)
        if benchmark_close is None:
            raise ValueError(
                "benchmark close data is required for regime-gated leverage"
            )
        regime_schedule = build_regime_leverage_schedule(
            benchmark_close,
            bundle.close,
            bundle.market_exposure,
            base_leverage=float(params["leverage"]),
            strong_leverage=params["regime_strong_leverage"],
            exceptional_leverage=params["regime_exceptional_leverage"],
            strong_breadth_threshold=params["regime_strong_breadth_threshold"],
            exceptional_breadth_threshold=params["regime_exceptional_breadth_threshold"],
            strong_volatility_max=params["regime_strong_volatility_max"],
            exceptional_volatility_max=params["regime_exceptional_volatility_max"],
        )
    equity, weights, trades, candidates = run_satellite_walk_forward(
        close=bundle.close,
        open_px=bundle.open_px,
        high=bundle.high,
        low=bundle.low,
        amount=bundle.amount,
        train_days=int(params["train_days"]),
        retrain_frequency=int(params["retrain_frequency"]),
        top_n=int(params["top_n"]),
        rebalance_frequency=int(params["rebalance_frequency"]),
        max_position_weight=float(params["max_position_weight"]),
        leverage=float(params["leverage"]),
        min_score=None if params["min_score"] is None else float(params["min_score"]),
        commission_bps=float(params["commission_bps"]),
        impact_bps=float(params["impact_bps"]),
        max_buy_open_gap=float(params["max_buy_open_gap"]),
        limit_buffer=float(params["limit_buffer"]),
        market_exposure=bundle.market_exposure,
        initial_capital=float(params["initial_capital"]),
        filter_kwargs=filter_kwargs,
        regime_schedule=regime_schedule,
        basket_guard_return_20d_min=params["basket_guard_return_20d_min"],
        basket_guard_distance_ma60_min=params["basket_guard_distance_ma60_min"],
        basket_guard_scale=float(params["basket_guard_scale"]),
        rebound_exit_return=params["rebound_exit_return"],
        rebound_exit_scale=float(params["rebound_exit_scale"]),
        rebound_exit_market_exposure_max=params["rebound_exit_market_exposure_max"],
        rebound_exit_market_exposure_min=params["rebound_exit_market_exposure_min"],
    )
    if equity.empty:
        raise ValueError("Trial generated no equity rows")
    return StrategyRun(equity, weights, trades, candidates)


def _slice_price_bundle(bundle: PriceBundle, test_end: pd.Timestamp) -> PriceBundle:
    end = pd.Timestamp(test_end)

    def frame_slice(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[pd.DatetimeIndex(frame.index) <= end].copy()

    def series_slice(series: pd.Series) -> pd.Series:
        return series.loc[pd.DatetimeIndex(series.index) <= end].copy()

    sliced = PriceBundle(
        close=frame_slice(bundle.close),
        open_px=frame_slice(bundle.open_px),
        high=frame_slice(bundle.high),
        low=frame_slice(bundle.low),
        amount=frame_slice(bundle.amount),
        market_exposure=series_slice(bundle.market_exposure),
        benchmark_effective_date=(
            min(pd.Timestamp(bundle.benchmark_effective_date), end)
            if bundle.benchmark_effective_date is not None
            else None
        ),
        benchmark_close=(
            series_slice(bundle.benchmark_close)
            if bundle.benchmark_close is not None
            else None
        ),
    )
    members = (
        sliced.close,
        sliced.open_px,
        sliced.high,
        sliced.low,
        sliced.amount,
        sliced.market_exposure,
    )
    if sliced.benchmark_close is not None:
        members = (*members, sliced.benchmark_close)
    if any(not member.index.empty and pd.Timestamp(member.index.max()) > end for member in members):
        raise AssertionError("Fold bundle extends beyond test_end")
    return sliced


def evaluate_strategy_folds(
    bundle: PriceBundle,
    folds: tuple[TradingFold | CoreTestSegment, ...],
    trial_executor: Callable[[PriceBundle, Mapping[str, object]], StrategyRun],
    params: Mapping[str, object],
) -> tuple[FoldMetrics, ...]:
    fold_runs = run_strong_pullback_folds(
        bundle,
        folds,
        trial_executor,
        params,
        slicer=_slice_price_bundle,
    )
    return tuple(
        calculate_fold_metrics(run.equity, run.trades, fold)
        for fold, run in fold_runs
    )


def write_trial_artifacts(
    trial_dir: Path,
    run: StrategyRun,
    metrics: Mapping[str, object],
    trial_state: Mapping[str, object],
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    run.equity.to_csv(trial_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    run.weights.to_csv(trial_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8-sig")
    run.trades.to_csv(trial_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
    run.candidates.to_csv(trial_dir / "selected_candidates.csv", index=False, encoding="utf-8-sig")
    (trial_dir / "metrics.json").write_text(
        json.dumps(
            _json_safe(metrics), ensure_ascii=False, allow_nan=False, indent=2
        ),
        encoding="utf-8",
    )
    (trial_dir / "trial_state.json").write_text(
        json.dumps(trial_state, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )


def _read_csv_or_empty(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def load_trial_artifacts(trial_dir: Path) -> StrategyRun:
    return StrategyRun(
        equity=_read_csv_or_empty(trial_dir / "equity_curve.csv", parse_dates=["date"]),
        weights=_read_csv_or_empty(trial_dir / "rolling_feature_weights.csv"),
        trades=_read_csv_or_empty(trial_dir / "trade_audit.csv"),
        candidates=_read_csv_or_empty(trial_dir / "selected_candidates.csv", dtype={"symbol": "string"}),
    )


def write_chinese_summary(
    run_dir: Path,
    asof_date: pd.Timestamp,
    champion_id: str,
    round_rows: list[dict[str, object]],
    final_metrics: Mapping[str, Mapping[str, float]],
    test_status: str,
    test_reason: str,
) -> Path:
    comparison = pd.DataFrame([
        {
            "版本": name,
            "测试期总收益": values["total_return"],
            "测试期最大回撤": values["max_drawdown"],
            "测试期Sharpe": values["sharpe_like"],
        }
        for name, values in final_metrics.items()
    ])
    lines = [
        f"# 强势回调策略自进化报告 {asof_date:%Y-%m-%d}", "",
        f"- 研究冠军：`{champion_id}`",
        f"- 测试状态：`{test_status}`",
        f"- 判定原因：{test_reason}", "",
        "## 每轮决定", "",
        pd.DataFrame(round_rows).to_markdown(index=False) if round_rows else "本次没有搜索组。", "",
        "## 保留测试期比较", "", comparison.to_markdown(index=False, floatfmt=".4f"), "",
        "## 风险提示", "",
        "该结果仅用于研究和人工复核，不连接券商、不自动下单，也不会自动覆盖现有策略配置。",
    ]
    path = run_dir / f"evolution_summary_{asof_date:%Y%m%d}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_evolution(
    config: EvolutionConfig,
    data_path: Path,
    config_path: Path,
    benchmark_path: Path | None,
    asof_date: pd.Timestamp,
    output_root: Path,
    run_id: str,
    resume: bool,
    bundle_loader: Callable = load_price_bundle,
    trial_executor: Callable = execute_strategy_trial,
    git_commit: str = "unknown",
    dry_run: bool = True,
    promote_shadow: bool = False,
    state_path: Path | None = None,
    expected_previous_fingerprint: str | StateWriteExpectation | None = None,
) -> EvolutionOutcome:
    run_started = time.perf_counter()
    regime_keys = (
        "regime_strong_leverage",
        "regime_exceptional_leverage",
        "regime_strong_breadth_threshold",
        "regime_exceptional_breadth_threshold",
        "regime_strong_volatility_max",
        "regime_exceptional_volatility_max",
    )
    config_uses_regime = any(
        config.baseline.get(key) is not None for key in regime_keys
    ) or any(
        candidate.overrides.get(key) is not None
        for group in config.search_groups
        for candidate in group.candidates
        for key in regime_keys
    )
    if config_uses_regime and benchmark_path is None:
        raise ValueError("benchmark is required for regime-gated leverage evolution")
    if _state_path_is_config(output_root):
        raise ValueError("output_root must not be under configs")
    data_asof_date = pd.Timestamp(asof_date).normalize().strftime("%Y-%m-%d")
    if not dry_run and promote_shadow:
        data_asof_date = validate_state_mutation_freshness(
            Path(data_path), benchmark_path, asof_date
        )
    output_root, run_dir = _resolve_run_dir(output_root, run_id)
    evidence = build_run_evidence(data_path, config_path, config, benchmark_path, git_commit)
    resolved_state_path = (
        Path(state_path)
        if state_path is not None
        else output_root.parent / "evolution_state" / "strong_pullback.json"
    )
    resolved_state_path = _validate_runtime_state_path(resolved_state_path)
    default_state = EvolutionState.initial("baseline", config.baseline)
    reconcile_startup_transition(output_root, resolved_state_path, default_state)
    input_state = load_evolution_state(resolved_state_path, default_state)
    if benchmark_path is None and any(
        input_state.champion_parameters.get(key) is not None for key in regime_keys
    ):
        raise ValueError("benchmark is required for regime-gated champion replay")
    locked_champion_id = input_state.champion_version
    locked_champion_params = dict(input_state.champion_parameters)
    selected_experiment_id: str | None = None
    champion_id = "unselected"
    final_legacy_status: str | None = None
    final_legacy_reasons: tuple[str, ...] = ()
    previous_fingerprint = expected_previous_fingerprint
    if previous_fingerprint is None:
        previous_fingerprint = (
            fingerprint_payload(input_state)
            if resolved_state_path.exists()
            else StateWriteExpectation.ABSENT
        )
    manifest_path = run_dir / "manifest.json"
    if resume:
        if not manifest_path.exists():
            raise ValueError(f"Cannot resume missing run: {run_id}")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("evidence_fingerprint") != evidence.fingerprint:
            raise ValueError("Resume evidence does not match the existing run")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": asdict(evidence),
        "evidence_fingerprint": evidence.fingerprint,
        "data_fingerprint": evidence.data_fingerprint,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "benchmark": str(benchmark_path.resolve()) if benchmark_path else None,
        "asof_date": asof_date.strftime("%Y-%m-%d"),
        "data_asof_date": data_asof_date,
        "global_state_changed": False,
        "promotion_persistence_status": "not_started",
        "selected_experiment_id": None,
    }
    _atomic_write_json(manifest_path, manifest)

    try:
        placeholder_snapshot = replace(
            input_state,
            last_completed_run_id=run_id,
            last_data_fingerprint=evidence.data_fingerprint,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        (run_dir / "folds.json").write_text("[]\n", encoding="utf-8")
        persist_evolution_outcome(
            run_dir=run_dir,
            snapshot=placeholder_snapshot,
            candidate_scores=[],
            experiments=[],
            decision_markdown=_promotion_decision_markdown(
                run_id, champion_id, None, "not_started", "纸面研究尚未完成。"
            ),
            dry_run=True,
            promote_shadow=False,
            state_path=resolved_state_path,
            expected_previous_fingerprint=None,
        )
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_config_dict(config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        selection_bundle = bundle_loader(
            data_path, config.periods.validation_end, benchmark_path, locked_champion_params
        )
        if _bundle_has_holdout_dates(selection_bundle, config.periods.validation_end):
            raise AssertionError("Selection bundle contains post-selection dates")

        core = config.evolution_core
        selection_segments = build_selection_segments(
            selection_bundle.close.index,
            config.periods.validation_start,
            config.periods.validation_end,
            core.min_folds,
        )
        selection_fold_rows = [
            {
                "fold_id": segment.fold_id,
                "test_start": segment.test_start,
                "test_end": segment.test_end,
                "fingerprint": fingerprint_payload(segment),
            }
            for segment in selection_segments
        ]
        (run_dir / "selection_folds.json").write_text(
            json.dumps(
                selection_fold_rows,
                default=str,
                sort_keys=True,
                ensure_ascii=True,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        fold_rows: list[dict[str, object]] = []

        trial_rows: list[dict[str, object]] = []
        round_rows: list[dict[str, object]] = []
        trial_metrics: dict[str, dict[str, dict[str, float]]] = {}
        trial_selection_fold_metrics: dict[str, tuple[FoldMetrics, ...]] = {}
        candidate_scores: list[dict[str, object]] = []
        experiments: list[dict[str, object]] = []
        experiment_ids: dict[str, str] = {}
        selection_generic_decisions: dict[str, object] = {}
        trial_params: dict[str, dict[str, object]] = {
            "baseline": dict(locked_champion_params)
        }

        def finish_without_holdout(
            *,
            final_champion_id: str,
            final_champion_params: Mapping[str, object],
            snapshot: EvolutionState,
            test_status: str,
            test_reason: str,
            failed_gates: list[str],
            core_test_status: str,
            persist_rollback: bool = False,
            comparison_metrics: Mapping[str, Mapping[str, float]] | None = None,
        ) -> EvolutionOutcome:
            state_transition_pending = (
                persist_rollback and not dry_run and promote_shadow
            )
            persistence_status = (
                "pending" if state_transition_pending else "not_changed"
            )
            if state_transition_pending:
                persistence_reason = "Waiting for durable shadow rollback CAS."
            elif persist_rollback:
                persistence_reason = (
                    "Shadow rollback recommended; dry-run did not change global state."
                )
            else:
                persistence_reason = (
                    "Research-only result; global shadow state was not changed."
                )
            manifest.update({
                "champion_id": final_champion_id,
                "test_status": test_status,
                "test_reason": test_reason,
                "core_test_status": core_test_status,
                "global_state_changed": False,
                "promotion_persistence_status": persistence_status,
                "selected_experiment_id": selected_experiment_id,
                "failed_gates": failed_gates,
                "final_legacy_status": final_legacy_status,
                "final_legacy_reasons": list(final_legacy_reasons),
                "shadow_reevaluation_reason": shadow_reevaluation_reason,
                "data_asof_date": snapshot.data_asof_date or data_asof_date,
            })
            _atomic_write_json(manifest_path, manifest)
            changed = persist_evolution_outcome(
                run_dir=run_dir,
                snapshot=snapshot,
                candidate_scores=candidate_scores,
                experiments=experiments,
                decision_markdown=_promotion_decision_markdown(
                    run_id,
                    final_champion_id,
                    selected_experiment_id,
                    persistence_status,
                    persistence_reason,
                    final_legacy_status,
                    final_legacy_reasons,
                ),
                dry_run=dry_run or not persist_rollback,
                promote_shadow=promote_shadow,
                state_path=resolved_state_path,
                expected_previous_fingerprint=previous_fingerprint,
            )
            if changed:
                persistence_status = "committed"
                persistence_reason = "Shadow rollback CAS committed."
            manifest.update({
                "global_state_changed": changed,
                "promotion_persistence_status": persistence_status,
            })
            _atomic_write_text(
                run_dir / "shadow_decision.md",
                _promotion_decision_markdown(
                    run_id,
                    final_champion_id,
                    selected_experiment_id,
                    persistence_status,
                    persistence_reason,
                    final_legacy_status,
                    final_legacy_reasons,
                ),
            )
            pd.DataFrame(trial_rows).to_csv(
                run_dir / "trials.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(round_rows).to_csv(
                run_dir / "rounds.csv", index=False, encoding="utf-8-sig"
            )
            resolved_comparison_metrics = comparison_metrics or {}
            pd.DataFrame([
                {"version": name, **metrics}
                for name, metrics in resolved_comparison_metrics.items()
            ], columns=("version", *TRIAL_METRIC_KEYS)).to_csv(
                run_dir / "test_comparison.csv", index=False, encoding="utf-8-sig"
            )
            (run_dir / "champion_candidate.yaml").write_text(
                yaml.safe_dump(
                    dict(final_champion_params), allow_unicode=True, sort_keys=False
                ),
                encoding="utf-8",
            )
            summary_path = write_chinese_summary(
                run_dir,
                asof_date,
                final_champion_id,
                round_rows,
                resolved_comparison_metrics,
                test_status,
                test_reason,
            )
            manifest.update({
                "summary": str(summary_path),
                "status": "success",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - run_started, 6),
                "peak_memory_bytes": _peak_memory_bytes(),
            })
            _atomic_write_json(manifest_path, manifest)
            publish_run_metadata(output_root, manifest)
            return EvolutionOutcome(
                run_id,
                run_dir,
                final_champion_id,
                dict(final_champion_params),
                test_status,
                test_reason,
            )

        def cached_trial(trial_id: str, params_hash: str) -> StrategyRun | None:
            trial_dir = _resolve_trial_dir(run_dir, trial_id)
            state_path = trial_dir / "trial_state.json"
            if not resume or not state_path.exists():
                return None
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(state, Mapping) or not can_resume_trial(
                    state, evidence.fingerprint, params_hash, trial_id
                ):
                    return None
                metrics = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
                if not _has_complete_cached_metrics(metrics):
                    return None
                fold_payloads = metrics.get("selection_folds")
                if not isinstance(fold_payloads, list):
                    return None
                expected_fold_fields = {field.name for field in fields(FoldMetrics)}
                parsed_fold_metrics: list[FoldMetrics] = []
                for payload in fold_payloads:
                    if not isinstance(payload, Mapping) or set(payload) != expected_fold_fields:
                        return None
                    metric = FoldMetrics(**dict(payload))
                    numeric_values = (
                        metric.total_return,
                        metric.max_drawdown,
                        metric.sharpe,
                        metric.filled_trades,
                        metric.average_turnover,
                        metric.pnl_concentration,
                    )
                    if not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in numeric_values
                    ):
                        return None
                    parsed_fold_metrics.append(metric)
                if [metric.fold_id for metric in parsed_fold_metrics] != [
                    segment.fold_id for segment in selection_segments
                ]:
                    return None
                run = load_trial_artifacts(trial_dir)
                if not _has_complete_trial_artifacts(trial_dir, run):
                    return None
                trial_metrics[trial_id] = {
                    "train": dict(metrics["train"]),
                    "validation": dict(metrics["validation"]),
                }
                trial_selection_fold_metrics[trial_id] = tuple(parsed_fold_metrics)
                return run
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return None

        def execute_one(trial_id: str, params: dict[str, object]) -> StrategyRun:
            trial_dir = _resolve_trial_dir(run_dir, trial_id)
            params_hash = stable_hash(params)
            cached = cached_trial(trial_id, params_hash)
            if cached is not None:
                return cached
            run = trial_executor(selection_bundle, params)
            train_metrics = calculate_segment_metrics(
                run.equity, config.periods.research_start, config.periods.train_end,
                config.selection.rolling_window_days,
            )
            validation_metrics = calculate_segment_metrics(
                run.equity, config.periods.validation_start, config.periods.validation_end,
                config.selection.rolling_window_days,
            )
            trial_metrics[trial_id] = {"train": train_metrics, "validation": validation_metrics}
            trial_selection_fold_metrics[trial_id] = evaluate_strategy_folds(
                selection_bundle,
                selection_segments,
                trial_executor,
                params,
            )
            write_trial_artifacts(
                trial_dir,
                run,
                {
                    **trial_metrics[trial_id],
                    "selection_folds": [
                        asdict(metric)
                        for metric in trial_selection_fold_metrics[trial_id]
                    ],
                },
                {
                    "status": "completed",
                    "trial_id": trial_id,
                    "params_hash": params_hash,
                    "evidence_fingerprint": evidence.fingerprint,
                },
            )
            return run

        try:
            execute_one("baseline", trial_params["baseline"])
        except Exception as exc:
            raise RuntimeError(f"Baseline trial failed: {exc}") from exc
        trial_rows.append({
            "group_id": "baseline",
            "trial_id": locked_champion_id,
            "parent_id": "",
            "status": "incumbent",
            "reason_cn": "锁定起始冠军",
            **{
                f"validation_{key}": value
                for key, value in trial_metrics["baseline"]["validation"].items()
            },
        })

        incumbent_id = locked_champion_id
        incumbent_trial_id = "baseline"
        incumbent_params = dict(locked_champion_params)
        incumbent_metrics = trial_metrics["baseline"]["validation"]
        shadow_reevaluation_reason: str | None = None
        if input_state.shadow_status == "shadow":
            shadow_trial_id = "__shadow_incumbent__"
            shadow_params = dict(input_state.shadow_parameters or {})
            shadow_id = input_state.shadow_version or "invalid-shadow"
            trial_params[shadow_trial_id] = shadow_params
            try:
                execute_one(shadow_trial_id, shadow_params)
                shadow_legacy_decision = evaluate_promotion(
                    trial_metrics[shadow_trial_id]["validation"],
                    trial_metrics["baseline"]["validation"],
                    config.selection,
                )
                shadow_generic_decision = evaluate_candidate(
                    trial_selection_fold_metrics[shadow_trial_id],
                    trial_selection_fold_metrics["baseline"],
                    _promotion_policy(config),
                )
            except Exception as exc:
                shadow_reevaluation_reason = (
                    "shadow reevaluation evidence failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                trial_rows.append({
                    "group_id": "shadow_reevaluation",
                    "trial_id": shadow_id,
                    "parent_id": locked_champion_id,
                    "status": "trial_error",
                    "reason_cn": shadow_reevaluation_reason,
                })
                raise RuntimeError(shadow_reevaluation_reason) from exc

            shadow_is_eligible = (
                shadow_legacy_decision.eligible
                and shadow_generic_decision.status == "eligible_for_shadow"
            )
            shadow_legacy_summary = ",".join(
                shadow_legacy_decision.reasons or ("passed",)
            )
            shadow_generic_summary = ",".join(
                shadow_generic_decision.failed_gates or ("passed",)
            )
            shadow_reevaluation_reason = (
                "shadow reevaluation passed"
                if shadow_is_eligible
                else (
                    "shadow reevaluation failed: "
                    f"legacy={shadow_legacy_summary}; "
                    f"generic={shadow_generic_summary} "
                    f"(status={shadow_generic_decision.status})"
                )
            )
            shadow_failed_gates = [
                f"legacy:{reason}" for reason in shadow_legacy_decision.reasons
            ] + list(shadow_generic_decision.failed_gates)
            trial_rows.append({
                "group_id": "shadow_reevaluation",
                "trial_id": shadow_id,
                "parent_id": locked_champion_id,
                "status": "eligible" if shadow_is_eligible else "rejected",
                "selection_generic_status": shadow_generic_decision.status,
                "selection_generic_failed_gates": json.dumps(
                    shadow_generic_decision.failed_gates, ensure_ascii=True
                ),
                "reason_cn": shadow_reevaluation_reason,
                **{
                    f"validation_{key}": value
                    for key, value in trial_metrics[shadow_trial_id]["validation"].items()
                },
            })
            if shadow_is_eligible:
                incumbent_id = shadow_id
                incumbent_trial_id = shadow_trial_id
                incumbent_params = shadow_params
                incumbent_metrics = trial_metrics[shadow_trial_id]["validation"]
            else:
                effective_data_asof = data_asof_date
                if not dry_run and promote_shadow:
                    freshness_bundle = bundle_loader(
                        data_path, asof_date, benchmark_path, locked_champion_params
                    )
                    effective_data_asof = validate_loaded_bundle_freshness(
                        freshness_bundle, asof_date
                    )
                rollback_snapshot = rollback_shadow(
                    input_state,
                    reason=shadow_reevaluation_reason,
                    data_asof_date=effective_data_asof,
                )
                rollback_snapshot = replace(
                    rollback_snapshot,
                    last_completed_run_id=run_id,
                    last_data_fingerprint=evidence.data_fingerprint,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                selected_experiment_id = input_state.shadow_experiment_id
                return finish_without_holdout(
                    final_champion_id=rollback_snapshot.champion_version,
                    final_champion_params=rollback_snapshot.champion_parameters,
                    snapshot=rollback_snapshot,
                    test_status="not_opened_shadow_rollback",
                    test_reason=shadow_reevaluation_reason,
                    failed_gates=shadow_failed_gates or ["shadow_reevaluation"],
                    core_test_status="not_opened",
                    persist_rollback=True,
                )

        for group in config.search_groups:
            candidate_pairs: list[tuple[str, Mapping[str, float]]] = []
            group_seed = _group_seed(core.random_seed, group.group_id)
            generated_candidates = build_group_candidates(
                incumbent_params,
                group,
                max_candidates=core.max_candidates_per_group,
                seed=group_seed,
            )
            for candidate_id, params in generated_candidates:
                trial_params[candidate_id] = params
                experiment_id = "exp_" + fingerprint_payload({
                    "group_id": group.group_id,
                    "candidate_id": candidate_id,
                    "parent_id": incumbent_id,
                    "parameters": params,
                    "data_fingerprint": evidence.data_fingerprint,
                    "code_fingerprint": evidence.code_fingerprint,
                    "selection_period": {
                        "start": config.periods.validation_start,
                        "end": config.periods.validation_end,
                    },
                })[:20]
                experiment_ids[candidate_id] = experiment_id
                try:
                    execute_one(candidate_id, params)
                    candidate_pairs.append((candidate_id, trial_metrics[candidate_id]["validation"]))
                    selection_generic_decision = evaluate_candidate(
                        trial_selection_fold_metrics[candidate_id],
                        trial_selection_fold_metrics[incumbent_trial_id],
                        _promotion_policy(config),
                    )
                    selection_generic_decisions[candidate_id] = selection_generic_decision
                    candidate_scores.append({
                        "group_id": group.group_id,
                        "candidate_id": candidate_id,
                        "parent_id": incumbent_id,
                        "experiment_id": experiment_id,
                        "status": "selection_evaluated",
                        "selection_status": "selection_evaluated",
                        "selection_generic_status": selection_generic_decision.status,
                        "selection_generic_gates": json.dumps(
                            selection_generic_decision.gates,
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        "core_status": "not_evaluated",
                        "failed_gates": "[]",
                        "gates": "{}",
                    })
                    experiments.append({
                        "experiment_id": experiment_id,
                        "group_id": group.group_id,
                        "candidate_id": candidate_id,
                        "parent_version": incumbent_id,
                        "parameters": params,
                        "data_fingerprint": evidence.data_fingerprint,
                        "code_fingerprint": evidence.code_fingerprint,
                        "selection_metrics": trial_metrics[candidate_id]["validation"],
                        "selection_fold_metrics": [
                            asdict(metric)
                            for metric in trial_selection_fold_metrics[candidate_id]
                        ],
                        "selection_generic_status": selection_generic_decision.status,
                        "selection_generic_gates": selection_generic_decision.gates,
                        "fold_metrics": [],
                        "status": "selection_evaluated",
                        "selection_status": "selection_evaluated",
                        "core_status": "not_evaluated",
                        "failed_gates": (),
                        "gates": {},
                    })
                except Exception as exc:
                    trial_rows.append({
                        "group_id": group.group_id,
                        "trial_id": candidate_id,
                        "status": "trial_error",
                        "selection_status": "trial_error",
                        "selection_generic_status": "not_evaluated",
                        "core_status": "not_evaluated",
                        "reason_cn": str(exc),
                    })
                    candidate_scores.append({
                        "group_id": group.group_id,
                        "candidate_id": candidate_id,
                        "parent_id": incumbent_id,
                        "experiment_id": experiment_id,
                        "status": "trial_error",
                        "selection_status": "trial_error",
                        "selection_generic_status": "not_evaluated",
                        "core_status": "not_evaluated",
                        "failed_gates": json.dumps(("trial_error",)),
                        "gates": "{}",
                    })
                    experiments.append({
                        "experiment_id": experiment_id,
                        "group_id": group.group_id,
                        "candidate_id": candidate_id,
                        "parent_version": incumbent_id,
                        "parameters": params,
                        "data_fingerprint": evidence.data_fingerprint,
                        "code_fingerprint": evidence.code_fingerprint,
                        "fold_metrics": [],
                        "status": "trial_error",
                        "selection_status": "trial_error",
                        "selection_generic_status": "not_evaluated",
                        "core_status": "not_evaluated",
                        "failed_gates": ("trial_error",),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            persist_evolution_outcome(
                run_dir=run_dir,
                snapshot=placeholder_snapshot,
                candidate_scores=candidate_scores,
                experiments=experiments,
                decision_markdown=(
                    "# 影子决定\n\n纸面研究尚未完成，未更新全局状态。\n"
                    "正式 YAML 未修改，也未产生任何券商订单。\n"
                ),
                dry_run=True,
                promote_shadow=False,
                state_path=resolved_state_path,
                expected_previous_fingerprint=None,
            )
            if not candidate_pairs:
                raise RuntimeError(f"All candidates failed in group {group.group_id}")
            _, decisions = choose_group_winner(
                incumbent_id, incumbent_metrics, tuple(candidate_pairs), config.selection
            )
            generic_eligible_pairs = tuple(
                pair
                for pair in candidate_pairs
                if selection_generic_decisions[pair[0]].status == "eligible_for_shadow"
            )
            winner_id, _ = choose_group_winner(
                incumbent_id,
                incumbent_metrics,
                generic_eligible_pairs,
                config.selection,
            )
            for decision in decisions:
                generic_decision = selection_generic_decisions[decision.candidate_id]
                combined_eligible = (
                    decision.promotion.eligible
                    and generic_decision.status == "eligible_for_shadow"
                )
                selection_status = (
                    "selection_eligible"
                    if combined_eligible
                    else "selection_rejected"
                )
                failed_selection_gates = tuple(
                    [f"legacy:{reason}" for reason in decision.promotion.reasons]
                    + [
                        f"generic:{gate}"
                        for gate in generic_decision.failed_gates
                    ]
                )
                for score in candidate_scores:
                    if score.get("candidate_id") == decision.candidate_id:
                        score.update({
                            "status": selection_status,
                            "selection_status": selection_status,
                            "selection_generic_status": generic_decision.status,
                            "failed_gates": json.dumps(
                                failed_selection_gates, ensure_ascii=True
                            ),
                        })
                for experiment in experiments:
                    if experiment.get("candidate_id") == decision.candidate_id:
                        experiment.update({
                            "status": selection_status,
                            "selection_status": selection_status,
                            "selection_generic_status": generic_decision.status,
                            "failed_gates": failed_selection_gates,
                        })
                trial_rows.append({
                    "group_id": group.group_id,
                    "trial_id": decision.candidate_id,
                    "parent_id": incumbent_id,
                    "status": "eligible" if combined_eligible else "rejected",
                    "reason_cn": (
                        "旧版与通用选择期门槛均通过"
                        if combined_eligible
                        else "；".join(failed_selection_gates)
                    ),
                    **{f"validation_{key}": value for key, value in decision.metrics.items()},
                    "turnover_ratio": decision.promotion.turnover_ratio,
                    "robust_score": decision.promotion.robust_score,
                })
            promoted = winner_id != incumbent_id
            round_rows.append({
                "group_id": group.group_id,
                "hypothesis_cn": group.hypothesis_cn,
                "parent_id": incumbent_id,
                "winner_id": winner_id,
                "decision": "保留" if promoted else "回滚",
                "reason_cn": (
                    "旧版与通用选择期门槛均通过"
                    if promoted
                    else "没有候选同时通过旧版与通用选择期门槛"
                ),
            })
            if promoted:
                incumbent_id = winner_id
                incumbent_trial_id = winner_id
                incumbent_params = dict(trial_params[winner_id])
                incumbent_metrics = trial_metrics[winner_id]["validation"]

        champion_id = incumbent_id
        champion_params = incumbent_params
        selected_experiment_id = experiment_ids.get(champion_id)
        locked_candidate_is_existing_shadow = (
            input_state.shadow_status == "shadow"
            and incumbent_trial_id == "__shadow_incumbent__"
            and champion_id == input_state.shadow_version
        )
        base_snapshot = replace(
            input_state,
            last_completed_run_id=run_id,
            last_data_fingerprint=evidence.data_fingerprint,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        if champion_id == locked_champion_id:
            return finish_without_holdout(
                final_champion_id=champion_id,
                final_champion_params=champion_params,
                snapshot=base_snapshot,
                test_status="not_opened_no_challenger",
                test_reason="No parameter-selection challenger was locked.",
                failed_gates=["no_selected_experiment"],
                core_test_status="not_opened",
            )

        selected_legacy_decision = evaluate_promotion(
            trial_metrics[incumbent_trial_id]["validation"],
            trial_metrics["baseline"]["validation"],
            config.selection,
        )
        final_legacy_status = (
            "eligible" if selected_legacy_decision.eligible else "rejected"
        )
        final_legacy_reasons = tuple(selected_legacy_decision.reasons)
        final_legacy_failed_gates = [
            f"legacy:{reason}" for reason in final_legacy_reasons
        ]
        manifest.update({
            "champion_id": champion_id,
            "selected_experiment_id": selected_experiment_id,
            "final_legacy_status": final_legacy_status,
            "final_legacy_reasons": list(final_legacy_reasons),
        })
        if final_legacy_failed_gates:
            manifest["failed_gates"] = final_legacy_failed_gates
        _atomic_write_json(manifest_path, manifest)
        for score in candidate_scores:
            if score.get("experiment_id") == selected_experiment_id:
                score.update({
                    "final_legacy_status": final_legacy_status,
                    "final_legacy_reasons": json.dumps(
                        final_legacy_reasons, ensure_ascii=False
                    ),
                })
                if not selected_legacy_decision.eligible:
                    score.update({
                        "status": "final_legacy_rejected",
                        "failed_gates": json.dumps(
                            final_legacy_failed_gates, ensure_ascii=False
                        ),
                    })
        for experiment in experiments:
            if experiment.get("experiment_id") == selected_experiment_id:
                experiment.update({
                    "final_legacy_status": final_legacy_status,
                    "final_legacy_reasons": final_legacy_reasons,
                })
                if not selected_legacy_decision.eligible:
                    experiment.update({
                        "status": "final_legacy_rejected",
                        "failed_gates": tuple(final_legacy_failed_gates),
                    })
        persist_evolution_outcome(
            run_dir=run_dir,
            snapshot=base_snapshot,
            candidate_scores=candidate_scores,
            experiments=experiments,
            decision_markdown=_promotion_decision_markdown(
                run_id,
                champion_id,
                selected_experiment_id,
                "not_changed",
                "Final locked-candidate legacy selection evaluated.",
                final_legacy_status,
                final_legacy_reasons,
            ),
            dry_run=True,
            promote_shadow=False,
            state_path=resolved_state_path,
            expected_previous_fingerprint=None,
        )
        if not selected_legacy_decision.eligible:
            rejection_reason = (
                "Final locked candidate failed legacy selection gates: "
                + ",".join(final_legacy_reasons)
            )
            return finish_without_holdout(
                final_champion_id=champion_id,
                final_champion_params=champion_params,
                snapshot=base_snapshot,
                test_status="not_opened_final_legacy_rejected",
                test_reason=rejection_reason,
                failed_gates=final_legacy_failed_gates,
                core_test_status="not_opened",
            )

        core_bundle = bundle_loader(
            data_path,
            config.periods.core_test_end,
            benchmark_path,
            locked_champion_params,
        )
        if _bundle_has_holdout_dates(core_bundle, config.periods.core_test_end):
            raise AssertionError("Core-test bundle contains final holdout dates")
        core_segments = build_core_test_segments(
            core_bundle.close.index,
            config.periods.core_test_start,
            config.periods.core_test_end,
            core.min_folds,
        )
        fold_rows = [
            {
                "fold_id": segment.fold_id,
                "test_start": segment.test_start,
                "test_end": segment.test_end,
                "fingerprint": fingerprint_payload(segment),
            }
            for segment in core_segments
        ]
        (run_dir / "folds.json").write_text(
            json.dumps(
                fold_rows,
                default=str,
                sort_keys=True,
                ensure_ascii=True,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        locked_core_metrics = evaluate_strategy_folds(
            core_bundle,
            core_segments,
            trial_executor,
            locked_champion_params,
        )
        candidate_core_metrics = evaluate_strategy_folds(
            core_bundle,
            core_segments,
            trial_executor,
            champion_params,
        )
        selected_core_decision = evaluate_candidate(
            candidate_core_metrics,
            locked_core_metrics,
            _promotion_policy(config),
        )
        for score in candidate_scores:
            if score.get("experiment_id") == selected_experiment_id:
                score.update({
                    "status": selected_core_decision.status,
                    "core_status": selected_core_decision.status,
                    "failed_gates": json.dumps(
                        selected_core_decision.failed_gates, ensure_ascii=True
                    ),
                    "gates": json.dumps(
                        selected_core_decision.gates,
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                })
        for experiment in experiments:
            if experiment.get("experiment_id") == selected_experiment_id:
                experiment.update({
                    "status": selected_core_decision.status,
                    "core_status": selected_core_decision.status,
                    "fold_metrics": [
                        asdict(metric) for metric in candidate_core_metrics
                    ],
                    "champion_fold_metrics": [
                        asdict(metric) for metric in locked_core_metrics
                    ],
                    "failed_gates": selected_core_decision.failed_gates,
                    "gates": selected_core_decision.gates,
                })
        persist_evolution_outcome(
            run_dir=run_dir,
            snapshot=base_snapshot,
            candidate_scores=candidate_scores,
            experiments=experiments,
            decision_markdown=_promotion_decision_markdown(
                run_id,
                champion_id,
                selected_experiment_id,
                "not_changed",
                "Core tests completed; final holdout remains gated.",
                final_legacy_status,
                final_legacy_reasons,
            ),
            dry_run=True,
            promote_shadow=False,
            state_path=resolved_state_path,
            expected_previous_fingerprint=None,
        )
        if selected_core_decision.status != "eligible_for_shadow":
            if locked_candidate_is_existing_shadow:
                effective_data_asof = data_asof_date
                if not dry_run and promote_shadow:
                    freshness_bundle = bundle_loader(
                        data_path, asof_date, benchmark_path, locked_champion_params
                    )
                    effective_data_asof = validate_loaded_bundle_freshness(
                        freshness_bundle, asof_date
                    )
                rollback_reason = (
                    "Existing shadow failed locked core tests: "
                    + ",".join(selected_core_decision.failed_gates)
                )
                rollback_snapshot = rollback_shadow(
                    input_state,
                    reason=rollback_reason,
                    data_asof_date=effective_data_asof,
                )
                rollback_snapshot = replace(
                    rollback_snapshot,
                    last_completed_run_id=run_id,
                    last_data_fingerprint=evidence.data_fingerprint,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                selected_experiment_id = input_state.shadow_experiment_id
                return finish_without_holdout(
                    final_champion_id=rollback_snapshot.champion_version,
                    final_champion_params=rollback_snapshot.champion_parameters,
                    snapshot=rollback_snapshot,
                    test_status="not_opened_shadow_core_rollback",
                    test_reason=rollback_reason,
                    failed_gates=list(selected_core_decision.failed_gates),
                    core_test_status=selected_core_decision.status,
                    persist_rollback=True,
                )
            return finish_without_holdout(
                final_champion_id=champion_id,
                final_champion_params=champion_params,
                snapshot=base_snapshot,
                test_status="not_opened_core_rejected",
                test_reason=(
                    "Locked candidate failed core tests: "
                    + ",".join(selected_core_decision.failed_gates)
                ),
                failed_gates=list(selected_core_decision.failed_gates),
                core_test_status=selected_core_decision.status,
            )

        test_end = min(
            asof_date,
            config.periods.test_end if config.periods.test_end is not None else asof_date,
        )
        full_bundle = bundle_loader(
            data_path, test_end, benchmark_path, locked_champion_params
        )
        effective_data_asof = data_asof_date
        if not dry_run and promote_shadow:
            effective_data_asof = validate_loaded_bundle_freshness(
                full_bundle, asof_date
            )
        final_runs = {
            "baseline": trial_executor(full_bundle, locked_champion_params),
            "champion": trial_executor(full_bundle, champion_params),
        }
        final_metrics = {
            name: calculate_segment_metrics(
                run.equity, config.periods.test_start, test_end, config.selection.rolling_window_days
            )
            for name, run in final_runs.items()
        }
        final_params = {
            "baseline": dict(locked_champion_params),
            "champion": dict(champion_params),
        }
        for name, run in final_runs.items():
            write_trial_artifacts(
                run_dir / "final" / name,
                run,
                {"test": final_metrics[name]},
                {
                    "status": "completed",
                    "trial_id": f"final_{name}",
                    "params_hash": stable_hash(final_params[name]),
                    "evidence_fingerprint": evidence.fingerprint,
                },
            )
        test_status, test_reason = assess_test_result(
            final_metrics["baseline"], final_metrics["champion"], config.selection
        )

        if locked_candidate_is_existing_shadow and test_status == "rollback_recommended":
            rollback_reason = f"Existing shadow failed final holdout: {test_reason}"
            rollback_snapshot = rollback_shadow(
                input_state,
                reason=rollback_reason,
                data_asof_date=effective_data_asof,
            )
            rollback_snapshot = replace(
                rollback_snapshot,
                last_completed_run_id=run_id,
                last_data_fingerprint=evidence.data_fingerprint,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            selected_experiment_id = input_state.shadow_experiment_id
            return finish_without_holdout(
                final_champion_id=rollback_snapshot.champion_version,
                final_champion_params=rollback_snapshot.champion_parameters,
                snapshot=rollback_snapshot,
                test_status=test_status,
                test_reason=rollback_reason,
                failed_gates=["holdout_rollback"],
                core_test_status=selected_core_decision.status,
                persist_rollback=True,
                comparison_metrics=final_metrics,
            )

        snapshot = base_snapshot
        shadow_eligible = (
            bool(selected_experiment_id)
            and selected_core_decision.status == "eligible_for_shadow"
            and selected_legacy_decision.eligible
            and test_status == "ready_for_manual_review"
        )
        if shadow_eligible:
            snapshot = promote_to_shadow(
                input_state,
                challenger_version=champion_id,
                challenger_parameters=champion_params,
                experiment_id=selected_experiment_id,
                run_id=run_id,
                data_fingerprint=evidence.data_fingerprint,
                data_asof_date=effective_data_asof,
            )
        state_transition_pending = (
            not dry_run
            and promote_shadow
            and shadow_eligible
        )
        persistence_status = "pending" if state_transition_pending else "not_changed"
        persistence_reason = (
            "等待影子状态转换的原子比较并替换。"
            if state_transition_pending
            else "纸面研究；未满足显式影子晋级持久化条件。"
        )
        manifest.update({
            "champion_id": champion_id,
            "test_status": test_status,
            "test_reason": test_reason,
            "core_test_status": selected_core_decision.status,
            "global_state_changed": False,
            "promotion_persistence_status": persistence_status,
            "selected_experiment_id": selected_experiment_id,
            "failed_gates": list(selected_core_decision.failed_gates),
            "final_legacy_status": final_legacy_status,
            "final_legacy_reasons": list(final_legacy_reasons),
            "shadow_reevaluation_reason": shadow_reevaluation_reason,
            "data_asof_date": effective_data_asof,
        })
        _atomic_write_json(manifest_path, manifest)
        decision_markdown = _promotion_decision_markdown(
            run_id,
            champion_id,
            selected_experiment_id,
            persistence_status,
            persistence_reason,
            final_legacy_status,
            final_legacy_reasons,
        )
        global_state_changed = persist_evolution_outcome(
            run_dir=run_dir,
            snapshot=snapshot,
            candidate_scores=candidate_scores,
            experiments=experiments,
            decision_markdown=decision_markdown,
            dry_run=dry_run or not shadow_eligible,
            promote_shadow=promote_shadow,
            state_path=resolved_state_path,
            expected_previous_fingerprint=previous_fingerprint,
        )
        if global_state_changed:
            persistence_status = "committed"
            persistence_reason = "影子状态转换 CAS 已提交。"
        manifest.update({
            "global_state_changed": global_state_changed,
            "promotion_persistence_status": persistence_status,
        })
        _atomic_write_text(
            run_dir / "shadow_decision.md",
            _promotion_decision_markdown(
                run_id,
                champion_id,
                selected_experiment_id,
                persistence_status,
                persistence_reason,
                final_legacy_status,
                final_legacy_reasons,
            ),
        )
        _atomic_write_json(manifest_path, manifest)

        pd.DataFrame(trial_rows).to_csv(run_dir / "trials.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(round_rows).to_csv(run_dir / "rounds.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([
            {"version": name, **metrics} for name, metrics in final_metrics.items()
        ]).to_csv(run_dir / "test_comparison.csv", index=False, encoding="utf-8-sig")
        (run_dir / "champion_candidate.yaml").write_text(
            yaml.safe_dump(champion_params, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        summary_path = write_chinese_summary(
            run_dir, asof_date, champion_id, round_rows, final_metrics, test_status, test_reason
        )
        manifest["summary"] = str(summary_path)
        if test_status != "ready_for_manual_review":
            raise RuntimeError(f"Holdout test {test_status}: {test_reason}")

        manifest.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - run_started, 6),
            "peak_memory_bytes": _peak_memory_bytes(),
        })
        _atomic_write_json(manifest_path, manifest)
        publish_run_metadata(output_root, manifest)
        return EvolutionOutcome(run_id, run_dir, champion_id, champion_params, test_status, test_reason)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        committed = _state_records_promotion(
            resolved_state_path,
            input_state,
            run_id,
            selected_experiment_id,
        )
        prior_persistence_status = str(
            manifest.get("promotion_persistence_status", "not_started")
        )
        if committed:
            persistence_status = "committed"
            persistence_reason = f"全局影子状态已提交；后续运行产物失败：{error_text}"
        elif prior_persistence_status in {"pending", "committed"}:
            persistence_status = "rejected"
            persistence_reason = f"影子状态未提交：{error_text}"
        else:
            persistence_status = prior_persistence_status
            persistence_reason = error_text
        manifest.update({
            "status": "failed",
            "global_state_changed": committed,
            "promotion_persistence_status": persistence_status,
            "selected_experiment_id": selected_experiment_id,
            "failed_gates": manifest.get("failed_gates", ["run_failed"]),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": error_text,
            "elapsed_seconds": round(time.perf_counter() - run_started, 6),
            "peak_memory_bytes": _peak_memory_bytes(),
        })
        _atomic_write_text(
            run_dir / "shadow_decision.md",
            _promotion_decision_markdown(
                run_id,
                champion_id,
                selected_experiment_id,
                persistence_status,
                persistence_reason,
                final_legacy_status,
                final_legacy_reasons,
            ),
        )
        _atomic_write_json(manifest_path, manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded evolution for the strong-pullback satellite strategy."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--output-root", default="outputs/evolution_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", default=None, metavar="RUN_ID")
    parser.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--promote-shadow", action="store_true", default=False)
    parser.add_argument(
        "--state-path", default="outputs/evolution_state/strong_pullback.json"
    )
    return parser.parse_args(argv)


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    data_path = Path(args.data).resolve()
    benchmark_path = Path(args.benchmark).resolve() if args.benchmark else None
    config = load_evolution_config(config_path)
    validate_input_schema(data_path)
    newest_date = pd.read_csv(data_path, usecols=["date"], parse_dates=["date"])["date"].max()
    asof_date = pd.Timestamp(args.asof_date) if args.asof_date else pd.Timestamp(newest_date)
    if asof_date < config.periods.test_start:
        raise ValueError("asof-date must reach the configured test period")
    run_id = args.resume or args.run_id or (
        f"strong_pullback_{asof_date:%Y%m%d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    outcome = run_evolution(
        config=config,
        data_path=data_path,
        config_path=config_path,
        benchmark_path=benchmark_path,
        asof_date=asof_date,
        output_root=Path(args.output_root).resolve(),
        run_id=run_id,
        resume=bool(args.resume),
        git_commit=_git_commit(project_root),
        dry_run=args.dry_run,
        promote_shadow=args.promote_shadow,
        state_path=Path(args.state_path).resolve(),
    )
    print(
        "Strong-pullback evolution completed.\n"
        f"  run_id: {outcome.run_id}\n"
        f"  champion: {outcome.champion_id}\n"
        f"  test_status: {outcome.test_status}\n"
        f"  output: {outcome.run_dir}"
    )


if __name__ == "__main__":
    main()
