from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from numbers import Real
from pathlib import Path


class EvolutionStateError(ValueError):
    pass


EVOLUTION_STATE_SCHEMA_VERSION = 2


class StateWriteExpectation(Enum):
    ABSENT = "absent"


@dataclass(frozen=True)
class EvolutionTransitionJournal:
    schema_version: int
    status: str
    operation: str
    run_id: str
    experiment_id: str | None
    target_state_fingerprint: str
    target_data_asof_date: str | None
    updated_at: str
    reason: str | None = None


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _FrozenDict(dict[str, object]):
    def __init__(self, *args: object, **kwargs: object) -> None:
        source = dict(*args, **kwargs)
        super().__init__((key, _freeze(value)) for key, value in source.items())

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("state mappings are immutable")

    __delitem__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, value: object) -> _FrozenDict:
        self._immutable(value)
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenDict:
        return self


class _FrozenList(list[object]):
    def __init__(self, values: Sequence[object] = ()) -> None:
        super().__init__(_freeze(value) for value in values)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("state mappings are immutable")

    __delitem__ = _immutable
    __setitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenList:
        return self


def _freeze(value: object) -> object:
    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value
    if isinstance(value, Mapping):
        return _FrozenDict(value)
    if isinstance(value, list):
        return _FrozenList(value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> _FrozenDict:
    if not isinstance(value, Mapping):
        raise TypeError("state parameters must be a mapping")
    return _FrozenDict(value)


def _canonicalize(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("fingerprint mappings require string keys")
            normalized[key] = _canonicalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint values must be finite")
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def fingerprint_payload(value: object) -> str:
    payload = json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_parameter_candidates(
    champion_parameters: Mapping[str, object],
    candidate_overrides: Sequence[Mapping[str, object]],
    max_candidates: int,
    seed: int,
) -> tuple[dict[str, object], ...]:
    if not isinstance(champion_parameters, Mapping):
        raise TypeError("champion_parameters must be a mapping")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 0:
        raise ValueError("max_candidates must be a non-negative integer")

    champion = copy.deepcopy(dict(champion_parameters))
    candidates: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for override in candidate_overrides:
        if not isinstance(override, Mapping) or not override:
            raise ValueError("each candidate override must be a non-empty mapping")
        candidate = copy.deepcopy(champion)
        candidate.update(copy.deepcopy(dict(override)))
        if candidate == champion:
            continue
        fingerprint = fingerprint_payload(candidate)
        if fingerprint in fingerprints:
            raise ValueError("conflicting duplicate candidate fingerprints")
        fingerprints.add(fingerprint)
        candidates.append(candidate)

    random.Random(seed).shuffle(candidates)
    return tuple(candidates[:max_candidates])


@dataclass(frozen=True)
class FoldMetrics:
    fold_id: str
    total_return: float
    max_drawdown: float
    sharpe: float
    filled_trades: int
    average_turnover: float
    pnl_concentration: float


@dataclass(frozen=True)
class PromotionPolicy:
    min_folds: int = 3
    min_filled_trades_per_fold: int = 5
    min_positive_fold_ratio: float = 2.0 / 3.0
    min_mean_return_improvement: float = 0.01
    max_drawdown_floor: float = -0.40
    max_drawdown_worsening: float = 0.05
    max_turnover_ratio: float = 1.50
    max_pnl_concentration: float = 0.50

    def __post_init__(self) -> None:
        for name in ("min_folds", "min_filled_trades_per_fold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "min_positive_fold_ratio",
            "min_mean_return_improvement",
            "max_drawdown_floor",
            "max_drawdown_worsening",
            "max_turnover_ratio",
            "max_pnl_concentration",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= self.min_positive_fold_ratio <= 1.0:
            raise ValueError("min_positive_fold_ratio must be between 0 and 1")
        if not 0.0 <= self.min_mean_return_improvement <= 1.0:
            raise ValueError("min_mean_return_improvement must be between 0 and 1")
        if not -1.0 <= self.max_drawdown_floor <= 0.0:
            raise ValueError("max_drawdown_floor must be between -1 and 0")
        if not 0.0 <= self.max_drawdown_worsening <= 1.0:
            raise ValueError("max_drawdown_worsening must be between 0 and 1")
        if self.max_turnover_ratio <= 0.0:
            raise ValueError("max_turnover_ratio must be positive")
        if not 0.0 <= self.max_pnl_concentration <= 1.0:
            raise ValueError("max_pnl_concentration must be between 0 and 1")


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    failed_gates: tuple[str, ...]
    gates: dict[str, dict[str, object]]


def _fold_map(folds: Sequence[FoldMetrics]) -> tuple[dict[str, FoldMetrics] | None, str | None]:
    try:
        values = tuple(folds)
    except TypeError:
        return None, "invalid_folds"
    result: dict[str, FoldMetrics] = {}
    for fold in values:
        if not isinstance(fold, FoldMetrics):
            return None, "invalid_folds"
        if not isinstance(fold.fold_id, str) or not fold.fold_id:
            return None, "invalid_fold_id"
        if fold.fold_id in result:
            return None, "duplicate_fold_ids"
        result[fold.fold_id] = fold
    return result, None


def _finite_metrics(fold: FoldMetrics) -> bool:
    values = (
        fold.total_return,
        fold.max_drawdown,
        fold.sharpe,
        fold.filled_trades,
        fold.average_turnover,
        fold.pnl_concentration,
    )
    return all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    )


def _gate(passed: bool, value: object, threshold: object) -> dict[str, object]:
    return {"passed": passed, "value": value, "threshold": threshold}


def _insufficient_metrics_decision(policy: PromotionPolicy, reason: str) -> PromotionDecision:
    thresholds = {
        "min_folds": policy.min_folds,
        "min_filled_trades": policy.min_filled_trades_per_fold,
        "positive_fold_ratio": policy.min_positive_fold_ratio,
        "mean_return_improvement": policy.min_mean_return_improvement,
        "max_drawdown": policy.max_drawdown_floor,
        "drawdown_worsening": policy.max_drawdown_worsening,
        "turnover_ratio": policy.max_turnover_ratio,
        "pnl_concentration": policy.max_pnl_concentration,
    }
    gates = {
        name: _gate(False, None, threshold)
        for name, threshold in thresholds.items()
    }
    gates["finite_metrics"] = {
        "passed": False,
        "value": False,
        "threshold": True,
        "reason": reason,
    }
    return PromotionDecision(
        "insufficient_evidence",
        tuple((*thresholds, "finite_metrics")),
        gates,
    )


def evaluate_candidate(
    challenger: Sequence[FoldMetrics],
    champion: Sequence[FoldMetrics],
    policy: PromotionPolicy,
) -> PromotionDecision:
    challenger_by_id, challenger_error = _fold_map(challenger)
    champion_by_id, champion_error = _fold_map(champion)
    if challenger_error or champion_error:
        error = challenger_error or champion_error or "invalid_folds"
        return PromotionDecision("rejected", (error,), {"fold_alignment": {"passed": False, "error": error}})
    assert challenger_by_id is not None and champion_by_id is not None
    if set(challenger_by_id) != set(champion_by_id):
        return PromotionDecision(
            "rejected",
            ("fold_alignment",),
            {
                "fold_alignment": {
                    "passed": False,
                    "challenger_fold_ids": tuple(sorted(challenger_by_id)),
                    "champion_fold_ids": tuple(sorted(champion_by_id)),
                }
            },
        )

    fold_ids = tuple(sorted(challenger_by_id))
    challenger_folds = tuple(challenger_by_id[fold_id] for fold_id in fold_ids)
    champion_folds = tuple(champion_by_id[fold_id] for fold_id in fold_ids)
    all_finite = all(_finite_metrics(fold) for fold in (*challenger_folds, *champion_folds))
    if not all_finite:
        return _insufficient_metrics_decision(policy, "missing, non-numeric, or non-finite metric")
    fold_count = len(fold_ids)
    min_trades = min((fold.filled_trades for fold in challenger_folds), default=0)
    positive_ratio = (
        sum(fold.total_return > 0 for fold in challenger_folds) / fold_count
        if fold_count
        else 0.0
    )
    mean_return_improvement = (
        sum(fold.total_return for fold in challenger_folds) / fold_count
        - sum(fold.total_return for fold in champion_folds) / fold_count
        if fold_count
        else 0.0
    )
    worst_drawdown = min((fold.max_drawdown for fold in challenger_folds), default=math.inf)
    worst_drawdown_worsening = max(
        (max(champion_fold.max_drawdown - challenger_fold.max_drawdown, 0.0)
         for challenger_fold, champion_fold in zip(challenger_folds, champion_folds)),
        default=math.inf,
    )
    champion_turnover = sum(fold.average_turnover for fold in champion_folds) / fold_count if fold_count else 0.0
    challenger_turnover = sum(fold.average_turnover for fold in challenger_folds) / fold_count if fold_count else 0.0
    if abs(champion_turnover) <= 1e-12:
        turnover_ratio = 1.0 if abs(challenger_turnover) <= 1e-12 else math.inf
    else:
        turnover_ratio = challenger_turnover / champion_turnover
    max_concentration = max((fold.pnl_concentration for fold in challenger_folds), default=math.inf)

    gates = {
        "min_folds": _gate(fold_count >= policy.min_folds, fold_count, policy.min_folds),
        "min_filled_trades": _gate(
            min_trades >= policy.min_filled_trades_per_fold,
            min_trades,
            policy.min_filled_trades_per_fold,
        ),
        "positive_fold_ratio": _gate(
            positive_ratio >= policy.min_positive_fold_ratio,
            positive_ratio,
            policy.min_positive_fold_ratio,
        ),
        "mean_return_improvement": _gate(
            mean_return_improvement >= policy.min_mean_return_improvement,
            mean_return_improvement,
            policy.min_mean_return_improvement,
        ),
        "max_drawdown": _gate(
            worst_drawdown >= policy.max_drawdown_floor,
            worst_drawdown,
            policy.max_drawdown_floor,
        ),
        "drawdown_worsening": _gate(
            worst_drawdown_worsening <= policy.max_drawdown_worsening,
            worst_drawdown_worsening,
            policy.max_drawdown_worsening,
        ),
        "turnover_ratio": _gate(
            turnover_ratio <= policy.max_turnover_ratio,
            turnover_ratio,
            policy.max_turnover_ratio,
        ),
        "pnl_concentration": _gate(
            max_concentration <= policy.max_pnl_concentration,
            max_concentration,
            policy.max_pnl_concentration,
        ),
    }
    failed = [name for name, result in gates.items() if not result["passed"]]
    if not all_finite:
        gates["finite_metrics"] = {"passed": False, "value": False, "threshold": True}
        failed.append("finite_metrics")
    status = "insufficient_evidence" if not all_finite or "min_folds" in failed or "min_filled_trades" in failed else (
        "rejected" if failed else "eligible_for_shadow"
    )
    return PromotionDecision(status, tuple(failed), gates)


def _copy_parameters(parameters: Mapping[str, object], name: str) -> dict[str, object]:
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(parameters))


def _state_time(now: object | None) -> str:
    if now is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(now, datetime):
        return now.isoformat()
    if isinstance(now, str) and now:
        return now
    raise ValueError("now must be a non-empty string or datetime")


def _transition_journal_path(state_path: Path) -> Path:
    path = Path(state_path)
    return path.with_name(f"{path.stem}.promotion_journal.json")


def _transition_lock_path(state_path: Path) -> Path:
    path = Path(state_path)
    return path.with_name(f".{path.name}.transition.lock")


def _write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    content = json.dumps(
        _canonicalize(payload), sort_keys=True, ensure_ascii=True, allow_nan=False
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _journal_from_payload(payload: object) -> EvolutionTransitionJournal:
    if not isinstance(payload, Mapping):
        raise EvolutionStateError("Evolution transition journal must be a JSON mapping")
    expected = {field.name for field in dataclasses.fields(EvolutionTransitionJournal)}
    actual = set(payload)
    if actual - expected:
        raise EvolutionStateError(
            f"Unknown transition journal fields: {sorted(actual - expected)}"
        )
    if expected - actual:
        raise EvolutionStateError(
            f"Missing transition journal fields: {sorted(expected - actual)}"
        )
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise EvolutionStateError("Unsupported transition journal schema_version")
    if payload.get("status") not in {"pending", "committed", "rejected"}:
        raise EvolutionStateError("Invalid transition journal status")
    if payload.get("operation") not in {"promote", "rollback"}:
        raise EvolutionStateError("Invalid transition journal operation")
    for name in ("run_id", "target_state_fingerprint", "updated_at"):
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise EvolutionStateError(f"Invalid transition journal {name}")
    experiment_id = payload.get("experiment_id")
    if experiment_id is not None and (
        not isinstance(experiment_id, str) or not experiment_id.strip()
    ):
        raise EvolutionStateError("Invalid transition journal experiment_id")
    reason = payload.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise EvolutionStateError("Invalid transition journal reason")
    try:
        return EvolutionTransitionJournal(**dict(payload))
    except TypeError as exc:
        raise EvolutionStateError(f"Invalid evolution transition journal: {exc}") from exc


def load_evolution_transition_journal(
    state_path: Path,
) -> EvolutionTransitionJournal | None:
    journal_path = _transition_journal_path(state_path)
    if not journal_path.exists():
        return None
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvolutionStateError(f"Could not read evolution transition journal: {exc}") from exc
    return _journal_from_payload(payload)


def _stage_evolution_transition_unlocked(
    state_path: Path,
    target_state: EvolutionState,
    *,
    operation: str,
    run_id: str,
    experiment_id: str | None,
    now: object | None,
) -> EvolutionTransitionJournal:
    journal = EvolutionTransitionJournal(
        schema_version=1,
        status="pending",
        operation=operation,
        run_id=run_id,
        experiment_id=experiment_id,
        target_state_fingerprint=fingerprint_payload(target_state),
        target_data_asof_date=target_state.data_asof_date,
        updated_at=_state_time(now),
        reason=None,
    )
    journal = _journal_from_payload(_canonicalize(journal))
    _write_json_atomic(_transition_journal_path(state_path), journal)
    return journal


def stage_evolution_transition(
    state_path: Path,
    target_state: EvolutionState,
    *,
    operation: str,
    run_id: str,
    experiment_id: str | None = None,
    now: object | None = None,
) -> EvolutionTransitionJournal:
    with _exclusive_file_lock(_transition_lock_path(state_path)):
        return _stage_evolution_transition_unlocked(
            state_path,
            target_state,
            operation=operation,
            run_id=run_id,
            experiment_id=experiment_id,
            now=now,
        )


def _reconcile_evolution_transition_unlocked(
    state_path: Path,
    default_state: EvolutionState,
    *,
    now: object | None,
) -> EvolutionTransitionJournal | None:
    journal = load_evolution_transition_journal(state_path)
    if journal is None or journal.status != "pending":
        return journal
    committed = False
    if Path(state_path).exists():
        actual_state = load_evolution_state(Path(state_path), default_state)
        committed = (
            fingerprint_payload(actual_state) == journal.target_state_fingerprint
        )
    reconciled = dataclasses.replace(
        journal,
        status="committed" if committed else "rejected",
        updated_at=_state_time(now),
        reason=(
            "target state fingerprint is committed"
            if committed
            else "target state was not committed"
        ),
    )
    _write_json_atomic(_transition_journal_path(state_path), reconciled)
    return reconciled


def reconcile_evolution_transition(
    state_path: Path,
    default_state: EvolutionState,
    *,
    now: object | None = None,
) -> EvolutionTransitionJournal | None:
    with _exclusive_file_lock(_transition_lock_path(state_path)):
        return _reconcile_evolution_transition_unlocked(
            state_path, default_state, now=now
        )


def commit_evolution_state_transition(
    state_path: Path,
    target_state: EvolutionState,
    *,
    operation: str,
    run_id: str,
    experiment_id: str | None,
    expected_previous_fingerprint: str | StateWriteExpectation | None,
    now: object | None = None,
) -> str:
    with _exclusive_file_lock(_transition_lock_path(state_path)):
        _reconcile_evolution_transition_unlocked(
            state_path, target_state, now=now
        )
        pending = _stage_evolution_transition_unlocked(
            state_path,
            target_state,
            operation=operation,
            run_id=run_id,
            experiment_id=experiment_id,
            now=now,
        )
        try:
            fingerprint = write_evolution_state_atomic(
                state_path,
                target_state,
                expected_previous_fingerprint=expected_previous_fingerprint,
            )
        except Exception as exc:
            rejected = dataclasses.replace(
                pending,
                status="rejected",
                updated_at=_state_time(now),
                reason=f"{type(exc).__name__}: {exc}",
            )
            _write_json_atomic(_transition_journal_path(state_path), rejected)
            raise
        committed = dataclasses.replace(
            pending,
            status="committed",
            updated_at=_state_time(now),
            reason="state CAS committed",
        )
        _write_json_atomic(_transition_journal_path(state_path), committed)
        return fingerprint


@dataclass(frozen=True)
class EvolutionState:
    schema_version: int
    champion_version: str
    champion_parameters: dict[str, object]
    shadow_status: str
    shadow_version: str | None = None
    shadow_parameters: dict[str, object] | None = None
    shadow_experiment_id: str | None = None
    shadow_run_id: str | None = None
    shadow_data_fingerprint: str | None = None
    previous_champion_version: str | None = None
    previous_champion_parameters: dict[str, object] | None = None
    last_completed_run_id: str | None = None
    last_data_fingerprint: str | None = None
    blocked_reason: str | None = None
    updated_at: str | None = None
    data_asof_date: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != EVOLUTION_STATE_SCHEMA_VERSION
        ):
            raise EvolutionStateError(
                f"schema_version must be {EVOLUTION_STATE_SCHEMA_VERSION}"
            )
        if not isinstance(self.champion_version, str) or not self.champion_version.strip():
            raise EvolutionStateError("champion_version must be a non-empty string")
        if self.shadow_status not in {"none", "shadow", "rolled_back"}:
            raise EvolutionStateError("shadow_status is invalid")
        optional_identifiers = (
            "shadow_version",
            "shadow_experiment_id",
            "shadow_run_id",
            "shadow_data_fingerprint",
            "previous_champion_version",
            "last_completed_run_id",
            "last_data_fingerprint",
            "blocked_reason",
        )
        for name in optional_identifiers:
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise EvolutionStateError(f"{name} must be a non-empty string or null")
        if not isinstance(self.updated_at, str) or not self.updated_at.strip():
            raise EvolutionStateError("updated_at must be an aware ISO timestamp")
        try:
            parsed_updated_at = datetime.fromisoformat(self.updated_at)
        except ValueError as exc:
            raise EvolutionStateError("updated_at must be an aware ISO timestamp") from exc
        if parsed_updated_at.tzinfo is None or parsed_updated_at.utcoffset() is None:
            raise EvolutionStateError("updated_at must be an aware ISO timestamp")
        if self.data_asof_date is not None:
            try:
                parsed_asof = date.fromisoformat(self.data_asof_date)
            except (TypeError, ValueError) as exc:
                raise EvolutionStateError("data_asof_date must be an ISO date") from exc
            if parsed_asof.isoformat() != self.data_asof_date:
                raise EvolutionStateError("data_asof_date must be an ISO date")

        object.__setattr__(self, "champion_parameters", _freeze_mapping(self.champion_parameters))
        if self.shadow_parameters is not None:
            object.__setattr__(self, "shadow_parameters", _freeze_mapping(self.shadow_parameters))
        if self.previous_champion_parameters is not None:
            object.__setattr__(self, "previous_champion_parameters", _freeze_mapping(self.previous_champion_parameters))
        shadow_fields = (
            self.shadow_version,
            self.shadow_parameters,
            self.shadow_experiment_id,
            self.shadow_run_id,
            self.shadow_data_fingerprint,
        )
        if self.shadow_status == "none":
            if any(value is not None for value in shadow_fields) or any(
                value is not None
                for value in (
                    self.previous_champion_version,
                    self.previous_champion_parameters,
                    self.blocked_reason,
                )
            ):
                raise EvolutionStateError("none state cannot contain shadow lifecycle fields")
        elif self.shadow_status == "shadow":
            if any(value is None for value in shadow_fields) or any(
                value is None
                for value in (
                    self.previous_champion_version,
                    self.previous_champion_parameters,
                    self.data_asof_date,
                )
            ):
                raise EvolutionStateError("shadow state requires complete lifecycle fields")
            if self.blocked_reason is not None:
                raise EvolutionStateError("shadow state cannot contain blocked_reason")
        else:
            if any(value is None for value in shadow_fields) or self.blocked_reason is None:
                raise EvolutionStateError("rolled_back state requires shadow fields and reason")
            if self.previous_champion_version is not None or self.previous_champion_parameters is not None:
                raise EvolutionStateError("rolled_back state cannot retain previous champion fields")
            if self.data_asof_date is None:
                raise EvolutionStateError("rolled_back state requires data_asof_date")

    @classmethod
    def initial(
        cls,
        champion_version: str,
        champion_parameters: Mapping[str, object],
        *,
        now: object | None = None,
    ) -> EvolutionState:
        return cls(
            schema_version=EVOLUTION_STATE_SCHEMA_VERSION,
            champion_version=champion_version,
            champion_parameters=_copy_parameters(champion_parameters, "champion_parameters"),
            shadow_status="none",
            updated_at=_state_time(now),
        )

    @property
    def experiment_id(self) -> str | None:
        return self.shadow_experiment_id

    @property
    def run_id(self) -> str | None:
        return self.shadow_run_id

    @property
    def data_fingerprint(self) -> str | None:
        return self.shadow_data_fingerprint

    @property
    def last_run_id(self) -> str | None:
        return self.last_completed_run_id


def _state_from_payload(payload: object) -> EvolutionState:
    if not isinstance(payload, Mapping):
        raise EvolutionStateError("Evolution state must be a JSON mapping")
    expected_fields = {field.name for field in dataclasses.fields(EvolutionState)}
    actual_fields = set(payload)
    unknown = actual_fields - expected_fields
    missing = expected_fields - actual_fields
    if unknown:
        raise EvolutionStateError(f"Unknown state fields: {sorted(unknown)}")
    if missing:
        raise EvolutionStateError(f"Missing state fields: {sorted(missing)}")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != EVOLUTION_STATE_SCHEMA_VERSION
    ):
        raise EvolutionStateError(
            f"Unsupported state schema_version: {payload.get('schema_version')!r}"
        )
    try:
        return EvolutionState(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise EvolutionStateError(f"Invalid evolution state: {exc}") from exc


def load_evolution_state(path: Path, default_state: EvolutionState) -> EvolutionState:
    path = Path(path)
    if not path.exists():
        return default_state
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvolutionStateError(f"Could not read evolution state: {exc}") from exc
    return _state_from_payload(payload)


def write_evolution_state_atomic(
    path: Path,
    state: EvolutionState,
    expected_previous_fingerprint: str | StateWriteExpectation | None = None,
) -> str:
    if not isinstance(state, EvolutionState):
        raise EvolutionStateError("state must be an EvolutionState")
    _state_from_payload(_canonicalize(state))
    path = Path(path)
    payload = json.dumps(
        _canonicalize(state), sort_keys=True, ensure_ascii=True
    )
    fingerprint = fingerprint_payload(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _exclusive_file_lock(lock_path):
        existing_state = load_evolution_state(path, state) if path.exists() else None
        if expected_previous_fingerprint is StateWriteExpectation.ABSENT:
            if path.exists():
                raise EvolutionStateError("expected evolution state to be absent")
        elif expected_previous_fingerprint is not None:
            if not isinstance(expected_previous_fingerprint, str):
                raise EvolutionStateError("invalid previous state expectation")
            if not path.exists():
                raise EvolutionStateError("previous state fingerprint does not match")
            assert existing_state is not None
            actual_previous = fingerprint_payload(existing_state)
            if actual_previous != expected_previous_fingerprint:
                raise EvolutionStateError("previous state fingerprint does not match")
        if existing_state is not None and existing_state.data_asof_date is not None:
            if state.data_asof_date is None:
                raise EvolutionStateError(
                    "data_asof_date cannot be removed from evolution state"
                )
            if date.fromisoformat(state.data_asof_date) < date.fromisoformat(
                existing_state.data_asof_date
            ):
                raise EvolutionStateError(
                    "data_asof_date cannot be older than the current evolution state"
                )

        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return fingerprint


def promote_to_shadow(
    state: EvolutionState,
    challenger_version: str,
    challenger_parameters: Mapping[str, object],
    experiment_id: str,
    run_id: str,
    data_fingerprint: str,
    data_asof_date: str,
    now: object | None = None,
) -> EvolutionState:
    return EvolutionState(
        schema_version=state.schema_version,
        champion_version=state.champion_version,
        champion_parameters=state.champion_parameters,
        shadow_status="shadow",
        shadow_version=challenger_version,
        shadow_parameters=_copy_parameters(challenger_parameters, "challenger_parameters"),
        shadow_experiment_id=experiment_id,
        shadow_run_id=run_id,
        shadow_data_fingerprint=data_fingerprint,
        previous_champion_version=state.champion_version,
        previous_champion_parameters=state.champion_parameters,
        last_completed_run_id=run_id,
        last_data_fingerprint=data_fingerprint,
        blocked_reason=None,
        updated_at=_state_time(now),
        data_asof_date=data_asof_date,
    )


def rollback_shadow(
    state: EvolutionState,
    reason: str,
    data_asof_date: str | None = None,
    now: object | None = None,
) -> EvolutionState:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    champion_version = state.previous_champion_version or state.champion_version
    champion_parameters = (
        state.previous_champion_parameters
        if state.previous_champion_parameters is not None
        else state.champion_parameters
    )
    return EvolutionState(
        schema_version=state.schema_version,
        champion_version=champion_version,
        champion_parameters=champion_parameters,
        shadow_status="rolled_back",
        shadow_version=state.shadow_version,
        shadow_parameters=state.shadow_parameters,
        shadow_experiment_id=state.shadow_experiment_id,
        shadow_run_id=state.shadow_run_id,
        shadow_data_fingerprint=state.shadow_data_fingerprint,
        previous_champion_version=None,
        previous_champion_parameters=None,
        last_completed_run_id=state.last_completed_run_id,
        last_data_fingerprint=state.last_data_fingerprint,
        blocked_reason=reason,
        updated_at=_state_time(now),
        data_asof_date=data_asof_date or state.data_asof_date,
    )
