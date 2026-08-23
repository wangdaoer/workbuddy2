from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import Callable, Mapping

import pandas as pd
import yaml

from run_backtest import max_drawdown, sharpe_like
from strategy_evolution_core import (
    FoldMetrics,
    fingerprint_payload,
    generate_parameter_candidates,
)


DEFAULT_STRATEGY_PARAMS: dict[str, object] = {
    "train_days": 252,
    "retrain_frequency": 20,
    "top_n": 8,
    "rebalance_frequency": 5,
    "max_position_weight": 0.08,
    "leverage": 0.60,
    "min_score": None,
    "commission_bps": 1.0,
    "impact_bps": 0.7,
    "max_buy_open_gap": 0.05,
    "limit_buffer": 0.995,
    "initial_capital": 1_000_000.0,
    "max_abs_daily_return": 0.22,
    "min_close": 2.0,
    "min_avg_amount_20d": 30_000_000.0,
    "min_pullback_5d": 0.03,
    "max_pullback_5d": 0.18,
    "min_prior_return_20": 0.08,
    "min_prior_return_60": 0.18,
    "min_return_20d": -0.12,
    "min_return_60d": 0.0,
    "min_distance_ma60": -0.10,
    "max_intraday_return": 0.05,
    "market_ma_window": 120,
    "market_risk_off_drawdown_20d": -0.08,
    "market_below_ma_exposure": 0.60,
    "market_crash_exposure": 0.0,
    "regime_strong_leverage": None,
    "regime_exceptional_leverage": None,
    "regime_strong_breadth_threshold": None,
    "regime_exceptional_breadth_threshold": None,
    "regime_strong_volatility_max": None,
    "regime_exceptional_volatility_max": None,
    "basket_guard_return_20d_min": None,
    "basket_guard_distance_ma60_min": None,
    "basket_guard_scale": 1.0,
    "rebound_exit_return": None,
    "rebound_exit_scale": 0.0,
    "rebound_exit_market_exposure_max": None,
    "rebound_exit_market_exposure_min": None,
}

ALLOWED_OVERRIDE_PARAMS = frozenset({
    "top_n", "rebalance_frequency", "max_position_weight", "leverage", "min_score",
    "min_close", "min_avg_amount_20d", "min_pullback_5d", "max_pullback_5d",
    "min_prior_return_20", "min_prior_return_60", "min_return_20d",
    "min_return_60d", "min_distance_ma60", "max_intraday_return",
    "basket_guard_return_20d_min", "basket_guard_distance_ma60_min",
    "basket_guard_scale", "rebound_exit_return", "rebound_exit_scale",
    "rebound_exit_market_exposure_max", "rebound_exit_market_exposure_min",
    "regime_strong_leverage", "regime_exceptional_leverage",
    "regime_strong_breadth_threshold", "regime_exceptional_breadth_threshold",
    "regime_strong_volatility_max", "regime_exceptional_volatility_max",
})

INTEGER_STRATEGY_PARAMS = frozenset({
    "train_days", "retrain_frequency", "top_n", "rebalance_frequency",
    "market_ma_window",
})
OPTIONAL_NUMERIC_STRATEGY_PARAMS = frozenset({
    "min_score", "basket_guard_return_20d_min",
    "basket_guard_distance_ma60_min", "rebound_exit_return",
    "rebound_exit_market_exposure_max", "rebound_exit_market_exposure_min",
    "regime_strong_leverage", "regime_exceptional_leverage",
    "regime_strong_breadth_threshold", "regime_exceptional_breadth_threshold",
    "regime_strong_volatility_max", "regime_exceptional_volatility_max",
})
NUMERIC_STRATEGY_PARAMS = frozenset(DEFAULT_STRATEGY_PARAMS) - (
    INTEGER_STRATEGY_PARAMS | OPTIONAL_NUMERIC_STRATEGY_PARAMS
)

SELECTION_KEYS = frozenset({
    "min_validation_days",
    "min_test_days",
    "max_drawdown_floor",
    "min_annualized_return_delta",
    "min_sharpe_delta",
    "max_turnover_ratio",
    "rolling_window_days",
    "max_negative_window_rate",
})

EVOLUTION_CORE_KEYS = frozenset({
    "max_candidates_per_group", "random_seed", "min_folds",
    "min_filled_trades_per_fold", "min_positive_fold_ratio",
    "min_mean_return_improvement", "max_drawdown_floor",
    "max_drawdown_worsening", "max_turnover_ratio", "max_pnl_concentration",
})

RESERVED_EVOLUTION_IDS = frozenset({
    "baseline", "champion", "final", "experiments", "trials",
    "__shadow_incumbent__",
})
TOP_LEVEL_KEYS = frozenset({
    "strategy", "evolution_core", "periods", "baseline", "search_groups", "selection",
})
PERIOD_KEYS = frozenset({
    "research_start", "train_end", "validation_start", "validation_end",
    "core_test_start", "core_test_end", "test_start", "test_end",
})
SEARCH_GROUP_KEYS = frozenset({"id", "hypothesis_cn", "candidates"})
SEARCH_CANDIDATE_KEYS = frozenset({"id", "overrides"})
WINDOWS_RESERVED_DEVICE_NAMES = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
})


@dataclass(frozen=True)
class EvolutionPeriods:
    research_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    core_test_start: pd.Timestamp
    core_test_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp | None


@dataclass(frozen=True)
class SelectionRules:
    min_validation_days: int
    min_test_days: int
    max_drawdown_floor: float
    min_annualized_return_delta: float
    min_sharpe_delta: float
    max_turnover_ratio: float
    rolling_window_days: int
    max_negative_window_rate: float


@dataclass(frozen=True)
class EvolutionCoreConfig:
    max_candidates_per_group: int
    random_seed: int
    min_folds: int
    min_filled_trades_per_fold: int
    min_positive_fold_ratio: float
    min_mean_return_improvement: float
    max_drawdown_floor: float
    max_drawdown_worsening: float
    max_turnover_ratio: float
    max_pnl_concentration: float


@dataclass(frozen=True)
class SearchCandidate:
    candidate_id: str
    overrides: dict[str, object]


@dataclass(frozen=True)
class SearchGroup:
    group_id: str
    hypothesis_cn: str
    candidates: tuple[SearchCandidate, ...]


@dataclass(frozen=True)
class EvolutionConfig:
    strategy: str
    evolution_core: EvolutionCoreConfig
    periods: EvolutionPeriods
    baseline: dict[str, object]
    search_groups: tuple[SearchGroup, ...]
    selection: SelectionRules


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    turnover_ratio: float
    robust_score: float


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    promotion: PromotionDecision
    metrics: dict[str, float]


@dataclass(frozen=True)
class TradingFold:
    fold_id: str
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]


@dataclass(frozen=True)
class CoreTestSegment:
    fold_id: str
    test_dates: tuple[pd.Timestamp, ...]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]


def _timestamp(value: object, name: str, allow_none: bool = False) -> pd.Timestamp | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Missing required date: {name}")
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date for {name}: {value!r}") from exc


def _validate_params(params: Mapping[str, object]) -> None:
    unknown = set(params) - set(DEFAULT_STRATEGY_PARAMS)
    if unknown:
        raise ValueError(f"Unknown strategy parameters: {sorted(unknown)}")
    for key in INTEGER_STRATEGY_PARAMS:
        if type(params[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in NUMERIC_STRATEGY_PARAMS:
        value = params[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{key} must be a finite real number")
    for key in OPTIONAL_NUMERIC_STRATEGY_PARAMS:
        value = params[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{key} must be null or a finite real number")
    for key in ("train_days", "retrain_frequency", "top_n", "rebalance_frequency"):
        if int(params[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("leverage", "max_position_weight"):
        value = float(params[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if float(params["leverage"]) > 1.5:
        raise ValueError("leverage must be <= 1.5")
    if float(params["max_position_weight"]) > 1.0:
        raise ValueError("max_position_weight must be <= 1.0")
    for key in ("commission_bps", "impact_bps", "max_buy_open_gap"):
        value = float(params[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be a finite non-negative number")
    if int(params["market_ma_window"]) <= 0:
        raise ValueError("market_ma_window must be positive")
    for key in ("initial_capital", "max_abs_daily_return"):
        value = float(params[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be a finite positive number")
    if float(params["max_abs_daily_return"]) > 1.0:
        raise ValueError("max_abs_daily_return must be <= 1.0")
    market_risk_off_drawdown = float(params["market_risk_off_drawdown_20d"])
    if not math.isfinite(market_risk_off_drawdown) or not -1.0 <= market_risk_off_drawdown <= 0.0:
        raise ValueError("market_risk_off_drawdown_20d must be between -1 and 0")
    for key in (
        "limit_buffer",
        "market_below_ma_exposure",
        "market_crash_exposure",
        "basket_guard_scale",
        "rebound_exit_scale",
    ):
        value = float(params[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    for key in ("basket_guard_return_20d_min", "basket_guard_distance_ma60_min"):
        value = params[key]
        if value is not None and (
            not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{key} must be between -1 and 1")
    rebound_exit_return = params["rebound_exit_return"]
    if rebound_exit_return is not None and (
        not math.isfinite(float(rebound_exit_return))
        or not 0.0 <= float(rebound_exit_return) <= 1.0
    ):
        raise ValueError("rebound_exit_return must be between 0 and 1")
    for key in ("rebound_exit_market_exposure_max", "rebound_exit_market_exposure_min"):
        value = params[key]
        if value is not None and (
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{key} must be between 0 and 1")
    regime_keys = (
        "regime_strong_leverage",
        "regime_exceptional_leverage",
        "regime_strong_breadth_threshold",
        "regime_exceptional_breadth_threshold",
        "regime_strong_volatility_max",
        "regime_exceptional_volatility_max",
    )
    regime_values = [params[key] for key in regime_keys]
    if any(value is not None for value in regime_values):
        if any(value is None for value in regime_values):
            raise ValueError("regime parameters must be configured together")
        strong_leverage = float(params["regime_strong_leverage"])
        exceptional_leverage = float(params["regime_exceptional_leverage"])
        strong_breadth = float(params["regime_strong_breadth_threshold"])
        exceptional_breadth = float(params["regime_exceptional_breadth_threshold"])
        strong_volatility = float(params["regime_strong_volatility_max"])
        exceptional_volatility = float(params["regime_exceptional_volatility_max"])
        if not float(params["leverage"]) <= strong_leverage <= exceptional_leverage <= 1.5:
            raise ValueError("regime leverage must satisfy base <= strong <= exceptional <= 1.5")
        if not 0.0 <= strong_breadth <= exceptional_breadth <= 1.0:
            raise ValueError("regime breadth must satisfy 0 <= strong <= exceptional <= 1")
        if not 0.0 < exceptional_volatility <= strong_volatility:
            raise ValueError("regime volatility must satisfy 0 < exceptional <= strong")
    if float(params["min_pullback_5d"]) > float(params["max_pullback_5d"]):
        raise ValueError("min_pullback_5d must be <= max_pullback_5d")


def is_windows_reserved_device_name(value: str) -> bool:
    normalized = value.strip().rstrip(". ").casefold()
    stem = normalized.split(".", 1)[0]
    return stem in WINDOWS_RESERVED_DEVICE_NAMES


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    identifier = value.strip()
    normalized = identifier.casefold()
    if (
        identifier in {".", ".."}
        or "/" in identifier
        or "\\" in identifier
        or Path(identifier).is_absolute()
        or PureWindowsPath(identifier).is_absolute()
        or any(ord(character) < 32 or ord(character) == 127 for character in identifier)
        or any(character in '<>:"|?*' for character in identifier)
        or identifier.endswith(".")
        or is_windows_reserved_device_name(identifier)
        or normalized in RESERVED_EVOLUTION_IDS
    ):
        raise ValueError(f"{label} must be a safe non-reserved filename component")
    return identifier


def _strict_integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _strict_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _parse_evolution_core(value: object) -> EvolutionCoreConfig:
    if not isinstance(value, Mapping):
        raise ValueError("evolution_core must be a mapping")
    raw = dict(value)
    unknown = set(raw) - EVOLUTION_CORE_KEYS
    missing = EVOLUTION_CORE_KEYS - set(raw)
    if unknown:
        raise ValueError(f"Unknown evolution_core keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing evolution_core keys: {sorted(missing)}")

    integer_values = {
        name: _strict_integer(raw[name], name, minimum=minimum)
        for name, minimum in {
            "max_candidates_per_group": 1,
            "random_seed": None,
            "min_folds": 1,
            "min_filled_trades_per_fold": 1,
        }.items()
    }
    number_values = {
        name: _strict_number(raw[name], name)
        for name in (
            "min_positive_fold_ratio", "min_mean_return_improvement",
            "max_drawdown_floor", "max_drawdown_worsening",
            "max_turnover_ratio", "max_pnl_concentration",
        )
    }
    for name in (
        "min_positive_fold_ratio", "min_mean_return_improvement",
        "max_drawdown_worsening", "max_pnl_concentration"
    ):
        if not 0.0 <= number_values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not -1.0 <= number_values["max_drawdown_floor"] <= 0.0:
        raise ValueError("max_drawdown_floor must be between -1 and 0")
    if number_values["max_turnover_ratio"] <= 0.0:
        raise ValueError("max_turnover_ratio must be positive")
    return EvolutionCoreConfig(**integer_values, **number_values)


def _parse_selection(value: object) -> SelectionRules:
    if not isinstance(value, Mapping):
        raise ValueError("selection must be a mapping")
    raw = dict(value)
    unknown = set(raw) - SELECTION_KEYS
    missing = SELECTION_KEYS - set(raw)
    if unknown:
        raise ValueError(f"Unknown selection keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing selection keys: {sorted(missing)}")

    integer_values = {
        name: _strict_integer(raw[name], name, minimum=1)
        for name in (
            "min_validation_days",
            "min_test_days",
            "rolling_window_days",
        )
    }
    number_values = {
        name: _strict_number(raw[name], name)
        for name in (
            "max_drawdown_floor",
            "min_annualized_return_delta",
            "min_sharpe_delta",
            "max_turnover_ratio",
            "max_negative_window_rate",
        )
    }
    if not -1.0 <= number_values["max_drawdown_floor"] <= 0.0:
        raise ValueError("max_drawdown_floor must be between -1 and 0")
    if number_values["min_annualized_return_delta"] < 0.0:
        raise ValueError("min_annualized_return_delta must be non-negative")
    if number_values["max_turnover_ratio"] < 1.0:
        raise ValueError("max_turnover_ratio must be >= 1")
    if not 0.0 <= number_values["max_negative_window_rate"] <= 1.0:
        raise ValueError("max_negative_window_rate must be between 0 and 1")
    return SelectionRules(**integer_values, **number_values)


def parse_evolution_config(raw: Mapping[str, object]) -> EvolutionConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("Evolution config must be a mapping")
    raw_config = dict(raw)
    unknown_top_level = set(raw_config) - TOP_LEVEL_KEYS
    if unknown_top_level:
        raise ValueError(f"Unknown top-level keys: {sorted(unknown_top_level)}")
    if raw_config.get("strategy") != "strong_pullback_satellite":
        raise ValueError("strategy must be strong_pullback_satellite")
    evolution_core = _parse_evolution_core(raw_config.get("evolution_core"))
    period_value = raw_config.get("periods")
    if not isinstance(period_value, Mapping):
        raise ValueError("periods must be a mapping")
    period_raw = dict(period_value)
    unknown_periods = set(period_raw) - PERIOD_KEYS
    if unknown_periods:
        raise ValueError(f"Unknown periods keys: {sorted(unknown_periods)}")
    periods = EvolutionPeriods(
        research_start=_timestamp(period_raw.get("research_start"), "research_start"),
        train_end=_timestamp(period_raw.get("train_end"), "train_end"),
        validation_start=_timestamp(period_raw.get("validation_start"), "validation_start"),
        validation_end=_timestamp(period_raw.get("validation_end"), "validation_end"),
        core_test_start=_timestamp(period_raw.get("core_test_start"), "core_test_start"),
        core_test_end=_timestamp(period_raw.get("core_test_end"), "core_test_end"),
        test_start=_timestamp(period_raw.get("test_start"), "test_start"),
        test_end=_timestamp(period_raw.get("test_end"), "test_end", allow_none=True),
    )
    if not (
        periods.research_start <= periods.train_end
        and periods.train_end < periods.validation_start
        and periods.validation_start <= periods.validation_end
        and periods.validation_end < periods.core_test_start
        and periods.core_test_start <= periods.core_test_end
        and periods.core_test_end < periods.test_start
        and (periods.test_end is None or periods.test_start <= periods.test_end)
    ):
        raise ValueError(
            "Periods must satisfy train_end < validation_start <= validation_end "
            "< core_test_start <= core_test_end < test_start"
        )

    baseline_value = raw_config.get("baseline")
    if not isinstance(baseline_value, Mapping):
        raise ValueError("baseline must be a mapping")
    baseline_raw = dict(baseline_value)
    unknown_baseline = set(baseline_raw) - set(DEFAULT_STRATEGY_PARAMS)
    if unknown_baseline:
        raise ValueError(f"Unknown baseline keys: {sorted(unknown_baseline)}")
    baseline = {**DEFAULT_STRATEGY_PARAMS, **baseline_raw}
    _validate_params(baseline)
    group_ids: set[str] = set()
    candidate_ids: set[str] = set()
    groups: list[SearchGroup] = []
    for group_raw in list(raw_config.get("search_groups") or []):
        if not isinstance(group_raw, Mapping):
            raise ValueError("search_groups entries must be mappings")
        unknown_group = set(group_raw) - SEARCH_GROUP_KEYS
        if unknown_group:
            raise ValueError(f"Unknown search group keys: {sorted(unknown_group)}")
        group_id = _identifier(group_raw.get("id"), "Group id")
        normalized_group_id = group_id.casefold()
        if normalized_group_id in group_ids:
            raise ValueError(f"Duplicate group id: {group_id!r}")
        group_ids.add(normalized_group_id)
        candidates: list[SearchCandidate] = []
        for candidate_raw in list(group_raw.get("candidates") or []):
            if not isinstance(candidate_raw, Mapping):
                raise ValueError("candidates entries must be mappings")
            unknown_candidate = set(candidate_raw) - SEARCH_CANDIDATE_KEYS
            if unknown_candidate:
                raise ValueError(
                    f"Unknown search candidate keys: {sorted(unknown_candidate)}"
                )
            candidate_id = _identifier(candidate_raw.get("id"), "Candidate id")
            normalized_candidate_id = candidate_id.casefold()
            if normalized_candidate_id in candidate_ids:
                raise ValueError(f"Duplicate candidate id: {candidate_id!r}")
            candidate_ids.add(normalized_candidate_id)
            overrides_raw = candidate_raw.get("overrides") or {}
            if not isinstance(overrides_raw, Mapping):
                raise ValueError("candidate overrides must be a mapping")
            overrides = dict(overrides_raw)
            unknown = set(overrides) - ALLOWED_OVERRIDE_PARAMS
            if unknown:
                raise ValueError(f"Unknown strategy parameters: {sorted(unknown)}")
            _validate_params({**baseline, **overrides})
            candidates.append(SearchCandidate(candidate_id, overrides))
        if not candidates:
            raise ValueError(f"Search group {group_id!r} has no candidates")
        groups.append(SearchGroup(group_id, str(group_raw.get("hypothesis_cn", "")).strip(), tuple(candidates)))

    selection = _parse_selection(raw_config.get("selection"))
    return EvolutionConfig(
        strategy="strong_pullback_satellite",
        evolution_core=evolution_core,
        periods=periods,
        baseline=baseline,
        search_groups=tuple(groups),
        selection=selection,
    )


def load_evolution_config(path: Path) -> EvolutionConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Evolution config must be a YAML mapping")
    return parse_evolution_config(raw)


def build_group_candidates(
    incumbent: Mapping[str, object],
    group: SearchGroup,
    *,
    max_candidates: int | None = None,
    seed: int | None = None,
) -> tuple[tuple[str, dict[str, object]], ...]:
    if max_candidates is None and seed is None:
        return tuple(
            (candidate.candidate_id, {**deepcopy(dict(incumbent)), **deepcopy(candidate.overrides)})
            for candidate in group.candidates
        )
    if max_candidates is None or seed is None:
        raise ValueError("max_candidates and seed must be provided together")
    generated = generate_parameter_candidates(
        incumbent,
        [candidate.overrides for candidate in group.candidates],
        max_candidates=max_candidates,
        seed=seed,
    )
    ids_by_fingerprint = {
        fingerprint_payload({**deepcopy(dict(incumbent)), **deepcopy(candidate.overrides)}):
        candidate.candidate_id
        for candidate in group.candidates
    }
    return tuple(
        (ids_by_fingerprint[fingerprint_payload(params)], params)
        for params in generated
    )


def _positive_window_size(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_trading_folds(
    trading_dates: object,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
) -> tuple[TradingFold, ...]:
    train_days = _positive_window_size(train_days, "train_days")
    validation_days = _positive_window_size(validation_days, "validation_days")
    test_days = _positive_window_size(test_days, "test_days")
    step_days = _positive_window_size(step_days, "step_days")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("trading_dates must be valid datetimes") from exc
    if dates.hasnans:
        raise ValueError("trading_dates must not contain NaT")
    if not dates.is_unique:
        raise ValueError("trading_dates must be unique")
    dates = dates.sort_values()

    folds: list[TradingFold] = []
    offset = 0
    while True:
        train_end = train_days + offset
        validation_end = train_end + validation_days
        test_end = validation_end + test_days
        if test_end > len(dates):
            break
        folds.append(
            TradingFold(
                fold_id=f"fold_{len(folds):03d}",
                train_dates=tuple(dates[:train_end]),
                validation_dates=tuple(dates[train_end:validation_end]),
                test_dates=tuple(dates[validation_end:test_end]),
            )
        )
        offset += step_days
    return tuple(folds)


def _build_period_segments(
    trading_dates: object,
    period_start: object,
    period_end: object,
    min_folds: int,
    *,
    prefix: str,
) -> tuple[CoreTestSegment, ...]:
    min_folds = _positive_window_size(min_folds, "min_folds")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize()
        start = pd.Timestamp(period_start).normalize()
        end = pd.Timestamp(period_end).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation-period dates must be valid datetimes") from exc
    if dates.hasnans:
        raise ValueError("trading_dates must not contain NaT")
    if not dates.is_unique:
        raise ValueError("trading_dates must be unique")
    if start > end:
        raise ValueError("period_start must not be after period_end")
    period_dates = dates.sort_values()
    period_dates = period_dates[(period_dates >= start) & (period_dates <= end)]
    if len(period_dates) < min_folds:
        raise ValueError("Evaluation period has fewer trading dates than min_folds")

    base_size, remainder = divmod(len(period_dates), min_folds)
    segments: list[CoreTestSegment] = []
    offset = 0
    for index in range(min_folds):
        size = base_size + (1 if index < remainder else 0)
        segment_dates = tuple(period_dates[offset:offset + size])
        segments.append(
            CoreTestSegment(
                fold_id=f"{prefix}_{index:03d}",
                test_dates=segment_dates,
            )
        )
        offset += size
    return tuple(segments)


def build_selection_segments(
    trading_dates: object,
    validation_start: object,
    validation_end: object,
    min_folds: int,
) -> tuple[CoreTestSegment, ...]:
    return _build_period_segments(
        trading_dates,
        validation_start,
        validation_end,
        min_folds,
        prefix="selection",
    )


def build_core_test_segments(
    trading_dates: object,
    core_test_start: object,
    core_test_end: object,
    min_folds: int,
) -> tuple[CoreTestSegment, ...]:
    return _build_period_segments(
        trading_dates,
        core_test_start,
        core_test_end,
        min_folds,
        prefix="core",
    )


def _slice_fold_rows(
    frame: pd.DataFrame,
    fold: TradingFold | CoreTestSegment | None,
    date_column: str,
) -> pd.DataFrame:
    if date_column not in frame:
        raise ValueError(f"Fold evidence requires {date_column}")
    if fold is None:
        return frame.copy()
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Fold evidence contains invalid {date_column}")
    return frame.loc[dates.isin(fold.test_dates)].copy()


def _positive_symbol_contributions(trades: pd.DataFrame) -> dict[str, float]:
    if "symbol_contributions_json" not in trades:
        raise ValueError("Fold evidence requires symbol_contributions_json")
    totals: dict[str, float] = {}
    for value in trades["symbol_contributions_json"]:
        try:
            contributions = json.loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed symbol_contributions_json") from exc
        if not isinstance(contributions, Mapping):
            raise ValueError("symbol_contributions_json must decode to a mapping")
        for symbol, contribution in contributions.items():
            if (
                isinstance(contribution, bool)
                or not isinstance(contribution, (int, float))
                or not math.isfinite(float(contribution))
            ):
                raise ValueError(
                    "symbol_contributions_json values must be finite numbers"
                )
            key = str(symbol).zfill(6)
            totals[key] = totals.get(key, 0.0) + float(contribution)
    return {symbol: value for symbol, value in totals.items() if value > 0.0}


def calculate_fold_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    fold: TradingFold | CoreTestSegment | None = None,
    *,
    fold_id: str | None = None,
) -> FoldMetrics:
    equity_slice = _slice_fold_rows(equity, fold, "date")
    if equity_slice.empty:
        raise ValueError("Fold contains no equity rows")
    trades_slice = _slice_fold_rows(trades, fold, "realize_date")
    if "turnover" not in trades_slice:
        raise ValueError("Fold evidence requires turnover")
    turnover = pd.to_numeric(trades_slice["turnover"], errors="coerce")
    if turnover.isna().any() or not all(math.isfinite(float(value)) for value in turnover):
        raise ValueError("Fold evidence turnover must contain finite numbers")
    filled_trades = int(turnover.abs().gt(1e-12).sum())
    positive_contributions = _positive_symbol_contributions(trades_slice)
    metrics = calculate_segment_metrics(
        equity_slice,
        equity_slice["date"].min(),
        equity_slice["date"].max(),
        rolling_window_days=max(1, len(equity_slice)),
    )
    total_positive = sum(positive_contributions.values())
    concentration = (
        max(positive_contributions.values()) / total_positive
        if total_positive > 0.0
        else math.inf
    )
    resolved_fold_id = fold.fold_id if fold is not None else fold_id
    if not isinstance(resolved_fold_id, str) or not resolved_fold_id:
        raise ValueError("fold_id must be a non-empty string")
    return FoldMetrics(
        fold_id=resolved_fold_id,
        total_return=metrics["total_return"],
        max_drawdown=metrics["max_drawdown"],
        sharpe=metrics["sharpe_like"],
        filled_trades=filled_trades,
        average_turnover=metrics["avg_turnover"],
        pnl_concentration=concentration,
    )


def run_strong_pullback_folds(
    panel: object,
    folds: tuple[TradingFold | CoreTestSegment, ...],
    executor: Callable[[object, Mapping[str, object]], object],
    params: Mapping[str, object],
    *,
    slicer: Callable[[object, pd.Timestamp], object] | None = None,
) -> tuple[tuple[TradingFold | CoreTestSegment, object], ...]:
    if slicer is None:
        if not isinstance(panel, pd.DataFrame) or "date" not in panel:
            raise ValueError("panel must contain a date column")
        dates = pd.to_datetime(panel["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("panel contains invalid dates")

        def slice_to_end(value: object, test_end: pd.Timestamp) -> object:
            assert isinstance(value, pd.DataFrame)
            return value.loc[dates.le(test_end)].copy()
    else:
        slice_to_end = slicer
    return tuple(
        (
            fold,
            executor(slice_to_end(panel, fold.test_end), params),
        )
        for fold in folds
    )


def calculate_segment_metrics(
    equity: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    rolling_window_days: int,
) -> dict[str, float]:
    rolling_window_days = _positive_window_size(
        rolling_window_days, "rolling_window_days"
    )
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("date")
    if frame.empty:
        raise ValueError(f"No equity rows between {start} and {end}")
    returns = pd.to_numeric(frame["gross_return"], errors="coerce") - pd.to_numeric(
        frame["cost"], errors="coerce"
    )
    if returns.isna().any() or (returns <= -1.0).any():
        raise ValueError("Segment contains invalid net returns")
    nav = (1.0 + returns).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annualized = float((1.0 + total_return) ** (252.0 / len(nav)) - 1.0)
    rolling = nav / nav.shift(rolling_window_days) - 1.0
    completed = rolling.dropna()
    negative_rate = float(completed.lt(0.0).mean()) if not completed.empty else 0.0
    worst_rolling = float(completed.min()) if not completed.empty else total_return
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
        "avg_turnover": float(pd.to_numeric(frame["turnover"], errors="coerce").mean()),
        "avg_gross_exposure": float(pd.to_numeric(frame["gross_exposure"], errors="coerce").mean()),
        "trade_days": int(len(frame)),
        "rolling_window_count": int(len(completed)),
        "negative_window_rate": negative_rate,
        "worst_rolling_return": worst_rolling,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Segment metrics contain non-finite values")
    return metrics


def _turnover_ratio(candidate_turnover: float, incumbent_turnover: float) -> float:
    if abs(incumbent_turnover) <= 1e-12:
        return 1.0 if abs(candidate_turnover) <= 1e-12 else math.inf
    return candidate_turnover / incumbent_turnover


def _robust_score(metrics: Mapping[str, float], turnover_ratio: float) -> float:
    return float(
        metrics["annualized_return"]
        + 0.15 * metrics["sharpe_like"]
        + 0.50 * metrics["max_drawdown"]
        - 0.10 * metrics["negative_window_rate"]
        - 0.05 * max(turnover_ratio - 1.0, 0.0)
    )


def evaluate_promotion(
    candidate_metrics: Mapping[str, float],
    incumbent_metrics: Mapping[str, float],
    rules: SelectionRules,
) -> PromotionDecision:
    required = (
        "annualized_return", "max_drawdown", "sharpe_like", "avg_turnover",
        "trade_days", "negative_window_rate",
    )
    reasons: list[str] = []
    if not all(math.isfinite(float(candidate_metrics[key])) for key in required):
        reasons.append("关键指标存在非有限值")
    turnover_ratio = _turnover_ratio(
        float(candidate_metrics["avg_turnover"]), float(incumbent_metrics["avg_turnover"])
    )
    if int(candidate_metrics["trade_days"]) < rules.min_validation_days:
        reasons.append("验证期有效交易日不足")
    rolling_window_count = candidate_metrics.get("rolling_window_count", 0)
    try:
        has_rolling_windows = math.isfinite(float(rolling_window_count)) and float(rolling_window_count) > 0
    except (TypeError, ValueError):
        has_rolling_windows = False
    if not has_rolling_windows:
        reasons.append("完成的滚动窗口不足")
    if float(candidate_metrics["max_drawdown"]) < rules.max_drawdown_floor:
        reasons.append("最大回撤超过限制")
    if float(candidate_metrics["annualized_return"]) - float(incumbent_metrics["annualized_return"]) < rules.min_annualized_return_delta - 1e-12:
        reasons.append("年化收益提升不足")
    if float(candidate_metrics["sharpe_like"]) - float(incumbent_metrics["sharpe_like"]) < rules.min_sharpe_delta - 1e-12:
        reasons.append("Sharpe 恶化超过限制")
    if turnover_ratio > rules.max_turnover_ratio + 1e-12:
        reasons.append("换手率放大超过限制")
    if float(candidate_metrics["negative_window_rate"]) > rules.max_negative_window_rate + 1e-12:
        reasons.append("负滚动窗口占比超过限制")
    return PromotionDecision(
        eligible=not reasons,
        reasons=tuple(reasons),
        turnover_ratio=float(turnover_ratio),
        robust_score=_robust_score(candidate_metrics, turnover_ratio),
    )


def choose_group_winner(
    incumbent_id: str,
    incumbent_metrics: Mapping[str, float],
    candidates: tuple[tuple[str, Mapping[str, float]], ...],
    rules: SelectionRules,
) -> tuple[str, tuple[CandidateDecision, ...]]:
    decisions = tuple(
        CandidateDecision(candidate_id, evaluate_promotion(metrics, incumbent_metrics, rules), dict(metrics))
        for candidate_id, metrics in candidates
    )
    eligible = [decision for decision in decisions if decision.promotion.eligible]
    if not eligible:
        return incumbent_id, decisions
    eligible.sort(
        key=lambda item: (
            item.metrics["annualized_return"], item.promotion.robust_score,
            item.metrics["sharpe_like"], item.metrics["max_drawdown"],
            -item.metrics["avg_turnover"],
        ),
        reverse=True,
    )
    return eligible[0].candidate_id, decisions


def assess_test_result(
    baseline_metrics: Mapping[str, float],
    champion_metrics: Mapping[str, float],
    rules: SelectionRules,
) -> tuple[str, str]:
    required = ("total_return", "max_drawdown", "sharpe_like", "trade_days")
    try:
        values = [float(metrics[key]) for metrics in (baseline_metrics, champion_metrics) for key in required]
    except (KeyError, TypeError, ValueError):
        return "test_warning", "测试期关键指标缺失或无效，暂不判断晋级"
    if not all(math.isfinite(value) for value in values):
        return "test_warning", "测试期关键指标缺失或无效，暂不判断晋级"
    if min(int(baseline_metrics["trade_days"]), int(champion_metrics["trade_days"])) < rules.min_test_days:
        return "test_warning", "测试期有效交易日不足，暂不判断晋级"
    failures: list[str] = []
    if float(champion_metrics["total_return"]) < float(baseline_metrics["total_return"]):
        failures.append("测试期收益低于基准")
    if float(champion_metrics["max_drawdown"]) < rules.max_drawdown_floor:
        failures.append("测试期最大回撤超过限制")
    if float(champion_metrics["sharpe_like"]) - float(baseline_metrics["sharpe_like"]) < -0.10:
        failures.append("测试期 Sharpe 恶化超过 0.10")
    if failures:
        return "rollback_recommended", "；".join(failures)
    return "ready_for_manual_review", "测试期收益、回撤和 Sharpe 门槛通过，等待人工确认"
