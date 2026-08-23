"""Guarded self-evolution primitives for the observation-factor strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from execution_rules import (
    LIMIT_DOWN_PRICE_COLUMNS,
    LIMIT_RATE_COLUMNS,
    LIMIT_UP_PRICE_COLUMNS,
    apply_open_constraints_with_diagnostics,
)
from legacy_observation_factors import compute_observation_factors, validate_panel
from run_backtest import max_drawdown, sharpe_like
from strategy_evolution_core import FoldMetrics, PromotionPolicy


STRATEGY_NAME = "multifactor_observation"
FACTOR_NAMES = (
    "momentum_20",
    "momentum_60",
    "breakout_distance_20",
    "trend_acceleration",
    "liquidity_20",
    "liquidity_stability_20",
)
WEIGHT_KEYS = tuple(f"{name}_weight" for name in FACTOR_NAMES)
FACTOR_RANK_COLUMNS = {
    name: f"_evolution_rank_{name}" for name in FACTOR_NAMES
}

LOCKED_EXECUTION = {
    "signal_lag_sessions": 1,
    "execution_model": "next_open",
    "historical_universe_only": True,
    "allow_short": False,
    "max_leverage": 1.0,
    "block_limit_up_buys": True,
    "block_limit_down_sells": True,
}

DEFAULT_PARAMETERS: dict[str, object] = {
    "momentum_20_weight": 0.25,
    "momentum_60_weight": 0.20,
    "breakout_distance_20_weight": 0.20,
    "trend_acceleration_weight": 0.15,
    "liquidity_20_weight": 0.10,
    "liquidity_stability_20_weight": 0.10,
    "top_n": 8,
    "rebalance_frequency": 5,
    "gross_exposure": 1.0,
    "max_position_weight": 0.15,
    "min_score": 0.60,
    "min_momentum_20": -1.0,
    "min_momentum_60": -1.0,
    "min_breakout_distance_20": -1.0,
    "max_momentum_20": 1_000_000.0,
    "max_momentum_60": 1_000_000.0,
    "max_breakout_distance_20": 1_000_000.0,
    "min_median_amount_20": 30_000_000.0,
    "max_daily_amount_participation": 0.05,
    "market_ma_window": 120,
    "market_below_ma_exposure": 1.0,
    "market_risk_off_drawdown_20d": -1.0,
    "market_crash_exposure": 1.0,
    "breadth_ma_window": 20,
    "breadth_risk_off_gap": -1.0,
    "breadth_risk_off_exposure": 1.0,
    "breadth_crash_gap": -1.0,
    "breadth_crash_exposure": 1.0,
    "portfolio_stop_drawdown": -1.0,
    "portfolio_stop_cooldown_sessions": 10,
    "initial_capital": 1_000_000.0,
    "commission_bps": 3.0,
    "impact_bps": 7.0,
    "max_buy_open_gap": 0.03,
    "limit_buffer": 0.995,
}

TUNABLE_PARAMETERS = frozenset(
    {
        *WEIGHT_KEYS,
        "top_n",
        "rebalance_frequency",
        "gross_exposure",
        "max_position_weight",
        "min_score",
        "min_momentum_20",
        "min_momentum_60",
        "min_breakout_distance_20",
        "max_momentum_20",
        "max_momentum_60",
        "max_breakout_distance_20",
        "min_median_amount_20",
        "max_daily_amount_participation",
        "market_ma_window",
        "market_below_ma_exposure",
        "market_risk_off_drawdown_20d",
        "market_crash_exposure",
        "breadth_ma_window",
        "breadth_risk_off_gap",
        "breadth_risk_off_exposure",
        "breadth_crash_gap",
        "breadth_crash_exposure",
        "portfolio_stop_drawdown",
        "portfolio_stop_cooldown_sessions",
    }
)


@dataclass(frozen=True)
class EvolutionPeriods:
    research_start: pd.Timestamp
    selection_start: pd.Timestamp
    selection_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp | None
    fold_sessions: int
    step_sessions: int
    warmup_sessions: int


@dataclass(frozen=True)
class SearchCandidate:
    candidate_id: str
    overrides: dict[str, object]


@dataclass(frozen=True)
class SearchGroup:
    group_id: str
    hypothesis: str
    candidates: tuple[SearchCandidate, ...]


@dataclass(frozen=True)
class HoldoutPolicy:
    min_sessions: int
    min_return_delta: float
    min_excess_return: float
    max_drawdown_floor: float
    min_double_cost_return: float
    min_double_cost_return_delta: float
    min_positive_excess_fold_ratio: float


@dataclass(frozen=True)
class EvolutionConfig:
    strategy: str
    target_mode: str
    annual_return_stretch: float
    benchmark: str
    locked_execution: dict[str, object]
    baseline: dict[str, object]
    periods: EvolutionPeriods
    promotion: PromotionPolicy
    holdout: HoldoutPolicy
    max_candidates_per_group: int
    random_seed: int
    search_groups: tuple[SearchGroup, ...]


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class ObservationBacktest:
    equity: pd.Series
    returns: pd.Series
    turnover: pd.Series
    filled_trades: pd.Series
    market_exposure: pd.Series
    symbol_pnl: pd.DataFrame
    trade_log: pd.DataFrame
    execution_counts: dict[str, int]
    position_weights: pd.DataFrame | None = None


@dataclass(frozen=True)
class ParameterEvaluation:
    folds: tuple[FoldMetrics, ...]
    fold_rows: tuple[dict[str, object], ...]
    backtest: ObservationBacktest


def _as_timestamp(value: object, name: str, allow_none: bool = False) -> pd.Timestamp | None:
    if value is None and allow_none:
        return None
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid date for {name}: {value!r}") from exc
    if pd.isna(result):
        raise ValueError(f"invalid date for {name}: {value!r}")
    return result


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def validate_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    unknown = set(parameters) - set(DEFAULT_PARAMETERS)
    missing = set(DEFAULT_PARAMETERS) - set(parameters)
    if unknown:
        raise ValueError(f"unknown strategy parameters: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing strategy parameters: {sorted(missing)}")

    out = dict(parameters)
    for name in WEIGHT_KEYS:
        value = _finite_number(out[name], name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        out[name] = value
    if sum(float(out[name]) for name in WEIGHT_KEYS) <= 0.0:
        raise ValueError("at least one observation factor weight must be positive")

    for name in (
        "top_n",
        "rebalance_frequency",
        "market_ma_window",
        "breadth_ma_window",
        "portfolio_stop_cooldown_sessions",
    ):
        if type(out[name]) is not int or int(out[name]) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if int(out["market_ma_window"]) < 20:
        raise ValueError("market_ma_window must be at least 20 sessions")
    if int(out["breadth_ma_window"]) < 5:
        raise ValueError("breadth_ma_window must be at least 5 sessions")
    for name in (
        "gross_exposure",
        "max_position_weight",
        "max_daily_amount_participation",
    ):
        value = _finite_number(out[name], name)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
        out[name] = value
    min_score = _finite_number(out["min_score"], "min_score")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    out["min_score"] = min_score

    for name in (
        "min_momentum_20",
        "min_momentum_60",
        "min_breakout_distance_20",
    ):
        value = _finite_number(out[name], name)
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [-1, 1]")
        out[name] = value

    ceiling_pairs = (
        ("min_momentum_20", "max_momentum_20"),
        ("min_momentum_60", "max_momentum_60"),
        ("min_breakout_distance_20", "max_breakout_distance_20"),
    )
    for minimum_name, maximum_name in ceiling_pairs:
        value = _finite_number(out[maximum_name], maximum_name)
        if not 0.0 <= value <= 1_000_000.0:
            raise ValueError(f"{maximum_name} must be in [0, 1000000]")
        if value < float(out[minimum_name]):
            raise ValueError(f"{maximum_name} cannot be below {minimum_name}")
        out[maximum_name] = value

    for name in (
        "market_below_ma_exposure",
        "market_crash_exposure",
        "breadth_risk_off_exposure",
        "breadth_crash_exposure",
    ):
        value = _finite_number(out[name], name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        out[name] = value
    drawdown_trigger = _finite_number(
        out["market_risk_off_drawdown_20d"],
        "market_risk_off_drawdown_20d",
    )
    if not -1.0 <= drawdown_trigger <= 0.0:
        raise ValueError("market_risk_off_drawdown_20d must be in [-1, 0]")
    out["market_risk_off_drawdown_20d"] = drawdown_trigger
    for name in ("breadth_risk_off_gap", "breadth_crash_gap"):
        value = _finite_number(out[name], name)
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [-1, 1]")
        out[name] = value
    if float(out["breadth_crash_gap"]) > float(out["breadth_risk_off_gap"]):
        raise ValueError("breadth_crash_gap cannot exceed breadth_risk_off_gap")
    portfolio_stop = _finite_number(
        out["portfolio_stop_drawdown"], "portfolio_stop_drawdown"
    )
    if not -1.0 <= portfolio_stop < 0.0:
        raise ValueError("portfolio_stop_drawdown must be in [-1, 0)")
    out["portfolio_stop_drawdown"] = portfolio_stop

    for name in (
        "min_median_amount_20",
        "initial_capital",
        "commission_bps",
        "impact_bps",
        "max_buy_open_gap",
        "limit_buffer",
    ):
        value = _finite_number(out[name], name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        out[name] = value
    if float(out["initial_capital"]) <= 0.0:
        raise ValueError("initial_capital must be positive")
    if not 0.0 < float(out["limit_buffer"]) <= 1.0:
        raise ValueError("limit_buffer must be in (0, 1]")
    return out


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def parse_evolution_config(raw: Mapping[str, object]) -> EvolutionConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("evolution config must be a mapping")
    allowed_top = {
        "strategy",
        "research",
        "locked_execution",
        "evolution_core",
        "periods",
        "baseline",
        "search_groups",
        "holdout",
    }
    unknown_top = set(raw) - allowed_top
    if unknown_top:
        raise ValueError(f"unknown evolution config keys: {sorted(unknown_top)}")
    if raw.get("strategy") != STRATEGY_NAME:
        raise ValueError(f"strategy must be {STRATEGY_NAME!r}")

    research = _mapping(raw.get("research"), "research")
    if research.get("target_mode") != "stretch_research":
        raise ValueError("research.target_mode must be 'stretch_research'")
    annual_return_stretch = _finite_number(
        research.get("annual_return_stretch"), "research.annual_return_stretch"
    )
    if annual_return_stretch != 10.0:
        raise ValueError("research.annual_return_stretch must remain 10.0")
    benchmark = research.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("research.benchmark must be a non-empty string")

    locked = _mapping(raw.get("locked_execution"), "locked_execution")
    if locked != LOCKED_EXECUTION:
        raise ValueError(
            "locked_execution must preserve lag-1 next-open, long-only, "
            "historical-universe and limit constraints"
        )

    baseline = dict(DEFAULT_PARAMETERS)
    baseline.update(_mapping(raw.get("baseline"), "baseline"))
    baseline = validate_parameters(baseline)

    period_raw = _mapping(raw.get("periods"), "periods")
    periods = EvolutionPeriods(
        research_start=_as_timestamp(period_raw.get("research_start"), "research_start"),
        selection_start=_as_timestamp(period_raw.get("selection_start"), "selection_start"),
        selection_end=_as_timestamp(period_raw.get("selection_end"), "selection_end"),
        holdout_start=_as_timestamp(period_raw.get("holdout_start"), "holdout_start"),
        holdout_end=_as_timestamp(period_raw.get("holdout_end"), "holdout_end", allow_none=True),
        fold_sessions=int(period_raw.get("fold_sessions", 126)),
        step_sessions=int(period_raw.get("step_sessions", 63)),
        warmup_sessions=int(period_raw.get("warmup_sessions", 80)),
    )
    if not (
        periods.research_start < periods.selection_start <= periods.selection_end
        < periods.holdout_start
    ):
        raise ValueError("periods must be strictly ordered without selection/holdout overlap")
    if periods.holdout_end is not None and periods.holdout_end < periods.holdout_start:
        raise ValueError("holdout_end cannot precede holdout_start")
    if min(periods.fold_sessions, periods.step_sessions) < 1:
        raise ValueError("fold_sessions and step_sessions must be positive")
    if periods.warmup_sessions < 61:
        raise ValueError("warmup_sessions must be at least 61")

    core = _mapping(raw.get("evolution_core"), "evolution_core")
    promotion = PromotionPolicy(
        min_folds=int(core.get("min_folds", 3)),
        min_filled_trades_per_fold=int(core.get("min_filled_trades_per_fold", 5)),
        min_positive_fold_ratio=float(core.get("min_positive_fold_ratio", 2.0 / 3.0)),
        min_mean_return_improvement=float(core.get("min_mean_return_improvement", 0.01)),
        max_drawdown_floor=float(core.get("max_drawdown_floor", -0.40)),
        max_drawdown_worsening=float(core.get("max_drawdown_worsening", 0.05)),
        max_turnover_ratio=float(core.get("max_turnover_ratio", 1.50)),
        max_pnl_concentration=float(core.get("max_pnl_concentration", 0.50)),
    )
    max_candidates = int(core.get("max_candidates_per_group", 8))
    random_seed = int(core.get("random_seed", 20260714))
    if max_candidates < 1:
        raise ValueError("max_candidates_per_group must be positive")

    holdout_raw = _mapping(raw.get("holdout"), "holdout")
    holdout = HoldoutPolicy(
        min_sessions=int(holdout_raw.get("min_sessions", 60)),
        min_return_delta=float(holdout_raw.get("min_return_delta", 0.0)),
        min_excess_return=float(holdout_raw.get("min_excess_return", 0.0)),
        max_drawdown_floor=float(holdout_raw.get("max_drawdown_floor", -0.40)),
        min_double_cost_return=float(holdout_raw.get("min_double_cost_return", 0.0)),
        min_double_cost_return_delta=float(
            holdout_raw.get("min_double_cost_return_delta", 0.0)
        ),
        min_positive_excess_fold_ratio=float(
            holdout_raw.get("min_positive_excess_fold_ratio", 2.0 / 3.0)
        ),
    )
    if holdout.min_sessions < 1:
        raise ValueError("holdout.min_sessions must be positive")
    if not 0.0 <= holdout.min_positive_excess_fold_ratio <= 1.0:
        raise ValueError("holdout.min_positive_excess_fold_ratio must be in [0, 1]")
    if not -1.0 <= holdout.max_drawdown_floor <= 0.0:
        raise ValueError("holdout.max_drawdown_floor must be in [-1, 0]")

    groups: list[SearchGroup] = []
    seen_ids: set[str] = set()
    for group_value in raw.get("search_groups", ()):
        group_raw = _mapping(group_value, "search group")
        group_id = group_raw.get("id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("search group id must be a non-empty string")
        candidates: list[SearchCandidate] = []
        for candidate_value in group_raw.get("candidates", ()):
            candidate_raw = _mapping(candidate_value, "search candidate")
            candidate_id = candidate_raw.get("id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError("candidate id must be a non-empty string")
            if candidate_id in seen_ids:
                raise ValueError(f"duplicate candidate id: {candidate_id}")
            seen_ids.add(candidate_id)
            overrides = _mapping(candidate_raw.get("overrides"), "candidate overrides")
            forbidden = set(overrides) - TUNABLE_PARAMETERS
            if forbidden:
                raise ValueError(
                    f"candidate {candidate_id} changes locked parameters: {sorted(forbidden)}"
                )
            candidate_parameters = dict(baseline)
            candidate_parameters.update(overrides)
            validate_parameters(candidate_parameters)
            candidates.append(SearchCandidate(candidate_id, overrides))
        if not candidates:
            raise ValueError(f"search group {group_id} has no candidates")
        groups.append(
            SearchGroup(
                group_id=group_id,
                hypothesis=str(group_raw.get("hypothesis", "")),
                candidates=tuple(candidates),
            )
        )
    if not groups:
        raise ValueError("at least one search group is required")

    return EvolutionConfig(
        strategy=STRATEGY_NAME,
        target_mode="stretch_research",
        annual_return_stretch=annual_return_stretch,
        benchmark=benchmark.strip(),
        locked_execution=locked,
        baseline=baseline,
        periods=periods,
        promotion=promotion,
        holdout=holdout,
        max_candidates_per_group=max_candidates,
        random_seed=random_seed,
        search_groups=tuple(groups),
    )


def load_evolution_config(path: Path) -> EvolutionConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_evolution_config(raw)


def combine_panel_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one market-data frame is required")
    prepared: list[pd.DataFrame] = []
    for source_order, frame in enumerate(frames):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("market-data sources must be pandas DataFrames")
        if frame.duplicated(["date", "symbol"]).any():
            raise ValueError("duplicate date-symbol rows inside one market-data source")
        item = frame.copy()
        item["_source_order"] = source_order
        prepared.append(item)
    combined = pd.concat(prepared, ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined = combined.sort_values(
        ["date", "symbol", "_source_order"], kind="mergesort"
    ).drop_duplicates(["date", "symbol"], keep="last")
    combined = combined.drop(columns="_source_order")
    market_columns = ("open", "high", "low", "close", "volume", "amount")
    quarantined_count = 0
    suspended_quote_count = 0
    if set(market_columns).issubset(combined.columns):
        market_values = combined.loc[:, market_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        all_zero = market_values.notna().all(axis=1) & market_values.eq(0.0).all(axis=1)
        quarantined_count = int(all_zero.sum())
        combined = combined.loc[~all_zero].copy()
        market_values = market_values.loc[~all_zero]
        suspended_quote = (
            market_values["close"].gt(0.0)
            & market_values[["open", "high", "low", "volume", "amount"]]
            .notna()
            .all(axis=1)
            & market_values[["open", "high", "low", "volume", "amount"]]
            .eq(0.0)
            .all(axis=1)
        )
        suspended_quote_count = int(suspended_quote.sum())
        combined.loc[suspended_quote, ["open", "high", "low"]] = np.nan
    panel = validate_panel(combined)
    if "open" not in panel:
        raise ValueError("next-open evolution requires an open column")
    panel["open"] = pd.to_numeric(panel["open"], errors="coerce")
    if not panel["open"].dropna().gt(0.0).all():
        raise ValueError("open prices must be positive when present")
    if panel["open"].notna().sum() == 0:
        raise ValueError("next-open evolution requires usable open prices")
    panel.attrs["quarantined_all_zero_market_rows"] = quarantined_count
    panel.attrs["suspended_no_open_quote_rows"] = suspended_quote_count
    return panel


def build_factor_scores(
    factor_panel: pd.DataFrame,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    params = validate_parameters(parameters)
    missing = set(FACTOR_NAMES).difference(factor_panel.columns)
    if missing:
        raise ValueError(f"missing observation factors: {sorted(missing)}")
    retained_columns = [
        name
        for name in (
            "date",
            "symbol",
            "open",
            "close",
            "momentum_20",
            "momentum_60",
            "breakout_distance_20",
            "liquidity_20",
            "score_eligible",
            *LIMIT_RATE_COLUMNS,
            *LIMIT_UP_PRICE_COLUMNS,
            *LIMIT_DOWN_PRICE_COLUMNS,
        )
        if name in factor_panel
    ]
    out = factor_panel.loc[:, retained_columns].copy()
    numerator = pd.Series(0.0, index=factor_panel.index)
    weight_total = sum(float(params[key]) for key in WEIGHT_KEYS)
    complete = pd.Series(True, index=factor_panel.index)
    for factor, weight_key in zip(FACTOR_NAMES, WEIGHT_KEYS):
        rank_column = FACTOR_RANK_COLUMNS[factor]
        if rank_column in factor_panel:
            ranks = pd.to_numeric(factor_panel[rank_column], errors="coerce")
        else:
            values = pd.to_numeric(factor_panel[factor], errors="coerce")
            ranks = values.groupby(factor_panel["date"], sort=False).rank(
                method="average", pct=True
            )
        complete &= ranks.notna()
        numerator = numerator.add(ranks.fillna(0.0) * float(params[weight_key]))
    out["evolution_score"] = numerator.div(weight_total).where(complete)
    out["median_amount_20"] = np.expm1(
        pd.to_numeric(out["liquidity_20"], errors="coerce")
    )
    out["evolution_eligible"] = (
        out["evolution_score"].ge(float(params["min_score"]))
        & pd.to_numeric(out["momentum_20"], errors="coerce").ge(
            float(params["min_momentum_20"])
        )
        & pd.to_numeric(out["momentum_60"], errors="coerce").ge(
            float(params["min_momentum_60"])
        )
        & pd.to_numeric(out["breakout_distance_20"], errors="coerce").ge(
            float(params["min_breakout_distance_20"])
        )
        & pd.to_numeric(out["momentum_20"], errors="coerce").le(
            float(params["max_momentum_20"])
        )
        & pd.to_numeric(out["momentum_60"], errors="coerce").le(
            float(params["max_momentum_60"])
        )
        & pd.to_numeric(out["breakout_distance_20"], errors="coerce").le(
            float(params["max_breakout_distance_20"])
        )
        & out["median_amount_20"].ge(float(params["min_median_amount_20"]))
        & out.get("score_eligible", True)
    )
    return out


def _pivot(panel: pd.DataFrame, column: str, dates: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
    if column not in panel:
        return pd.DataFrame(np.nan, index=dates, columns=symbols)
    return (
        panel.pivot(index="date", columns="symbol", values=column)
        .reindex(index=dates, columns=symbols)
        .apply(pd.to_numeric, errors="coerce")
    )


def _target_weights(
    rows: pd.DataFrame,
    parameters: Mapping[str, object],
    equity: float,
    symbols: pd.Index,
    exposure_scale: float = 1.0,
) -> pd.Series:
    params = parameters
    if not math.isfinite(exposure_scale) or not 0.0 <= exposure_scale <= 1.0:
        raise ValueError("exposure_scale must be finite and in [0, 1]")
    effective_gross_exposure = float(params["gross_exposure"]) * exposure_scale
    if effective_gross_exposure <= 0.0:
        return pd.Series(0.0, index=symbols)
    eligible = rows.loc[rows["evolution_eligible"].fillna(False)].copy()
    if eligible.empty:
        return pd.Series(0.0, index=symbols)
    desired_weight = min(
        float(params["max_position_weight"]),
        effective_gross_exposure / int(params["top_n"]),
    )
    required_amount = (
        equity * desired_weight / float(params["max_daily_amount_participation"])
    )
    eligible = eligible.loc[eligible["median_amount_20"].ge(required_amount)]
    eligible = eligible.sort_values(
        ["evolution_score", "symbol"], ascending=[False, True], kind="mergesort"
    ).head(int(params["top_n"]))
    target = pd.Series(0.0, index=symbols)
    if eligible.empty:
        return target
    total = min(
        effective_gross_exposure,
        float(params["max_position_weight"]) * len(eligible),
    )
    target.loc[eligible["symbol"]] = total / len(eligible)
    return target


def _uses_market_regime(parameters: Mapping[str, object]) -> bool:
    return not (
        float(parameters["market_below_ma_exposure"]) == 1.0
        and float(parameters["market_crash_exposure"]) == 1.0
    )


def _uses_breadth_regime(parameters: Mapping[str, object]) -> bool:
    return not (
        float(parameters["breadth_risk_off_exposure"]) == 1.0
        and float(parameters["breadth_crash_exposure"]) == 1.0
    )


def build_cross_sectional_momentum_breadth(
    factor_panel: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> pd.Series:
    if "momentum_20" not in factor_panel:
        raise ValueError("momentum breadth requires momentum_20")
    index = pd.DatetimeIndex(sorted(pd.Timestamp(value) for value in dates)).unique()
    eligible = factor_panel.get(
        "score_eligible", pd.Series(True, index=factor_panel.index)
    ).fillna(False).astype(bool)
    working = factor_panel.loc[eligible, ["date", "momentum_20"]].copy()
    working["momentum_20"] = pd.to_numeric(
        working["momentum_20"], errors="coerce"
    )
    working = working.dropna(subset=["momentum_20"])
    breadth = working.groupby("date", sort=True)["momentum_20"].apply(
        lambda values: float(values.gt(0.0).mean())
    )
    breadth.index = pd.to_datetime(breadth.index)
    return breadth.reindex(index).rename("momentum_breadth_20")


def build_market_exposure(
    dates: Sequence[pd.Timestamp],
    benchmark: pd.Series | None,
    parameters: Mapping[str, object],
    breadth_20: pd.Series | None = None,
) -> pd.Series:
    """Build point-in-time market exposure without carrying stale quotes forward."""
    params = validate_parameters(parameters)
    index = pd.DatetimeIndex(sorted(pd.Timestamp(value) for value in dates)).unique()
    if index.empty:
        return pd.Series(dtype=float, index=index, name="market_exposure")

    exposure = pd.Series(1.0, index=index, name="market_exposure")
    if _uses_market_regime(params):
        if benchmark is None:
            raise ValueError("active market regime parameters require benchmark data")
        close = pd.Series(benchmark, copy=True)
        close.index = pd.to_datetime(close.index)
        close = pd.to_numeric(close, errors="coerce")
        close = close.groupby(level=0).last().sort_index()
        close = close.where(close.gt(0.0)).reindex(index)
        available = close.notna()
        ma = close.rolling(
            int(params["market_ma_window"]),
            min_periods=int(params["market_ma_window"]),
        ).mean()
        rolling_peak = close.rolling(20, min_periods=20).max()
        drawdown_20d = close.div(rolling_peak).sub(1.0)
        below_ma = ma.notna() & close.lt(ma)
        exposure.loc[below_ma] = np.minimum(
            exposure.loc[below_ma], float(params["market_below_ma_exposure"])
        )
        crash = drawdown_20d.le(float(params["market_risk_off_drawdown_20d"]))
        exposure.loc[crash.fillna(False)] = np.minimum(
            exposure.loc[crash.fillna(False)],
            float(params["market_crash_exposure"]),
        )
        exposure.loc[~available] = 0.0

    if _uses_breadth_regime(params):
        if breadth_20 is None:
            raise ValueError("active breadth regime parameters require breadth data")
        breadth = pd.Series(breadth_20, copy=True)
        breadth.index = pd.to_datetime(breadth.index)
        breadth = pd.to_numeric(breadth, errors="coerce").groupby(level=0).last()
        breadth = breadth.sort_index().reindex(index)
        window = int(params["breadth_ma_window"])
        breadth_ma = breadth.rolling(window, min_periods=window).mean()
        gap = breadth.sub(breadth_ma)
        risk_off = gap.lt(float(params["breadth_risk_off_gap"]))
        exposure.loc[risk_off.fillna(False)] = np.minimum(
            exposure.loc[risk_off.fillna(False)],
            float(params["breadth_risk_off_exposure"]),
        )
        crash = gap.le(float(params["breadth_crash_gap"]))
        exposure.loc[crash.fillna(False)] = np.minimum(
            exposure.loc[crash.fillna(False)],
            float(params["breadth_crash_exposure"]),
        )
        exposure.loc[breadth.isna() | breadth_ma.isna()] = 0.0
    return exposure


def run_observation_backtest(
    factor_panel: pd.DataFrame,
    parameters: Mapping[str, object],
    *,
    benchmark: pd.Series | None = None,
    end: pd.Timestamp | None = None,
    cost_multiplier: float = 1.0,
    record_positions: bool = False,
) -> ObservationBacktest:
    params = validate_parameters(parameters)
    if not math.isfinite(cost_multiplier) or cost_multiplier <= 0.0:
        raise ValueError("cost_multiplier must be finite and positive")
    scored = build_factor_scores(factor_panel, params)
    if end is not None:
        scored = scored.loc[scored["date"].le(pd.Timestamp(end))].copy()
    dates = pd.DatetimeIndex(sorted(scored["date"].unique()))
    symbols = pd.Index(sorted(scored["symbol"].unique()), dtype=object)
    if len(dates) < 2 or symbols.empty:
        raise ValueError("not enough market data for next-open backtest")
    breadth_20 = build_cross_sectional_momentum_breadth(scored, dates)
    market_exposure = build_market_exposure(
        dates, benchmark, params, breadth_20=breadth_20
    )

    open_px = _pivot(scored, "open", dates, symbols)
    close_px = _pivot(scored, "close", dates, symbols)
    limit_rate = next((name for name in LIMIT_RATE_COLUMNS if name in scored), None)
    limit_up = next((name for name in LIMIT_UP_PRICE_COLUMNS if name in scored), None)
    limit_down = next((name for name in LIMIT_DOWN_PRICE_COLUMNS if name in scored), None)
    limit_rate_px = _pivot(scored, limit_rate, dates, symbols) if limit_rate else None
    limit_up_px = _pivot(scored, limit_up, dates, symbols) if limit_up else None
    limit_down_px = _pivot(scored, limit_down, dates, symbols) if limit_down else None

    open_returns = open_px.ffill().pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    rows_by_date = {date: group for date, group in scored.groupby("date", sort=False)}
    positions = pd.Series(0.0, index=symbols)
    queued_target = positions.copy()
    equity = float(params["initial_capital"])
    risk_peak = equity
    cooldown_remaining = 0
    nav_values: list[float] = []
    turnover_values: list[float] = []
    filled_values: list[int] = []
    pnl_rows: list[pd.Series] = []
    position_rows: list[pd.Series] | None = [] if record_positions else None
    trade_rows: list[dict[str, object]] = []
    execution_counts = {
        "blocked_limit_up_buys": 0,
        "blocked_limit_down_sells": 0,
        "blocked_open_gap_buys": 0,
        "blocked_missing_quotes": 0,
        "blocked_orders_total": 0,
        "portfolio_stop_triggers": 0,
    }

    for index, date in enumerate(dates):
        held_positions = positions.copy()
        contribution = held_positions * open_returns.loc[date].reindex(symbols).fillna(0.0)
        gross_return = float(contribution.sum())
        equity *= 1.0 + gross_return

        target = queued_target.reindex(symbols).fillna(0.0)
        turnover = 0.0
        filled = 0
        if index > 0:
            adjusted, counts = apply_open_constraints_with_diagnostics(
                current=positions,
                target=target,
                open_row=open_px.loc[date],
                prev_close_row=close_px.iloc[index - 1],
                max_buy_open_gap=float(params["max_buy_open_gap"]),
                limit_buffer=float(params["limit_buffer"]),
                block_limit_up_buys=True,
                block_limit_down_sells=True,
                limit_rate_row=limit_rate_px.loc[date] if limit_rate_px is not None else None,
                limit_up_price_row=limit_up_px.loc[date] if limit_up_px is not None else None,
                limit_down_price_row=limit_down_px.loc[date] if limit_down_px is not None else None,
            )
            valid_quotes = open_px.loc[date].gt(0.0) & close_px.iloc[index - 1].gt(0.0)
            missing_change = target.ne(positions) & ~valid_quotes
            adjusted = adjusted.where(~missing_change, positions)
            counts["blocked_missing_quotes"] = int(missing_change.sum())
            counts["blocked_orders_total"] += int(missing_change.sum())
            for name, count in counts.items():
                execution_counts[name] += int(count)
            changes = adjusted.sub(positions).abs()
            turnover = float(changes.sum())
            filled = int(changes.gt(1e-12).sum())
            cost = turnover * (
                float(params["commission_bps"]) + float(params["impact_bps"])
            ) * cost_multiplier / 10_000.0
            equity *= max(0.0, 1.0 - cost)
            positions = adjusted
            if turnover > 0.0 or counts["blocked_orders_total"] > 0:
                trade_rows.append(
                    {
                        "date": date,
                        "turnover": turnover,
                        "filled_trades": filled,
                        "cost_rate": cost,
                        **counts,
                    }
                )

        nav_values.append(equity)
        turnover_values.append(turnover)
        filled_values.append(filled)
        contribution.name = date
        pnl_rows.append(contribution)
        if position_rows is not None:
            position_snapshot = positions.copy()
            position_snapshot.name = date
            position_rows.append(position_snapshot)

        stop_threshold = float(params["portfolio_stop_drawdown"])
        if cooldown_remaining <= 0:
            risk_peak = max(risk_peak, equity)
            risk_drawdown = equity / risk_peak - 1.0 if risk_peak > 0.0 else -1.0
            if index > 0 and stop_threshold > -1.0 and risk_drawdown <= stop_threshold:
                cooldown_remaining = int(params["portfolio_stop_cooldown_sessions"])
                risk_peak = equity
                execution_counts["portfolio_stop_triggers"] += 1

        if cooldown_remaining > 0:
            queued_target = pd.Series(0.0, index=symbols)
            cooldown_remaining -= 1
            if cooldown_remaining == 0:
                risk_peak = equity
        elif index % int(params["rebalance_frequency"]) == 0:
            queued_target = _target_weights(
                rows_by_date[date],
                params,
                equity,
                symbols,
                exposure_scale=float(market_exposure.loc[date]),
            )
        else:
            queued_target = positions.copy()

    equity_series = pd.Series(nav_values, index=dates, name="equity")
    returns = equity_series.pct_change(fill_method=None).fillna(0.0)
    return ObservationBacktest(
        equity=equity_series,
        returns=returns,
        turnover=pd.Series(turnover_values, index=dates, name="turnover"),
        filled_trades=pd.Series(filled_values, index=dates, name="filled_trades"),
        market_exposure=market_exposure,
        symbol_pnl=pd.DataFrame(pnl_rows, index=dates).fillna(0.0),
        trade_log=pd.DataFrame(trade_rows),
        execution_counts=execution_counts,
        position_weights=(
            pd.DataFrame(position_rows, index=dates).fillna(0.0)
            if position_rows is not None
            else None
        ),
    )


def build_walk_forward_folds(
    dates: Sequence[pd.Timestamp],
    periods: EvolutionPeriods,
) -> tuple[WalkForwardFold, ...]:
    index = pd.DatetimeIndex(sorted(pd.Timestamp(value) for value in dates))
    before_selection = index[index < periods.selection_start]
    if len(before_selection) < periods.warmup_sessions:
        raise ValueError("insufficient point-in-time history before selection_start")
    selection = index[
        (index >= periods.selection_start) & (index <= periods.selection_end)
    ]
    folds: list[WalkForwardFold] = []
    stop = len(selection) - periods.fold_sessions + 1
    for start_index in range(0, max(stop, 0), periods.step_sessions):
        window = selection[start_index : start_index + periods.fold_sessions]
        if len(window) != periods.fold_sessions:
            continue
        folds.append(
            WalkForwardFold(
                fold_id=f"wf_{len(folds) + 1:02d}",
                start=window[0],
                end=window[-1],
            )
        )
    return tuple(folds)


def benchmark_total_return(
    benchmark: pd.Series | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float | None:
    if benchmark is None:
        return None
    ordered = benchmark.sort_index().dropna()
    if ordered.empty or ordered.index[0] > start or ordered.index[-1] < end:
        return None
    series = ordered.loc[
        (ordered.index >= start) & (ordered.index <= end)
    ]
    if len(series) < 2 or float(series.iloc[0]) <= 0.0:
        return None
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


def extract_fold_metrics(
    backtest: ObservationBacktest,
    fold: WalkForwardFold,
    benchmark: pd.Series | None = None,
) -> tuple[FoldMetrics, dict[str, object]]:
    equity = backtest.equity
    period_nav = equity.loc[(equity.index >= fold.start) & (equity.index <= fold.end)]
    prior = equity.loc[equity.index < fold.start].tail(1)
    if prior.empty or period_nav.empty:
        raise ValueError(f"fold {fold.fold_id} has insufficient equity observations")
    nav = pd.concat([prior, period_nav])
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    daily = nav.pct_change(fill_method=None).dropna()
    turnover = backtest.turnover.loc[
        (backtest.turnover.index >= fold.start)
        & (backtest.turnover.index <= fold.end)
    ]
    fills = backtest.filled_trades.loc[
        (backtest.filled_trades.index >= fold.start)
        & (backtest.filled_trades.index <= fold.end)
    ]
    market_exposure = backtest.market_exposure.loc[
        (backtest.market_exposure.index >= fold.start)
        & (backtest.market_exposure.index <= fold.end)
    ]
    pnl = backtest.symbol_pnl.loc[
        (backtest.symbol_pnl.index >= fold.start)
        & (backtest.symbol_pnl.index <= fold.end)
    ].sum(axis=0)
    positive_pnl = pnl.clip(lower=0.0)
    pnl_concentration = (
        float(positive_pnl.max() / positive_pnl.sum())
        if float(positive_pnl.sum()) > 1e-12
        else 1.0
    )
    benchmark_return = benchmark_total_return(benchmark, fold.start, fold.end)
    excess_return = (
        total_return - benchmark_return if benchmark_return is not None else None
    )
    metrics = FoldMetrics(
        fold_id=fold.fold_id,
        total_return=total_return,
        max_drawdown=float(max_drawdown(nav)),
        sharpe=float(sharpe_like(daily)),
        filled_trades=int(fills.sum()),
        average_turnover=float(turnover.mean()) if not turnover.empty else 0.0,
        pnl_concentration=pnl_concentration,
    )
    row = {
        "fold_id": fold.fold_id,
        "start": fold.start.date().isoformat(),
        "end": fold.end.date().isoformat(),
        "sessions": int(len(period_nav)),
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "filled_trades": metrics.filled_trades,
        "average_turnover": metrics.average_turnover,
        "average_market_exposure": float(market_exposure.mean()),
        "risk_off_sessions": int(market_exposure.lt(1.0).sum()),
        "pnl_concentration": metrics.pnl_concentration,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
    }
    return metrics, row


def evaluate_parameter_set(
    factor_panel: pd.DataFrame,
    parameters: Mapping[str, object],
    periods: EvolutionPeriods,
    *,
    benchmark: pd.Series | None = None,
    cost_multiplier: float = 1.0,
    record_positions: bool = False,
) -> ParameterEvaluation:
    dates = pd.DatetimeIndex(sorted(factor_panel["date"].unique()))
    folds = build_walk_forward_folds(dates, periods)
    if not folds:
        raise ValueError("no complete walk-forward folds are available")
    backtest = run_observation_backtest(
        factor_panel,
        parameters,
        benchmark=benchmark,
        end=periods.selection_end,
        cost_multiplier=cost_multiplier,
        record_positions=record_positions,
    )
    metrics: list[FoldMetrics] = []
    rows: list[dict[str, object]] = []
    for fold in folds:
        fold_metrics, row = extract_fold_metrics(backtest, fold, benchmark)
        metrics.append(fold_metrics)
        rows.append(row)
    return ParameterEvaluation(tuple(metrics), tuple(rows), backtest)


def evaluate_holdout(
    factor_panel: pd.DataFrame,
    parameters: Mapping[str, object],
    periods: EvolutionPeriods,
    *,
    benchmark: pd.Series | None = None,
    cost_multiplier: float = 1.0,
) -> tuple[FoldMetrics, dict[str, object], ObservationBacktest]:
    available_dates = pd.DatetimeIndex(sorted(factor_panel["date"].unique()))
    end = periods.holdout_end or available_dates[-1]
    holdout_dates = available_dates[
        (available_dates >= periods.holdout_start) & (available_dates <= end)
    ]
    if len(holdout_dates) < 2:
        raise ValueError("holdout period is not yet available")
    backtest = run_observation_backtest(
        factor_panel,
        parameters,
        benchmark=benchmark,
        end=end,
        cost_multiplier=cost_multiplier,
    )
    fold = WalkForwardFold("holdout", holdout_dates[0], holdout_dates[-1])
    metrics, row = extract_fold_metrics(backtest, fold, benchmark)
    return metrics, row, backtest


def prepare_factor_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    factors, metadata = compute_observation_factors(panel)
    unavailable = [
        name
        for name in ("liquidity_20", "liquidity_stability_20")
        if not metadata["factor_availability"].get(name, False)
    ]
    if unavailable:
        raise ValueError(
            "evolution requires point-in-time amount data for factors: "
            + ", ".join(unavailable)
        )
    for factor, rank_column in FACTOR_RANK_COLUMNS.items():
        factors[rank_column] = pd.to_numeric(
            factors[factor], errors="coerce"
        ).groupby(factors["date"], sort=False).rank(method="average", pct=True)
    return factors, metadata
