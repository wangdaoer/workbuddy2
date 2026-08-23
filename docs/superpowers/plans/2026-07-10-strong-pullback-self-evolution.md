# Guarded Strong Pullback Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, validation-gated evolution runner for the existing strong-pullback satellite strategy without exposing the holdout test period during parameter search.

**Architecture:** Keep all pure configuration, metric, promotion, and test-decision logic in `strong_pullback_evolution.py`. Put data loading, the adapter to `run_satellite_walk_forward()`, versioned artifact I/O, resume checks, and the CLI in `run_strong_pullback_evolution.py`; this keeps the existing trading engine as the single source of execution behavior and makes the evolution decisions inexpensive to unit test.

**Tech Stack:** Python 3.12, pandas, NumPy, PyYAML, unittest/pytest, the existing `run_strong_pullback_satellite.py` engine.

## Global Constraints

- Research and validation trials must receive price matrices physically truncated at `validation_end`.
- The holdout test period may be opened only after the research champion is locked.
- Only initial baseline and locked champion may be evaluated on the holdout period.
- Search candidates may change configuration values only; Python strategy code is never generated or mutated.
- Maximum accepted validation drawdown is `-40%`.
- Validation annualized return is the primary objective; Sharpe, drawdown, rolling stability, and turnover are guardrails.
- A successful run produces a candidate for manual review and never overwrites an existing strategy configuration.
- No broker integration, order placement, or account mutation is in scope.
- Input panel selection remains explicit through required `--data`; no hard-coded dated panel is selected silently.

---

### Task 1: Define and validate the evolution configuration protocol

**Files:**
- Create: `strong_pullback_evolution.py`
- Create: `tests/test_strong_pullback_evolution.py`

**Interfaces:**
- Produces: `EvolutionPeriods`, `SelectionRules`, `SearchCandidate`, `SearchGroup`, and `EvolutionConfig` dataclasses.
- Produces: `parse_evolution_config(raw: Mapping[str, object]) -> EvolutionConfig`.
- Produces: `load_evolution_config(path: Path) -> EvolutionConfig`.
- Produces: `build_group_candidates(incumbent: Mapping[str, object], group: SearchGroup) -> tuple[tuple[str, dict[str, object]], ...]`.
- Consumes later: Tasks 2-5 use these types and fully resolved baseline parameters.

- [ ] **Step 1: Write failing tests for date isolation, unknown parameters, duplicate identifiers, and independent candidate construction**

```python
import copy
import unittest

from strong_pullback_evolution import (
    build_group_candidates,
    parse_evolution_config,
)


def valid_raw_config() -> dict[str, object]:
    return {
        "strategy": "strong_pullback_satellite",
        "periods": {
            "research_start": "2022-01-01",
            "train_end": "2024-12-31",
            "validation_start": "2025-01-01",
            "validation_end": "2025-12-31",
            "test_start": "2026-01-01",
            "test_end": None,
        },
        "baseline": {"top_n": 8, "leverage": 0.60, "max_position_weight": 0.08},
        "search_groups": [
            {
                "id": "risk_budget",
                "hypothesis_cn": "扩大风险预算",
                "candidates": [
                    {"id": "risk_075", "overrides": {"leverage": 0.75}},
                    {"id": "risk_090", "overrides": {"leverage": 0.90}},
                ],
            }
        ],
        "selection": {
            "min_validation_days": 120,
            "min_test_days": 60,
            "max_drawdown_floor": -0.40,
            "min_annualized_return_delta": 0.01,
            "min_sharpe_delta": -0.10,
            "max_turnover_ratio": 1.50,
            "rolling_window_days": 126,
            "max_negative_window_rate": 0.60,
        },
    }


class EvolutionConfigTest(unittest.TestCase):
    def test_rejects_overlapping_validation_and_test_periods(self):
        raw = valid_raw_config()
        raw["periods"]["test_start"] = "2025-12-31"

        with self.assertRaisesRegex(ValueError, "train_end < validation_start <= validation_end < test_start"):
            parse_evolution_config(raw)

    def test_rejects_unknown_strategy_override(self):
        raw = valid_raw_config()
        raw["search_groups"][0]["candidates"][0]["overrides"] = {"future_leak": 1}

        with self.assertRaisesRegex(ValueError, "Unknown strategy parameters"):
            parse_evolution_config(raw)

    def test_rejects_duplicate_candidate_ids_across_groups(self):
        raw = valid_raw_config()
        raw["search_groups"].append(copy.deepcopy(raw["search_groups"][0]))
        raw["search_groups"][1]["id"] = "entry_depth"

        with self.assertRaisesRegex(ValueError, "Duplicate candidate id"):
            parse_evolution_config(raw)

    def test_group_candidates_share_incumbent_but_not_each_other(self):
        config = parse_evolution_config(valid_raw_config())
        generated = build_group_candidates(config.baseline, config.search_groups[0])

        self.assertEqual(generated[0][1]["leverage"], 0.75)
        self.assertEqual(generated[1][1]["leverage"], 0.90)
        self.assertEqual(config.baseline["leverage"], 0.60)
        self.assertIsNot(generated[0][1], generated[1][1])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'strong_pullback_evolution'`.

- [ ] **Step 3: Implement dataclasses, defaults, validation, and independent group candidates**

Create `strong_pullback_evolution.py` with these exact public shapes and checks:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


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
})


@dataclass(frozen=True)
class EvolutionPeriods:
    research_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
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
    periods: EvolutionPeriods
    baseline: dict[str, object]
    search_groups: tuple[SearchGroup, ...]
    selection: SelectionRules


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
    if float(params["min_pullback_5d"]) > float(params["max_pullback_5d"]):
        raise ValueError("min_pullback_5d must be <= max_pullback_5d")


def parse_evolution_config(raw: Mapping[str, object]) -> EvolutionConfig:
    if raw.get("strategy") != "strong_pullback_satellite":
        raise ValueError("strategy must be strong_pullback_satellite")
    period_raw = dict(raw.get("periods") or {})
    periods = EvolutionPeriods(
        research_start=_timestamp(period_raw.get("research_start"), "research_start"),
        train_end=_timestamp(period_raw.get("train_end"), "train_end"),
        validation_start=_timestamp(period_raw.get("validation_start"), "validation_start"),
        validation_end=_timestamp(period_raw.get("validation_end"), "validation_end"),
        test_start=_timestamp(period_raw.get("test_start"), "test_start"),
        test_end=_timestamp(period_raw.get("test_end"), "test_end", allow_none=True),
    )
    if not (
        periods.research_start <= periods.train_end
        and periods.train_end < periods.validation_start
        and periods.validation_start <= periods.validation_end
        and periods.validation_end < periods.test_start
        and (periods.test_end is None or periods.test_start <= periods.test_end)
    ):
        raise ValueError("Periods must satisfy train_end < validation_start <= validation_end < test_start")

    baseline = {**DEFAULT_STRATEGY_PARAMS, **dict(raw.get("baseline") or {})}
    _validate_params(baseline)
    group_ids: set[str] = set()
    candidate_ids: set[str] = set()
    groups: list[SearchGroup] = []
    for group_raw in list(raw.get("search_groups") or []):
        group_id = str(group_raw.get("id", "")).strip()
        if not group_id or group_id in group_ids:
            raise ValueError(f"Duplicate or empty group id: {group_id!r}")
        group_ids.add(group_id)
        candidates: list[SearchCandidate] = []
        for candidate_raw in list(group_raw.get("candidates") or []):
            candidate_id = str(candidate_raw.get("id", "")).strip()
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError(f"Duplicate candidate id: {candidate_id!r}")
            candidate_ids.add(candidate_id)
            overrides = dict(candidate_raw.get("overrides") or {})
            unknown = set(overrides) - ALLOWED_OVERRIDE_PARAMS
            if unknown:
                raise ValueError(f"Unknown strategy parameters: {sorted(unknown)}")
            _validate_params({**baseline, **overrides})
            candidates.append(SearchCandidate(candidate_id, overrides))
        if not candidates:
            raise ValueError(f"Search group {group_id!r} has no candidates")
        groups.append(SearchGroup(group_id, str(group_raw.get("hypothesis_cn", "")).strip(), tuple(candidates)))

    selection_raw = dict(raw.get("selection") or {})
    selection = SelectionRules(**selection_raw)
    if selection.min_validation_days <= 0 or selection.min_test_days <= 0:
        raise ValueError("minimum period days must be positive")
    if not -1.0 <= selection.max_drawdown_floor <= 0.0:
        raise ValueError("max_drawdown_floor must be between -1 and 0")
    if selection.max_turnover_ratio < 1.0:
        raise ValueError("max_turnover_ratio must be >= 1")
    if not 0.0 <= selection.max_negative_window_rate <= 1.0:
        raise ValueError("max_negative_window_rate must be between 0 and 1")
    return EvolutionConfig("strong_pullback_satellite", periods, baseline, tuple(groups), selection)


def load_evolution_config(path: Path) -> EvolutionConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Evolution config must be a YAML mapping")
    return parse_evolution_config(raw)


def build_group_candidates(
    incumbent: Mapping[str, object], group: SearchGroup
) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(
        (candidate.candidate_id, {**dict(incumbent), **candidate.overrides})
        for candidate in group.candidates
    )
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit the configuration protocol**

```powershell
git add strong_pullback_evolution.py tests/test_strong_pullback_evolution.py
git commit -m "feat: add evolution config protocol"
```

---

### Task 2: Add period metrics, promotion gates, ranking, and holdout decisions

**Files:**
- Modify: `strong_pullback_evolution.py`
- Modify: `tests/test_strong_pullback_evolution.py`

**Interfaces:**
- Produces: `PromotionDecision` and `CandidateDecision` dataclasses.
- Produces: `calculate_segment_metrics(equity, start, end, rolling_window_days) -> dict[str, float]`.
- Produces: `evaluate_promotion(candidate_metrics, incumbent_metrics, rules) -> PromotionDecision`.
- Produces: `choose_group_winner(incumbent_id, incumbent_metrics, candidates, rules) -> tuple[str, tuple[CandidateDecision, ...]]`.
- Produces: `assess_test_result(baseline_metrics, champion_metrics, rules) -> tuple[str, str]`.
- Consumes later: Task 4 calls these functions after each trial and for final holdout comparison.

- [ ] **Step 1: Write failing metric and boundary tests**

Append tests that use a real equity audit frame:

```python
import pandas as pd

from strong_pullback_evolution import (
    assess_test_result,
    calculate_segment_metrics,
    choose_group_winner,
    evaluate_promotion,
)


class EvolutionDecisionTest(unittest.TestCase):
    def test_segment_metrics_compound_net_returns_inside_requested_period(self):
        equity = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=4, freq="B"),
                "gross_return": [0.50, 0.10, -0.05, 0.02],
                "cost": [0.0, 0.01, 0.0, 0.0],
                "turnover": [0.0, 0.2, 0.1, 0.0],
                "gross_exposure": [0.0, 0.6, 0.6, 0.6],
            }
        )

        metrics = calculate_segment_metrics(
            equity, "2025-01-02", "2025-01-06", rolling_window_days=2
        )

        self.assertAlmostEqual(metrics["total_return"], 1.09 * 0.95 * 1.02 - 1.0)
        self.assertEqual(metrics["trade_days"], 3)
        self.assertAlmostEqual(metrics["avg_turnover"], 0.1)

    def test_promotion_accepts_exact_drawdown_and_sharpe_boundaries(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        incumbent = {
            "annualized_return": 0.10, "max_drawdown": -0.30, "sharpe_like": 0.50,
            "avg_turnover": 0.10, "trade_days": 200, "negative_window_rate": 0.20,
        }
        candidate = {
            "annualized_return": 0.11, "max_drawdown": -0.40, "sharpe_like": 0.40,
            "avg_turnover": 0.15, "trade_days": 200, "negative_window_rate": 0.60,
        }

        decision = evaluate_promotion(candidate, incumbent, rules)

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reasons, ())

    def test_group_keeps_incumbent_when_no_candidate_passes(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        incumbent = {
            "annualized_return": 0.10, "max_drawdown": -0.20, "sharpe_like": 0.50,
            "avg_turnover": 0.10, "trade_days": 200, "negative_window_rate": 0.20,
        }
        candidate = {**incumbent, "annualized_return": 0.105}

        winner, decisions = choose_group_winner(
            "incumbent", incumbent, (("weak_gain", candidate),), rules
        )

        self.assertEqual(winner, "incumbent")
        self.assertFalse(decisions[0].promotion.eligible)

    def test_holdout_recommends_rollback_when_champion_loses(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        baseline = {"total_return": 0.20, "max_drawdown": -0.20, "sharpe_like": 0.50, "trade_days": 100}
        champion = {"total_return": 0.10, "max_drawdown": -0.25, "sharpe_like": 0.45, "trade_days": 100}

        status, reason = assess_test_result(baseline, champion, rules)

        self.assertEqual(status, "rollback_recommended")
        self.assertIn("收益", reason)
```

- [ ] **Step 2: Run the four new tests and verify RED**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: import errors for the four new functions.

- [ ] **Step 3: Implement metrics and exact gate behavior**

Add these implementations to `strong_pullback_evolution.py`:

```python
import numpy as np

from run_backtest import max_drawdown, sharpe_like


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


def calculate_segment_metrics(
    equity: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    rolling_window_days: int,
) -> dict[str, float]:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("date")
    if frame.empty:
        raise ValueError(f"No equity rows between {start} and {end}")
    returns = (
        pd.to_numeric(frame["gross_return"], errors="coerce")
        - pd.to_numeric(frame["cost"], errors="coerce")
    )
    if returns.isna().any() or (returns <= -1.0).any():
        raise ValueError("Segment contains invalid net returns")
    nav = (1.0 + returns).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annualized = float((1.0 + total_return) ** (252.0 / len(nav)) - 1.0)
    rolling = nav / nav.shift(int(rolling_window_days)) - 1.0
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
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: `8 passed`.

- [ ] **Step 5: Commit the metric and promotion logic**

```powershell
git add strong_pullback_evolution.py tests/test_strong_pullback_evolution.py
git commit -m "feat: add evolution promotion gates"
```

---

### Task 3: Build the existing-engine adapter and trial artifact I/O

**Files:**
- Create: `run_strong_pullback_evolution.py`
- Modify: `tests/test_strong_pullback_evolution.py`

**Interfaces:**
- Produces: `PriceBundle` and `StrategyRun` dataclasses.
- Produces: `validate_input_schema(path: Path) -> None`.
- Produces: `load_price_bundle(data_path, end_date, benchmark_path, params) -> PriceBundle`.
- Produces: `execute_strategy_trial(bundle, params) -> StrategyRun`.
- Produces: `write_trial_artifacts(trial_dir, run, metrics, trial_state) -> None` and `load_trial_artifacts(trial_dir) -> StrategyRun`.
- Consumes: `run_satellite_walk_forward`, `load_prices`, `pivot_prices`, `clean_matrix`, and `load_market_exposure` from existing modules.

- [ ] **Step 1: Write failing adapter tests for schema enforcement, physical truncation, and artifact round-trip**

```python
import tempfile
from pathlib import Path

import pandas as pd

from strong_pullback_evolution import DEFAULT_STRATEGY_PARAMS
from run_strong_pullback_evolution import (
    StrategyRun,
    load_price_bundle,
    load_trial_artifacts,
    validate_input_schema,
    write_trial_artifacts,
)


class EvolutionAdapterTest(unittest.TestCase):
    def test_schema_requires_real_ohlcv_and_amount_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            pd.DataFrame({"date": ["2025-01-01"], "symbol": ["000001"], "close": [10.0]}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Missing evolution input columns"):
                validate_input_schema(path)

    def test_price_bundle_is_physically_truncated_at_requested_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            rows = []
            for date in pd.date_range("2025-01-01", periods=4, freq="B"):
                rows.append({
                    "date": date, "symbol": "000001", "open": 10.0, "high": 11.0,
                    "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10_000.0,
                })
            pd.DataFrame(rows).to_csv(path, index=False)

            bundle = load_price_bundle(path, pd.Timestamp("2025-01-03"), None, {**DEFAULT_STRATEGY_PARAMS, "max_abs_daily_return": 0.22})

            self.assertEqual(bundle.close.index.max(), pd.Timestamp("2025-01-03"))

    def test_trial_artifacts_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "trial"
            run = StrategyRun(
                equity=pd.DataFrame({"date": ["2025-01-02"], "equity": [1_010_000.0], "gross_return": [0.01], "cost": [0.0], "turnover": [0.1], "gross_exposure": [0.6]}),
                weights=pd.DataFrame({"date": ["2025-01-01"], "momentum_20": [1.0]}),
                trades=pd.DataFrame({"signal_date": ["2025-01-01"], "gross_return": [0.01]}),
                candidates=pd.DataFrame({"signal_date": ["2025-01-01"], "symbol": ["000001"]}),
            )

            write_trial_artifacts(trial_dir, run, {"validation": {"total_return": 0.01}}, {"status": "completed"})
            loaded = load_trial_artifacts(trial_dir)

            self.assertEqual(float(loaded.equity.loc[0, "equity"]), 1_010_000.0)
            self.assertEqual(str(loaded.candidates.loc[0, "symbol"]), "000001")
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: import fails for `run_strong_pullback_evolution`.

- [ ] **Step 3: Implement the adapter without copying strategy logic**

Create `run_strong_pullback_evolution.py` with these core definitions:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from pandas.errors import EmptyDataError

from run_backtest import load_prices, pivot_prices
from run_strong_pullback_satellite import run_satellite_walk_forward
from train_next_open_rank_model import clean_matrix, load_market_exposure


REQUIRED_EVOLUTION_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
FILTER_KEYS = (
    "min_close", "min_avg_amount_20d", "min_pullback_5d", "max_pullback_5d",
    "min_prior_return_20", "min_prior_return_60", "min_return_20d",
    "min_return_60d", "min_distance_ma60", "max_intraday_return",
)


@dataclass(frozen=True)
class PriceBundle:
    close: pd.DataFrame
    open_px: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    amount: pd.DataFrame
    market_exposure: pd.Series


@dataclass(frozen=True)
class StrategyRun:
    equity: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    candidates: pd.DataFrame


def validate_input_schema(path: Path) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_EVOLUTION_COLUMNS - columns
    if missing:
        raise ValueError(f"Missing evolution input columns: {sorted(missing)}")


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
    if close.index.max() > end_date:
        raise AssertionError("Price bundle extends beyond requested end date")
    return PriceBundle(close, open_px, high, low, amount, market_exposure)


def execute_strategy_trial(bundle: PriceBundle, params: Mapping[str, object]) -> StrategyRun:
    filter_kwargs = {key: float(params[key]) for key in FILTER_KEYS}
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
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (trial_dir / "trial_state.json").write_text(json.dumps(trial_state, ensure_ascii=False, indent=2), encoding="utf-8")


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
```

- [ ] **Step 4: Run adapter tests and verify GREEN**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: `11 passed`.

- [ ] **Step 5: Commit the engine adapter**

```powershell
git add run_strong_pullback_evolution.py tests/test_strong_pullback_evolution.py
git commit -m "feat: add strong pullback trial adapter"
```

---

### Task 4: Orchestrate isolated rounds, resume matching, version manifests, and reports

**Files:**
- Modify: `run_strong_pullback_evolution.py`
- Modify: `tests/test_strong_pullback_evolution.py`

**Interfaces:**
- Produces: `RunEvidence` and `EvolutionOutcome` dataclasses.
- Produces: `build_run_evidence(data_path, config_path, config, git_commit) -> RunEvidence`.
- Produces: `can_resume_trial(trial_state, evidence_fingerprint, params_hash, trial_id) -> bool`.
- Produces: `run_evolution(config, data_path, config_path, benchmark_path, asof_date, output_root, run_id, resume, bundle_loader, trial_executor) -> EvolutionOutcome`.
- Produces: `write_chinese_summary(...) -> Path`.
- Consumes: all interfaces from Tasks 1-3.

- [ ] **Step 1: Write failing orchestration tests using injected deterministic boundaries**

The injected loader and executor are intentional here: they test that orchestration cannot request test data early, while Task 3 separately tests the real engine adapter.

```python
import json

from run_strong_pullback_evolution import (
    PriceBundle,
    StrategyRun,
    can_resume_trial,
    run_evolution,
)


class EvolutionOrchestrationTest(unittest.TestCase):
    @staticmethod
    def _flat_bundle(end_date: pd.Timestamp) -> PriceBundle:
        dates = pd.date_range("2022-01-03", end=end_date, freq="B")
        frame = pd.DataFrame({"000001": 10.0}, index=dates)
        return PriceBundle(frame, frame.copy(), frame.copy(), frame.copy(), frame * 1_000_000, pd.Series(1.0, index=dates))

    @staticmethod
    def _run_for_params(bundle: PriceBundle, params: dict[str, object]) -> StrategyRun:
        dates = bundle.close.index[bundle.close.index >= pd.Timestamp("2024-01-01")]
        daily = 0.0002 + float(params["leverage"]) * 0.0001
        equity = pd.DataFrame({
            "date": dates,
            "equity": 1_000_000.0 * (1.0 + daily) ** pd.RangeIndex(1, len(dates) + 1),
            "gross_return": daily,
            "cost": 0.0,
            "turnover": 0.10,
            "gross_exposure": float(params["leverage"]),
        })
        return StrategyRun(equity, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def test_research_load_finishes_at_validation_end_before_test_load(self):
        config = parse_evolution_config(valid_raw_config())
        requested_ends: list[pd.Timestamp] = []

        def loader(data_path, end_date, benchmark_path, params):
            requested_ends.append(pd.Timestamp(end_date))
            return self._flat_bundle(pd.Timestamp(end_date))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            outcome = run_evolution(
                config=config,
                data_path=data,
                config_path=config_path,
                benchmark_path=None,
                asof_date=pd.Timestamp("2026-07-09"),
                output_root=root / "runs",
                run_id="isolation-test",
                resume=False,
                bundle_loader=loader,
                trial_executor=self._run_for_params,
                git_commit="test-commit",
            )

        self.assertEqual(requested_ends[0], pd.Timestamp("2025-12-31"))
        self.assertTrue(all(value <= pd.Timestamp("2025-12-31") for value in requested_ends[:-1]))
        self.assertEqual(requested_ends[-1], pd.Timestamp("2026-07-09"))
        self.assertIn(outcome.test_status, {"ready_for_manual_review", "rollback_recommended"})

    def test_resume_requires_exact_trial_evidence(self):
        state = {"status": "completed", "trial_id": "risk_075", "evidence_fingerprint": "abc", "params_hash": "def"}

        self.assertTrue(can_resume_trial(state, "abc", "def", "risk_075"))
        self.assertFalse(can_resume_trial(state, "changed", "def", "risk_075"))
        self.assertFalse(can_resume_trial(state, "abc", "changed", "risk_075"))

    def test_failed_run_does_not_update_latest_pointer(self):
        config = parse_evolution_config(valid_raw_config())

        def broken_executor(bundle, params):
            raise RuntimeError("trial failed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Baseline trial failed"):
                run_evolution(
                    config, data, config_path, None, pd.Timestamp("2026-07-09"), root / "runs",
                    "failed-run", False,
                    lambda data_path, end_date, benchmark_path, params: self._flat_bundle(pd.Timestamp(end_date)),
                    broken_executor, "test-commit"
                )
            self.assertFalse((root / "runs" / "latest.json").exists())
            manifest = json.loads((root / "runs" / "failed-run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
```

- [ ] **Step 2: Run orchestration tests and verify RED**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: import errors for `run_evolution` and `can_resume_trial`.

- [ ] **Step 3: Implement evidence hashing and exact resume checks**

Use stable JSON serialization so hashes do not depend on dictionary insertion order:

```python
import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import yaml

from strong_pullback_evolution import (
    EvolutionConfig,
    build_group_candidates,
    calculate_segment_metrics,
    choose_group_winner,
    assess_test_result,
)


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunEvidence:
    data_path: str
    data_size: int
    data_mtime_ns: int
    config_path: str
    config_hash: str
    git_commit: str

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class EvolutionOutcome:
    run_id: str
    run_dir: Path
    champion_id: str
    champion_params: dict[str, object]
    test_status: str
    test_reason: str


def build_run_evidence(data_path: Path, config_path: Path, config: EvolutionConfig, git_commit: str) -> RunEvidence:
    stat = data_path.resolve().stat()
    return RunEvidence(
        data_path=str(data_path.resolve()),
        data_size=int(stat.st_size),
        data_mtime_ns=int(stat.st_mtime_ns),
        config_path=str(config_path.resolve()),
        config_hash=stable_hash({
            "strategy": config.strategy,
            "periods": asdict(config.periods),
            "baseline": config.baseline,
            "search_groups": [asdict(group) for group in config.search_groups],
            "selection": asdict(config.selection),
        }),
        git_commit=git_commit,
    )


def resolved_config_dict(config: EvolutionConfig) -> dict[str, object]:
    return {
        "strategy": config.strategy,
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
    trial_state: Mapping[str, object], evidence_fingerprint: str, params_hash: str, trial_id: str
) -> bool:
    return (
        trial_state.get("status") == "completed"
        and trial_state.get("trial_id") == trial_id
        and trial_state.get("evidence_fingerprint") == evidence_fingerprint
        and trial_state.get("params_hash") == params_hash
    )
```

- [ ] **Step 4: Implement the isolated round loop and fail-closed manifest updates**

`run_evolution()` must use this exact phase order:

```python
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
) -> EvolutionOutcome:
    run_dir = output_root / run_id
    evidence = build_run_evidence(data_path, config_path, config, git_commit)
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
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "benchmark": str(benchmark_path.resolve()) if benchmark_path else None,
        "asof_date": asof_date.strftime("%Y-%m-%d"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config_dict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    try:
        research_bundle = bundle_loader(
            data_path, config.periods.validation_end, benchmark_path, config.baseline
        )
        if research_bundle.close.index.max() > config.periods.validation_end:
            raise AssertionError("Research bundle contains holdout dates")

        trial_rows: list[dict[str, object]] = []
        round_rows: list[dict[str, object]] = []
        trial_runs: dict[str, StrategyRun] = {}
        trial_metrics: dict[str, dict[str, dict[str, float]]] = {}
        trial_params: dict[str, dict[str, object]] = {"baseline": dict(config.baseline)}

        def execute_one(trial_id: str, params: dict[str, object]) -> StrategyRun:
            trial_dir = run_dir / "trials" / trial_id
            params_hash = stable_hash(params)
            state_path = trial_dir / "trial_state.json"
            if resume and state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if can_resume_trial(state, evidence.fingerprint, params_hash, trial_id):
                    trial_metrics[trial_id] = json.loads(
                        (trial_dir / "metrics.json").read_text(encoding="utf-8")
                    )
                    return load_trial_artifacts(trial_dir)
            run = trial_executor(research_bundle, params)
            train_metrics = calculate_segment_metrics(
                run.equity, config.periods.research_start, config.periods.train_end,
                config.selection.rolling_window_days,
            )
            validation_metrics = calculate_segment_metrics(
                run.equity, config.periods.validation_start, config.periods.validation_end,
                config.selection.rolling_window_days,
            )
            trial_metrics[trial_id] = {"train": train_metrics, "validation": validation_metrics}
            write_trial_artifacts(
                trial_dir, run, trial_metrics[trial_id],
                {"status": "completed", "trial_id": trial_id, "params_hash": params_hash, "evidence_fingerprint": evidence.fingerprint},
            )
            return run

        try:
            trial_runs["baseline"] = execute_one("baseline", trial_params["baseline"])
        except Exception as exc:
            raise RuntimeError(f"Baseline trial failed: {exc}") from exc
        trial_rows.append({
            "group_id": "baseline",
            "trial_id": "baseline",
            "parent_id": "",
            "status": "incumbent",
            "reason_cn": "初始基准",
            **{
                f"validation_{key}": value
                for key, value in trial_metrics["baseline"]["validation"].items()
            },
        })

        incumbent_id = "baseline"
        incumbent_params = dict(config.baseline)
        incumbent_metrics = trial_metrics["baseline"]["validation"]
        for group in config.search_groups:
            candidate_pairs: list[tuple[str, Mapping[str, float]]] = []
            for candidate_id, params in build_group_candidates(incumbent_params, group):
                trial_params[candidate_id] = params
                try:
                    trial_runs[candidate_id] = execute_one(candidate_id, params)
                    candidate_pairs.append((candidate_id, trial_metrics[candidate_id]["validation"]))
                except Exception as exc:
                    trial_rows.append({"group_id": group.group_id, "trial_id": candidate_id, "status": "trial_error", "reason_cn": str(exc)})
            if not candidate_pairs:
                raise RuntimeError(f"All candidates failed in group {group.group_id}")
            winner_id, decisions = choose_group_winner(
                incumbent_id, incumbent_metrics, tuple(candidate_pairs), config.selection
            )
            for decision in decisions:
                trial_rows.append({
                    "group_id": group.group_id,
                    "trial_id": decision.candidate_id,
                    "parent_id": incumbent_id,
                    "status": "eligible" if decision.promotion.eligible else "rejected",
                    "reason_cn": "通过全部门槛" if decision.promotion.eligible else "；".join(decision.promotion.reasons),
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
                "reason_cn": "验证期收益优先且全部门槛通过" if promoted else "没有候选通过全部门槛",
            })
            if promoted:
                incumbent_id = winner_id
                incumbent_params = dict(trial_params[winner_id])
                incumbent_metrics = trial_metrics[winner_id]["validation"]

        champion_id = incumbent_id
        champion_params = incumbent_params
        test_end = min(
            asof_date,
            config.periods.test_end if config.periods.test_end is not None else asof_date,
        )
        full_bundle = bundle_loader(data_path, test_end, benchmark_path, config.baseline)
        final_runs = {
            "baseline": trial_executor(full_bundle, config.baseline),
            "champion": trial_executor(full_bundle, champion_params),
        }
        final_metrics = {
            name: calculate_segment_metrics(
                run.equity, config.periods.test_start, test_end, config.selection.rolling_window_days
            )
            for name, run in final_runs.items()
        }
        final_params = {"baseline": dict(config.baseline), "champion": dict(champion_params)}
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

        manifest.update({
            "status": "success", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "champion_id": champion_id, "test_status": test_status, "summary": str(summary_path),
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_root / "evolution_registry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        return EvolutionOutcome(run_id, run_dir, champion_id, champion_params, test_status, test_reason)
    except Exception as exc:
        manifest.update({
            "status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
```

The resume branch above must load both the data artifacts and `metrics.json`; a cached trial is unusable for promotion if either side is missing or malformed.

- [ ] **Step 5: Implement the Chinese report with explicit reasons and risk warning**

```python
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
        {"版本": name, "测试期总收益": values["total_return"], "测试期最大回撤": values["max_drawdown"], "测试期Sharpe": values["sharpe_like"]}
        for name, values in final_metrics.items()
    ])
    lines = [
        f"# 强势回调策略自进化报告 {asof_date:%Y-%m-%d}", "",
        f"- 研究冠军：`{champion_id}`", f"- 测试状态：`{test_status}`", f"- 判定原因：{test_reason}", "",
        "## 每轮决定", "",
        pd.DataFrame(round_rows).to_markdown(index=False) if round_rows else "本次没有搜索组。", "",
        "## 保留测试期比较", "", comparison.to_markdown(index=False, floatfmt=".4f"), "",
        "## 风险提示", "",
        "该结果仅用于研究和人工复核，不连接券商、不自动下单，也不会自动覆盖现有策略配置。",
    ]
    path = run_dir / f"evolution_summary_{asof_date:%Y%m%d}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
```

- [ ] **Step 6: Run orchestration tests and verify GREEN**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: `14 passed`.

- [ ] **Step 7: Commit orchestration and version management**

```powershell
git add run_strong_pullback_evolution.py tests/test_strong_pullback_evolution.py
git commit -m "feat: orchestrate versioned strategy evolution"
```

---

### Task 5: Add the production configuration, CLI, documentation, and full verification

**Files:**
- Create: `configs/evolution_strong_pullback.yaml`
- Modify: `run_strong_pullback_evolution.py`
- Modify: `tests/test_strong_pullback_evolution.py`
- Modify: `README.md`
- Modify: `MODEL_OPERATING_SYSTEM.md`

**Interfaces:**
- Produces CLI: `python run_strong_pullback_evolution.py --config ... --data ... [--benchmark ...] [--asof-date ...] [--output-root ...] [--run-id ...] [--resume ...]`.
- Produces default config matching the approved design.
- Produces an end-to-end smoke test using the real parser and version-writing path.

- [ ] **Step 1: Write failing CLI and smoke tests**

```python
import numpy as np
import yaml

from strong_pullback_evolution import load_evolution_config
from run_strong_pullback_evolution import parse_args


class EvolutionCliTest(unittest.TestCase):
    def test_cli_requires_explicit_data_path(self):
        with self.assertRaises(SystemExit):
            parse_args(["--config", "configs/evolution_strong_pullback.yaml"])

    def test_default_config_parses_and_has_three_hypothesis_groups(self):
        config = load_evolution_config(Path("configs/evolution_strong_pullback.yaml"))

        self.assertEqual([group.group_id for group in config.search_groups], ["risk_budget", "entry_depth", "rebound_exit"])
        self.assertEqual(config.selection.max_drawdown_floor, -0.40)

    def test_real_engine_evolution_writes_versioned_outputs(self):
        dates = pd.bdate_range("2024-01-02", periods=150)
        raw = valid_raw_config()
        raw["periods"] = {
            "research_start": dates[0].strftime("%Y-%m-%d"),
            "train_end": dates[89].strftime("%Y-%m-%d"),
            "validation_start": dates[90].strftime("%Y-%m-%d"),
            "validation_end": dates[119].strftime("%Y-%m-%d"),
            "test_start": dates[120].strftime("%Y-%m-%d"),
            "test_end": dates[-1].strftime("%Y-%m-%d"),
        }
        raw["baseline"].update({
            "train_days": 65,
            "retrain_frequency": 10,
            "top_n": 4,
            "rebalance_frequency": 5,
            "max_position_weight": 0.20,
            "leverage": 0.60,
            "min_avg_amount_20d": 1.0,
            "min_pullback_5d": 0.0,
            "max_pullback_5d": 1.0,
            "min_prior_return_20": -1.0,
            "min_prior_return_60": -1.0,
            "min_return_20d": -1.0,
            "min_return_60d": -1.0,
            "min_distance_ma60": -1.0,
            "max_intraday_return": 1.0,
        })
        raw["search_groups"] = [{
            "id": "risk_budget",
            "hypothesis_cn": "合成样本风险预算测试",
            "candidates": [{"id": "risk_065", "overrides": {"leverage": 0.65}}],
        }]
        raw["selection"].update({
            "min_validation_days": 20,
            "min_test_days": 15,
            "max_drawdown_floor": -0.90,
            "min_annualized_return_delta": 0.0,
            "min_sharpe_delta": -10.0,
            "max_turnover_ratio": 10.0,
            "rolling_window_days": 10,
            "max_negative_window_rate": 1.0,
        })
        rows: list[dict[str, object]] = []
        for symbol_index in range(32):
            symbol = f"{symbol_index + 1:06d}"
            for day_index, date in enumerate(dates):
                base = 8.0 + symbol_index * 0.08
                close = base * (1.0 + day_index * 0.001) * (
                    1.0 + 0.03 * np.sin(day_index / 6.0 + symbol_index * 0.2)
                )
                open_price = close * (1.0 - 0.002 * np.cos(day_index / 5.0))
                volume = 1_000_000.0 + symbol_index * 1_000.0
                rows.append({
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                })

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "panel.csv"
            config_path = root / "evolution.yaml"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config_path.write_text(
                yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            config = load_evolution_config(config_path)
            outcome = run_evolution(
                config=config,
                data_path=data_path,
                config_path=config_path,
                benchmark_path=None,
                asof_date=dates[-1],
                output_root=root / "runs",
                run_id="real-engine-smoke",
                resume=False,
                git_commit="test-commit",
            )

            self.assertTrue((outcome.run_dir / "manifest.json").exists())
            self.assertTrue((outcome.run_dir / "champion_candidate.yaml").exists())
            self.assertTrue((outcome.run_dir / "test_comparison.csv").exists())
            self.assertTrue((outcome.run_dir / "final" / "baseline" / "equity_curve.csv").exists())
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: import failure for `parse_args` or missing default config.

- [ ] **Step 3: Add the default YAML configuration exactly as approved**

Create `configs/evolution_strong_pullback.yaml` with this complete content. Runtime defaults absent from the YAML are materialized by `DEFAULT_STRATEGY_PARAMS` and written into `resolved_config.yaml`.

```yaml
strategy: strong_pullback_satellite

periods:
  research_start: 2022-01-01
  train_end: 2024-12-31
  validation_start: 2025-01-01
  validation_end: 2025-12-31
  test_start: 2026-01-01
  test_end:

baseline:
  train_days: 252
  retrain_frequency: 20
  top_n: 8
  rebalance_frequency: 5
  max_position_weight: 0.08
  leverage: 0.60
  min_score:
  commission_bps: 1.0
  impact_bps: 0.7
  max_buy_open_gap: 0.05
  limit_buffer: 0.995
  min_close: 2.0
  min_avg_amount_20d: 30000000
  min_pullback_5d: 0.03
  max_pullback_5d: 0.18
  min_prior_return_20: 0.08
  min_prior_return_60: 0.18
  min_return_20d: -0.12
  min_return_60d: 0.0
  min_distance_ma60: -0.10
  max_intraday_return: 0.05
  rebound_exit_return:
  rebound_exit_scale: 0.0
  rebound_exit_market_exposure_max:
  rebound_exit_market_exposure_min:

search_groups:
  - id: risk_budget
    hypothesis_cn: 适度扩大风险敞口能否提高收益且不突破回撤边界
    candidates:
      - id: risk_075
        overrides:
          leverage: 0.75
          max_position_weight: 0.10
      - id: risk_090
        overrides:
          leverage: 0.90
          max_position_weight: 0.12
  - id: entry_depth
    hypothesis_cn: 收紧回调深度能否减少单边下跌标的
    candidates:
      - id: pullback_02_12
        overrides:
          min_pullback_5d: 0.02
          max_pullback_5d: 0.12
      - id: pullback_04_15
        overrides:
          min_pullback_5d: 0.04
          max_pullback_5d: 0.15
  - id: rebound_exit
    hypothesis_cn: 强市场中的反弹退出能否改善收益回撤比
    candidates:
      - id: rebound_080_strong
        overrides:
          rebound_exit_return: 0.08
          rebound_exit_market_exposure_min: 0.99
      - id: rebound_085_strong
        overrides:
          rebound_exit_return: 0.085
          rebound_exit_market_exposure_min: 0.99
      - id: rebound_095_all
        overrides:
          rebound_exit_return: 0.095

selection:
  min_validation_days: 100
  min_test_days: 60
  max_drawdown_floor: -0.40
  min_annualized_return_delta: 0.01
  min_sharpe_delta: -0.10
  max_turnover_ratio: 1.50
  rolling_window_days: 63
  max_negative_window_rate: 0.60
```

- [ ] **Step 4: Add a testable CLI and main entrypoint**

```python
import argparse

from strong_pullback_evolution import load_evolution_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded evolution for the strong-pullback satellite strategy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--output-root", default="outputs/evolution_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", default=None, metavar="RUN_ID")
    return parser.parse_args(argv)


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        text=True, capture_output=True, check=False,
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
    run_id = args.resume or args.run_id or f"strong_pullback_{asof_date:%Y%m%d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
```

`run_evolution` writes `resolved_config.yaml` immediately after the running manifest. Its content includes `strategy`, ISO date strings, `baseline` after defaults, all search groups, and all selection rules.

- [ ] **Step 5: Update the operating documentation**

Add this command to `README.md` and `MODEL_OPERATING_SYSTEM.md` under a new “月度自进化研究” section:

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_YYYYMMDD.csv `
  --asof-date YYYY-MM-DD
```

Document that this is a manual or monthly research command, does not run in the daily default chain, and never changes production configuration automatically.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: all tests in the file pass.

- [ ] **Step 7: Run the CLI help smoke test**

Run: `python run_strong_pullback_evolution.py --help`

Expected: exit code `0`; output lists `--config`, required `--data`, `--benchmark`, `--asof-date`, `--output-root`, `--run-id`, and `--resume`.

- [ ] **Step 8: Run the complete repository test suite**

Run: `python -m pytest -q`

Expected: all existing and new tests pass with zero failures.

- [ ] **Step 9: Check repository diff and generated-file hygiene**

Run: `git diff --check`

Expected: no output and exit code `0`.

Run: `git status --short`

Expected: only the planned source, config, documentation, and test files are modified; no market-data or `outputs/` artifacts are staged.

- [ ] **Step 10: Commit the completed v1 interface**

```powershell
git add configs/evolution_strong_pullback.yaml run_strong_pullback_evolution.py strong_pullback_evolution.py tests/test_strong_pullback_evolution.py README.md MODEL_OPERATING_SYSTEM.md
git commit -m "feat: add guarded strong pullback evolution"
```

---

## Final Verification Checklist

- [ ] Read `docs/superpowers/specs/2026-07-10-strong-pullback-self-evolution-design.md` and map every acceptance criterion to a passing test or generated artifact.
- [ ] Confirm the first bundle request ends at `validation_end` and the holdout bundle is requested only after the champion is fixed.
- [ ] Confirm `latest.json` changes only after a successful run.
- [ ] Confirm `champion_candidate.yaml` is new output and no existing YAML is overwritten.
- [ ] Confirm failed candidates are recorded and do not abort a group while at least one candidate succeeds.
- [ ] Confirm a failed baseline, an all-failed group, or a failed final test marks the manifest `failed`.
- [ ] Confirm all tests pass from a fresh `python -m pytest -q` run.
- [ ] Confirm no broker, order, credential, or account code was introduced.
