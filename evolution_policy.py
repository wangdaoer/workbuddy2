"""Risk/execution firewall: lock risk & execution parameters across strategy evolution.

Adopted from model4's ``model4_evolution_policy.py`` (2026-08-05) -- the only
module model4 has that our root did not. Strategy candidates may change any
*logic* parameters, but may NEVER touch the locked risk/execution fields below.
Produces a fingerprint audit so any drift in these values is detectable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from strategy_evolution_core import fingerprint_payload


LOCKED_RISK_AND_EXECUTION_PARAMETERS = frozenset(
    {
        "max_position_weight",
        "leverage",
        "commission_bps",
        "impact_bps",
        "max_buy_open_gap",
        "limit_buffer",
        "initial_capital",
        "max_abs_daily_return",
        "market_ma_window",
        "market_risk_off_drawdown_20d",
        "market_below_ma_exposure",
        "market_crash_exposure",
        "regime_strong_leverage",
        "regime_exceptional_leverage",
        "regime_strong_breadth_threshold",
        "regime_exceptional_breadth_threshold",
        "regime_strong_volatility_max",
        "regime_exceptional_volatility_max",
        "basket_guard_scale",
        "rebound_exit_scale",
    }
)


def validate_candidate_overrides(overrides: Mapping[str, object]) -> None:
    if not isinstance(overrides, Mapping):
        raise TypeError("candidate overrides must be a mapping")
    locked = sorted(set(overrides) & LOCKED_RISK_AND_EXECUTION_PARAMETERS)
    if locked:
        raise ValueError(f"Locked risk/execution parameters: {locked}")


def assert_locked_parameters_unchanged(
    incumbent: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    changed = []
    for field in sorted(LOCKED_RISK_AND_EXECUTION_PARAMETERS):
        if field not in incumbent:
            continue
        if field not in candidate or candidate[field] != incumbent[field]:
            changed.append(field)
    if changed:
        raise ValueError(f"Locked risk/execution parameters changed: {changed}")


def build_risk_firewall_audit(
    baseline: Mapping[str, object],
    candidate_overrides: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(baseline, Mapping):
        raise TypeError("baseline must be a mapping")
    override_fields: set[str] = set()
    candidate_count = 0
    for overrides in candidate_overrides:
        validate_candidate_overrides(overrides)
        override_fields.update(overrides)
        candidate_count += 1

    locked_values = {
        field: baseline[field]
        for field in sorted(LOCKED_RISK_AND_EXECUTION_PARAMETERS)
        if field in baseline
    }
    locked_values_fingerprint = fingerprint_payload(locked_values)
    policy_fingerprint = fingerprint_payload(
        {
            "mode": "fixed_risk_and_execution",
            "locked_fields": sorted(LOCKED_RISK_AND_EXECUTION_PARAMETERS),
            "locked_values_fingerprint": locked_values_fingerprint,
        }
    )
    return {
        "schema_version": 1,
        "status": "enforced",
        "mode": "fixed_risk_and_execution",
        "candidate_count": candidate_count,
        "candidate_override_fields": sorted(override_fields),
        "locked_fields": sorted(LOCKED_RISK_AND_EXECUTION_PARAMETERS),
        "locked_values": locked_values,
        "locked_values_fingerprint": locked_values_fingerprint,
        "policy_fingerprint": policy_fingerprint,
    }
