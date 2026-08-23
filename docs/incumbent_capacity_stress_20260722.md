# Incumbent Capacity Stress (2026-07-22)

> **Invalidated on 2026-07-23.** This run used a panel containing 4,579 rows
> mislabeled as the 2026-06-19 session, when the A-share market was closed.
> Its metrics are not valid evidence. Use
> `docs/incumbent_capacity_stress_calendar_clean_20260723.md` instead.

Status: completed research-only execution audit. No production ranking, position,
or execution setting changed.

## Contract

The scenarios and gates were frozen before valid execution in
`configs/incumbent_capacity_stress_preregistration_20260722.json`.

- Capacity uses the signal-date-known trailing 20-session median traded amount.
- A symbol may use at most 5% of that amount per day.
- Buy and sell changes are both partially filled; missing or nonpositive amount
  provides zero capacity.
- Low-liquidity symbols remain in ranking so the test measures execution damage
  instead of silently improving the universe.
- The no-capacity control must reproduce the registered incumbent before any
  scenario is accepted.

This audit uses previously observed history. It estimates execution capacity and
does not provide fresh out-of-sample alpha evidence.

## Control Reconciliation

The valid run reproduced the corrected incumbent exactly:

| Metric | Registered | Reproduced |
| --- | ---: | ---: |
| Total return | 27.71% | 27.71% |
| Annualized return | 7.55% | 7.55% |
| Maximum drawdown | -28.26% | -28.26% |
| Sharpe-like | 0.460 | 0.460 |
| Average gross exposure | 78.79% | 78.79% |

An earlier run was discarded before interpretation. A research-only feature had
entered the default `unconstrained` sleeve through automatic feature discovery,
so its no-pressure path did not reproduce the incumbent. Production now uses an
explicit 15-feature whitelist, and the runner fails closed on any future control
mismatch.

## Results

| Capital | Total return | Annualized | Max drawdown | Return retention | Exposure retention | Fill ratio | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CNY 10m | 29.66% | 8.03% | -28.56% | 107.06% | 100.68% | 99.57% | pass |
| CNY 50m | 30.19% | 8.16% | -26.17% | 108.96% | 96.10% | 95.56% | pass |
| CNY 100m | 26.24% | 7.18% | -22.11% | 94.70% | 81.13% | 84.50% | pass |
| CNY 500m | 4.01% | 1.18% | -8.13% | 14.48% | 29.04% | 40.35% | fail |

The largest preregistered scenario passing every frozen gate is CNY 100m. It is
close to the 80% minimum exposure-retention boundary, so it is a coarse historical
upper bound rather than a recommended operating size.

The 10m and 50m paths slightly outperform the control because partial fills alter
the timing and composition of holdings. This is path-dependent execution noise,
not evidence that capacity constraints create alpha. No strategy promotion or
expected-return uplift is assigned to it.

`avg_market_exposure_target` is the benchmark-regime overlay target and must not
be read as deployed capital. Actual deployment is represented by
`avg_gross_exposure`; capacity severity also requires the fill ratio because a
count of limited sessions alone does not describe how much trading was blocked.

## Decision

- Retain the incumbent unchanged.
- Record CNY 100m only as the largest passing preregistered historical scenario.
- Treat CNY 500m as failed because both return and exposure retention collapse.
- Use a materially lower operational size for any future simulation, with an
  explicit liquidity reserve and prospective fill tracking.
- Do not convert this result into a live-trading instruction or a return promise.

Machine-readable outputs are in
`outputs/incumbent_capacity_stress_20260722/summary.json`; scenario equity and
trade-audit files are stored in the same directory.
