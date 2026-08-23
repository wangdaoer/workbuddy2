# Trend-Gated Pullback Research (2026-07-22)

> **Calendar-contaminated result, invalidated on 2026-07-23.** The historical
> panel included a false 2026-06-19 session. Do not use this report's metrics or
> rejection decision until the frozen experiment is replayed on the clean panel.

Status: rejected as an incumbent replacement. No production, ranking, shadow,
or portfolio-weight effect.

## Contract

The formula and sequential gates were frozen before execution in
`configs/trend_gated_pullback_preregistration_20260722.json`.

- Eligibility requires `momentum_20 > 0` and `breakout_20 > -0.10`.
- Eligible stocks are ranked only by
  `reversal_5_rank * (1 - volatility_20_rank)`.
- The sleeve contains one positive-direction feature.
- Universe, strict mature one-day next-open label, 40-stock cap, five-session
  rebalance, 0.93 leverage, market filter, costs, and execution constraints are
  identical to the incumbent.
- Stage 2 robustness tests are forbidden after any stage 1 failure.

This is exploratory research on previously observed history. Even a full pass
could only qualify the candidate for new forward observation.

## Stage 1 Results

| Variant | Total return | Annualized return | Max drawdown | Sharpe-like | Average exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict incumbent | 27.71% | 7.55% | -28.26% | 0.460 | 78.79% |
| Trend-gated pullback | 20.63% | 5.74% | -20.04% | 0.423 | 78.79% |

| Frozen gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Total return | > 27.71% | 20.63% | fail |
| Annualized return | > 7.55% | 5.74% | fail |
| Maximum drawdown | >= -28.26% | -20.04% | pass |
| Sharpe-like | > 0.460 | 0.423 | fail |

The candidate improved maximum drawdown by about `8.23` percentage points but
gave up about `7.07` percentage points of total return and `1.81` percentage
points of annualized return. It failed three of four strict superiority gates.

Metrics use the corrected initial-capital policy and include the first realized
trading return. Equity paths and the three-of-four rejection did not change.

## Sequential Stop

Stage 2 was not opened. No double-cost run, three-percent open-gap run,
subperiod selection, factor-correlation gate, or top-decile-overlap gate was
used to rescue or reinterpret the failed candidate. This is the intended
fail-fast behavior, not missing evidence.

## Decision

- Reject `trend_gated_pullback` as production champion, challenger, or shadow.
- Retain the strict unconstrained one-day model as production champion.
- Do not tune the `0%` trend threshold, `-10%` near-high threshold, volatility
  window, or ranking weights after seeing this result.
- Record only the lower-drawdown behavior as a future defensive-risk hypothesis.
  Any such experiment requires a new registration and cannot reuse this sample
  as unseen evidence.
- Money-flow fields remain forward-observation-only because their historical
  coverage is materially incomplete before 2026-06-22.
