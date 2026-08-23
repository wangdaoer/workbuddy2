# Hybrid Pullback Attribution Research (2026-07-22)

> **Calendar-contaminated result, invalidated on 2026-07-23.** The historical
> panel included a false 2026-06-19 session. Do not use this report's metrics or
> rejection decision until the frozen experiment is replayed on the clean panel.

Status: rejected research experiment. No production, ranking, or portfolio-weight effect.

## Contract

The experiment was frozen before the result was generated in
`configs/hybrid_pullback_preregistration_20260722.json`. It tests one primary
hypothesis only: `strong_pullback_20_5` with a fixed positive direction. The
other correlated pullback derivatives were not eligible for winner selection.

The candidate uses the incumbent's strict mature one-day next-open label,
universe, 40-stock cap, five-session rebalance, 0.93 leverage, market filter,
next-open constraints, and normal trading costs. A second run doubles both
commission and impact. This is exploratory attribution on previously observed
history, not fresh out-of-sample evidence.

## Results

| Variant | Total return | Annualized return | Max drawdown | Sharpe-like | Average exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict incumbent | 27.71% | 7.55% | -28.26% | 0.460 | 78.79% |
| Hybrid pullback, normal cost | -51.05% | -19.15% | -54.13% | -0.575 | 78.51% |
| Hybrid pullback, double cost | -53.05% | -20.15% | -55.85% | -0.618 | 78.51% |

Candidate calendar returns were negative in every observed year: `-12.73%`
in 2023, `-27.06%` in 2024, `-10.06%` in 2025, and `-13.23%` in 2026 YTD.
The candidate/incumbent daily-return correlation was `0.650` over 847 common
sessions.

## Gate Decision

Every preregistered attribution gate that depends on portfolio performance
failed:

| Gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Total return | >= 0.00% | -51.05% | fail |
| Annualized return | >= 0.00% | -19.15% | fail |
| Sharpe-like | >= 0.000 | -0.575 | fail |
| Maximum drawdown floor | >= -35.00% | -54.13% | fail |
| Drawdown worsening vs incumbent | <= 5.00pp | 25.86pp | fail |
| Positive calendar-year ratio | >= 50.00% | 0.00% | fail |
| Double-cost total return | >= 0.00% | -53.05% | fail |

Metrics use the corrected initial-capital policy and include the first realized
trading return. Equity paths and gate decisions did not change.

The replacement gates therefore were not opened.

## IC/Portfolio Divergence

The factor's available-day mean one-day RankIC was `0.0166`, and its positive
IC ratio was `57.96%`. All 43 retraining points assigned the factor a positive
weight, yet the executable portfolio lost money in every calendar segment.
This is evidence that a mildly positive all-universe rank relationship does not
guarantee profitable top-tail selection after next-open timing, market exposure,
turnover, and execution constraints.

The current interaction also overlaps its windows: `momentum_20` contains the
same recent five sessions represented inversely by `reversal_5`. Multiplying
their percentile ranks can favor deeper declines that retain only residual
20-day strength. The observed result is consistent with falling-price exposure,
not a reliable definition of capital absorption.

## Decision

- Reject `hybrid_pullback` as an attribution factor and incumbent replacement.
- Do not select `strong_pullback_60_5`, `breakout_pullback_20_5`, or
  `liquid_pullback` after seeing this result.
- Keep the strict unconstrained one-day model as production champion.
- Do not backfill historical money-flow confirmation. The panel's
  `main_net_inflow` and `main_net_volume_ratio` history is materially incomplete
  before 2026-06-22, so those fields remain forward-observation evidence only.
- Any trend-gated pullback follow-up must be a new frozen experiment and must
  beat the incumbent on executable portfolio metrics, not RankIC alone.
