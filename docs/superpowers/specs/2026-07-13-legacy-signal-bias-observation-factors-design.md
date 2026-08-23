# Legacy Signal Bias Audit and Observation Factors

## Goal

Explain which parts of the legacy 10000%+ curves came from reusable price/volume information and which parts depended on biased universe construction or unrealistic execution. Convert only the reusable parts into research-only observation factors.

This work must not change the current production candidate list, simulation account, or trade instructions. The first version writes an independent audit package and a shadow observation table.

## Scope

The audit covers four primary risks:

1. Dynamic high-return universe construction.
2. Static-universe look-through, survivorship, and full-period selection.
3. Same-close versus next-open execution timing.
4. Limit-up buys and limit-down sells that cannot be executed.

Secondary sensitivity checks cover opening gaps, transaction costs, turnover, and simple liquidity capacity. Leverage is reported separately so it cannot be confused with signal quality.

## Inputs

- Legacy selected Top300 panel: `data_panel_history_high_return_top300_20220101_20260626.csv`.
- Broad historical panel: `data_panel_history_main_chinext_20220101_20260626.csv`.
- Legacy configurations under `configs/high_risk_strategy_high_return_v2*.yaml`.
- Existing backtest engine and execution constraints in `run_backtest.py` and `execution_rules.py`.
- Existing legacy reproduction audit under `outputs/high_return_v2/legacy_10000pct_reproduction_audit`.

Every output records the actual input path, date range, row count, symbol count, configuration fingerprint, and generation time.

## Data Timing Contract

For observation date `t`, every feature may use data dated no later than the close of `t`. The earliest actionable or evaluation date is `t+1`.

- Rolling returns use prices through `t` only.
- Breakout distance compares the close at `t` with a prior rolling high shifted by one day.
- Liquidity features use trailing observations through `t`.
- Forward returns are labels used only for evaluation and never enter factor construction or ranking.
- Static full-period winner lists may be audited but may not enter the clean observation factor.

## Bias Audit

The audit produces a comparable case table rather than a single headline return.

### Universe Cases

- `legacy_static_selected_top300`: the historical preselected panel, marked as look-through suspect.
- `broad_point_in_time`: all symbols available in the broad panel on each date, subject to minimum history and liquidity checks.
- `dynamic_trailing_only`: daily TopN selected only from trailing 20/60-day momentum, prior-high breakout distance, trailing volatility, and trailing amount.
- `dynamic_lagged_one_day`: the same universe eligibility shifted by one trading day. This is the clean reference for next-day observation.

The report compares full-period return distributions and membership overlap. It flags a static universe when its constituents have materially stronger full-period outcomes than the broad population.

### Execution Cases

- Legacy close-to-close execution.
- Next-open execution.
- Next-open with limit-up buy and limit-down sell blocks.
- Next-open with limit blocks and a maximum opening-gap threshold.
- Cost and liquidity sensitivity variants.

The audit reports total and annualized return, maximum drawdown, turnover, gross exposure, blocked-order counts, average opening gap, and the delta from the preceding case.

## Observation Factors

The clean factor table is research-only and contains one row per observation date and symbol.

Reusable features:

- `momentum_20`: trailing 20-day return.
- `momentum_60`: trailing 60-day return.
- `breakout_distance_20`: close versus the prior 20-day high.
- `trend_acceleration`: recent short-horizon return acceleration using only trailing closes.
- `volatility_20`: trailing daily-return volatility.
- `liquidity_20`: trailing median traded amount, log transformed.
- `liquidity_stability_20`: trailing amount stability.
- `flow_persistence`: included only when a point-in-time money-flow column is present; otherwise explicitly unavailable.

Risk and execution flags:

- `limit_up_risk` and `limit_down_risk` based on the applicable board price limit and prior close.
- `opening_gap_risk`, populated only when next-day open is evaluated; it is never used to rank date `t` observations.
- `capacity_risk` from position notional versus trailing traded amount.
- `history_eligible` and `liquidity_eligible`.

The composite observation score uses percentile ranks of momentum, breakout, acceleration, liquidity, and stability. Volatility and capacity are penalties. Limit and opening-gap fields remain execution-risk annotations; they do not create positive alpha.

## Outputs

Write under:

`outputs/high_return_v2/legacy_signal_bias_audit/`

- `legacy_signal_bias_audit.md`: Chinese decision report with bias findings and case deltas.
- `legacy_signal_bias_audit.csv`: comparable audit cases.
- `legacy_signal_bias_audit.json`: machine-readable metadata and verdicts.
- `observation_factors_latest.csv`: latest-date clean shadow factors and ranks.
- `observation_factor_history.csv`: dated factor history used for forward evaluation.
- `observation_factor_dictionary.json`: definitions, timing, direction, and availability.

All artifacts carry `research_only=true` and `trade_instruction=false`.

## Failure Handling

- Stop with a clear error when required price/date/symbol columns are missing.
- Mark optional amount or money-flow factors unavailable instead of silently filling them with fabricated values.
- Reject duplicate date-symbol rows unless an explicit deterministic deduplication rule is supplied.
- Reject forward-looking factor columns from the ranking input.
- Do not promote an audit case when exact source configuration or data fingerprint is missing.

## Tests and Acceptance

Automated tests must demonstrate:

1. Features at date `t` do not change when rows after `t` are modified.
2. Prior-high breakout is shifted and cannot see the current or future high.
3. Static full-period winners are flagged as look-through suspect.
4. Lagged dynamic membership uses the previous trading day's eligibility.
5. Limit-up buys and limit-down sells are blocked in strict execution cases.
6. Opening-gap data never enters the date-`t` observation score.
7. Optional money-flow absence is reported, not imputed.
8. Outputs are explicitly research-only and contain no buy/sell instruction.

The first release is accepted when the focused tests and existing backtest tests pass, the audit package is generated from local data, and the report clearly separates reusable factors from rejected assumptions.
