# Holding-Horizon Alignment Research (2026-07-22)

> **Calendar-contaminated result, invalidated on 2026-07-23.** The historical
> panel included a false 2026-06-19 session. The implementation finding remains
> testable, but all reported performance metrics require a calendar-clean replay.

Status: research only. Production promotion is not allowed by this study.

## Finding

The historical trainer used the IC row immediately before each signal date even
though its next-open exit was not observable yet. Strict training now purges all
labels whose exit open is unavailable at the signal-date close.

For a signal at position `i` and a holding horizon `h`, the latest mature label
is `i - h - 1`; the Python training slice must therefore end at `i - h`.

## Strict comparison

All variants use the same 2026-07-22 panel, stock universe, next-open execution,
40-stock portfolio, five-session rebalance schedule, 0.93 leverage, costs, and
market exposure rules. Portfolio PnL remains one-session next-open PnL; longer
horizons affect IC training only.

| Variant | Training horizons | Total return | Annualized return | Max drawdown | Sharpe-like | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Historical daily model | legacy one-day window | 20.89% | 5.81% | -28.62% | 0.385 | invalid as strict reference |
| Strict one-day | 1 | 27.71% | 7.55% | -28.26% | 0.460 | new research baseline |
| Strict multi-horizon | 1, 3, 5 | 21.24% | 5.93% | -28.34% | 0.390 | rejected |

Strict rows use the corrected initial-capital metric policy, which includes the
first realized trading return. The earlier presentation divided final equity by
the first post-trade equity observation. Net-value paths and decisions did not
change.

The multi-horizon challenger does not improve return, drawdown, or Sharpe-like
performance over the strict one-day baseline. It must not enter the daily
production workflow.

## Next action

Use the strict one-day result as the correctness baseline. The next independent
research change should separate trend-following and mean-reversion sleeves,
because the current learned weights are predominantly contrarian despite the
trend-momentum reporting label.
