# Signal Sleeve Split Research (2026-07-22)

> **Calendar-contaminated result, invalidated on 2026-07-23.** The historical
> panel included a false 2026-06-19 session. Do not use this report's metrics or
> rejection decision until the frozen experiment is replayed on the clean panel.

Status: rejected research experiment. No production or portfolio-weight effect.

## Contract

All variants use the strict mature one-day next-open label, identical universe,
40-stock cap, five-session rebalance schedule, 0.93 leverage, costs, next-open
constraints, and market exposure rules.

Economic feature directions are immutable. A feature whose mature mean IC has
the wrong sign receives zero weight. If every feature is unavailable, the sleeve
holds cash rather than selecting arbitrary stocks.

### Pure trend sleeve

- Positive: `momentum_20`, `momentum_60`, `breakout_20`, `distance_ma20`

### Pure mean-reversion sleeve

- Positive: `reversal_5`
- Negative: `intraday_return`, `close_position`

Hybrid pullback features and derived duplicates are excluded from both sleeves.

## Results

| Variant | Total return | Annualized return | Max drawdown | Sharpe-like | Average exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict incumbent | 27.71% | 7.55% | -28.26% | 0.460 | 78.79% |
| Pure trend | -28.25% | -9.42% | -28.83% | -1.349 | 9.09% |
| Pure mean reversion | 8.75% | 2.53% | -30.82% | 0.227 | 78.54% |
| Fixed 50/50 sleeve blend | -9.03% | -2.78% | -22.49% | -0.121 | derived |

The trend/mean-reversion daily-return correlation is `0.194`. Low correlation
does not rescue a blend whose trend sleeve loses money and whose reversal sleeve
underperforms the incumbent with a worse drawdown.

## Calendar attribution

| Year | Incumbent | Trend | Mean reversion | 50/50 blend | Blend excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | -0.61% | -5.58% | 2.36% | -1.43% | -0.83% |
| 2024 | -2.28% | -24.01% | 13.13% | -6.08% | -3.80% |
| 2025 | 37.33% | 0.00% | 15.16% | 8.16% | -29.17% |
| 2026 YTD | -4.26% | 0.00% | -18.45% | -9.15% | -4.89% |

All rows use the corrected initial-capital metric policy. The earlier table
omitted the first realized trading return; equity paths and rejection decisions
did not change.

## Decision

- Reject the pure trend sleeve under the one-day label.
- Reject the pure mean-reversion sleeve as an incumbent replacement.
- Reject the fixed sleeve blend; do not run parameter optimization or forward promotion.
- Retain the strict unconstrained one-day model as production champion.
- The result shows that the incumbent's value is not explained by either pure
  atomic sleeve alone. Hybrid pullback interactions may contain useful signal,
  but they require a separately preregistered experiment and duplicate-exposure
  controls.
