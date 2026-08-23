# Incumbent Attribution Review (2026-07-22)

> **Invalidated on 2026-07-23.** This run used a panel containing 4,579 rows
> mislabeled as the 2026-06-19 session, when the A-share market was closed.
> Its metrics are not valid evidence. Use
> `docs/incumbent_attribution_calendar_clean_20260723.md` instead.

Status: research-only diagnosis. No production, ranking, or position-weight change.

## Scope

The strict one-day incumbent was decomposed from its recorded daily net returns,
costs, turnover, realized gross exposure, and market-exposure state. Conditional
log-return contributions reconcile to the full-period log return. Rolling factor
weights are reported only as scoring tendencies, not as factor PnL attribution.
The market-exposure state is the risk-overlay target; realized deployment is
reported separately as gross exposure.

## Findings

- Full-period return is `27.71%`, annualized return `7.55%`, maximum drawdown
  `-28.26%`, and Sharpe-like `0.460` under the corrected initial-capital policy.
- Only one of four calendar segments was positive. The 2025 return was `37.33%`;
  2023, 2024, and 2026 YTD were negative.
- The only positive year supplied `100%` of positive calendar log-return
  contribution. Cross-year stability is therefore not established.
- `risk_on` sessions supplied `82.84%` of full-period net log return and a
  conditional compounded return of `22.46%`. This state was not uniformly
  profitable: its 2023 and 2026 YTD segments were negative.
- `liquidity_20` had a negative weight at every retraining point. This is a
  persistent lower-liquidity tilt and requires capacity stress before any
  increase in capital or concentration.
- On 2025-04-09 and 2026-07-22 the market-risk signal was zero while actual
  gross exposure remained positive. Both were outside the five-session
  rebalance schedule, so they are deferred risk-response observations, not
  evidence of blocked selling. No risk-off rebalance day retained residual
  exposure in this sample.

## Flow Readiness

- `main_net_volume_ratio` has seven source sessions through 2026-07-22. It meets
  the five-session rolling minimum but remains a short forward-only history.
- The institutional-accumulation observer has `0/80` completed primary-horizon
  samples. Gate evaluation and promotion remain prohibited.
- Neither flow observer affects selection or portfolio weights.

## Decision

- Keep the strict incumbent unchanged.
- Do not treat the 2025 result as evidence of a stable annual return.
- The point-in-time capacity audit is complete; CNY 100m is only the largest
  passing frozen historical scenario, with a narrow exposure-retention margin.
- Prioritize a separately registered daily risk-response challenger before
  testing more concentrated portfolios.
- Continue flow observation until its frozen maturity gates are met.

Generated evidence is stored in `outputs/incumbent_attribution_20260722`.
