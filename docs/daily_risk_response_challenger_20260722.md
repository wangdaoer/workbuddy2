# Daily Decrease-Only Risk Response (2026-07-22)

> **Invalidated on 2026-07-23.** This run used a panel containing 4,579 rows
> mislabeled as the 2026-06-19 session, when the A-share market was closed.
> Its metrics are not valid evidence. Use
> `docs/daily_risk_response_calendar_clean_20260723.md` instead.

Status: rejected. No production, shadow, ranking, or position effect.

## Contract

The rule and sequential gates were frozen before execution in
`configs/daily_risk_response_preregistration_20260722.json`.

- Alpha selection and the normal five-session rebalance remain unchanged.
- A non-scheduled session may only reduce existing long positions when the
  market-exposure target falls below its running ceiling since the last normal
  rebalance.
- Risk-only sessions cannot buy or restore exposure. Recovery waits for the next
  normal rebalance.
- Costs and next-open execution constraints are identical to the incumbent.
- A passing result could only enter prospective shadow observation.

This experiment uses previously observed history and is not fresh out-of-sample
evidence.

## Control Reconciliation

The scheduled-only control reproduced the corrected incumbent exactly:

| Metric | Control |
| --- | ---: |
| Total return | 27.71% |
| Annualized return | 7.55% |
| Maximum drawdown | -28.26% |
| Sharpe-like | 0.460 |
| Average turnover | 29.46% |
| Zero-target non-scheduled sessions retaining positions | 2 |

## Results

| Metric | Scheduled only | Daily decrease only |
| --- | ---: | ---: |
| Total return | 27.71% | 17.33% |
| Annualized return | 7.55% | 4.87% |
| Maximum drawdown | -28.26% | -29.79% |
| Sharpe-like | 0.460 | 0.342 |
| Average turnover | 29.46% | 29.58% |
| Independent risk-target decreases | 20 | 20 |
| Risk-reduction execution attempts | 0 | 21 |
| Zero-target sessions left unattempted | 2 | 0 |

Operationally, the candidate worked as designed: it attempted daily risk sales,
removed both unattempted zero-target sessions, and left no unexplained residual.
It nevertheless failed three frozen performance gates:

- Total-return retention was `62.57%`, below the required `95%`.
- Maximum drawdown worsened by `1.53` percentage points, above the allowed
  `0.50` percentage points.
- Sharpe retention was `74.36%`, below the required `90%`.

Annualized return and turnover gates passed.

## Failure Attribution

There were 20 independent risk-target decreases. The candidate produced 21
execution attempts because the constrained 2026-02-02 signal required one retry
on 2026-02-03. The 21 execution rows improved relative log return by about
`0.51%` in aggregate. The following 43 wait-for-rebalance rows lost about
`9.10%` of relative log return. All four calendar-year segments had negative
relative log-return contribution.

The first zero target was observed on the 2025-04-07 signal and appears on the
2025-04-09 realized-return row. The largest single wait-state loss was the next
realized row, 2025-04-10: the candidate remained out while the control earned
about `2.72%`, producing a relative log-return loss near `2.69%`. The rule
reduced same-day downside in some cases but systematically paid for delayed
re-entry.

## Decision

- Reject `daily_decrease_only` as champion, challenger, and shadow candidate.
- Keep the scheduled-only incumbent unchanged.
- Do not tune re-entry timing, exposure thresholds, or cooldown length on this
  observed sample to rescue the rejected rule.
- Preserve the separate trigger-day versus post-trigger-wait attribution as a
  required diagnostic for any future risk-response proposal.

Machine-readable evidence is stored in
`outputs/daily_risk_response_challenger_20260722`, including
`relative_phase_attribution.csv`.
