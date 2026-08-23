# Configuration index

All configurations are research or simulation artifacts. None authorizes live trading.

Status meanings:

- `active-research`: used by the current daily research workflow.
- `shadow`: evaluated without changing the formal ranking or placing orders.
- `experimental`: retained for active manual research.
- `rejected`: failed a preregistered or holdout gate.
- `legacy-reproduction`: retained only to reproduce historical results or audits.
- `utility`: schema or mapping input, not a strategy.
- `example`: example sweep or starter configuration.

| Configuration | Status | Entry point / evidence |
| --- | --- | --- |
| `personal_trade_habit_overlay.yaml` | active-research | `run_daily_model_pipeline.py`; personal overlay only |
| `factor_replacement_preregistration.json` | shadow | Frozen 2026-07-14 shortlist; unseen tracking starts 2026-07-15 |
| `hybrid_pullback_preregistration_20260722.json` | rejected | Single-factor attribution failed every portfolio gate; `docs/hybrid_pullback_attribution_research_20260722.md` |
| `trend_gated_pullback_preregistration_20260722.json` | rejected | Lower drawdown but failed 3/4 strict superiority gates; `docs/trend_gated_pullback_research_20260722.md` |
| `incumbent_capacity_stress_preregistration_20260722.json` | invalidated | Superseded after the non-session 2026-06-19 panel row was quarantined |
| `incumbent_capacity_stress_calendar_clean_replay_20260723.json` | experimental | Unchanged 10m/50m/100m/500m capacity replay on the calendar-clean panel |
| `daily_risk_response_preregistration_20260722.json` | invalidated | Superseded after the non-session 2026-06-19 panel row was quarantined |
| `daily_risk_response_calendar_clean_replay_20260723.json` | experimental | Unchanged daily decrease-only replay on the calendar-clean panel |
| `dynamic_breadth_overlay_preregistration.json` | shadow | Reset and frozen 2026-07-22 after strict label-maturity correction; 60-day forward observation starts 2026-07-23; automatic promotion disabled |
| `market_breadth_mismatch_observer_20260723.json` | shadow | Frozen diagnostic labels for weak MA60 breadth and high benchmark risk target; 60 matured signal outcomes required; no ranking, position, order, or automatic-promotion effect |
| `institutional_accumulation_shadow.yaml` | shadow | Frozen 2026-07-18 price-volume + THS flow proxy; prospective tracking starts 2026-07-20; no ranking effect |
| `evolution_strong_pullback.yaml` | shadow | `run_strong_pullback_evolution.py`; `docs/superpowers/specs/2026-07-10-strong-pullback-self-evolution-design.md` |
| `evolution_multifactor_observation.yaml` | shadow | `run_multifactor_observation_evolution.py`; observation-only gates |
| `trend_ignition_shortlist_preregistered.yaml` | rejected | `docs/research-results/trend_ignition_shortlist_20260714_v1.md` |
| `external_strategy_absorption.yaml` | experimental | `docs/external_strategy_absorption_20260709.md` |
| `field_map_stock_targets.yaml` | utility | field mapping for target exports |
| `prelive_gf_manual.yaml` | utility | Guangfa manual-review pre-live draft; 100% is a hard ceiling, approved benchmark risk sets the effective exposure, and whole-lot allocation is order-independent; no broker connection or order placement |
| `high_risk_strategy.yaml` | example | baseline `run_backtest.py` example |
| `high_risk_strategy_high_return.yaml` | legacy-reproduction | historical high-return research |
| `high_risk_strategy_high_return_toy.yaml` | example | reduced toy research |
| `high_risk_strategy_high_return_v2.yaml` | legacy-reproduction | historical v2 research |
| `high_risk_strategy_high_return_v2_selected.yaml` | legacy-reproduction | selected historical universe audit |
| `high_risk_strategy_high_return_v2_dynamic_universe.yaml` | legacy-reproduction | legacy bias audit baseline |
| `high_risk_strategy_high_return_v2_dynamic_universe_balanced.yaml` | legacy-reproduction | historical balanced variant |
| `high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30.yaml` | legacy-reproduction | historical drawdown variant |
| `high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30_v2.yaml` | legacy-reproduction | historical drawdown variant v2 |
| `high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30_v3.yaml` | legacy-reproduction | historical drawdown variant v3 |
| `high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30_v3_next_open.yaml` | legacy-reproduction | historical next-open variant |
| `high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30_v3_next_open_gap3.yaml` | legacy-reproduction | historical next-open gap variant |
| `high_risk_strategy_high_return_v2_dynamic_universe_conservative.yaml` | legacy-reproduction | historical conservative variant |
| `high_risk_strategy_high_return_v2_toy.yaml` | example | v2 toy research |
| `high_risk_strategy_high_return_v2_toy_small.yaml` | example | reduced v2 toy research |
| `high_risk_strategy_high_return_v2_toy_small_best.yaml` | legacy-reproduction | retained toy winner, not approved |
| `high_risk_strategy_next_open_reversal_dynamic.yaml` | experimental | manual next-open reversal research |
| `high_risk_strategy_next_open_reversal_dynamic_low_turnover.yaml` | experimental | low-turnover reversal research |
| `high_risk_strategy_next_open_trend_dynamic.yaml` | experimental | manual next-open trend research |
| `sweep_grid_example.yaml` | example | generic sweep example |
| `sweep_grid_high_return.yaml` | legacy-reproduction | historical high-return sweep |
| `sweep_grid_high_return_expanded.yaml` | legacy-reproduction | expanded historical sweep |
| `sweep_grid_high_return_fast.yaml` | legacy-reproduction | fast historical sweep |
| `sweep_grid_high_return_toy_fast.yaml` | example | toy fast sweep |

The current daily rank model is configured by explicit `PipelineConfig` and CLI fields in
`run_daily_model_pipeline.py`; it does not silently select one of the
`high_risk_strategy_high_return_v2*` files.

`prelive_gf_manual.yaml` is also the single source for the forward whole-lot shadow-account
execution assumptions: initial cash, commission, minimum commission, sell stamp tax,
slippage, open-gap limit, and price-limit buffer. The account executes only a prior trading
day's manual draft and remains research-only.

## Concentrated return research

`research_concentrated_return_frontier.py` pairs 3/5/8/10-stock portfolios
with feasible position caps and evaluates normal costs, doubled costs, and a
3% next-open gap limit. It is research-only and cannot promote a strategy into
the daily production pipeline. The 900% cumulative-return threshold is an
exploration target, not a promised result.

`run_legacy_alpha_strict_replay.py` is the frozen follow-up audit. It replays
the legacy momentum/trend/reversal/low-volatility/liquidity Top10 score under
the current next-open execution contract and compares only the preregistered
control, market-filter, and breadth-filter variants. Its outputs are historical
research evidence only and cannot update daily production state.

`track_dynamic_breadth_overlay.py` is the forward-only observer for the daily
rank incumbent and breadth-guard challenger. It excludes every date before
2026-07-23 from gate statistics, requires 60 valid shared trading days, and
cannot promote or modify the production model automatically.

`track_market_breadth_mismatch.py` records only the latest source-date snapshot
seen on each run. Recorded breadth and regime-target values are immutable; a
return is attached later only through the explicit
`trade_audit.signal_date -> realize_date` mapping. The observer starts on
2026-07-23, requires 60 matured outcomes before diagnostic review, and never
changes selection, position sizing, order drafts, or the incumbent model.
