# World Event Shadow

`world_event_shadow.py` adds external macro and event context to the Model 3
daily audit. It is an observation layer only.

## Governance contract

- `research_only` is always `true`.
- `trade_instruction` is always `false`.
- `selection_effect` is always `false`.
- `portfolio_weight_effect` is always `0.0`.
- Missing or unavailable data is reported as degraded. It is never converted
  into a neutral or low-risk reading.
- Promotion into portfolio decisions requires a separate preregistration and
  prospective out-of-sample review.

## Live API

Set the API key in the environment without writing it into source control:

```powershell
$env:WORLDMONITOR_API_KEY = "wm_xxx"
python world_event_shadow.py `
  --asof-date 2026-07-22 `
  --config configs/world_event_shadow.yaml `
  --output outputs/high_return_v2/world_event_shadow_20260722.json
```

Without the key, the command still writes JSON, CSV and Chinese Markdown audit
artifacts with `status=degraded`.

The daily pipeline also supplies a local payload cache. A failed endpoint may
reuse a payload for at most `max_cache_age_hours`; its original observation
timestamp remains authoritative, so cached data is still labelled stale or
expired rather than fresh.

## Reviewed snapshot

For reproducible research, use a reviewed local JSON snapshot:

```powershell
python world_event_shadow.py `
  --asof-date 2026-07-22 `
  --config configs/world_event_shadow.yaml `
  --snapshot data/world_event_snapshot.json `
  --output outputs/high_return_v2/world_event_shadow_20260722.json
```

The snapshot may contain the documented World Monitor response objects under
`macro_signals`, `fear_greed`, and `economic_stress`. Reviewed normalized
0-100 observations can be supplied under `normalized`:

```json
{
  "macro_signals": {},
  "fear_greed": {},
  "economic_stress": {},
  "normalized": {
    "observed_at": "2026-07-22T08:30:00Z",
    "china_external_risk": 65,
    "energy_shock": 40,
    "shipping_disruption": 55,
    "trade_policy_pressure": 70
  }
}
```

## License boundary

The implementation consumes documented external contracts and does not copy
World Monitor source code. Do not vendor AGPL source into Model 3 without a
separate license review.
