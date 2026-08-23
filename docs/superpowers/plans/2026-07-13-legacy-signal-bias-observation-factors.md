# Legacy Signal Bias Observation Factors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how universe construction and execution assumptions created the legacy high-return curves, then generate point-in-time, research-only observation factors from the reusable signal components.

**Architecture:** Extend the existing backtest engine only where an isolated comparison requires it: universe membership lag and execution-block diagnostics. Keep feature construction and audit orchestration in two new focused Python modules so the current daily pipeline and candidate list remain unchanged. Run every ablation from one base configuration and mutate only the audited assumption.

**Tech Stack:** Python 3, pandas, NumPy, PyYAML, unittest/pytest, existing `run_backtest.py` and `execution_rules.py`.

## Global Constraints

- Every feature dated `t` uses data no later than the close of `t`; the earliest evaluation or action date is `t+1`.
- The first release writes only independent audit and shadow-observation outputs.
- Outputs must include `research_only=true` and `trade_instruction=false`.
- Missing optional amount or money-flow inputs must be reported as unavailable, never silently fabricated.
- Forward-return labels and next-day opening gaps must never enter the observation score.
- Do not modify the current production candidate list, simulation account, or trade instructions.

## File Structure

- Modify `run_backtest.py`: add lagged dynamic-universe membership and aggregate execution-block metrics.
- Modify `execution_rules.py`: expose deterministic masks and counts for open-execution constraints.
- Modify `tests/test_run_backtest.py`: cover membership lag and engine diagnostics.
- Modify `tests/test_execution_rules.py`: cover each independent block reason.
- Create `legacy_observation_factors.py`: validate panels, construct point-in-time factors, score observations, and measure static-universe look-through.
- Create `tests/test_legacy_observation_factors.py`: verify timing invariance, breakout shift, optional fields, and research-only output semantics.
- Create `run_legacy_signal_bias_audit.py`: run same-parameter ablations and write CSV/JSON/Markdown artifacts.
- Create `tests/test_legacy_signal_bias_audit.py`: verify case construction, single-variable overrides, verdicts, and artifact metadata.

---

### Task 1: Lagged Dynamic Universe Membership

**Files:**
- Modify: `run_backtest.py:24-113`
- Modify: `run_backtest.py:315-343`
- Modify: `tests/test_run_backtest.py`

**Interfaces:**
- Consumes: YAML field `universe.selection_lag_days` with default `0`.
- Produces: `StrategyConfig.universe_selection_lag_days: int` and a lagged boolean universe panel from `BacktestEngine._precompute_universe_panel()`.

- [ ] **Step 1: Write the failing configuration and lag tests**

```python
def test_strategy_config_defaults_dynamic_universe_lag_to_zero():
    cfg = load_config(Path("configs/high_risk_strategy_high_return_v2_dynamic_universe.yaml"))
    assert cfg.universe_selection_lag_days == 0


def test_dynamic_universe_lag_uses_previous_trading_day_membership():
    engine = BacktestEngine(make_config(
        universe_dynamic_top_n=1,
        universe_selection_mode="high_return",
        universe_selection_min_history=2,
        universe_selection_lag_days=1,
    ))
    dates = pd.date_range("2026-01-01", periods=65, freq="D")
    close = pd.DataFrame({
        "000001": np.linspace(10, 30, 65),
        "000002": np.r_[np.linspace(10, 11, 64), 40],
    }, index=dates)
    amount = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    lagged = engine._precompute_universe_panel(close, amount)
    same_day = BacktestEngine(make_config(
        universe_dynamic_top_n=1,
        universe_selection_mode="high_return",
        universe_selection_min_history=2,
        universe_selection_lag_days=0,
    ))._precompute_universe_panel(close, amount)
    pd.testing.assert_series_equal(lagged.iloc[-1], same_day.iloc[-2], check_names=False)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_run_backtest.py -q`

Expected: FAIL because `universe_selection_lag_days` is not accepted or defined.

- [ ] **Step 3: Add the configuration field and shift membership**

```python
# Add beside the existing universe selection fields in StrategyConfig:
universe_selection_lag_days: int

# Add beside universe_selection_min_history in StrategyConfig.from_dict():
universe_selection_lag_days=max(
    0, int(cfg.get("universe", {}).get("selection_lag_days", 0))
),

# At the end of _precompute_universe_panel:
membership = rank.le(top_n)
lag_days = self.cfg.universe_selection_lag_days
if lag_days:
    membership = membership.shift(lag_days).fillna(False).astype(bool)
return membership
```

- [ ] **Step 4: Run focused and existing backtest tests**

Run: `python -m pytest tests/test_run_backtest.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the isolated change**

```powershell
git add run_backtest.py tests/test_run_backtest.py
git commit -m "feat: add point-in-time universe lag"
```

### Task 2: Execution Constraint Diagnostics

**Files:**
- Modify: `execution_rules.py`
- Modify: `run_backtest.py:180-220`
- Modify: `run_backtest.py:646-719`
- Modify: `tests/test_execution_rules.py`
- Modify: `tests/test_run_backtest.py`

**Interfaces:**
- Produces: `open_constraint_masks(current, target, open_row, prev_close_row, max_buy_open_gap, limit_buffer, block_limit_up_buys=True, block_limit_down_sells=True) -> dict[str, pd.Series]`.
- Produces: `apply_open_constraints_with_diagnostics(current, target, open_row, prev_close_row, max_buy_open_gap, limit_buffer, block_limit_up_buys=True, block_limit_down_sells=True) -> tuple[pd.Series, dict[str, int]]`.
- Adds metric keys `blocked_limit_up_buys`, `blocked_limit_down_sells`, and `blocked_open_gap_buys` to next-open results.

- [ ] **Step 1: Write failing mask and count tests**

```python
def test_open_constraint_diagnostics_separate_limit_and_gap_reasons():
    current = pd.Series({"000001": 0.0, "000002": 0.2, "300001": 0.0})
    target = pd.Series({"000001": 0.2, "000002": 0.0, "300001": 0.2})
    prev_close = pd.Series(10.0, index=target.index)
    open_row = pd.Series({"000001": 11.0, "000002": 9.0, "300001": 11.6})
    adjusted, counts = apply_open_constraints_with_diagnostics(
        current, target, open_row, prev_close,
        max_buy_open_gap=0.15, limit_buffer=0.995,
    )
    assert counts == {
        "blocked_limit_up_buys": 1,
        "blocked_limit_down_sells": 1,
        "blocked_open_gap_buys": 1,
    }
    assert adjusted.equals(current)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_execution_rules.py -q`

Expected: FAIL importing `apply_open_constraints_with_diagnostics`.

- [ ] **Step 3: Implement shared masks and diagnostics without changing wrapper behavior**

```python
def open_constraint_masks(current, target, open_row, prev_close_row,
                          max_buy_open_gap, limit_buffer,
                          block_limit_up_buys=True,
                          block_limit_down_sells=True):
    current = current.reindex(target.index).fillna(0.0)
    gap = open_row.reindex(target.index) / (
        prev_close_row.reindex(target.index) + 1e-12
    ) - 1.0
    limit = limit_thresholds(target.index)
    increasing = target.gt(current)
    decreasing = target.lt(current)
    return {
        "blocked_limit_up_buys": increasing & gap.ge(limit * limit_buffer)
        if block_limit_up_buys else pd.Series(False, index=target.index),
        "blocked_limit_down_sells": decreasing & gap.le(-limit * limit_buffer)
        if block_limit_down_sells else pd.Series(False, index=target.index),
        "blocked_open_gap_buys": increasing & gap.gt(max_buy_open_gap)
        if max_buy_open_gap is not None else pd.Series(False, index=target.index),
    }


def apply_open_constraints_with_diagnostics(
    current, target, open_row, prev_close_row,
    max_buy_open_gap, limit_buffer,
    block_limit_up_buys=True,
    block_limit_down_sells=True,
):
    masks = open_constraint_masks(
        current=current,
        target=target,
        open_row=open_row,
        prev_close_row=prev_close_row,
        max_buy_open_gap=max_buy_open_gap,
        limit_buffer=limit_buffer,
        block_limit_up_buys=block_limit_up_buys,
        block_limit_down_sells=block_limit_down_sells,
    )
    blocked = masks["blocked_limit_up_buys"] | masks["blocked_limit_down_sells"] | masks["blocked_open_gap_buys"]
    adjusted = target.where(~blocked, current.reindex(target.index).fillna(0.0))
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    return adjusted, counts
```

Keep `apply_open_constraints()` as a compatibility wrapper returning only the adjusted target.

- [ ] **Step 4: Accumulate diagnostics in `BacktestEngine`**

Initialize the three counters to zero. During each next-open execution, add the returned counts. Include them in the metrics dictionary. Close-to-close cases report all three counters as zero.

- [ ] **Step 5: Run execution and engine tests**

Run: `python -m pytest tests/test_execution_rules.py tests/test_run_backtest.py -q`

Expected: PASS with the existing constraint behavior unchanged.

- [ ] **Step 6: Commit the diagnostics**

```powershell
git add execution_rules.py run_backtest.py tests/test_execution_rules.py tests/test_run_backtest.py
git commit -m "feat: report blocked backtest executions"
```

### Task 3: Point-in-Time Observation Factor Library

**Files:**
- Create: `legacy_observation_factors.py`
- Create: `tests/test_legacy_observation_factors.py`

**Interfaces:**
- Produces: `validate_panel(df: pd.DataFrame) -> pd.DataFrame`.
- Produces: `compute_observation_factors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]`.
- Produces: `audit_static_universe(selected: pd.DataFrame, broad: pd.DataFrame) -> dict[str, object]`.
- Produces columns `momentum_20`, `momentum_60`, `breakout_distance_20`, `trend_acceleration`, `volatility_20`, `liquidity_20`, `liquidity_stability_20`, `flow_persistence`, eligibility fields, `observation_score`, `research_only`, and `trade_instruction`.

- [ ] **Step 1: Write failing timing, validation, and scoring tests**

```python
def make_panel(periods=80):
    dates = pd.date_range("2026-01-01", periods=periods, freq="D")
    rows = []
    for symbol, offset in [("000001", 0.0), ("300001", 2.0)]:
        for idx, date in enumerate(dates):
            close = 10.0 + offset + idx * 0.10
            rows.append({
                "date": date,
                "symbol": symbol,
                "close": close,
                "open": close * 0.999,
                "amount": 10_000_000.0 + idx * 1_000.0,
            })
    return pd.DataFrame(rows)


def make_extreme_future_rows(base):
    last_date = pd.Timestamp(base["date"].max())
    rows = []
    for symbol in sorted(base["symbol"].unique()):
        rows.append({
            "date": last_date + pd.Timedelta(days=1),
            "symbol": symbol,
            "close": 1_000_000.0,
            "open": 1_000_000.0,
            "amount": 1_000_000_000.0,
        })
    return pd.DataFrame(rows)


def make_monotonic_panel(periods=25):
    dates = pd.date_range("2026-01-01", periods=periods, freq="D")
    close = np.arange(10.0, 10.0 + periods)
    return pd.DataFrame({
        "date": dates,
        "symbol": "000001",
        "close": close,
        "open": close,
        "amount": 10_000_000.0,
    })


def test_future_rows_cannot_change_existing_observation_factors():
    base = make_panel(periods=80)
    before, _ = compute_observation_factors(base)
    changed = pd.concat([base, make_extreme_future_rows(base)])
    after, _ = compute_observation_factors(changed)
    cutoff = base["date"].max()
    cols = ["momentum_20", "momentum_60", "breakout_distance_20", "trend_acceleration", "observation_score"]
    pd.testing.assert_frame_equal(
        before.loc[before.date.eq(cutoff), cols].reset_index(drop=True),
        after.loc[after.date.eq(cutoff), cols].reset_index(drop=True),
    )


def test_breakout_uses_prior_twenty_day_high():
    panel = make_monotonic_panel(periods=25)
    factors, _ = compute_observation_factors(panel)
    row = factors.iloc[-1]
    prior_high = panel.sort_values("date")["close"].iloc[-21:-1].max()
    assert row["breakout_distance_20"] == pytest.approx(row["close"] / prior_high - 1.0)


def test_missing_money_flow_is_marked_unavailable():
    factors, meta = compute_observation_factors(make_panel(periods=80).drop(columns=["amount"]))
    assert factors["flow_persistence"].isna().all()
    assert meta["factor_availability"]["flow_persistence"] is False


def test_output_never_contains_trade_instruction():
    factors, _ = compute_observation_factors(make_panel(periods=80))
    assert factors["research_only"].eq(True).all()
    assert factors["trade_instruction"].eq(False).all()
    assert "opening_gap_risk" not in OBSERVATION_SCORE_INPUTS
```

Add duplicate date-symbol and missing required-column rejection tests.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_legacy_observation_factors.py -q`

Expected: FAIL because `legacy_observation_factors` does not exist.

- [ ] **Step 3: Implement panel validation and trailing features**

Use stable long-form group operations sorted by `symbol,date`:

```python
grouped = out.groupby("symbol", sort=False)
out["momentum_20"] = grouped["close"].pct_change(20, fill_method=None)
out["momentum_60"] = grouped["close"].pct_change(60, fill_method=None)
prior_high = grouped["close"].transform(lambda s: s.rolling(20).max().shift(1))
out["breakout_distance_20"] = out["close"] / (prior_high + 1e-12) - 1.0
ret_3 = grouped["close"].pct_change(3, fill_method=None)
ret_6 = grouped["close"].pct_change(6, fill_method=None)
out["trend_acceleration"] = ret_3 - ret_6 / 2.0
daily_return = grouped["close"].pct_change(fill_method=None)
out["volatility_20"] = daily_return.groupby(out["symbol"]).transform(
    lambda s: s.rolling(20).std()
)
```

When `amount` exists, compute trailing median amount and coefficient-of-variation stability. When a recognized point-in-time money-flow column exists, compute its trailing positive-day share; otherwise leave `flow_persistence` as `NaN` and set availability false.

- [ ] **Step 4: Implement cross-sectional score and static-universe audit**

Rank each usable feature by date. Use positive weights for momentum, prior-high breakout, acceleration, liquidity, and liquidity stability; subtract volatility and capacity penalties. Do not include forward labels, next-day open, opening gaps, or limit outcomes.

The static audit must compute selected and broad full-period return medians, percentile distribution, overlap with broad full-period winners, and return `lookthrough_suspect=true` when the selected median percentile is at least `0.75`.

- [ ] **Step 5: Run factor tests**

Run: `python -m pytest tests/test_legacy_observation_factors.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the factor library**

```powershell
git add legacy_observation_factors.py tests/test_legacy_observation_factors.py
git commit -m "feat: add point-in-time legacy observation factors"
```

### Task 4: Same-Parameter Ablation Audit Runner

**Files:**
- Create: `run_legacy_signal_bias_audit.py`
- Create: `tests/test_legacy_signal_bias_audit.py`

**Interfaces:**
- Produces: `load_raw_yaml(path: Path) -> dict[str, object]`.
- Produces: `build_ablation_cases(base: dict) -> list[AuditCase]`.
- Produces: `run_ablation_case(case: AuditCase, data: pd.DataFrame) -> dict[str, object]`.
- Produces: `build_audit_verdict(cases: pd.DataFrame, universe_audit: dict) -> dict[str, object]`.
- CLI accepts `--broad-data`, `--selected-data`, `--base-config`, `--output-dir`, and optional `--skip-backtests`.

- [ ] **Step 1: Write failing same-parameter case tests**

```python
def test_ablation_cases_change_only_the_named_assumption():
    base = load_raw_yaml(Path("configs/high_risk_strategy_high_return_v2_dynamic_universe.yaml"))
    cases = {case.case_id: case for case in build_ablation_cases(base)}
    assert cases["dynamic_same_close"].config["portfolio"] == base["portfolio"]
    assert cases["dynamic_lag1_close"].config["universe"]["selection_lag_days"] == 1
    assert cases["dynamic_lag1_next_open"].config["execution"]["model"] == "next_open"
    assert cases["dynamic_lag1_next_open_limits"].config["execution"]["block_limit_up_buys"] is True
    assert cases["dynamic_lag1_next_open_limits_gap3"].config["execution"]["max_buy_open_gap"] == 0.03
    assert cases["dynamic_lag1_next_open_limits_gap3_cost2x"].config["cost"]["commission_bps"] == base["cost"]["commission_bps"] * 2
```

Also assert that the base portfolio, signal, and risk dictionaries remain identical across cases unless the case explicitly names cost.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_legacy_signal_bias_audit.py -q`

Expected: FAIL because the audit runner does not exist.

- [ ] **Step 3: Implement immutable case construction**

Use `copy.deepcopy(base)` for each case. Build this ordered ladder:

```python
@dataclass(frozen=True)
class AuditCase:
    case_id: str
    changed_assumption: str
    config: dict[str, object]


def load_raw_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config: {path}")
    return loaded
```

1. `dynamic_same_close`.
2. `dynamic_lag1_close`.
3. `dynamic_lag1_next_open`.
4. `dynamic_lag1_next_open_limits`.
5. `dynamic_lag1_next_open_limits_gap3`.
6. `dynamic_lag1_next_open_limits_gap3_cost2x`.

Each row stores total return, annualized return, max drawdown, turnover, gross exposure, three blocked-order counts, final equity, and delta versus the preceding case.

- [ ] **Step 4: Implement factor and artifact outputs**

Write these exact files under the output directory:

- `legacy_signal_bias_audit.md`
- `legacy_signal_bias_audit.csv`
- `legacy_signal_bias_audit.json`
- `observation_factors_latest.csv`
- `observation_factor_history.csv`
- `observation_factor_dictionary.json`

The JSON root includes:

```python
{
    "research_only": True,
    "trade_instruction": False,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "inputs": input_metadata,
    "universe_audit": universe_audit,
    "ablation_results": case_rows,
    "reusable_factors": [
        "momentum_20", "momentum_60", "breakout_distance_20",
        "trend_acceleration", "liquidity_20", "liquidity_stability_20",
    ],
    "rejected_assumptions": [
        "static_full_period_winner_universe", "same_close_fill",
        "unrestricted_limit_fill", "leverage_as_alpha",
    ],
}
```

The Chinese Markdown report must label dynamic trailing momentum, prior-high breakout, acceleration, and liquidity as observation candidates. It must reject static full-period winner membership, same-close fill, unrestricted limit fills, and leverage as alpha.

- [ ] **Step 5: Run audit-runner tests**

Run: `python -m pytest tests/test_legacy_signal_bias_audit.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the runner**

```powershell
git add run_legacy_signal_bias_audit.py tests/test_legacy_signal_bias_audit.py
git commit -m "feat: add legacy signal ablation audit"
```

### Task 5: Full Local Run and Verification

**Files:**
- Generate: `outputs/high_return_v2/legacy_signal_bias_audit/*`
- Modify only if verification exposes a tested defect: files from Tasks 1-4 and their matching tests.

**Interfaces:**
- Consumes the local broad and selected historical panels and the legacy dynamic-universe configuration.
- Produces the final research-only audit package and latest shadow observation ranking.

- [ ] **Step 1: Run all focused tests before the expensive backtests**

Run:

```powershell
python -m pytest tests/test_execution_rules.py tests/test_run_backtest.py tests/test_legacy_observation_factors.py tests/test_legacy_signal_bias_audit.py -q
```

Expected: all focused tests PASS with no warnings attributable to the new code.

- [ ] **Step 2: Generate the real audit package**

Run:

```powershell
python .\run_legacy_signal_bias_audit.py `
  --broad-data .\data_panel_history_main_chinext_20220101_20260626.csv `
  --selected-data .\data_panel_history_high_return_top300_20220101_20260626.csv `
  --base-config .\configs\high_risk_strategy_high_return_v2_dynamic_universe.yaml `
  --output-dir .\outputs\high_return_v2\legacy_signal_bias_audit
```

Expected: exit code 0 and all six required artifacts exist.

- [ ] **Step 3: Verify output contracts**

Check that:

- The JSON has `research_only=true` and `trade_instruction=false`.
- All ablation rows share identical signal, portfolio, and risk fingerprints.
- The lagged and strict cases have non-empty metrics.
- The strict limit case reports blocked orders when the historical data contains qualifying gaps.
- The latest observation file contains no buy/sell action column.
- The factor dictionary marks forward return and opening gap as evaluation-only fields.
- The report explicitly separates verified evidence, interpretation, and unavailable fields.

- [ ] **Step 4: Run regression tests**

Run: `python -m pytest -q`

Expected: the complete repository test suite passes. If unrelated pre-existing tests fail, record their exact names and rerun all tests touched by this plan to confirm they remain green.

- [ ] **Step 5: Commit any verification-driven tested fixes**

```powershell
git add run_backtest.py execution_rules.py legacy_observation_factors.py run_legacy_signal_bias_audit.py tests/test_execution_rules.py tests/test_run_backtest.py tests/test_legacy_observation_factors.py tests/test_legacy_signal_bias_audit.py
git commit -m "fix: finalize legacy signal bias audit"
```

Skip this commit when verification required no source changes.
