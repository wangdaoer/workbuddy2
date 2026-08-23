# A股通用策略进化内核 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有强势回调策略增加市场无关、可复现、可审计的多滚动折影子进化内核，同时保留现有 A 股回测、严格留出测试和人工确认边界。

**Architecture:** 新建纯标准库模块 `strategy_evolution_core.py` 管理指纹、逐折晋级门槛和 champion/shadow/rollback 状态；`strong_pullback_evolution.py` 负责把 A 股交易日与回测结果适配成通用折指标；`run_strong_pullback_evolution.py` 负责运行产物和可选的原子状态写入。默认 dry-run，不修改正式 YAML。

**Tech Stack:** Python 3.12、标准库 dataclasses/hashlib/json/pathlib/tempfile、pandas、PyYAML、unittest/pytest。

## Global Constraints

- 通用内核不得 import pandas、行情模块或策略执行器。
- 强势回调策略必须继续调用现有 A 股回测函数，不能复制交易逻辑。
- 默认只写单次运行目录；只有显式 `--promote-shadow` 才可更新独立影子状态。
- 即使影子晋级，也不得修改 `configs/*.yaml` 或每日正式策略。
- 不接券商、不下单、不生成订单、不自由修改策略 Python 代码。
- 训练、验证、测试按实际交易日物理切片，测试期不得反向参与候选搜索。
- 缺失、非有限或样本不足的指标必须判定为 `insufficient_evidence`。

---

### Task 1: 市场无关的进化内核

**Files:**
- Create: `strategy_evolution_core.py`
- Create: `tests/test_strategy_evolution_core.py`

**Interfaces:**
- Produces: `fingerprint_payload(value: object) -> str`
- Produces: `generate_parameter_candidates(champion_parameters, candidate_overrides, max_candidates, seed) -> tuple[dict[str, object], ...]`
- Produces: `FoldMetrics`, `PromotionPolicy`, `PromotionDecision`, `EvolutionState`
- Produces: `evaluate_candidate(challenger, champion, policy) -> PromotionDecision`
- Produces: `promote_to_shadow(state, challenger_version, challenger_parameters, experiment_id, run_id, data_fingerprint, now=None) -> EvolutionState`
- Produces: `rollback_shadow(state, reason, now=None) -> EvolutionState`

- [ ] **Step 1: Write failing fingerprint and promotion tests**

```python
from strategy_evolution_core import (
    FoldMetrics,
    PromotionPolicy,
    evaluate_candidate,
    fingerprint_payload,
)


def _fold(fold_id: str, total_return: float, *, trades: int = 20) -> FoldMetrics:
    return FoldMetrics(
        fold_id=fold_id,
        total_return=total_return,
        max_drawdown=-0.12,
        sharpe=1.1,
        filled_trades=trades,
        average_turnover=0.20,
        pnl_concentration=0.30,
    )


def test_fingerprint_is_canonical_and_sensitive():
    assert fingerprint_payload({"a": 1, "b": [2, 3]}) == fingerprint_payload({"b": [2, 3], "a": 1})
    assert fingerprint_payload({"a": 1}) != fingerprint_payload({"a": 2})


def test_candidate_passes_all_fold_gates():
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(3))
    challenger = tuple(_fold(f"f{i}", 0.08) for i in range(3))
    decision = evaluate_candidate(challenger, champion, PromotionPolicy())
    assert decision.status == "eligible_for_shadow"
    assert decision.failed_gates == ()


def test_candidate_with_too_few_trades_is_insufficient_evidence():
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(3))
    challenger = tuple(_fold(f"f{i}", 0.08, trades=1) for i in range(3))
    decision = evaluate_candidate(challenger, champion, PromotionPolicy(min_filled_trades_per_fold=5))
    assert decision.status == "insufficient_evidence"
    assert "min_filled_trades" in decision.failed_gates


def test_candidate_generation_is_seeded_and_does_not_mutate_champion():
    champion = {"leverage": 0.60, "top_n": 8}
    overrides = (
        {"leverage": 0.75},
        {"leverage": 0.90},
        {"top_n": 6},
        {"top_n": 10},
    )
    first = generate_parameter_candidates(champion, overrides, max_candidates=4, seed=20260712)
    second = generate_parameter_candidates(champion, overrides, max_candidates=4, seed=20260712)
    assert first == second
    assert len(first) == 4
    assert champion == {"leverage": 0.60, "top_n": 8}
    assert all(candidate != champion for candidate in first)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_strategy_evolution_core.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'strategy_evolution_core'`.

- [ ] **Step 3: Implement immutable metrics, policy, canonical fingerprint, and gate audit**

```python
@dataclass(frozen=True)
class FoldMetrics:
    fold_id: str
    total_return: float
    max_drawdown: float
    sharpe: float
    filled_trades: int
    average_turnover: float
    pnl_concentration: float


@dataclass(frozen=True)
class PromotionPolicy:
    min_folds: int = 3
    min_filled_trades_per_fold: int = 5
    min_positive_fold_ratio: float = 2.0 / 3.0
    min_mean_return_improvement: float = 0.01
    max_drawdown_floor: float = -0.40
    max_drawdown_worsening: float = 0.05
    max_turnover_ratio: float = 1.50
    max_pnl_concentration: float = 0.50


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    failed_gates: tuple[str, ...]
    gates: dict[str, dict[str, object]]
```

`evaluate_candidate()` must align folds by `fold_id`, reject duplicate or mismatched folds, distinguish `insufficient_evidence` from `rejected`, and evaluate every configured hard gate without short-circuiting so the report contains the full failure set.

`generate_parameter_candidates()` must validate each override as a non-empty mapping, apply every override independently to a fresh champion copy, reject conflicting duplicate fingerprints, remove an unchanged champion, shuffle with a local `random.Random(seed)`, and return at most `max_candidates` independent dictionaries. The strong-pullback adapter will feed it one existing hypothesis group at a time, preserving the current one-hypothesis-per-round contract.

- [ ] **Step 4: Add failing boundary, NaN, shadow, and rollback tests**

```python
def test_non_finite_metric_cannot_promote():
    challenger = (_fold("f1", float("nan")), _fold("f2", 0.1), _fold("f3", 0.1))
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(1, 4))
    assert evaluate_candidate(challenger, champion, PromotionPolicy()).status == "insufficient_evidence"


def test_shadow_promotion_and_rollback_restore_previous_champion():
    initial = EvolutionState.initial("v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00")
    shadow = promote_to_shadow(
        initial,
        challenger_version="v2",
        challenger_parameters={"leverage": 0.75},
        experiment_id="exp-1",
        run_id="run-1",
        data_fingerprint="data-1",
        now="2026-07-12T01:00:00+00:00",
    )
    restored = rollback_shadow(shadow, reason="drawdown", now="2026-07-12T02:00:00+00:00")
    assert shadow.shadow_status == "shadow"
    assert restored.champion_version == "v1"
    assert restored.shadow_status == "rolled_back"
    assert restored.blocked_reason == "drawdown"
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_strategy_evolution_core.py -q`

Expected: all tests pass with no warnings.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- strategy_evolution_core.py tests/test_strategy_evolution_core.py
git commit -m "feat: add generic strategy evolution core"
```

---

### Task 2: A股交易日滚动折与指标适配

**Files:**
- Modify: `strong_pullback_evolution.py`
- Modify: `run_strong_pullback_satellite.py`
- Modify: `tests/test_strong_pullback_evolution.py`
- Modify: `tests/test_strong_pullback_satellite.py`

**Interfaces:**
- Consumes: `FoldMetrics` from Task 1
- Produces: `TradingFold`
- Produces: `build_trading_folds(trading_dates, train_days, validation_days, test_days, step_days) -> tuple[TradingFold, ...]`
- Produces: `calculate_fold_metrics(equity, trades, fold) -> FoldMetrics`

- [ ] **Step 1: Write failing actual-trading-day fold tests**

```python
def test_build_trading_folds_uses_actual_index_positions():
    dates = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-16"])
    folds = build_trading_folds(dates, train_days=2, validation_days=2, test_days=1, step_days=1)
    assert folds[0].train_dates == (pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"))
    assert folds[0].validation_dates == (pd.Timestamp("2026-01-08"), pd.Timestamp("2026-01-09"))
    assert folds[0].test_dates == (pd.Timestamp("2026-01-12"),)


def test_build_trading_folds_rejects_duplicate_dates():
    dates = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-05"])
    with pytest.raises(ValueError, match="unique"):
        build_trading_folds(dates, train_days=1, validation_days=1, test_days=1, step_days=1)
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run: `python -m pytest tests/test_strong_pullback_evolution.py -q`

Expected: import failure for `TradingFold` or `build_trading_folds`.

- [ ] **Step 3: Implement `TradingFold` and deterministic fold construction**

```python
@dataclass(frozen=True)
class TradingFold:
    fold_id: str
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]
```

The builder must sort and normalize the supplied trading dates, reject duplicates/NaT/non-positive window sizes, generate expanding chronological windows by index position, and return an empty tuple when no complete fold fits.

- [ ] **Step 4: Write failing concentration and physical-slice tests**

```python
def test_calculate_fold_metrics_uses_realized_symbol_pnl_concentration():
    equity = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "gross_return": [0.0, 0.02, 0.01],
        "cost": [0.0, 0.0, 0.0],
        "turnover": [0.0, 0.2, 0.1],
        "gross_exposure": [0.0, 0.6, 0.6],
    })
    trades = pd.DataFrame({
        "symbol_contributions_json": [
            '{"000001": 0.012, "000002": 0.008}',
            '{"000001": 0.006, "000002": 0.004}',
        ]
    })
    metrics = calculate_fold_metrics(equity, trades, fold_id="f1")
    assert metrics.pnl_concentration == pytest.approx(0.60)


def test_fold_runner_never_receives_rows_after_test_end():
    seen = []
    run_strong_pullback_folds(panel, folds, lambda sliced, params: seen.append(sliced["date"].max()) or synthetic_run, params)
    assert seen == [fold.test_end for fold in folds]
```

- [ ] **Step 5: Implement metrics adapter and fold runner**

Before calculating concentration, add a non-behavioral audit field to each existing `trade_rows` entry in `run_strong_pullback_satellite.py`:

```python
symbol_contributions = (target * realized).replace(0.0, np.nan).dropna()
trade_row["symbol_contributions_json"] = json.dumps(
    {str(symbol).zfill(6): float(value) for symbol, value in symbol_contributions.items()},
    sort_keys=True,
    ensure_ascii=True,
)
```

Add a satellite regression test asserting that the sum of decoded daily symbol contributions equals `gross_return` within floating-point tolerance. `calculate_fold_metrics()` must reuse `calculate_segment_metrics()` for return/risk fields, aggregate only positive contribution by symbol for concentration, and set `pnl_concentration` to non-finite evidence when no positive contribution exists. `run_strong_pullback_folds()` must pass a panel physically truncated at each `fold.test_end` to the injected existing strategy executor.

- [ ] **Step 6: Run Task 2 and existing evolution tests**

Run: `python -m pytest tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py -q`

Expected: all tests pass, including existing holdout/resume checks.

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- strong_pullback_evolution.py run_strong_pullback_satellite.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py
git commit -m "feat: add A-share rolling fold adapter"
```

---

### Task 3: 影子状态持久化与命令行接入

**Files:**
- Modify: `strategy_evolution_core.py`
- Modify: `strong_pullback_evolution.py`
- Modify: `run_strong_pullback_evolution.py`
- Modify: `configs/evolution_strong_pullback.yaml`
- Modify: `tests/test_strategy_evolution_core.py`
- Modify: `tests/test_strong_pullback_evolution.py`

**Interfaces:**
- Consumes: Task 1 state and Task 2 fold evaluations
- Produces: `EvolutionCoreConfig` parsed as part of existing `EvolutionConfig`
- Produces: `load_evolution_state(path, default_state) -> EvolutionState`
- Produces: `write_evolution_state_atomic(path, state, expected_previous_fingerprint=None) -> str`
- Produces: `persist_evolution_outcome(run_dir, snapshot, candidate_scores, experiments, decision_markdown, dry_run, promote_shadow, state_path, expected_previous_fingerprint) -> bool`
- Extends CLI: `--dry-run`, `--promote-shadow`, `--state-path`

- [ ] **Step 1: Write failing atomic-state tests**

```python
def test_atomic_state_write_round_trips_and_checks_previous_fingerprint(tmp_path):
    path = tmp_path / "state.json"
    first = EvolutionState.initial("v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00")
    first_fp = write_evolution_state_atomic(path, first)
    assert load_evolution_state(path, first) == first

    second = replace(first, last_completed_run_id="run-2")
    with pytest.raises(EvolutionStateError, match="previous state fingerprint"):
        write_evolution_state_atomic(path, second, expected_previous_fingerprint="stale")
    assert write_evolution_state_atomic(path, second, expected_previous_fingerprint=first_fp)
```

- [ ] **Step 2: Run focused state test and verify RED**

Run: `python -m pytest tests/test_strategy_evolution_core.py -q`

Expected: import failure for state read/write functions.

- [ ] **Step 3: Implement strict schema read and atomic replacement**

Use `json.dumps(..., sort_keys=True, ensure_ascii=True)`, write a sibling temporary file with UTF-8, flush and close it, then replace the destination with `Path.replace()`. Reject unknown/missing fields and unsupported `schema_version`; compare the actual previous canonical fingerprint before replacing an existing state.

- [ ] **Step 4: Write failing CLI safety and artifact tests**

```python
def test_cli_defaults_to_dry_run_and_requires_explicit_shadow_promotion():
    args = parse_args(["--config", "configs/evolution_strong_pullback.yaml", "--data", "panel.csv"])
    assert args.dry_run is True
    assert args.promote_shadow is False


def test_dry_run_writes_snapshot_without_changing_global_state(tmp_path):
    state_path = tmp_path / "global" / "strong_pullback.json"
    run_dir = tmp_path / "run"
    snapshot = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )
    changed = persist_evolution_outcome(
        run_dir=run_dir,
        snapshot=snapshot,
        candidate_scores=[{"candidate_id": "c1", "status": "rejected"}],
        experiments=[{"experiment_id": "e1", "status": "rejected"}],
        decision_markdown="# 影子决定\n\n纸面研究，未更新全局状态。\n",
        dry_run=True,
        promote_shadow=False,
        state_path=state_path,
        expected_previous_fingerprint=None,
    )
    assert (run_dir / "evolution_state_snapshot.json").exists()
    assert (run_dir / "candidate_scores.csv").exists()
    assert not state_path.exists()
    assert changed is False
```

- [ ] **Step 5: Extend config and runner without changing formal YAML**

Add an `evolution_core` config block:

```yaml
evolution_core:
  max_candidates_per_group: 8
  random_seed: 20260712
  min_folds: 3
  min_filled_trades_per_fold: 5
  min_positive_fold_ratio: 0.6666666667
  min_mean_return_improvement: 0.01
  max_drawdown_floor: -0.40
  max_drawdown_worsening: 0.05
  max_turnover_ratio: 1.50
  max_pnl_concentration: 0.50
```

Add `EvolutionCoreConfig` as a frozen dataclass in `strong_pullback_evolution.py` and make `parse_evolution_config()` reject missing/unknown keys, booleans passed as integers, `min_folds < 1`, ratios outside `[0, 1]`, positive drawdown floors, and non-positive candidate/trade limits. The disjoint `periods` dates are the sole orchestration timing contract; legacy `evolution_core` walk-forward window fields are not accepted. The checked-in selection rules use `min_validation_days: 100` and `rolling_window_days: 63` for the 117-session 2025-01-01..2025-06-30 period without changing risk gates. Extend the existing default-config test to assert the parsed values.

For each existing `search_groups` entry, the runner must pass that group's explicit overrides to `generate_parameter_candidates()` using a seed derived from `random_seed` and the stable group ID fingerprint. Candidates remain independent copies of the current champion, so random ordering cannot create path dependence within a group. The runner must always write run-local `folds.json`, `candidate_scores.csv`, `experiments/*.json`, `evolution_state_snapshot.json`, and `shadow_decision.md`. Only `promote_shadow=True` with `dry_run=False` may call `write_evolution_state_atomic()`. It must never open a path under `configs/` for writing.

- [ ] **Step 6: Run CLI/state tests and regressions**

Run: `python -m pytest tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py -q`

Expected: all tests pass, and no test creates or modifies a repository `configs/*.yaml` file.

- [ ] **Step 7: Commit Task 3**

```powershell
git add -- strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py configs/evolution_strong_pullback.yaml tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py
git commit -m "feat: add shadow evolution state workflow"
```

---

### Task 4: 全链路验证与研究演练

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-12-generic-strategy-evolution-core-design.md` only if verified behavior requires wording correction

**Interfaces:**
- Consumes: all prior tasks
- Produces: verified dry-run artifacts under `outputs/evolution_runs/<run_id>/`

- [ ] **Step 1: Add exact research-only usage documentation**

Document these commands and state boundaries:

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_20260710.csv `
  --benchmark D:\codex\daily-market-data\benchmarks\510300.csv `
  --asof-date 2026-07-10 `
  --dry-run
```

Shadow registry update must be documented as a separate, explicit research command using `--no-dry-run --promote-shadow`; the documentation must state that it still does not change formal YAML or produce broker orders.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py -q`

Expected: all focused tests pass.

- [ ] **Step 3: Run the full project regression suite**

Run: `python -m pytest -q`

Expected: zero failures and zero errors. Existing unrelated warnings must be reported rather than hidden.

- [ ] **Step 4: Run compile verification**

Run: `python -m compileall strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py -q`

Expected: exit code 0.

- [ ] **Step 5: Run a real dry-run against the latest confirmed panel**

Run the documented command using the newest existing `data_panel_history_main_chinext_*.csv` whose maximum date matches its filename. If benchmark coverage is missing, stop and report the mismatch rather than silently omitting the benchmark.

Verify:

```text
global_state_changed == false
formal configs unchanged
folds.json exists and has at least min_folds complete folds
candidate_scores.csv contains every candidate and failed-gate audit
evolution_state_snapshot.json contains the input data fingerprint
shadow_decision.md labels the run paper-only
```

- [ ] **Step 6: Verify determinism**

Repeat the same dry-run with a second run-local output root. Compare canonical JSON/CSV content after excluding timestamp and run-directory fields. Candidate order, experiment IDs, fold metrics, decision status, and state content must match exactly.

- [ ] **Step 7: Inspect scope and repository diff**

Run:

```powershell
git diff --check
git status --short
git diff --name-only HEAD~3..HEAD
```

Confirm no broker module, daily pipeline file, formal production strategy config, or unrelated user file was modified by these tasks.

- [ ] **Step 8: Commit documentation if changed**

```powershell
git add -- README.md docs/superpowers/specs/2026-07-12-generic-strategy-evolution-core-design.md
git commit -m "docs: document shadow evolution workflow"
```

Skip this commit only when neither file changed.
