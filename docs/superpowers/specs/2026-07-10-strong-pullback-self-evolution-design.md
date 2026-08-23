# 强势回调策略自进化 v1 设计

> 发布状态说明（2026-07-13）：本文最初记录的是二段式“验证/测试”v1 方案。该历史方案已被三段式“选择期/锁定核心测试/最终保留集”实现取代；下文时间契约和命令语义已按当前实现校正，旧二段式片段不再是可执行规范。

## 目标

在现有 `strong_pullback_satellite` 回测引擎之外增加一个受约束的实验控制器。控制器自动执行滚动参数试验、验证集比较、版本留存和失败回滚，在最大回撤约束内优先寻找更高收益的强势回调策略候选。

本功能只用于本地研究和人工决策辅助，不连接券商、不自动下单，也不自动替换每日流水线当前使用的策略配置。

## 设计原则

- 自进化只修改配置参数，不修改 Python 策略代码。
- 每轮只检验一个明确假设组，避免无边界的全参数组合搜索。
- 研究阶段完全看不到测试期指标；测试期只比较最终冠军和初始基准。
- 验证集没有合格改进时保留当前版本，并记录回滚原因。
- 高收益优先，但最大回撤不得低于 `-40%`，且不能用换手率或单一窗口异常膨胀换取表面收益。
- 所有结果必须可复现、可审计，并保留输入数据、Git 版本和解析后参数的证据。

## 范围

### v1 包含

- `strong_pullback_satellite` 策略参数的分组迭代。
- 训练记录期、参数选择期、锁定核心测试期和最终保留集的严格隔离。
- 基准、每轮试验和最终冠军的指标、净值、交易与候选明细。
- 自动保留或回滚研究版本。
- 最终测试比较和中文总结报告。
- 支持中断后按试验标识继续运行。

### v1 不包含

- AI 自由生成或修改策略代码。
- Optuna、贝叶斯优化或遗传算法等额外优化依赖。
- 自动提升为每日正式配置。
- 每日自动调参。
- 券商连接、订单生成或资金账户操作。
- 同时进化趋势动量、隐性吸筹等其他策略家族。

## 文件结构

- `run_strong_pullback_evolution.py`：命令行入口、配置解析、实验编排和产物写入。
- `strong_pullback_evolution.py`：时间切分、候选生成、指标计算、晋级判定和最终测试判定。
- `configs/evolution_strong_pullback.yaml`：基准参数、时间区间、搜索组和门槛。
- `tests/test_strong_pullback_evolution.py`：纯函数和小样本端到端测试。
- `outputs/evolution_runs/<run_id>/`：单次进化运行的版本目录。

现有 `run_strong_pullback_satellite.py` 继续作为唯一策略执行逻辑来源。新控制器直接复用其公开函数，不复制标签、涨跌停约束或交易成本逻辑。

## 配置协议

`configs/evolution_strong_pullback.yaml` 使用以下顶层结构：

```yaml
strategy: strong_pullback_satellite

periods:
  research_start: 2022-01-01
  train_end: 2024-12-31
  validation_start: 2025-01-01
  validation_end: 2025-06-30
  core_test_start: 2025-07-01
  core_test_end: 2025-12-31
  test_start: 2026-01-01
  test_end: null

baseline:
  train_days: 252
  retrain_frequency: 20
  top_n: 8
  rebalance_frequency: 5
  max_position_weight: 0.08
  leverage: 0.60
  min_score: null
  commission_bps: 1.0
  impact_bps: 0.7
  max_buy_open_gap: 0.05
  limit_buffer: 0.995
  min_close: 2.0
  min_avg_amount_20d: 30000000
  min_pullback_5d: 0.03
  max_pullback_5d: 0.18
  min_prior_return_20: 0.08
  min_prior_return_60: 0.18
  min_return_20d: -0.12
  min_return_60d: 0.0
  min_distance_ma60: -0.10
  max_intraday_return: 0.05
  rebound_exit_return: null
  rebound_exit_scale: 0.0
  rebound_exit_market_exposure_max: null
  rebound_exit_market_exposure_min: null

search_groups:
  - id: risk_budget
    hypothesis_cn: 适度扩大风险敞口能否提高收益且不突破回撤边界
    candidates:
      - id: risk_075
        overrides: {leverage: 0.75, max_position_weight: 0.10}
      - id: risk_090
        overrides: {leverage: 0.90, max_position_weight: 0.12}
  - id: entry_depth
    hypothesis_cn: 收紧回调深度能否减少单边下跌标的
    candidates:
      - id: pullback_02_12
        overrides: {min_pullback_5d: 0.02, max_pullback_5d: 0.12}
      - id: pullback_04_15
        overrides: {min_pullback_5d: 0.04, max_pullback_5d: 0.15}
  - id: rebound_exit
    hypothesis_cn: 强市场中的反弹退出能否改善收益回撤比
    candidates:
      - id: rebound_080_strong
        overrides: {rebound_exit_return: 0.08, rebound_exit_market_exposure_min: 0.99}
      - id: rebound_085_strong
        overrides: {rebound_exit_return: 0.085, rebound_exit_market_exposure_min: 0.99}
      - id: rebound_095_all
        overrides: {rebound_exit_return: 0.095}

selection:
  min_validation_days: 100
  min_test_days: 60
  max_drawdown_floor: -0.40
  min_annualized_return_delta: 0.01
  min_sharpe_delta: -0.10
  max_turnover_ratio: 1.50
  rolling_window_days: 63
  max_negative_window_rate: 0.60
```

配置读取后必须拒绝未知策略参数、重复的组或候选标识、交叉或倒置的时间区间，以及不在合理范围内的风险参数。默认选择期 2025-01-01 至 2025-06-30 按实际 A 股日历有 117 个交易日；100 日有效样本门槛与 63 日滚动窗口可形成 54 个完整窗口，同时不改变回撤、Sharpe、换手率和负窗口率门槛。

## 迭代流程

1. 读取并标准化行情，参数搜索只加载至 `validation_end`，不得读取核心测试或最终保留集日期。
2. 用基准参数执行训练记录期和选择期回放；选择期按实际交易日形成互不重叠的本地分段。
3. 按 `search_groups` 顺序运行。每个候选都从当前保留版本复制参数，再应用该候选所属组的覆盖值。
4. 每组候选必须同时通过整段旧版选择门槛和选择期本地通用硬门槛，才可成为下一组父参数。
5. 所有组结束后，最终锁定候选还必须直接对起始 champion 通过旧版选择门槛；失败即停止，不能打开后续数据。
6. 通过选择门槛后，只在 `core_test_start..core_test_end` 的锁定核心分段比较最终候选与起始 champion，不重开候选搜索。
7. 核心门槛通过后才加载 `test_start` 起的最终保留集，只运行初始基准和锁定候选，并生成测试比较和人工确认状态。

同一个搜索组内的候选互相独立，不能以前一个候选为起点。下一组只能继承上一组已经正式保留的版本。

## 样本隔离

- 训练期：`research_start` 至 `train_end`，用于记录长期拟合和稳定性，不直接决定晋级。
- 参数选择期：`validation_start` 至 `validation_end`（当前为 2025-01-01..2025-06-30），是组路径的唯一决策区间；117 个实际交易日使用 `min_validation_days: 100` 和 `rolling_window_days: 63`。
- 锁定核心测试期：`core_test_start` 至 `core_test_end`（当前为 2025-07-01..2025-12-31），只比较锁定候选与起始 champion，不能反馈到组搜索。
- 最终保留集：`test_start` 至 `test_end` 或数据最新日期（当前从 2026-01-01 开始），仅在最终旧版选择门槛和核心门槛均通过后打开。
- 研究试验调用的价格矩阵必须在 `validation_end` 处物理截断，不能只在完整回测结果上隐藏测试列。
- 最终测试运行可以使用测试期之前的历史作为指标预热，但任何测试期收益都不能反向参与候选搜索。
- 时间区间必须满足 `train_end < validation_start <= validation_end < core_test_start <= core_test_end < test_start`。

## 指标和晋级规则

每个训练期、验证期和测试期都计算：

- 总收益和年化收益。
- 最大回撤。
- `sharpe_like`。
- 平均换手率和平均总仓位。
- 有效交易日数。
- 63 交易日滚动收益为负的窗口占比。
- 最差 63 交易日滚动收益。

候选相对当前版本必须同时满足以下条件才可晋级：

- 验证期有效交易日不少于 `100`。
- 验证期最大回撤不低于 `-40%`。
- 验证期年化收益至少提高 `1` 个百分点。
- 验证期 Sharpe 差值不低于 `-0.10`。
- 验证期平均换手率不超过当前版本的 `1.50` 倍；当前版本换手率为零时，候选也必须为零。
- 验证期负滚动窗口占比不超过 `60%`。
- 所有关键指标均为有限数值。

多个候选同时合格时，依次按验证期年化收益、稳健评分、Sharpe、较小回撤和较低换手率排序。稳健评分定义为：

```text
annualized_return
+ 0.15 * sharpe_like
+ 0.50 * max_drawdown
- 0.10 * negative_window_rate
- 0.05 * max(turnover_ratio - 1.0, 0.0)
```

其中 `max_drawdown` 为负数，因此回撤扩大将降低评分。

## 测试期判定

测试期只产生建议，不自动晋级：

- `ready_for_manual_review`：冠军测试期总收益不低于初始基准，最大回撤不低于 `-40%`，且 Sharpe 不比基准低 `0.10` 以上。
- `test_warning`：测试期有效交易日少于 `60`，或数据不足，无法可靠比较。
- `rollback_recommended`：其余情况，继续保留初始基准。

即使状态为 `ready_for_manual_review`，也只生成 `champion_candidate.yaml`，不覆盖其他配置文件。

## 产物和版本管理

单次运行目录为 `outputs/evolution_runs/<run_id>/`，其中 `run_id` 默认由策略名、截止日期和 UTC 时间组成。目录至少包含：

- `manifest.json`：数据绝对路径、文件大小与修改时间、Git 提交、Python 和依赖版本、配置哈希、时间区间及运行状态。
- `resolved_config.yaml`：本次运行完全解析后的配置。
- `trials.csv`：每个候选的组、父版本、覆盖参数、训练/验证指标、门槛结果和排序。
- `rounds.csv`：每个搜索组保留或回滚的决定及中文原因。
- `trials/<trial_id>/metrics.json`。
- `trials/<trial_id>/equity_curve.csv`。
- `trials/<trial_id>/trade_audit.csv`。
- `trials/<trial_id>/selected_candidates.csv`。
- `champion_candidate.yaml`：研究冠军的完整策略参数。
- `test_comparison.csv`：初始基准与研究冠军的测试期比较。
- `evolution_summary_<asof>.md`：中文总结、每轮原因、收益回撤权衡和最终人工确认状态。

运行成功后更新 `outputs/evolution_runs/latest.json`，并向 `outputs/evolution_runs/evolution_registry.jsonl` 追加一条记录。失败运行保留目录和错误状态，但不得更新 `latest.json`。

## 命令行接口

默认运行命令：

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_20260709.csv `
  --benchmark D:\codex\daily-market-data\benchmarks\510300.csv `
  --asof-date 2026-07-09 `
  --dry-run
```

`--data` 必填，避免静默选错历史面板。`--benchmark` 可选；未提供时使用全仓位市场暴露，并在报告中明确记录。`--asof-date` 默认取输入数据最新日期。`--resume <run_id>` 只复用配置哈希、数据证据和试验标识完全匹配的已完成试验。CLI 默认就是 dry-run；写入全局 shadow 状态必须同时显式提供 `--no-dry-run --promote-shadow`，任一单独出现都不能授权状态变更。早期文档中“仅 `--promote-shadow` 即可写状态”的表述属于历史方案，现已失效。

## 错误处理

以下情况立即失败并保留失败清单：

- 行情缺少 `date/symbol/open/high/low/close/volume/amount` 必要字段。
- 输入数据覆盖不到配置的训练期或验证期。
- 时间区间重叠、倒置或测试期不在验证期之后。
- 验证期有效交易日不足。
- 参数未知、类型错误或超出边界。
- 指标出现无穷值，或净值为空、非正数。
- 试验子进程或策略执行函数报错。

单个候选失败时记录 `trial_error` 并继续同组其他候选；基准失败、所有候选失败或最终测试失败时整次运行失败。

## 测试设计

单元测试覆盖：

- 配置时间区间、未知参数和重复标识校验。
- 候选只覆盖当前搜索组，且同组候选互相独立。
- 研究阶段传入策略引擎的数据最大日期不超过 `validation_end`。
- 年化收益、最大回撤、滚动窗口和换手率指标。
- 回撤恰好为 `-40%`、Sharpe 差值恰好为 `-0.10` 等边界。
- 没有候选合格时保留当前版本。
- 测试期较差时给出 `rollback_recommended`，且不改正式配置。
- `resume` 仅跳过证据完全匹配的试验。

端到端测试使用小型合成面板和缩短的 `train_days`，验证基准、两个候选、冠军、测试比较、中文报告和版本清单全部生成。完整项目测试继续使用：

```powershell
python -m pytest -q
```

## 接入节奏

v1 作为独立研究命令运行。完成至少一次完整历史运行并人工确认报告后，第二阶段才考虑在 `run_daily_model_pipeline.py` 增加默认关闭的月度进化开关。每日流水线仍只读取人工确认后的固定策略参数，避免每日自动追逐短期最优。

## 验收标准

- 参数搜索期间没有读取或计算锁定核心测试或最终保留集指标。
- 每轮最多改变一个配置中的搜索组，并能解释保留或回滚原因。
- 验证期最大回撤超过 `-40%` 的候选永不晋级。
- 没有合格候选时结果与初始基准一致。
- 核心测试和最终保留集都只运行初始基准和锁定冠军，且最终保留集只在前置选择/核心门槛通过后打开。
- 每个结果可由 `manifest.json`、解析配置和输入数据证据复现。
- 所有新增测试和项目完整测试通过。
- 不连接券商、不自动下单、不自动覆盖现有配置。
