# 高风险量化交易模型3 · 回测骨架

该目录用于策略建模工作，先给出“可落地的第一版”：

- 可切换信号类型（可替换）：
  - `trend_momentum_ratio`：短线与中线动量比值分数（默认保底）
  - `trend_breakout_accel`：突破+加速动量（高风险更激进）
  - `mean_reversion_mr`：均值回归反转（防御性）
- 多标的多空头寸配资与仓位归一
- 风控钩子（最大回撤、仓位上限、杠杆/净中性、目标波动率）
- 成本模型（滑点+交易成本）
- 回测结果与指标输出（权益曲线、交易日志、绩效摘要）

> 说明：本框架为研究用途示例，不构成投资建议；请先在纸面环境复核。

## Vercel 部署

仓库包含一个位于 `api/index.py` 的轻量状态接口，以及一个位于 `index.html` 的只读研究
工作台。部署后的 `/` 显示项目界面，`/api` 和 `/health` 返回研究项目状态。

量化回测依赖本地行情文件、SQLite 数据库和科学计算依赖，不在 Vercel 函数内运行；
`.vercelignore` 会把这些研究文件排除在部署包之外。每日流水线仍应在已配置数据目录的
本地或专用计算环境中执行。

日更、数据、回测、风险和发布的强制行为守则见
[`MODEL_OPERATING_SYSTEM.md`](MODEL_OPERATING_SYSTEM.md)。出现重复问题时，应先按该规范完成
上游修复、回归测试和下游重跑，再继续讨论模型结果。

## 路径配置

仓库默认只使用相对目录。不同机器的数据位置通过环境变量配置，命令行显式参数仍有
最高优先级：

| 环境变量 | 用途 | 仓库内默认值 |
| --- | --- | --- |
| `QUANT_DATA_ROOT` | 每日行情、快照和基准 | `external_data/daily-market-data` |
| `QUANT_FALLBACK_ROOT` | 备用抓取与状态工具 | `external_data/exchange-data-ingest` |
| `QUANT_STOCK_DATA_ROOT` | 个股历史与市值文件 | `external_data/stock-data` |
| `QUANT_TDX_ROOT` | 通达信历史目录 | `external_data/tdx` |
| `QUANT_DASHBOARD_ROOT` | Dashboard 项目目录 | `external_data/stock-analysis-dashboard` |
| `QUANT_PERSONAL_TRADES_FILE` | 个人成交记录 | `inputs/personal_trades.xls` |

当前机器可在 PowerShell 用户环境中设置，例如：

```powershell
[Environment]::SetEnvironmentVariable("QUANT_DATA_ROOT", (Join-Path $HOME "quant-data"), "User")
[Environment]::SetEnvironmentVariable("QUANT_FALLBACK_ROOT", (Join-Path $HOME "exchange-data-ingest"), "User")
```

配置用途与状态见 [`configs/README.md`](configs/README.md)。设计决策和修复记录见
[`docs/superpowers`](docs/superpowers) 与 [`.superpowers/sdd`](.superpowers/sdd)。

## 本地研究数据库

研究行情、每日候选和通达信历史数据可以汇总到本地 SQLite。默认仅用于研究查询，
不包含券商、订单或账户操作。构建器会自动选择当前目录中日期最新的历史面板，
并导入输出目录中最新交易日的 CSV；也可用 `--panel` 和 `--asof-date` 显式固定输入。

```powershell
python build_research_database.py `
  --db data/research.sqlite3 `
  --daily-dir "$env:QUANT_DATA_ROOT\ths_exports\normalized" `
  --output-dir outputs\high_return_v2 `
  --tdx-root $env:QUANT_TDX_ROOT
```

通达信历史库可以单独查询：

```powershell
python query_tdx_history.py `
  --db data/research.sqlite3 `
  --tdx-db data/tdx_history.sqlite3 `
  --market SZ `
  --symbol 000001
```

趋势起爆与生命线研究也可直接读取独立 TDX 历史库。`--tdx-symbols` 和
`--tdx-symbols-file` 支持纯六位代码以及 `SZ000001`、`600000.SH` 格式；不启用
TDX 模式时仍必须显式提供 `--data`，避免误用陈旧面板。

```powershell
python analyze_trend_ignition_lifelines.py `
  --use-tdx-history `
  --research-db data/research.sqlite3 `
  --tdx-db data/tdx_history.sqlite3 `
  --tdx-symbols 000001,600000.SH `
  --start 2025-01-01 `
  --end 2025-12-31
```

### 趋势起爆评分研究

训练样本只使用起爆日收盘时已经可知的价格、均线、波动和成交额特征。标签向后观察
365 个自然日；未走满观察窗的事件会保留在原始样本中，但不会进入训练集。相邻起爆
事件使用固定 20 个交易日冷却，不允许再用未来峰值日期决定是否保留样本。

按时间段分别生成样本后，用多个 `--sample-csv` 合并训练集，再执行严格的过去训练、
未来验证。训练入口要求完整特征合同，缺列会直接报错；缺失值使用独立分箱，不会混入
最低数值分箱。

```powershell
python build_trend_ignition_training_set.py `
  --sample-csv outputs\trend_ignition_training\period_2018_2020_v3\trend_ignition_samples.csv `
  --sample-csv outputs\trend_ignition_training\period_2021_2023_v3\trend_ignition_samples.csv `
  --sample-csv outputs\trend_ignition_training\period_2024_2026_v3\trend_ignition_samples.csv `
  --output-dir outputs\trend_ignition_training\training_set_v3

python train_trend_ignition_scorer.py `
  --training-set outputs\trend_ignition_training\training_set_v3\trend_ignition_training_set.csv `
  --output-dir outputs\trend_ignition_training\scorer_v3
```

`scorer_summary.json` 中的 `passes_research_gate` 仅代表跨期研究门槛，不代表已获准接入
每日选股或实盘。逐特征结果保存在 `walk_forward_feature_diagnostics*.csv`；使用
`--feature-columns` 运行的事后短名单仍应标记为探索性，等待新的未见数据验证。

冻结评分器可在不改变原观察清单和排序的前提下做影子记录。工具只保留当日满足原始
起爆合同的观察行，强制要求观察日期晚于训练截止日，并把固定训练阈值、研究门槛和
`exploratory_posthoc` 身份写入产物。每日流水线默认不启用；当前未见样本不足以授权
自动接入排序。

```powershell
python run_daily_model_pipeline.py `
  --asof-date 2026-07-14 `
  --enable-trend-ignition-shadow `
  --trend-ignition-scorer outputs\trend_ignition_training\scorer_v3_shortlist_exploratory\binned_scorer.json `
  --trend-ignition-scorer-summary outputs\trend_ignition_training\scorer_v3_shortlist_exploratory\scorer_summary.json
```

影子结果写入 `outputs/high_return_v2/trend_ignition_shadow_<YYYYMMDD>`，每日运行卡记录
合格行数、覆盖率、固定分桶和训练截止日。该步骤始终标记 `research_only=true`、
`trade_instruction=false`、`ranking_modified=false`。

两特征短名单已冻结为下一未见时期的预注册研究对象，不能继续调参后沿用同一注册编号。
配置锁定特征、标签、训练截止日、验收门槛和模型文件 SHA-256；每日排名、仓位和实盘控制
均保持关闭。验证配置与冻结模型：

```powershell
python validate_trend_ignition_preregistration.py `
  --config configs\trend_ignition_shortlist_preregistered.yaml
```

只有信号日严格晚于 `2025-06-12`、完成至少 500 个一年期结果，并且每个固定分桶和每个
未见时期都通过预注册门槛的证据 JSON 才可提交复核。届时使用 `--evidence <path>` 校验；
校验通过仍需人工审查，不能自动晋级。

把已有主库中的通达信表合并到独立历史库时，默认保留源数据。目标库允许保留此前汇总的
其他历史记录；只有核验本次源库的每一行都已写入目标库，并显式添加 `--delete-source` 时，
才会删除源库中的通达信行。需要覆盖已有重建库时也必须显式添加 `--overwrite-rebuilt`。

通达信归档按内部 `.day` 成员记录断点，重复执行会逐成员跳过已完成文件并继续补齐剩余文件，
不会因为归档中已有部分数据而跳过整个压缩包。

```powershell
python migrate_tdx_history.py `
  --source-db data/research.sqlite3 `
  --target-db data/tdx_history.sqlite3
```

## 每日基准刷新

默认每日入口会在股票面板更新和双轨影子回测之前刷新
`$env:QUANT_DATA_ROOT\benchmarks\510300.csv`。刷新器使用新浪实时行情、
搜狐历史行情和 Yahoo 日线交叉核验日期、OHLC 与成交量，验证通过后才以原子替换方式写入；
任一数值冲突都会终止流水线。若搜狐或 Yahoo 仅缺少目标交易日，但另一个历史源与收盘后的
新浪行情逐字段一致，则允许以 `degraded_two_source` 模式原子更新，并在状态文件和运行卡中
保留警告；缺少两个有效来源、缺少更早日期或任一数值不一致时仍会终止。

每日命令默认启用严格步骤缓存。只有目标日期、基础面板、当日行情、基准、全部 Python
代码和配置，以及步骤命令与既有产物的 SHA-256 全部一致时才会命中；任一可缓存上游重算，
其可缓存下游会自动重算。命中情况写入运行状态和运行卡。需要强制全量复算时使用
`--disable-step-cache`。

因子衰减监控支持跨交易日增量检查点。只有上一期源面板哈希与本次基础面板完全一致、相关
计算代码指纹一致且日期严格前进时才复用已成熟 RankIC；系统按每只股票的实际交易记录确定
受新增数据影响的信号日，并为停牌或稀疏记录补足滚动窗口。任一校验失败会自动全量计算，
`calculation_mode`、重算起点和父检查点日期会写入监控 JSON 与每日运行卡。

每日流程的最后一步会把当天标准化行情和当天模型 CSV 产物增量写入
`data/research.sqlite3`，不会重新扫描完整历史面板或 TDX 历史库。同步结果写入
`outputs/high_return_v2/research_database_sync_<YYYYMMDD>.json`，并进入
`daily_run_state.jsonl`。完全相同的数据重复运行不会新增记录；当天行情或模型产物修正后，
对应记录会更新。历史回放或故障诊断时可显式添加 `--skip-research-db-sync`。

核心步骤成功后，默认日更会使用尚未发布的当日运行状态严格校验优先观察表、早期形态表和
模型决策表，再原子生成 `outputs/high_return_v2/marketlens_model3_latest.json`，并同步到
`$env:QUANT_DASHBOARD_ROOT/data/quant-model3-latest.json`。状态日期、行数、正式产物路径或
必需文件任一不一致都会让 `marketlens_export` 失败，整次运行不能标记成功；`_pending` 文件
不会替代正式文件发布。离线诊断可显式添加 `--skip-marketlens-export`。

```powershell
python run_daily_model_pipeline.py --asof-date 2026-07-13
```

日更默认最多并行执行两个依赖安全的子任务，并在运行卡中记录逐步骤耗时。资源受限或排查并发
问题时可使用 `--max-parallel-steps 1` 恢复严格串行执行。

日更内部历史面板默认写为 Parquet，以减少重复读取和磁盘占用；已有 CSV 历史面板仍可直接作为
基础输入，统一读取层会保持两种格式的计算口径一致。

刷新审计写入 `outputs/high_return_v2/benchmark_refresh_status_<YYYYMMDD>.json`，
并同步进入 `daily_run_state.jsonl`。离线历史复盘可显式添加
`--skip-benchmark-refresh`，该开关不会在默认每日运行中自动启用。

## 月度自进化研究

本验证流程是只读研究：必须显式提供沪深 300 基准并使用 `--dry-run`，不接受省略 `--benchmark` 的验证结果。CLI 本身仍为遗留/一般纸面研究保留可选的 `--benchmark`；省略时会回退为全程满仓市场暴露，但不能据此更新 shadow。默认时间契约把参数选择限制在 2025-01-01 至 2025-06-30；该区间按实际 A 股日历有 117 个交易日，因此整段选择门槛使用 `min_validation_days: 100`，允许少量停牌或数据缺口，同时使用 `rolling_window_days: 63` 产生 54 个完整滚动窗口。回撤、Sharpe、换手率、负窗口率以及通用核心门槛均未因可行性修正而降低。每个组赢家必须同时通过整段旧版选择门槛和仅使用该选择期非重叠分段的通用硬门槛，才能成为下一组父参数。候选路径锁定后，才在 2025-07-01 至 2025-12-31 的独立非重叠核心测试段上直接比较锁定候选与起始 champion；只有核心门槛通过后才打开 2026-01-01 起的最终保留集。每个选择/核心测试段都会把价格和市场暴露物理截断到该段 `test_end` 后独立重放，指标只来自该段日期。时间划分只由 `periods` 中的互斥日期边界决定；旧 `evolution_core.train_days`、`validation_days`、`test_days`、`step_days` 字段已从严格 schema 删除，继续提供会被明确拒绝。运行专属产物写入 `<output-root>/<run_id>`（默认 `outputs/evolution_runs/<run_id>`）；成功完成的运行还会在跨进程锁下原子更新 output root 下的 `latest.json` 和去重后的 `evolution_registry.jsonl`。即使是 `--dry-run`，它也绝不修改全局影子状态、正式配置或券商/订单。

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_20260710.csv `
  --benchmark "$env:QUANT_DATA_ROOT\benchmarks\510300.csv" `
  --asof-date 2026-07-10 `
  --dry-run
```

该命令仅供人工或每月研究使用，不接入每日默认流水线，也不会自动修改任何生产配置。

若人工复核后需要更新影子注册表，必须单独、显式执行以下研究命令：

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_20260710.csv `
  --benchmark "$env:QUANT_DATA_ROOT\benchmarks\510300.csv" `
  --asof-date 2026-07-10 `
  --no-dry-run `
  --promote-shadow
```

即使带有 `--no-dry-run --promote-shadow`，它也只会在资格门槛通过、`--asof-date` 等于面板原始最大日期、且清洗后的全部价格矩阵和基准有效序列都以有限数值实际到达该日期时更新影子注册表 `outputs/evolution_state/strong_pullback.json`；状态保存的是这个有效加载日期。自定义状态路径也必须是专用 `evolution_state` 目录下的 JSON。已有 shadow 会先重新评估；选择期、锁定核心测试或最终保留集给出失败结论时，按适用阶段先以独立 CAS/journal 事件持久化回滚并立即结束本次运行，替代候选不能在同一运行晋级。状态旁的 `strong_pullback.promotion_journal.json` 记录 `pending`、`committed` 或 `rejected`；下次启动会恢复 pending 事实，也会把已 committed 但尚未完成清单/决定发布的运行幂等收尾。状态 schema v2 强制保存语义数据日期；缺少该字段的旧 v1 状态会被明确拒绝，必须先人工复核并重建。该流程不会修改正式 YAML，也不会产生任何券商订单。

## 使用说明

1. 准备日线数据文件 `csv`，至少包含：
   - `date`（交易日）
   - `symbol`（标的）
   - `close`（收盘价）
   - 可选：`open`、`high`、`low`、`volume`

2. 安装依赖：
   - `pandas`
   - `numpy`
   - `pyyaml`

3. 运行：

```bash
python .\run_backtest.py --data "你的数据.csv" --config .\configs\high_risk_strategy.yaml
```

4. 输出文件写入 `outputs` 目录：
   - `equity_curve.csv`
   - `metrics.json`
   - `trade_log.csv`
   - `backtest_report.md`

## 目标与风控默认参数（第一阶段）

- `signal_type`：可在上述三种信号间切换
- `long_exposure`: `1.0`，`short_exposure`: `1.0`（默认净中性）
- `max_position_weight`: 每只标的最大仓位 `0.08`
- `max_drawdown`: 单位时间最大回撤 `18%`，触发后硬平仓 `2` 天
- `target_annualized_vol`: `0.55`，按名义波动自动缩放
- `commission_bps`: `1.2`，`impact_bps`: `0.8`

如果你确认策略方向（比如“情绪反转/突破/事件驱动”），我会在下一步把信号生成替换成对应规则并增加更完整的风险审计项。

### 参数扫描（下一步）

当你准备比较参数时，可直接运行：

```bash
python .\sweep_backtests.py --data "你的数据.csv" --config .\configs\high_risk_strategy.yaml
```

默认会自动扫描一组保守网格；如需自定义网格，请提供 YAML 文件，例如：

```yaml
signal.short_window: [3, 5, 8]
signal.long_window: [15, 20, 25]
risk.max_drawdown: [0.12, 0.16, 0.2]
portfolio.leverage: [1.1, 1.3, 1.5]
```

并执行：

```bash
python .\sweep_backtests.py --data "你的数据.csv" --config .\configs\high_risk_strategy.yaml --sweep .\configs\sweep_grid.yaml
```

结果文件：

- `outputs/sweep_YYYYMMDD_HHMMSS/sweep_results.csv`
- `outputs/sweep_YYYYMMDD_HHMMSS/top_k.md`
- `outputs/sweep_YYYYMMDD_HHMMSS/sweep_summary.md`

## 真实数据接入前的标准化步骤（阶段性）

当前仓库中的回测脚本仍要求输入标准面板：`date,symbol,open,high,low,close,volume`。  
若你先只想基于现有产物（如 `paper_account_*\\stock_targets.csv`）做一次参数探索，可先执行标准化再回测：

1. 先生成映射后的标准面板

```bash
python .\normalize_data.py `
  --input .\paper_account_today\stock_targets.csv `
  --output .\high_risk_quant_model3\data_panel.csv `
  --mapping .\configs\field_map_stock_targets.yaml
```

2. 如果不想单独维护 mapping 文件，可直接把映射放到命令里（示例）

```bash
python .\normalize_data.py `
  --input .\paper_account_today\stock_targets.csv `
  --output .\high_risk_quant_model3\data_panel.csv `
  --mapping "{ \"date\": \"price_date\", \"symbol\": \"code\", \"close\": \"close_price\" }"
```

3. 然后回测/扫参

```bash
python .\run_backtest.py --data .\high_risk_quant_model3\data_panel.csv --config .\configs\high_risk_strategy.yaml
```

```bash
python .\sweep_backtests.py --data .\high_risk_quant_model3\data_panel.csv --config .\configs\high_risk_strategy.yaml
```

说明：
- `--mapping` 支持 YAML 文件或 JSON 字符串，字段建议至少提供：`date`、`symbol`、`close`。
- 标准化脚本会自动用 `close` 填补缺失的 `open/high/low`，`volume` 缺省时填为空值。
- 注意：`stock_targets.csv` 这类文件通常是每日截面快照，不一定包含足够历史日线用于严格回测，结果仅可用于流程连通验证。

可选 mapping 文件示例（保存到 `high_risk_quant_model3/configs/field_map_stock_targets.yaml`）：

```yaml
date: price_date
symbol: code
close: close_price
open: close_price
high: close_price
low: close_price
volume: ""   # 留空则自动置空
```

### 宽表格式的补充用法

如果你的历史文件是“宽表”（例如 `date,000001,000002,...`，或 `close_000001,close_000002,...`），可加 `--wide` 运行：

```bash
python .\normalize_data.py `
  --input .\some_wide_prices.csv `
  --output .\high_risk_quant_model3\data_panel.csv `
  --wide
```

若列名为 `code_open`、`code_close` 这种 `symbol_field` 形式，脚本会自动识别；  
若只有 `000001,000002...` 这类纯 symbol 列，默认会把它们当 `close` 处理（可用 `--wide-value-field` 修改）。

## TDX 实时行情源验证

TDX 实时行情源来自 `oficcejo/tdx-api`，当前部署在：

- 源码：`D:\codex\_repo_cache\tdx-api`
- 编译测试服务：`D:\codex\_repo_cache\tdx-api\_bin\tdx-api-server.exe`
- Go 探针：`D:\codex\_repo_cache\tdx-api\_probe\realtime_probe.go`
- 主模型采样器：`D:\codex\智能化\high_risk_quant_model3\tdx_realtime_probe.py`
- 默认配置：`D:\codex\智能化\high_risk_quant_model3\configs\tdx_realtime_probe.yaml`

单次快照验证：

```bash
python .\tdx_realtime_probe.py --samples 1
```

交易时间连续验证：

```powershell
.\run_tdx_realtime_probe.ps1 -Samples 8 -IntervalSeconds 30
```

说明：
- 默认读取最新 `merged_priority_watchlist_*_cn.csv` 前 20 只，并额外读取 `510300` 做基准连通性对照。
- 输出在 `outputs/realtime_tdx/`，包括行情明细、采样状态、汇总 JSON 和 Markdown 报告。
- 该模块只做研究观察和网站实时状态候选数据源验证，不产生买卖指令，也不替代统一每日数据目录。

## 疑似机构提前建仓影子观察

该观察层组合未过热的价格结构、连续成交额放大、同花顺主力净量和大单净额。公开数据无法识别
真实机构账户，因此输出只称为“疑似建仓”，不改变主模型排序、仓位或交易边界。

```powershell
python .\institutional_accumulation_shadow.py `
  --data .\data_panel_history_main_chinext_20220101_YYYYMMDD.csv `
  --output .\outputs\high_return_v2\institutional_accumulation_shadow_YYYYMMDD.csv `
  --asof-date YYYY-MM-DD `
  --config .\configs\institutional_accumulation_shadow.yaml `
  --names-source "$env:QUANT_DATA_ROOT\ths_exports\normalized\ths_hs_a_share_YYYY-MM-DD.xls"
```

每日流水线会随后执行 1/3/5/10 日的次日开盘前视跟踪。v1 注册日为 2026-07-18，独立验证从
2026-07-20 开始；5 日完整信号至少需要 5 个连续资金流交易日，验证门槛需要 80 个已完成样本。
