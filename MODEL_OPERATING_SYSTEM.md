# 模型操作系统

本文件定义 `high_risk_quant_model3` 的日常运行边界、检查点和质量要求。目标不是把模型变成自动交易系统，而是把研究、筛选、复盘、审计和风险提示做成可重复、可追踪的本地流程。

## 当前定位

- 项目根目录：当前仓库根目录
- 数据源优先级：先读 `$env:QUANT_DATA_ROOT\ths_exports\normalized`，缺少最新数据时再参考 `$env:QUANT_FALLBACK_ROOT`
- 研究方向：A 股主板 + 创业板，高波动/高弹性标的，重点观察涨幅大于 5%、强趋势、趋势二波、强势平台蓄势和模型排序靠前的股票
- 输出用途：研究、观察、复盘和人工决策辅助
- 明确边界：默认不连接券商、不自动下单、不把模型结果视为买卖建议；实盘前置阶段只允许生成
  人工复核草稿，禁止券商 API 提交。

## 强制行为守则（由历史问题沉淀）

本节优先级高于临时操作习惯。凡是重复发生过的数据误用、字段缺失、路径错误、
回测穿越、报告口径混淆或下游产物未刷新，都必须在这里固化为长期约束。发现新问题时，
不能只修当天文件；必须同时补回归测试、更新本节或其引用的配置契约，并留下可复核证据。

### 1. 数据源唯一且不可偷换

- 2026-06-22 及之后的常规全市场日更，只能读取
  `$env:QUANT_DATA_ROOT\ths_exports\normalized\ths_hs_a_share_YYYY-MM-DD.xls` 或同名
  `.csv`。`asof_date` 必须与文件名和文件内交易日一致。
- `ths_money_flow_YYYY-MM-DD.xls/.csv` 是每日自选股测试输入，不是全市场行情源；禁止把它
  改名、转换或伪装成 `ths_hs_a_share_YYYY-MM-DD` 后接入常规流水线。
- 2026-06-22 之前的历史回测继续使用已经冻结的原始历史面板。除非有单独的数据修订任务和
  前后校验报告，禁止为了“补齐”而重新抓取、覆盖或混拼旧数据。
- 主数据缺失、日期不符或内容异常时必须停止。禁止静默回退到昨日文件、`input` 目录、
  自选股文件或另一个同名文件后仍把结果标记为当日成功。
- 每次成功运行必须在 `daily_run_card_YYYYMMDD.json/md` 中记录实际数据源、文件存在性、
  `asof_date`、产物哈希和数据库最新日期，作为数据源没有被偷换的证据。
- `get_latest_fetch_status().status=ok` 只证明抓取任务完成，不证明各字段已经满足模型要求。
  任何步骤开始前仍须检查该步骤必需字段的非空覆盖率、有效证券数和实际最大日期；状态正常但
  必需字段缺失时，必须降级或停止，不能把“抓取成功”写成“模型数据完整”。
- `510300` 基准刷新采用 Yahoo、搜狐、新浪三源二取二：目标交易日任意两路数据一致即可继续，
  第三路缺失、格式异常、收盘前快照或与一致多数偏离时记入降级警告，不等待三路全部到齐。
  找不到两路一致数据时必须停止；若需要补目标日前的多日历史缺口，中间日期仍要求搜狐与
  Yahoo 两套历史序列逐日一致，不能用只有当日快照的新浪数据回填历史。
- 文件名、目录名、`asof_date` 和采集任务“成功”状态都不能单独证明某日是交易日。基准交易日
  序列刷新后，面板在首末日期范围内的日期集合必须与该序列完全一致；出现任一额外日期或缺失
  日期时，面板更新、训练、回测、容量审计和风险挑战者全部必须 fail-closed。
- 非交易日上的全市场快照不得通过“证券数足够”“价格非空”或沿用昨日数据进入历史面板。
  发现这类记录时必须隔离原始行、保留修复审计，并禁止用前值或后值合成缺失交易日。
- 任一非法交易日进入历史面板后，所有依赖该面板的下游回测和研究报告一律视为失效；必须先
  修复面板，再按原冻结配置重放。旧产物需置顶标注失效并保留在隔离目录，不得与重放结果混用。

### 2. 字段问题在入口修复，不在报告末端打补丁

- 每个步骤必须声明必需字段和可选字段。`date`、`symbol`、`close` 以及该步骤真实参与计算的
  字段缺失时应立即失败；可选字段缺失只能显式降级，并在运行卡或报告中写明影响。
- `.xls`、`.csv`、制表符文本和不同编码应在读取层按内容识别并统一规范化。字段别名只能在
  集中的映射/读取模块维护，禁止在多个报告脚本中反复增加临时列名判断。
- 同一缺字段问题第二次出现时，视为数据契约缺陷：必须修读取层、增加对应真实格式的测试
  样本，并验证后续步骤；不得继续手工生成一份“看起来正确”的替代文件。
- `symbol` 必须按六位字符串保存；每日面板必须检查 `date + symbol` 唯一、日期可解析、价格为
  有限正数。名称映射后，主观察与模拟入选表的 `stock_name` 缺失数必须为 0。
- 源文件中的停牌、待上市或异常证券可能以 `--` 提供空 OHLC：这类行必须计入源覆盖并列出代码，
  但不得写入价格库或参与收益/因子计算。只要当日仍有可用价格，报告应分别给出源证券数、价格
  可用数和覆盖率；禁止把占位行填零，也禁止用源证券总数冒充可交易价格行数。
- 资金流字段必须保留来源语义和单位。`大单净额` 映射为 `main_net_inflow` 时，不得与比例型
  `主力净量` 混为同一数值；`主力净量` 映射为十进制比例 `main_net_volume_ratio`，源文件百分比点
  必须在入口除以 `100`。两者禁止相互推算回填；未达到因子要求的历史覆盖天数时，因子保持不可用
  而不是填零。
- 数据新鲜度必须至少区分三种口径：源文件/数据库的 `source_latest_date`、OHLC 可用于收益和
  标签计算的 `price_usable_latest_date`，以及成交量、成交额或资金流可用于因子计算的
  `factor_usable_latest_date`。运行卡必须分别记录，禁止只用数据库 `max(date)` 代表全部字段可用。
- 执行规范新增的审计字段必须同时落实到上游状态产物、每日运行卡和回归测试，并在改变状态
  合同时提升 `schema_version`。文档中出现字段名但程序未生成、运行卡未转录或测试未断言，均
  视为规范尚未实施，不能用人工查看源文件替代机器可审计证据。
- “字段存在”与“当日字段可用”必须分开记录。历史面板中曾出现某列，只能证明历史研究可用性；
  当日因子和影子信号还必须记录该列最新非空日期、当日覆盖率及当日可用状态。历史有值但请求日
  全空时，不得把列存在、历史滚动值或另一资金流字段写成该字段今日可用。
- 缺失字段按步骤契约处理，不能使用一套全局 `dropna`。例如后续 OHLC 完整但成交额缺失时，
  价格结果标签可以继续计算；依赖成交额的当日新信号必须标记不可用。禁止因可选成交额为空而
  删除本来可用于价格标签的日期，也禁止反过来用价格完整掩盖因子字段缺失。
- `analysis_end_date`、随访天数和“样本已成熟”必须由实际可用价格序列的最大日期推导，不能直接
  复制用户请求的 `asof_date`。请求日期晚于实际可用日期时，状态只能是 `waiting/incomplete`，
  不得生成完成证据。

### 3. 路径必须可迁移且只有一个权威入口

- 新代码和配置不得增加 `C:\Users\...`、`D:\codex\...` 等私人绝对路径默认值。外部位置统一
  通过 `workspace_paths.py`、环境变量或显式 CLI 参数解析，仓库内默认值使用相对路径。
- Windows 盘符路径不能只依赖当前平台的 `Path.is_absolute()` 判断；涉及跨平台测试、CI、
  WSL 或 Linux 时，必须同时正确识别 Windows 和 POSIX 绝对路径，并有回归测试覆盖。
- 项目代码、数据库、面板和报告统一保存在 D 盘项目/数据根目录。禁止重新在 C 盘生成大型
  面板、缓存或报告副本；临时文件也应写入项目的受控临时目录并在成功后清理。
- 同一种产物只能有一个权威路径。README、配置、自动任务和运行卡必须指向同一位置，禁止
  依靠人工记忆在多个目录中寻找“最新版本”。
- 写入数据库的文件来源必须先规范为解析后的绝对路径；相对路径和绝对路径不得生成两套观察
  记录。同步器发现指向同一文件的旧路径别名时必须清理，并在状态文件和运行卡记录清理行数。
- 历史日更面板的内部权威计算格式为 Parquet，CSV 仅作为冻结历史输入和兼容导出。两种格式必须
  通过统一 `panel_io` 契约读取，保持日期、六位证券代码、数值列和缺失值语义一致；禁止各脚本
  自行按扩展名增加临时分支。Parquet 写入必须先落临时文件再原子替换，失败时不得覆盖上一份
  有效面板。

### 4. 时间、股票池和成交假设必须点时可得

- 信号、特征、指数成分、股票名称、ST 状态、停牌状态和可交易性都只能使用决策时已经可得的
  信息。禁止用未来高收益股票反选历史股票池、用最新成分覆盖历史成分或用未来峰值筛事件。
- 收盘后形成的信号最早只能在下一可观察交易时点执行。若回测使用次日开盘，停牌、涨跌停、
  一字板、开盘跳空和无法成交必须按同一执行合同处理，不能用当日收盘价补成交。
- 复权方法必须前后一致，并显式审查除权除息、价格跳变和重复记录。异常收益不能先归因于
  策略，必须先排除复权错误、代码映射错误和成分穿越。
- ST、`*ST`、`SST`、`S*ST` 在点时 5% 涨跌停规则和成交状态未完整建模前，只能进入
  `risk_watch`，`personal_selected` 必须为 `False`，目标权重必须为 0。名称识别应限定为前缀，
  不得误伤名称中间偶然含有字母 `ST` 的普通标的。
- 成交成本、冲击成本、换手、成交额限制和单票容量必须与策略频率相匹配。缺少容量证据的
  高收益结果只能标记为研究上限，不能作为可执行收益。

### 5. 异常高收益先审计，后讨论

- `annual_return_stretch = 10.0` 只是年化 1000% 的上限探索标签，不是验收目标、收益承诺或
  自动晋级条件。正式研究验收仍看样本外超额收益、RankIC、回撤、换手、容量和稳定性。
- 任何年化达到或接近 1000%、总收益达到 10000% 量级、或显著偏离上一可信版本的结果，默认视为
  “待证伪异常”。在完成未来函数、幸存者偏差、动态股票池穿越、复权、涨跌停、停牌、成本、
  容量和参数搜索审计前，禁止把它表述为当前策略可实现收益。
- 报告必须区分历史回测总收益、年化收益、当日收益、观察样本收益和影子组合收益。重叠观察
  样本的平均涨跌不能当作组合收益，策略族命中率也不能替代净值曲线。
- 新因子、新评分器和动态风险档位先进入预注册或影子跟踪；在独立样本未成熟前不得自动改动
  主模型排序、仓位或当前日更配置。自动晋级始终关闭，最终切换必须人工复核。
- 公开行情、供应商主力净量和大单净额都不能识别真实账户身份。“机构建仓”只能命名为疑似
  建仓代理信号，并同时披露数据口径、预热天数和误判风险；不得把代理信号写成已确认机构交易。
- 预注册验证只有在 `minimum_completed_samples`、每桶最小样本数、完整随访期和数据日期全部
  满足后才能执行门槛判定。门槛未满足时，输出必须明确写入 `provisional_only=true`、
  `gate_evaluation_allowed=false`、`promotion_allowed=false`；临时正收益、正相关或正分桶差不得
  被解释为模型有效，也不得据此调参、改阈值或挑选有利子区间。
- 已失败的独立未见区间必须作为失败审计保留。若其数据进入下一轮训练，它即转为训练数据，
  新版本必须使用新的注册编号、模型哈希、冻结日期和更晚的未见起点，不能继续沿用原注册身份。
- 高收益策略迁移必须先迁移并消融信号来源，再研究集中度或杠杆。仅把持仓数从 40 只缩减为
  3/5/8/10 只不构成旧策略复现；若相同 alpha 在匹配执行口径下全周期收益非正，禁止继续用
  单票上限、杠杆或止损参数搜索“修复”收益。
- 集中持仓只能放大已经通过严格样本外验证的正期望信号。集中版本至少要同时通过：全周期收益
  为正、双倍成本收益为正、3% 开盘追高限制后收益为正、正收益年份占比不低于三分之二、最大
  回撤不触及 65% 研究失效线；任一失败即记录明确 `failed_gates` 并拒绝进入每日主链路。
- 回撤停机后净值长期不变是失效后的非活跃尾段，不是低波动或稳定性证据。研究报告必须记录
  最后一次有效净值变化后的交易日数；触发失效线且连续 20 个交易日无净值变化时，必须标记
  `inactive_tail_after_strategy_failure`，年度零收益不得计作稳定年份。
- 历史 900% 等累计收益必须同时报告样本跨度和对应年化。八年十倍与四年十倍不是同一目标，
  禁止只比较累计百分比。旧版全样本曲线、滚动样本外拼接曲线和单个爆发窗口必须分列；若单一
  窗口贡献大部分利润，只能判为市场状态依赖，不能表述为“维持”长期收益。
- 日频净值年化必须统一按有效净值观测数计算，即 `252 / (净值行数 - 1)`；禁止把日历天数与
  252 交易日指数混用。不同模块不得各自实现另一套 CAGR 公式，历史报告若使用旧口径必须重建
  或显式标记不可比较。
- 旧策略的固定股票池或自选池结果只能作为待重放工件。没有点时股票池证据时，不得与当前
  全市场历史面板直接排名；迁移时必须固定信号权重、成交合同和验证区间，一次只增加市场过滤
  或宽度保护一个变量，禁止在看过同一保留集后扩大参数网格。
- 大型研究矩阵必须复用输入哈希一致的价格、信号和点时股票池缓存。性能优化不得改变信号日期、
  成交日期或仓位路径；缓存实现必须有“没有重新计算且结果一致”的回归测试。失败或超时的半成品
  不得写入权威结果目录。

### 6. 日更必须整链一致、重复运行幂等

- 默认执行完整日更。只有当训练权重的日期、配置哈希、数据哈希和代码版本均匹配时，才允许
  `--skip-train`；“文件已经存在”本身不是跳过训练的充分条件。
- 当日源文件被重新更新时，必须从第一个受影响步骤开始重跑所有下游步骤。禁止只更新面板或
  候选表，却继续沿用旧日报、旧观察池、旧影子评分、旧数据库同步状态或旧运行卡。
- 定向重跑必须列出步骤范围并保持依赖顺序。若中途超时或失败，应重新执行同一组幂等步骤，
  直到整组成功，不能把半套新产物与半套旧产物拼成一次成功运行。
- 所有当日产物的日期、来源和生成批次必须一致；数据库最新日期必须到达 `asof_date`。
  重复执行相同输入不得制造重复的 `date + symbol` 或重复观察记录。
- 前瞻状态观察器每天只能登记当次实际见到的最新信号日，禁止从更新后的历史文件回填“前瞻”
  样本。已登记的宽度、收益中位数和风险目标必须保持不可变；后续只能通过明确的
  `trade_audit.signal_date -> realize_date` 映射补充成熟收益，禁止按净值表同日或相邻行猜测。
- 流水线成功不等于名单正确。结束前必须检查行数、名称、分层、模拟入选数、ST 入选数、
  日期边界和关键指标，并把这些验证写入运行卡或可复核报告。
- 基准刷新默认要求搜狐、Yahoo 和收盘后新浪交叉验证。仅当搜狐或 Yahoo 单独缺少目标交易日，
  且另一个历史源与收盘后新浪逐字段一致时，才允许 `degraded_two_source` 降级；状态文件和
  运行卡必须保留警告。数值冲突、两个源均缺失或更早交易日缺口仍必须停止。
- 输出目录只能发布通过校验的一致批次。试运行、字段误筛或中途失败产生的文件必须删除、隔离或
  明确标记 `invalid`，不能与有效证据并列留在权威目录。正式 JSON/CSV 应先写临时文件，完成
  日期、行数、字段和哈希校验后再原子替换，防止下游读取半成品。
- MarketLens feed 是核心日更成功后的最终发布阶段，必须使用本次运行尚未写入日志的当日状态
  校验日期、行数和正式产物路径。禁止读取昨日状态后发布当日名单，也禁止使用 `_pending` 文件
  或缺失必需输入生成网站快照；导出失败必须把整次运行记为失败并保留旧网站快照。
- 日更允许对无数据依赖且不写同一权威产物的步骤做受控并行，但依赖关系必须在执行器中显式
  声明；数据库同步必须等待全部研究产物，MarketLens 必须保持最终发布。并行失败后不得继续调度
  新步骤。运行状态和运行卡必须记录每步开始时间、结束时间、耗时、状态、并发上限和缓存命中，
  以便区分真正提速与口径缺失造成的“假快”。
- 缓存或增量模式只能在输入文件哈希、配置哈希、相关代码哈希、上游产物哈希和目标日期全部
  匹配时启用，并必须有全量模式对照测试。仅凭文件存在、文件名日期或昨日运行成功不得复用；
  缓存命中必须写入运行卡，关键校验失败时必须自动退回全量计算。
- 涉及按个股记录位移的前瞻收益增量重算，必须按每只股票的实际记录定位受影响信号，不能用
  统一自然日或市场交易日窗口近似；停牌、上市较晚和稀疏记录必须补足各自滚动历史。增量结果
  必须与同日全量结果逐行对照后才能接入每日入口。

### 7. 测试和运行卡是完成条件，不是装饰

- 每个已定位缺陷都要增加最小回归测试；涉及公共路径、数据契约、执行规则或每日流水线时，
  还要覆盖端到端命令参数或产物合同。
- 修改后先跑相关定向测试，再跑 `python -m pytest tests -q`。只有退出码为 0 才能声明完成；
  必须如实报告通过、失败、子测试和警告数量，不能用含糊的“测试正常”替代。
- 运行卡应接受详细的全通过描述，但任何非零 `failed` 都必须产生警告。被配置明确关闭的可选
  报告不得登记成缺失产物；真正必需的产物缺失则必须使运行状态失败或警告。
- 报告、数据库和运行卡刷新后应再次核对内容，而不只检查文件存在。运行卡必须最终显示
  `run_status=success`、正确步骤数、最新测试状态和可解释的警告列表。
- 数据契约测试至少覆盖：OHLC 完整但成交量/成交额缺失、请求日期晚于实际可用日期、数据库
  最新日期与必需字段最新日期不同、同一输入重复运行，以及未达到预注册总样本/分桶样本门槛。
  测试必须证明价格标签不会被可选因子字段误删，同时依赖缺失字段的新信号不会被悄悄放行。

### 8. 配置、版本和代码改动必须可追踪

- `configs/README.md` 必须标明当前日更配置、实验配置和已弃用配置，并链接对应验证证据。
  禁止仅靠文件名后缀猜测 `v2/v3/final/latest` 哪个有效。
- 参数或候选选择一旦进入未见样本验证就必须冻结。继续调参必须使用新的注册编号和新的验证
  起点，不能反复查看同一未见区间后仍称其为样本外。
- 不得回退或覆盖与当前任务无关的本地修改。生成数据和报告应由 `.gitignore` 管理；代码、
  配置和规范的提交范围必须清楚。只有用户明确要求时才提交或推送 GitHub。
- 自动化只负责研究流水线和实盘前置人工复核草稿，不连接券商、不下单、不使用融资融券、场外杠杆
  或高频报撤单。
- `prelive_order_draft_YYYYMMDD.*` 只能作为人工确认表。该产物必须始终写入
  `trade_instruction=false`、`broker_order_allowed=false`、`manual_confirmation_required=true`，
  且不得包含券商密码、令牌、会话、自动提交字段或可直接导入交易端的订单协议。
  任何改变这些边界的需求都必须暂停并单独审核。
- 实盘前置配置中的 `max_deploy_ratio=1.0` 只表示账户仓位硬上限，不表示每天固定满仓。
  实际草稿仓位必须同时受正式主模型总仓位和已批准的基准市场过滤约束；实验中的动态宽度、
  策略竞技场或尚未完成独立观察的风险状态只能写入提示，不得影响正式草稿权重。
- 小账户整手草稿必须采用输入顺序无关的显式分配方法，记录分配优先分和有效总仓位上限。
  禁止通过候选表行顺序让先出现的股票静默占用全部现金；整手分配方法或优先级规则变更必须
  重新生成草稿并补回归测试。
- 整手影子账户只能在下一实际交易日执行前一交易日草稿；当日收盘后新生成的草稿不得使用
  当日开盘价回填成交。首日没有更早草稿时状态为 `waiting_first_execution`，不得伪造历史成交。
  影子账户必须先卖后买、逐笔扣除配置中的最低佣金、卖出税费和滑点，并记录涨跌停、开盘跳空
  与现金不足造成的阻塞或部分成交；同一日期重复运行必须从更早快照重建并保持字节级幂等。

### 9. 外部策略与竞技场必须隔离

- 外部项目先做许可证、数据口径、执行时点、依赖、测试和未来函数审计；未经本地重写与验证，
  不得直接进入正式流水线。借鉴的机制、拒绝吸收的部分和本地边界必须写入 `docs/`。
- 组合决策器只有在同一策略族、同一执行合同和共同交易日上才可比较。不同策略族必须分联赛；
  没有独立净值曲线的结构或资金流信号只能进入观察分区，不能参与组合排名。
- 历史总收益、Sharpe 或 Pareto 占优只用于发现挑战者，不能作为晋级证据。挑战者必须从预注册
  启用日之后积累新独立样本，完成门槛后也只能进入人工复核。
- 竞技场必须记录正式冠军、联赛基准、源净值日期、共同区间、执行合同、源文件哈希、观察日数和
  晋级禁用状态。重复运行同一日期必须幂等，不得增加虚假的独立观察日。
- `automatic_promotion`、跨策略族晋级和大模型直接交易始终关闭。竞技场报告不得改变选股、
  目标权重、正式配置或纸面账户。

### 10. 问题复发时的处理顺序

1. 立即停止发布新结论，保存原始输入、错误输出和运行卡，不手工美化结果。
2. 判断问题属于数据源、字段契约、时间/成交假设、路径、配置、报告口径还是执行环境。
3. 在最上游的责任模块修复；删除仅为当天绕过问题的临时转换或重复文件生成逻辑。
4. 增加能重现该问题的回归测试，并先确认测试在修复前失败、修复后通过。
5. 从首个受影响步骤重跑全部下游，检查数据库和运行卡，不混用旧产物。
6. 把新教训补入本守则或明确引用的配置契约。只有证据闭环后才恢复模型讨论。

每个重复问题还必须留下最小复发记录，至少包含：问题指纹、触发输入、错误的“假成功”表现、
最上游责任模块、受影响产物范围、修复提交、回归测试和新增守则条目。相同问题第三次出现时，
必须把对应检查前移到默认流水线预检或 CI，不能继续依赖人工记忆。

多周期次日开盘标签必须按退出开盘的真实可见时间清除未成熟样本。信号位置为 `i`、持有期为
`h` 时，最后一个可用于训练的标签位置只能是 `i-h-1`，训练切片右端必须为 `i-h`。组合
包含多个周期时按最大周期统一清除；训练器调用必须显式传入最大周期，禁止依赖默认值。

按经济含义拆分策略袖套时，因子方向必须冻结。历史 IC 与定义方向相反时只能降为零权重，
不能反向后继续沿用原策略名称；全部因子失效时必须空仓，禁止退化为任意等权选股。由同一
原始变量派生的因子不得伪装成独立分散来源，趋势回调等混合项必须单独登记和归因。

平均 RankIC 为正不能替代可成交组合验收。任何因子只要在固定的次日开盘、涨跌停、停牌、
成本和市场敞口规则下出现负收益或回撤越界，就必须按组合结果拒绝，不能用全股票池相关性
为亏损策略辩护。趋势窗口包含近期回调窗口时必须明确标记重叠暴露；不得在主假设失败后从
同源派生项中挑选最好结果。历史资金流字段覆盖不足时只能做前向观察，禁止回填或据此声称
完成多年资金流回测。

分阶段实验必须执行顺序停止：一级收益/风险硬门槛任一失败后，不得继续运行二级成本压力、
相关性筛选或参数微调来寻找可接受解释。失败方案即使改善某一个指标，也只能按预登记用途
判定；例如回撤改善但收益和 Sharpe 未超过 champion，仍不能改称替代者或 shadow。若要把
局部优点改造成新的防守假设，必须重新登记，并把已经看过的区间标明为非样本外。

生产 `unconstrained` 模型必须使用显式冻结的特征白名单，禁止把 `build_features()` 中新增的
研究因子自动纳入默认训练。任何研究因子上线前必须有单独配置、对照回测和人工决定；测试必须
断言生产白名单不包含研究专用特征。容量、成本等压力运行在开始场景前必须先以同一代码执行
无压力 control，并与登记 champion 指标对账；control 不一致时全部压力结果作废。
容量约束后的收益偶然高于 control 只能解释为部分成交改变持仓路径，禁止当作新增 alpha、收益
增强证据或放宽容量上限的理由。容量结论必须同时审查收益留存、仓位留存和成交填充率。
风控 overlay 的目标暴露与成交约束后的实际总仓位必须使用不同字段；禁止把目标暴露字段解释为
实际已部署资金。
提高风险响应频率必须把触发日减损与减仓后等待恢复期间的机会损失分开归因。仅消除延迟会话不
代表风险策略有效；若收益留存、回撤或 Sharpe 的预登记门槛失败，禁止用事后恢复规则或阈值微调
救回同一候选。

champion 复核必须拆分年度和市场状态，并用每日净收益复利及对数收益核对全周期；只有一个
正收益年份或单一状态贡献超过 80% 时，必须标记收益来源集中，禁止把全周期年化外推为稳定
年度收益。滚动因子权重只能解释打分倾向，不能冒充逐因子利润归因。流动性因子长期为负权重
时，在提高资金规模、单票仓位或组合集中度前必须先做点时容量压力。风险信号为零但实际仓位
仍大于零时，必须先区分非调仓日的风险响应延迟与调仓日已尝试卖出后的残余仓位；没有显式的
调仓标记、目标仓位和约束前后差额时，禁止把它直接称为涨跌停/停牌阻塞成交。
### 11. 参数寻优的阈值一致性：被扫参数必须覆盖其影响的所有代码路径

- 凡参数同时作用于「信号构造」与「执行闸门」两处（如 regime 阈值 THR_HI/THR_LO 既决定 soft 分里
  线性/MLP 配比 adv，又决定 gross 降权闸门），扫参时必须让两处使用【同一个】值，并把一致性校验
  做成 fail-fast 守卫（如 `production_soft_score.assert_regime_thresholds_consistent`），在回测入口
  拦截，绝不静默放行。
- P10j 曾因 soft 混合用模块 THR_HI=0.03、gross 闸门扫到 0.04 的错配，误报 THR_HI=0.04 更优
  (val 1.677)；一致化后真实 val 仅 1.328。错配与正确配置夏普差 ~0.18，足以反向误导部署，故必须固化。
- 夏普口径强制为 `net_returns = gross_return − cost`（ddof=1，年化 ×√252），禁用 `equity.pct_change()`
  （含预热首日跳变，虚高且不与 `calculate_walk_forward_metrics` 自洽）。
- 网格最优值落在边界时（如 THR_HI=0.04 看似更优），必须向两侧延伸确认非边界假象。
- 适配回归测试：`tests/test_p10j_threshold_consistency.py` 必须始终通过（修复前失败、修复后通过）。


### 12. 监控必须生产化：告警要可见、可消费、可阻断

- regime 监控（trailing IC 状态 + 告警）必须纳入每日流水线步骤 `production_regime_monitor`（依赖
  `update_panel`），产出日期令牌化 `regime_monitor_{token}.json` / `regime_alert_{token}.json` / `.md`，
  并挂载进 `daily_run_card` 与 `build_daily_run_state`；不允许只作为孤立脚本存在。
- 告警不是报告：ALERT=True（dead 区）必须给出可消费的降权信号 `recommended_gross_scale`
  （缺省 EXPOSURE_FLOOR=0.25）。执行层（P9-7 / prelive 下单）在下单/簿构建前须调用
  `production_soft_score.effective_book_gross_scale(alert, base_scale)` 得到当日生效 gross 上限，不得忽略。
- 运行卡必须把活跃告警通告为可见信号：`build_daily_run_state` 汇总进顶层 `monitoring_alerts`，
  `daily_run_card` 在 Warnings 增加 `production_regime_alert` 并在 `## Monitoring Alerts` 段落展示；
  状态解析 `production_regime_monitor_status` 须处理 skipped/missing/invalid/stale（面板滞后即 stale，
  提示需先刷新生产面板）。
- 硬阻断（可选）：`--block-on-alert` 让步骤在 ALERT=True 时以退出码 2 失败，触发整条流水线 failure 路径；
  默认关闭（`config.block_on_regime_alert=False`），由运维按风控要求开启。
- 回归测试 `tests/test_production_regime_monitor_pipeline.py` 必须始终通过（含步骤接线、状态分支、闸门助手、
  skip_backtest 下阈值一致性守卫、运行卡告警可见性）。
- 执行层闭环已接线（P10m）：日更流水线中 `prelive_order_draft` 步骤依赖 `production_regime_monitor`，并把当日令牌化
  `regime_alert_{token}.json` 通过 `--book-regime-alert` 传入；`build_draft` 用 `effective_book_gross_scale` 把 regime 死区告警
  压成的 gross 上限并入 `effective_deploy_ratio`（与 market_risk / base_strategy 限制取三者最小），并写入 metadata 的
  `book_regime_gross_scale` / `book_regime_alert_active` 以及 markdown 复核说明。monitor 关闭时省略该参数，由 prelive 自动发现
  outputs 下最新告警（canonical 或历史令牌化产物）。P9-7 簿构建无需改动：其消费的 `book_soft_equity.csv` 已是 `joint_schedule`
  把 regime 降权烤进 `market_exposure` 之后的全样本回测权益，簿侧闸门在内部已闭合。



### 13. 成本口径统一：万三佣金 + 法定印花税（仅卖出侧）

- 所有回测 / 影子账户 / 生产簿的成本必须采用统一实盘口径：佣金万分之三（3bps，双边），
  法定印花税万分之五（5bps，仅卖出侧），市场冲击 / 滑点另计（production_soft_score 默认
  含 0.7bps 于 impact，执行层单独 5bps 滑点）。禁止用旧的固定 bps 假设（如 2.5bps 佣金、忽略
  印花税）冒充实盘成本。
- 落地点：
  - `production_soft_score.py`：`COMMISSION_BPS=3.0`、`IMPACT_BPS=0.7`、`STAMP_TAX_BPS=5.0`，
    `produce_book_score` 完整回测使用该口径。
  - `train_next_open_rank_model.py`：`run_walk_forward(..., stamp_tax_bps=5.0)` 在成本中追加
    `(turnover/2.0)*stamp_tax_bps/1e4`（卖出侧约半换手）。
  - `run_backtest.py`：`StrategyConfig.stamp_tax_bps`（默认 5.0），成本公式同步加卖出侧印花税。
  - `prelive_account_shadow.py`：`ShadowAccountConfig.commission_bps=3.0`，
    `sell_stamp_tax_bps=5.0` 已在计收逻辑正确收取。
- 任何新增回测入口必须显式带 `stamp_tax_bps`；回归测试
  `tests/test_prelive_account_shadow.py::test_shadow_cost_schedule_is_wan3_commission_plus_legal_stamp`
  守护该口径。

### 14. 生产簿须每日基于新鲜数据重建（禁止滞后）

- `production_soft_score` 的 `book_soft_equity.csv` / `latest_soft_score.csv` 是 P9-7 等下游消费的
  生产簿，必须由日更流水线用当日新鲜面板全量重算（生产软分完整回测，无 `--skip-backtest`），
  不得依赖陈旧产物。
- 流水线：`run_daily_model_pipeline.py` 的 `production_book_build` 步骤（依赖 `update_panel`，
  默认开启，`--skip-production-book-build` 可关）每日运行完整模式重建。
- 可见 / 纪律：`build_daily_run_state` 经 `production_book_build_status` 校验最新日期须 == asof_date，
  状态分 skipped/missing/invalid/stale/ok；`daily_run_card` 在 `## Production Book` 段落与 Warnings
  展示（stale/missing/invalid 即预警）。
- 监控模式（`--monitor --skip-backtest`）只写 regime 令牌化告警、不写簿；簿重建交由
  `production_book_build` 步骤负责，避免“簿滞后于新鲜数据”的缺口。

### 15. 因子衰减监控必须是闸门（不只是报告）

- `monitor_factor_decay.py` 的因子衰减监控升级为闸门：当 `overall_status == "direction_reversal"`
  （因子方向性反转）时输出告警并进入统一 `monitoring_alerts`，要求“评估因子替换并暂停新增资本部署”。
- 落地点：
  - `monitor_factor_decay.factor_decay_alert(payload)` 返回 alert dict（`alert=True`、
    `severity=high`、`recommended_action=evaluate_factor_replacement_and_hold_new_capital`）；
    `main()` 写 `factor_decay_alert_{token}.json` + canonical `factor_decay_alert.json`，
    `--block-on-alert` 时以退出码 2 阻断。
  - 流水线 `build_daily_run_state` 经纯函数 `factor_decay_gate_alert(status)` 把闸门并入
    `monitoring_alerts`，运行卡 `## Monitoring Alerts` 展示（无 `recommended_gross_scale` 时不渲染该字段）。
  - `factor_decay_monitor` 步骤在 `config.block_on_factor_decay_alert=True` 时追加 `--block-on-alert`
    （默认关闭）。
- 边界保持：`weakened` / `stable` / `insufficient_history` 仅为软观察、不直接触发闸门；整体仍为
  `research_only`（不改模型参数），闸门只产生“暂停新增资本 + 评估替换”信号，模型参数替换仍走既有
  preregistration / tracking 流程。
- 回归测试守护：`tests/test_daily_model_pipeline.py` 的 `factor_decay_gate_alert_triggers_on_direction_reversal`、
  `factor_decay_gate_alert_silent_for_stable_or_weakened`、`test_factor_decay_alert_helper_shape`、
  `test_pipeline_passes_block_on_alert_to_factor_decay_monitor`、
  `test_parse_args_accepts_block_on_factor_decay_alert_flag`。



### 16. P10 生产产物每日刷新并可见

- P10 模型侧核心（b/c/e/g/h/i/j）已内嵌 `production_soft_score.py`（import `p10c_ensemble`、加载
  `p10e_regime_gated/linear_mlp_scores.npz`、内联 `joint_schedule`、用 `p10h` 容量配置 + `p10j` 阈值），
  随 `production_regime_monitor` / `production_book_build` 每日调用，无需重复接线。
- 面向生产的 P10 报告产物须每日刷新并作为标准步骤消费（MOS 规则缺口清单中“P10 产物未作为标准步骤消费”的收口）：
  - `p10g_production_monitor_chart.py`：P10g 生产监控图表，依赖 `production_regime_monitor`，
    输出令牌化 `regime_monitor_chart_{token}.png`，运行卡 Artifacts 自动渲染。
  - `p10g_production_integration.py`：P10g 生产集成报告（主簿接入 P9-7 组合 + 容量重算），
    依赖 `production_book_build`，输出 `p10g_production_integration/metrics.json`，运行卡展示
    `p10g_production_integration_status` / `book_capacity` / `combined_capacity`，缺失则告警。
- 脚本须带 CLI 且日更友好（MOS 规则 16 的落地约束）：`--panel/--scores-npz/--output-dir/--asof-date`
  等由流水线注入；图表/集成产物带日期令牌或每日原地刷新；卫星腿（sleeve/hk_passive/hk_active）
  与基线簿缺失时自动跳过对应段（不崩溃）；容量段依赖 `linear_mlp_scores.npz`（外部前提，缺失则跳过）。
- **分数缓存命名（2026-09-02 强制）**：`production_soft_score` 将可选因子（wq/broker/a158）与
  MLP 配置（cap/iters）**无条件**纳入内容寻址，缓存名随配置演进变化（如 `linear_mlp_scores_pit_broker.npz`
  → `_broker_mlp16c30000i400.npz`）。**任何下游禁止硬编码 npz 文件名**，一律调用
  `production_soft_score.latest_pit_broker_npz()`（读生产维护的 LATEST 指针文件，mtime 仅指针缺失时回退）；缺失时该 helper
  SystemExit 并提示先跑 production_soft_score 重建。硬编码旧名已两次导致 shape 错
  （08-31 `_broker` 后缀、09-01 `_mlp` 参数后缀），同类问题第三次出现将把该解析前移到默认预检。
- 研究型 P10 报告（`p10i` 联合调度、`p10h` 容量工程、`p10j` 阈值优化、`p10e_regime_gated` 分数构建）
  属上游验证，结论已固化进 `production_soft_score` 固定配置，**不接入每日流水线**，保持离线手动跑。
- 回归测试守护：`tests/test_daily_model_pipeline.py` 的 `test_default_pipeline_includes_p10g_*`、
  `test_pipeline_can_skip_p10g_*`、`test_parse_args_accepts_skip_p10g_*`、`test_p10g_production_*_status_branches`。


## 日更入口

每天数据更新后，从项目根目录运行：

```powershell
Set-Location <repository-root>
python run_daily_model_pipeline.py --asof-date YYYY-MM-DD
```

如果当天训练权重已经存在，可走快速刷新：

```bat
python run_daily_model_pipeline.py --asof-date YYYY-MM-DD --skip-train
```

## 月度自进化研究

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_YYYYMMDD.csv `
  --benchmark "$env:QUANT_DATA_ROOT\benchmarks\510300.csv" `
  --asof-date YYYY-MM-DD `
  --dry-run
```

这是月度复核认可的验证入口。默认参数选择期为 2025-01-01 至 2025-06-30，只允许该区间决定组赢家和最终候选路径。该区间按实际 A 股日历共有 117 个交易日；`min_validation_days: 100` 为少量停牌或数据缺口保留余量，`rolling_window_days: 63` 在完整样本上形成 54 个滚动窗口。此次可行性修正没有降低选择、核心测试或最终保留集的回撤、Sharpe、换手率、负窗口率及 PnL 集中度门槛。每个组候选必须同时通过整段旧版选择门槛，以及至少 `min_folds` 个选择期本地、按时间排序且互不重叠分段上的通用硬门槛，才能成为下一组父参数。路径锁定后，系统再把 2025-07-01 至 2025-12-31 划为独立的非重叠核心测试段；每段都把全部价格矩阵和市场暴露物理截断到段末，并只比较锁定候选与起始 champion。锁定候选通过核心门槛后，才允许打开 2026-01-01 起的最终保留集。时间边界完全由 `periods` 定义；旧 `evolution_core.train_days`、`validation_days`、`test_days`、`step_days` 已从严格 schema 删除，不能再作为兼容输入。

CLI 为遗留或一般纸面研究保留可选 `--benchmark`；省略时市场暴露回退为全程 `1.0`，这种结果不能作为基准化验证证据。任何全局 shadow 状态变更都要求同时使用 `--no-dry-run --promote-shadow`、`--asof-date` 等于面板原始最大日期，并要求清洗后的全部价格矩阵和基准有效序列以有限数值实际到达该日期。可写状态只能位于专用 `evolution_state` 目录；正式 YAML、券商和订单路径始终不在该流程内。

全局状态转换使用同目录的 promotion journal 记录 `pending`、`committed` 或 `rejected`。已有 shadow 在选择期复核、锁定核心测试或最终保留集被判定失败时，回滚会先作为独立持久化事件提交并结束本次运行；dry-run 只记录回滚建议，不改变状态，也不会在同一运行晋级替代者。下一次启动会核对 journal 与实际状态，并幂等完成已 committed 但清单或决定仍为 pending/running 的旧运行；单次运行清单同时记录耗时、峰值内存和覆盖重放/指标直接依赖的策略代码指纹。

运行前必须确认：

- 当天文件存在，例如 `$env:QUANT_DATA_ROOT\ths_exports\normalized\ths_hs_a_share_YYYY-MM-DD.xls`
- `$env:QUANT_FALLBACK_ROOT\scripts\market_data_utils.py` 中的 `get_latest_fetch_status()` 或 `ensure_latest_fetch_ok()` 可作为备用状态检查
- 如果状态函数落后，但主目录已有当天标准化文件，以主目录文件时间为更强执行依据

## 固定输出

每日核心产物在 `outputs\high_return_v2` 下：

- `daily_personal_overlay_report_YYYYMMDD.md`：中文日报
- `daily_personal_overlay_selected_YYYYMMDD.csv`：个人行为叠加后的入选表
- `daily_personal_overlay_changes_YYYYMMDD.csv`：新增、移除、降权变化
- `early_pattern_watchlist_YYYYMMDD.csv` / `_cn.csv`：早期形态观察池
- `merged_model_decision_table_YYYYMMDD.csv` / `_cn.csv`：模型决策明细
- `merged_priority_watchlist_YYYYMMDD.csv` / `_cn.csv`：优先观察表
- `core_risk_filter_finalist_stability_YYYYMMDD.md`：核心风控稳定性报告
- `strategy_arena_YYYYMMDD.md`：分联赛策略竞技场中文报告
- `strategy_arena_portfolio_YYYYMMDD_cn.csv`：组合决策器中文对比表
- `strategy_arena_history.csv`：幂等竞技场观察历史账本
- `dynamic_breadth_overlay_tracking.csv`：严格标签修复后自 2026-07-23 起重新登记的宽度覆盖前向观察账本
- `dynamic_breadth_overlay_tracking_summary.json` / `_report.md`：60 个有效交易日门槛摘要；始终为研究观察，禁止自动晋级
- `market_breadth_mismatch_tracking.csv`：自 2026-07-23 起逐日登记的市场宽度与基准风险目标
  错配观察账本；信号特征不可改写，收益按交易审计中的实现日成熟
- `market_breadth_mismatch_tracking_summary.json` / `_report.md`：显示 MA60 宽度、横截面 20 日
  收益中位数、实际总仓位、510300 风险目标和 60 个成熟样本进度；阈值仅用于诊断标签，
  不得触发减仓、改排名、改订单或自动晋级
- `marketlens_model3_latest.json`：通过当日状态校验后生成的网站研究快照
- `prelive_order_draft_YYYYMMDD_cn.csv/md`：广发证券人工复核草稿，只读、不可自动下单
- `prelive_account_shadow/prelive_account_shadow_summary.json/md`：一万元、100 股整手、次日开盘
  模拟执行账本；只用于前向验证，不连接券商

快速查看优先使用：

```text
outputs\high_return_v2\merged_priority_watchlist_YYYYMMDD_cn.csv
```

## 分层规则

优先观察表按以下层级理解：

- `action_focus`：人工明确决策为买入或强关注时的最高层
- `model_focus`：模型排序 + 个人行为过滤后的主观察层
- `risk_watch`：风险名称或风险状态，仅保留可见，不作为执行优先层
- `pattern_watch`：形态观察层，主要用于发现早期结构
- `review_later`：延后复查层

ST / *ST 名称必须进入 `risk_watch`，不能进入执行优先层。默认可见 `risk_watch` 数量应保持受控，避免挤占正常观察名单。

## 每日质量检查

每次日更完成后至少确认：

- 优先观察表行数正常，通常为 50 行
- `stock_name` 缺失数为 0
- `model_focus`、`pattern_watch`、`risk_watch` 分层合理
- 早期形态数量与类别正常，重点看 `趋势二波启动` 和 `强势平台蓄势`
- ST / *ST 已被降到 `risk_watch`
- 自动测试通过：`python -m pytest -q`
- 将本次证据追加到 `daily_run_state.jsonl`
- `marketlens_export_status=success`，模型 feed 与网站镜像的 `asofDate` 等于当日交易日

## 风险边界

允许为了收益率适度提高风险，但以下边界不变：

- 当前回撤容忍上限按约 `-40%` 管理，超过后必须优先做归因和风控审计
- 不因为单日高收益放松 ST / *ST、异常数据、未来函数和路径漂移检查
- 不用未确认来源的数据覆盖主日更数据
- 不把成交额缺失或估算成交额作为硬性失败，除非策略重新依赖成交额过滤
- 不做 ETF 和基金

## 需要人工介入的情况

出现以下情况时，应暂停自动推进并让用户确认：

- 当天主数据目录缺少最新文件
- 数据文件行数明显异常
- 优先表缺失股票名称
- 测试失败
- 输出写入 `_pending` 文件且原文件可能被表格软件占用
- 风控稳定性报告显示最大回撤、收益、胜率同时恶化
- 新策略会改变交易边界、股票池范围或风险容忍上限

## 每次任务的完成定义

只有同时满足以下条件，日更、修复或模型推进任务才算完成：

- 输入来源、文件日期、文件内交易日和 `asof_date` 一致，且没有静默回退数据源。
- 面板和数据库通过行数、唯一性、名称完整性、必需字段、最新日期和交易日集合一致性检查。
- 回测与观察结果通过点时数据、股票池、复权、成交、涨跌停、停牌、成本和容量审查。
- 当天所有受影响的下游产物已刷新，模拟入选表中 ST 数量为 0，报告明确标注研究/模拟盘。
- 定向测试和全量测试均通过，新增问题已有回归测试；遗留警告已记录且不会改变当前结论。
- `daily_run_card_YYYYMMDD.json/md` 状态成功，步骤数、测试数、警告、产物哈希和数据库日期正确。
- 已检查 `git status`，未覆盖用户修改，未把大型数据、数据库、缓存或当日报告误加入版本控制。
- 对外表述区分回测、样本观察和实盘，不承诺收益；未获明确授权时不提交、不推送、不交易。

## 复盘和改进节奏

- 每日：跑日更、看优先观察表、记录状态账本
- 每周：汇总入选后表现，检查成功/失败原因
- 每月：复查参数稳定性，避免只追逐短期最优
- 策略改动前：先写清楚改变了什么、为什么、如何验证
- 策略改动后：必须和上一版输出做对比，而不是只看单次收益
