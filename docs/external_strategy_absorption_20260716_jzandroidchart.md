# JZAndroidChart 主力净量吸收记录

## 结论

`JingZhuanDuoYing/JZAndroidChart` 是 Android 股票图表渲染库，不提供行情采集、
大单分类或主力净量计算公式。本项目不引入其 Android 依赖，只吸收
`ZeroCenterBarDataSet` 的零轴对称展示思想，用于同时呈现正向净流入和负向净流出。

- 仓库：https://github.com/JingZhuanDuoYing/JZAndroidChart
- 零轴数据集：https://github.com/JingZhuanDuoYing/JZAndroidChart/blob/master/chart/src/main/java/cn/jingzhuan/lib/chart3/data/dataset/ZeroCenterBarDataSet.kt
- 柱状图绘制：https://github.com/JingZhuanDuoYing/JZAndroidChart/blob/master/chart/src/main/java/cn/jingzhuan/lib/chart3/draw/BarDraw.kt

## 本地数据契约

- 同花顺源字段 `大单净额` 继续映射为 `main_net_inflow`，单位为金额。
- 同花顺源字段 `主力净量` 新增映射为 `main_net_volume_ratio`。
- 源文件中的 `主力净量` 使用百分比点，例如 `0.77` 表示 `0.77%`；入口统一除以
  `100`，内部保存为十进制比例 `0.0077`。
- 两个字段不可互换，也不得用 `main_net_inflow / amount` 回填缺失的
  `main_net_volume_ratio`。
- 缺失值保持缺失，不填零。零值是有效观测，表示供应商口径下当日净量接近中性。

## 已吸收能力

1. 每日入口保留 `main_net_volume_ratio`，并记录来源字段。
2. 生成当日横截面分位、5/10 日平均、5/10 日正值占比。
3. 用“5日资金分位 - 5日价格收益分位”做资金领先价格的连续型交叉验证。
4. 输出中文 CSV、JSON 审计信息、Markdown 报告和零轴正负柱图。
5. 与提前形态和高质量隐性吸筹名单做交集，但不改变原始候选排序、入选结果或仓位。

## 当前状态与晋级规则

本字段从 `2026-07-14` 开始形成稳定全市场覆盖。未满 5 个有效交易日前，状态必须为
`warmup`。满 5 日后仍只进入影子观察，至少完成以下检查才可申请进入正式模型：

- 数据覆盖率、缺失模式和异常值稳定；
- 下一开盘口径的 1/3/5/10 日样本外收益可复核；
- 对现有隐性吸筹、强势平台和趋势二波因子的增量信息有效；
- 行业、市值、成交额暴露不是主要收益来源；
- 加入成本、涨跌停和容量约束后仍有稳定增益；
- 参数和阈值预注册，不依据最新收益临时调整。

该字段用于回答“资金流是否支持价格形态”，不能单独证明吸筹、出货或未来涨跌。
