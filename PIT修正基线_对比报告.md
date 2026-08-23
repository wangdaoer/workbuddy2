# Point-in-Time 宇宙修正基线 · 对比报告

> 目的：落实用户提出的 **option 2（point-in-time 筛选框架）**，根治"筛选过之后的自选股"
> 必然带有的前视 / 幸存者偏差，给出诚实的 Route C 真实基线，对比旧的 `0.998` 上界。

## 1. 方法

在 `production_soft_score.py` 的 `produce_book_score()` 中新增 `universe` 开关（默认 `pit`），
PIT 掩码同时喂给 **训练**（`build_linear_mlp_scores` 的 live 集）与 **回测**
（`run_walk_forward` 的 `current_live_cols`），训练/回测宇宙严格一致：

| 模式 | 训练 + 回测宇宙定义 | 含义 |
|------|--------------------|------|
| `pit`（默认） | `pit_universe.pit_eligible(close, amount, thr=LIQ_THR, lookback=LIQ_LOOKBACK, min_days=60)` | 每再平衡日 `t` 仅用 **≤t 信息**判合格（已上市 / 未退市 / 未停牌 / trailing 流动性 / 最小历史），**无前视** |
| `full`（旧行为） | 训练=`liquid_mask_aligned`（当日流动性），回测=`liquid_mask=None`（全部有分名的票） | 接近全市场面板，但允许"在退市/停牌后仍以旧分交易"，带轻度幸存者/静态偏差；= 旧 `0.998` 上界 |
| `static-survivor`（反例） | 把末日存在的标的广播到所有历史日 | 用"未来名单"回测，量化前视会虚高到什么程度（诊断用） |

面板：与旧基线严格一致 —— `external_data/daily-market-data-tdx/data_panel.csv`
（Tdx 880 日 / 5910 只 / 484 万行，2022-12-05 → 2026-07-23）。
Route C 生产配置：佣金 3.0bps / 印花税 5.0bps / 冲击 0.7bps、freq=5、top_n=40、ADV=2%、AUM=1e8、
regime 联合调度（`joint`）。

## 2. 结果

| 宇宙 | Sharpe | 总收益 | 最大回撤 | Regime 状态 | 建议 gross |
|------|-------:|-------:|--------:|------------|-----------:|
| `full`（旧） | **0.998** | +63.4% | −19.0% | dead | 0.25 |
| `pit`（诚实） | **0.764** | +43.6% | −20.7% | decaying | 0.758 |
| `static-survivor`（反例） | **0.219** | +5.4% | −19.2% | decaying | 0.261 |

## 3. 前视 / 幸存者偏差虚高量（三点对照）

**① `full` 是上界，不是真实 OOS。** `full` 模式回测时 `liquid_mask=None`、训练用当日流动性掩码，
导致在再平衡日 `t` 仍可用**停牌中 / 上市不足 60 日**的票的"旧分"进场（这些票在 `t` 实际不可交易，
属于非 point-in-time 宇宙泄漏）。这把 Sharpe 从诚实值 **0.764 虚高到 0.998（+31%）**，
总收益 +63.4% vs +43.6%（虚高 ≈ 19.8 个百分点）。

**② `pit` 是诚实基线 = 0.764。** 每再平衡日 `t` 只用 ≤t 信息判合格，杜绝"用未来名单 / 未来退市状态 /
停牌中仍可交易"三类泄漏。Route C 真实 OOS 基线应记为 **≈0.76**。

**③ `static-survivor` 是反向失真（不是虚高，是压低）。** 该掩码把**末日名单**广播到所有历史日，
效果是：在日期 `t` 把"在 `t` 其实在交易、但后来退市"的票**剔除**（它们不在末日名单里），
同时加入"末日存在、但 `t` 时尚未 IPO"的票（无数据→drop）。本面板的**早期退市票是净赢家**，
把它们从可交易集里剔除会压低收益——于是 `static-survivor` 反而只有 **0.219（+5.4%）**。

> 这点很关键：**前视 / 幸存者偏差不总是"虚高"**。方向取决于名单怎么造：
> - 用"未来名单"去**多保留赢家**（如人工精选 watchlist 事后挑牛股）→ 虚高；
> - 用"末日幸存名单"去**剔除早期赢家**（如 `static-survivor` 这里）→ 压低。
> 本面板自带轻度**幸存者偏差在保留集**（早期赢家被留下来），所以 `full`/`pit` 仍可能比
> "退市完全补全"的真值略高；但 `full` 的 0.998 主要是**非 PIT 宇宙泄漏**造成的夸大，
> 已由 `pit` 的 0.764 修正。

> 若把模型接到**人工/规则筛选过的 watchlist** 上（最可能虚高的场景），接入前必用
> `pit_universe.detect_static_leak(close, amount, static_list=watchlist)` 量化污染度，
> 污染度高的名单不能直接喂模型。

## 4. 关键发现：regime 信号本身也被污染

`full` 与 `pit` 不仅回测结果不同，**regime 监控结论也不同**：

- `full`：trailing IC = **−0.0086 → status=dead**，建议 gross 降到 **0.25**（ALERT=True，去风险）。
- `pit`：trailing IC = **+0.0203 → status=decaying**，建议 gross **0.758**（维持）。

原因：`daily_ic(soft, label)` 在两种宇宙下覆盖的票不同——`full` 把退市/停牌票的分数也算进 IC，
拉低了 IC 估计，使 regime 误判为 dead。这意味着**旧的"alpha 已死、应大幅降权"结论，
部分是由宇宙泄漏制造的假象**。在 PIT 诚实宇宙下，模型 alpha 只是"衰减"（IC≈0.02），并未彻底死亡。

## 5. 结论与建议

1. **默认用 `pit`**：`production_soft_score.py` 已默认 `universe="pit"`，所有日常打分/监控
   都在诚实宇宙下进行，避免用未来名单污染训练与回测。
2. **重述基线**：Route C 真实 OOS 基线应记为 **sharpe≈0.76**（非 0.998）；0.998 仅作上界参考。
3. **regime 闸门以 PIT 为准**：不要再基于 `full` 的 dead 信号做硬降权；PIT 下当前为 decaying，
   按 `joint` 调度给 ~0.76 gross 即可。
4. **接入自选股/因子池前必做**：用 `pit.detect_static_leak(close, amount, static_list=watchlist)`
   量化该名单的前视污染度，污染度高的名单不能直接喂模型。
5. **仍待补**：完整幸存者校正需补"早期/窗口前退市票"（当前面板已按真实 IPO 日进表、无回填，
   但更早退市票缺失），属残余轻度偏差，对 0.76 影响有限。

## 6. 复现命令

```bash
# 诚实基线（默认）
python production_soft_score.py --universe pit

# 旧上界（复现 0.998）
python production_soft_score.py --universe full

# 前视反例（诊断）
python production_soft_score.py --universe static-survivor

# 量化某 watchlist 的前视污染度
python -c "import pit_universe as pit, pandas as pd; c=pd.read_pickle('...'); a=pd.read_pickle('...'); print(pit.detect_static_leak(c,a,static_list=wl))"
```

产物：`outputs/production_soft_score/{soft_score_feed.csv, book_soft_equity.csv, regime_monitor.json, regime_alert.json}`
（每次运行按 `universe` 分缓存 `outputs/p10e_regime_gated/linear_mlp_scores_{universe}.npz`，互不串味）。
