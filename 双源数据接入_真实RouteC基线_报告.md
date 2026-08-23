# 双源行情数据接入 + 真实 Route C 基线报告

> 生成日期：2026-08-05
> 项目：`workbuddy_strategy / high_risk_quant_model3`
> 任务：行情数据接入（通达信历史 + 腾讯自选股最新），并产出首个**真实数据** Route C 生产基线

---

## 0. TL;DR

- **数据管线端到端跑通**：通达信 `hsjday.zip` → 归一化日线面板（880 交易日 / 5910 只 / 484 万行），腾讯自选股补齐最新尾巴，双源合并为统一 `data_panel.csv`。
- **真实 Route C 生产基线：sharpe = 0.998**（总收益 +63.4%，最大回撤 -19.0%，AUM=1 亿）。与项目记忆 "Route C 生产配置 sharpe≈1.0" 完全吻合。
- **regime 监控给出关键信号**：截至 2026-07-23，模型 alpha 已**失效（dead）**——trailing IC 从 1 月 +0.075 跌到 -0.0086，系统自动建议把 gross 暴露降至 **0.25** 防御。`ALERT=True`。
- **默认配置不是生产配置**：朴素 `train_next_open_rank_model.py --data` 在真实数据上是**负 alpha**（sharpe -0.14），印证必须使用 Route C 生产链路。

---

## 1. 接入架构

```
            ┌──────────────┐         ┌────────────────────┐
            │ 通达信 Tdx    │         │ 腾讯自选股 (gtimg)  │
            │ hsjday.zip   │         │ 公开K线接口(无鉴权) │
            └──────┬───────┘         └─────────┬──────────┘
                   │                           │
        ingest_tdx_zip.py              ingest_westock_recent.py
                   │ 归一化逐日CSV               │ 归一化逐日CSV
                   ▼                           ▼
   external_data/daily-market-data-tdx/    external_data/daily-market-data-westock/
   ths_exports/normalized/                  ths_exports/normalized/
   (ths_hs_a_share_YYYY-MM-DD.csv)          (ths_hs_a_share_YYYY-MM-DD.csv)
                   │                           │
                   └─────────┬─────────────────┘
                             ▼
                  merge_panel_sources.py  （按 (date,symbol) 去重，
                            腾讯覆盖 Tdx 同日期）
                             ▼
                  external_data/daily-market-data/data_panel.csv
                             ▼
        ┌─────────────────────────────────────────────────┐
        │ production_soft_score.py  (Route C 生产链路)       │
        │  → soft_score_feed / regime 监控 / book 回测权益  │
        └─────────────────────────────────────────────────┘
```

两端产出**完全同构**：列名 `date,symbol,open,high,low,close,volume,amount`，`symbol` 为纯 6 位零填充字符串，与下游 `build_data_panel.py` / `train_next_open_rank_model.py` 契约一致。

---

## 2. 通达信历史接入（ingest_tdx_zip.py）

| 项 | 值 |
|---|---|
| 数据源 | `D:\数据源\6月12日\hsjday.zip`（542.8 MB，12299 个 `.day`） |
| 输出 | 880 个逐日归一化 CSV（2022-12-05 → 2026-07-23） |
| 标的数量 | 5910（沪 2366 / 深 2980 / 北 564） |
| 总行数 | 4,840,000+ |
| volume 单位自检 | "股"（shares），量价一致性 = 1.0（无单位错配） |

> 远超 walk-forward 的 ≥300 交易日门槛，足以支撑真实训练。

---

## 3. 腾讯自选股最新尾巴（ingest_westock_recent.py）

- **接口**：`web.ifzq.gtimg.cn/appstock/app/kline/kline`（已验证可直连，无需鉴权，沙箱可通）
- **K 线数组顺序（腾讯 quirky）**：`[date, open, close, high, low, volume]`
- **冒烟测试**：5 只样本 × 15 交易日真实拉取，价格量纲自洽（如 600519 茅台 1309.60、600000 浦发 9.27）。
- **全市场尾巴（2026-07-24 → 今天）**：5910 只顺序限速拉取，**仍在后台运行中**（写盘在收尾一次性落，详见 §6）。完成后由 `merge_panel_sources.py` 拼入完整面板。

---

## 4. 双源合并（merge_panel_sources.py）

- 按 `(date, symbol)` 去重，**腾讯自选股覆盖 Tdx 同日期**（优先级验证通过：合成测试里 07-16 重叠日取真实 8.85 而非 Tdx 假值 99.2）。
- 当前已合并 **Tdx 单源**面板 → `data_panel.csv`（484 万行 / 880 日 / 5910 只）。
- 完整双源面板（含腾讯尾巴）待 §6 尾巴到位后一键产出。

---

## 5. 真实 Route C 生产基线（production_soft_score.py）

在真实 Tdx 面板上运行 Route C 生产配置：

| 配置项 | 值 |
|---|---|
| 佣金 / 印花税(仅卖) / 冲击 | 3.0 bps / 5.0 bps / 0.7 bps |
| 再平衡频率 | 5（降频甜区，砍 4× 换手） |
| 持仓名数 / 单票上限 / ADV 参与 | 40 / 4% / 2% |
| AUM | 1 亿 |
| regime 调度 | joint（regime × 容量联合，dead 自动降权） |

**回测结果（asof = 2026-07-23）：**

| 指标 | 值 |
|---|---|
| 总收益 | **+63.43%** |
| **Sharpe** | **0.998** |
| 最大回撤 | -19.01% |

**regime 监控（核心信号）：**

| 字段 | 值 | 含义 |
|---|---|---|
| status | **dead** | alpha 已失效 |
| trailing_ic_current | **-0.0086** | 当前 60 日滚动 IC 为负 |
| recent_dead%_60d | 0.383 | 近 60 日 38.3% 交易日 alpha 死亡 |
| adv_today | 1.000 | 全线性防御配比 |
| recommended_gross_scale | **0.25** | 建议把簿暴露压到地板 |
| ALERT | **True** | 触发降权 |

**trailing IC 衰减全过程（关键发现）：**

```
2026-01  ~+0.075   ← 强 alpha
2026-03  ~+0.060
2026-04  ~+0.030   ← 开始衰减
2026-05  ~+0.004   ← 逼近 0
2026-06  ~-0.003   ← 跌破 0
2026-07-23 -0.0086 ← 持续负，触发 dead
```

> 这正是 regime 门控存在的意义：模型在 1 月还有 +0.075 的强 alpha，到 7 月已彻底失效。生产配置**不是在失效期硬扛**，而是自动识别并降权到 0.25 防御——这正是它能长期维持 sharpe≈1.0 的鲁棒性来源。

---

## 6. 对照：默认配置 vs 生产配置

| 链路 | 真实数据结果 | 说明 |
|---|---|---|
| `train_next_open_rank_model.py --data`（默认 unconstrained / 日频 / 无印花税） | final_equity 873,962（**-12.6%**），sharpe **-0.14**，年化 -5.3%，最大回撤 -33.7% | 朴素弱基线，**证明工程链路在真实数据上通，但默认参数非生产配置** |
| `p10c_ensemble.py`（集成 IC 测试，小 AUM=1M） | 线性 IC=0.0634 / MLP IC=0.0427 / 集成 IC≈0.062；MLP-only **sharpe=0.792**（+41.6%，dd -23.6%）；incumbent sharpe=0.535 | 生产链路在真实数据上**有真实 edge** |
| `production_soft_score.py`（Route C 生产配置，AUM=1 亿） | **sharpe=0.998**（+63.4%，dd -19.0%） | **首个真实 Route C 基线，与项目记忆一致** |

---

## 7. 交付物清单

| 文件 | 说明 |
|---|---|
| `ingest_westock_recent.py` | 腾讯自选股近期日线接入器（新增） |
| `merge_panel_sources.py` | 双源面板合并器（新增） |
| `INGEST_RUNBOOK.md` | 接入运行手册（新增） |
| `external_data/daily-market-data-tdx/...` | Tdx 880 日归一化面板（880 个 CSV） |
| `external_data/daily-market-data/data_panel.csv` | 合并面板（当前为 Tdx 单源，484 万行） |
| `outputs/production_soft_score/soft_score_feed.csv` | Route C soft 分 feed |
| `outputs/production_soft_score/regime_monitor.json` | regime 监控（含 120 日 trailing IC 序列） |
| `outputs/production_soft_score/regime_alert.json` | regime 告警（ALERT=True → 降权 0.25） |
| `outputs/production_soft_score/book_soft_equity.csv` | book 回测权益曲线 |
| `outputs/p10c_ensemble/metrics.json` | 集成 IC / 多 AUM 回测指标 |

---

## 8. 待完成 & 下一步

1. **腾讯全市场尾巴仍在拉取**（5910 只顺序限速，预计再需一段时间）。完成后运行：
   ```bash
   python merge_panel_sources.py \
     --source-dirs external_data/daily-market-data-tdx/ths_exports/normalized \
                     external_data/daily-market-data-westock/ths_exports/normalized \
     --output external_data/daily-market-data/data_panel.csv
   ```
   即得到 **2022-12-05 → 今天** 的完整双源面板。
2. **日更流程**：此后每日只需跑步骤 3（腾讯尾巴）+ 4（合并）+ 生产打分，Tdx 历史不动。
3. **当前建议**：regime 为 `dead`，若实盘部署，按 `regime_alert.json` 把 gross 暴露压到 **0.25** 再进场。

---

## 9. 运行环境备注

- 本机使用系统 Python 3.12（`C:\Users\86176\AppData\Local\Programs\Python\Python312\python.exe`，managed 3.13 未装 pandas）。
- `production_soft_score.py` 首次运行需 `mkdir outputs/p10e_regime_gated`（np.savez 不自动建目录，已排查修复）。
- 腾讯接口为公开端点，沙箱可直连，无需任何密钥。
