# 行情数据接入运行手册（通达信历史 + 腾讯自选股最新）

> 目标：把 `high_risk_quant_model3` 的空数据面板补齐为**真实可训练**的长表面板。
> 架构：**历史主体走通达信（Tdx）**，`hsjday.zip` → 逐日归一化 CSV；
> **最新尾巴走腾讯自选股（腾讯财经公开行情）**，补 Tdx 末日之后的交易日。
> 两套归一化 CSV 由 `merge_panel_sources.py` 按 (date,symbol) 拼接为统一 `data_panel.csv`，
> 再喂给 `train_next_open_rank_model.py --data`。

---

## 0. 目录约定（已建好骨架）

```
high_risk_quant_model3/
├── external_data/
│   ├── daily-market-data-tdx/ths_exports/normalized/   # Tdx 历史归一化 CSV 落点（ingest_tdx_zip 默认输出）
│   ├── daily-market-data-westock/ths_exports/normalized/ # 腾讯自选股近期归一化 CSV 落点（ingest_westock_recent 默认输出）
│   └── daily-market-data/                                # QUANT_DATA_ROOT；合并后的 data_panel.csv 落点
├── ingest_tdx_zip.py          # 已有：Tdx .day → 归一化 CSV
├── ingest_westock_recent.py   # 新增：腾讯自选股日线 → 归一化 CSV
├── merge_panel_sources.py     # 新增：多源合并 → data_panel.csv
└── train_next_open_rank_model.py  # 训练入口，读 --data <面板>
```

---

## 1. 前置条件

- Python 3.11/3.12 + `pandas`、`numpy`（本机实测用 `C:\Users\86176\AppData\Local\Programs\Python\Python312\python.exe`，已带 pandas 2.3.3）。
- **`hsjday.zip`**（通达信沪深京合一原始 `.day` 归档）—— **需你提供，当前工作区没有**。放任意位置，步骤 2 用 `--zip` 指向它。
- 拉腾讯自选股需要**能访问公网** `web.ifzq.gtimg.cn`（已实测沙箱可通）。

---

## 2. 全流程（在 `high_risk_quant_model3/` 下执行）

### 步骤 1 — 放好 Tdx 归档
把 `hsjday.zip` 放到项目内某处，例如 `external_data/hsjday.zip`。

### 步骤 2 — Tdx 历史接入（产出历史主体）
```bash
python ingest_tdx_zip.py --zip external_data/hsjday.zip --limit-days 880
```
- 默认 `--limit-days 880`（≈3.5 年，足够 walk-forward 的 ≥300 交易日窗口）。
- 产出：`external_data/daily-market-data-tdx/ths_exports/normalized/ths_hs_a_share_YYYY-MM-DD.csv`
- 自检会打印交易日范围、标的数、市场分布、`volume` 单位推断、剔除的指数/基金数。

### 步骤 3 — 腾讯自选股近期接入（补最新尾巴）
```bash
python ingest_westock_recent.py --from-tdx-dir external_data/daily-market-data-tdx/ths_exports/normalized
```
- `--from-tdx-dir` 会自动取 Tdx 面板**最近 5 天**的全部标的作为 universe，只补这些标的的近期日线，
  窗口 = `[Tdx 末日+1, 今天]`，无需手填代码。
- 也可显式指定：`--symbols sh600000,sz000001,600519` 或 `--universe symbols.txt`。
- 产出：`external_data/daily-market-data-westock/ths_exports/normalized/ths_hs_a_share_YYYY-MM-DD.csv`
- 全市场 5000+ 标的对公网接口是~分钟级；若遇限流报错，调大脚本内 `time.sleep` 或分批。

### 步骤 4 — 合并为统一面板
```bash
python merge_panel_sources.py \
  --source-dirs external_data/daily-market-data-tdx/ths_exports/normalized \
                   external_data/daily-market-data-westock/ths_exports/normalized \
  --output external_data/daily-market-data/data_panel.csv
```
- **顺序即优先级**：Tdx 在前、腾讯自选股在后 → 重叠日期（拼接边界）由腾讯自选股覆盖，保证最新价优先。
- 输出列固定：`date,symbol,open,high,low,close,volume,amount`，按 (date,symbol) 排序。

### 步骤 5 — 训练首个真实基线
```bash
python train_next_open_rank_model.py --data external_data/daily-market-data/data_panel.csv
```
- 默认配置下 `amount` 可缺（合成样本已验证无 amount 也能跑）；若开启容量/冲击/流动性特征，脚本要求 amount 列存在（腾讯自选股侧已用 `volume(手)*100*均价` 估算填入）。

---

## 3. 日常增量刷新（日更）

每天只需重复**步骤 3 + 步骤 4**（腾讯自选股自动延展到当天，合并覆盖边界），无需重跑 Tdx：
```bash
python ingest_westock_recent.py --from-tdx-dir external_data/daily-market-data-tdx/ths_exports/normalized
python merge_panel_sources.py --source-dirs external_data/daily-market-data-tdx/ths_exports/normalized external_data/daily-market-data-westock/ths_exports/normalized --output external_data/daily-market-data/data_panel.csv
```
如需让 `build_data_panel.py` 也能直接识别，可设环境变量 `QUANT_DATA_ROOT=external_data/daily-market-data`（合并后的归一化 CSV 也可直接放其 `ths_exports/normalized/` 下）。

---

## 4. 单位与边界注意事项（重要）

- **symbol 统一为纯 6 位字符串**（如 `000001`），与 Tdx 面板一致；写入时强制零填充，避免下游 int/str 不一致。
- **腾讯自选股 `volume` 单位为「手」(100 股)**，`amount` 由 `volume*100*均价` 估算（元）。Tdx 侧 volume 单位由 `ingest_tdx_zip` 自检推断。两者覆盖**不同日期区间**，仅在拼接边界有一日量纲切换；如开启容量类特征且边界出现离群，可在 `ingest_westock_recent.py` 增加 `--volume-unit` 归一化开关。
- **腾讯 K 线数组顺序特殊**：`[date, open, close, high, low, volume]`，已在接入器内正确映射为 OHLC。
- 若某标的腾讯侧缺失某日而 Tdx 有，则保留 Tdx 值（不强行补齐）。

---

## 5. 验证清单

- [ ] `ingest_tdx_zip` 打印 `n_days ≥ 300`、`n_symbols` 全市场量级（数千）
- [ ] `ingest_westock_recent` 打印 `symbols_ok` 接近请求数、`date_end == 今天`
- [ ] 合并后 `data_panel.csv` 的 `date_end` 为今天、`date_start` 为 Tdx 起点，重叠日 close 取自腾讯自选股
- [ ] `train_next_open_rank_model.py` 正常产出 metrics.json / run_card（非空权益曲线）

---

## 6. 实测记录（本工作区冒烟）

- 腾讯自选股接口 `web.ifzq.gtimg.cn/appstock/app/kline/kline` 已实测可拉取真实日线（5 只样本 × 15 交易日，价格量纲自洽）。
- `merge_panel_sources` 已用「合成 Tdx + 真实 westock」验证：Tdx 独有日保留、重叠日由 westock 覆盖（600000 在 07-16 取真实 8.85 而非合成 99.2）。
- 当前 `external_data/daily-market-data-westock/ths_exports/normalized/` 留有 5 只样本的真实近期 CSV 作为接入器可用性的证明；跑真实流水线时会被全市场数据自然覆盖/扩展。
