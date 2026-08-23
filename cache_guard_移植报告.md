# P0-1：把 model4 内容寻址缓存移植到 model3（根治换面板 npz 形状错）

> 行动来源：用户"继续" → 落实 `model4_对model3项目的帮助分析.md` 中 P0-1 建议。
> 时间：2026-08-05。

## 问题回顾（必踩的坑）
`production_soft_score.py` 的 `linear_mlp_scores_{universe}.npz` 只按 `universe` 分文件，
**不绑定面板**。当合并面板从 880 行增长到 889 行后，旧 npz 被读出时
`pd.DataFrame(d["linear"], index=..., columns=...)` 直接报错：

```
ValueError: Shape of passed values is (880,5910), indices imply (889,5910)
```

上一轮我们靠"手动 `find`+`rm` 删缓存"临时绕过，但换面板必再踩。

## 移植方案（来自 model4 `pipeline_cache.py` 的思想，但瘦身）
model4 用 `base_panel + benchmark + 每日文件 + 全部代码/配置 sha256` 做内容寻址。
对 model3 这个具体陷阱，**面板指纹** 已足够，不必搬整套 manifest 系统。

新增 `score_cache.py`，核心 `load_or_build(scores_npz, idx, symbols, build_fn, panel_csv=None, use_cache=True)`：
- 把 **面板指纹**（n_dates / n_symbols / first·last_date + 源文件 size·mtime）以 `meta` 数组**内嵌进 npz**；
- 每次加载都重新算当前面板指纹并比对：
  - 一致 → 命中缓存，直接 `pd.DataFrame(d["linear"], index=idx, columns=symbols)`；
  - **不一致 / 无 meta（legacy）/ 缺数组 / 损坏** → `unlink` 旧缓存 + 用当前面板**重建**；
  - 因此换面板（880→889）或重合并（mtime/size 变）都会自动失效，永不再触发形状错。

接入点（两处同构 bug，均已修）：
- `production_soft_score.py:331` —— 主生产路径（带 `panel_csv` 强寻址）。
- `p10e_causal.py:106` —— 同样有裸缓存（带 `PANEL` 强寻址）。
- `p10e_regime_gated.py:122` 经核查**根本不读缓存**（每次重建），无陷阱，未动。

## 实现中踩到的一个真 bug（已修，值得记）
第一版守卫在保存分支写成 `meta=np.array(_fp_str(cur_fp))`，而 `cur_fp` 已是
`_fp_str(panel_fingerprint(...))` 产出的 JSON 字符串 → **双重 `json.dumps`**，
npz 里存成了 `"{\"first_date\":...}"`（字符串的 JSON 形式）。加载时比对恒不相等 → 永远重建。
修复：`meta=np.array(cur_fp)`（cur_fp 即最终字符串，只编码一次）。

## 验证
- 单元守卫测试 7 个场景全绿：
  - 同面板二次命中（不重建）；
  - **面板形状变更（5→6 行，模拟 880→889）→ 自动重建且不崩**；
  - legacy 无 meta npz → 当 stale 重建；
  - `use_cache=False` → 强制重建；
  - 加 `panel_csv` 收紧指纹 → 重建；相同 `panel_csv` → 命中；
  - **重合并（改 mtime/size）→ 重建**（这正是真正触发陷阱的场景）。
- 三文件 `py_compile` 通过。
- **真实端到端**：后台在完整合并面板（→08-05）上重跑 `production_soft_score.py --universe pit`，
  应触发一次干净重建且无形状错（task `DRbR0T`，结果见 `logs/run_pit_guard_verify.log`）。

## 收益
- 换面板 / 重合并 / 换 universe 不再需要人工删缓存，缓存**按内容自动失效**。
- 比 model4 完整 manifest 系统轻；只解决当前最痛的一类错误，未引入额外复杂度。
- 若未来要更严（按代码/配置哈希），可在 `panel_fingerprint` 里加 `_included_project_files` 思想扩展。
