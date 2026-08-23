"""Forward-test 跟踪器 —— 把每日推荐清单落地为可验证的样本外(OOS)证据。

设计原则（与项目 PIT 方法论一致）：
  - 推荐清单 = 每日 run_daily_overlay.py 产物的「最终 20 只」(overlay CSV 中 selected==True)。
  - 前瞻收益只使用「决策日之后、且已真实落地」的行情，绝不前视。
  - 面板按交易日索引，horizon=1/5/20 表示「往后第 1/5/20 个交易日的收盘价」，天然规避周末/节假日。
  - 每条推荐随时间推移自动补齐：今天只有 1d，过几天自动补 5d/20d，状态 pending→partial→complete。

三个子命令：
  snapshot  把某日 overlay 报告的 20 只写入台账（默认取面板末日为推荐日；推荐日必须显式传入历史日）
  update    用最新面板重算所有台账行的 1/5/20 日前瞻收益（幂等，可每日跑）
  report    输出逐期 + 累计组合前瞻收益、命中率、对比 510300 基准

台账：outputs/forward_test/ledger.csv
报告：outputs/forward_test/report.md
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import cost_model

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data/daily-market-data/data_panel.csv"
BENCH = HERE / "external_data/benchmarks/510300.csv"
# 默认快照「真实推荐」= 降仓版 derisked overlay (方案B blended 已隔离, 见 .QUARANTINE 标记)。
# blended 仅作研究 A/B 候选 (--overlay 显式指定)。
OVERLAY = HERE / "outputs/watchlist_audit/full_overlay_calibrated_derisked.csv"
OVERLAY_BLENDED = HERE / "outputs/watchlist_audit/full_overlay_calibrated_blended.csv"
OVERLAY_DERISKED = HERE / "outputs/watchlist_audit/full_overlay_calibrated_derisked.csv"
LEDGER_DIR = HERE / "outputs/forward_test"
LEDGER = LEDGER_DIR / "ledger.csv"
REPORT = LEDGER_DIR / "report.md"

HORIZONS = [1, 5, 20]


# --------------------------------------------------------------------------- #
# 面板 / 基准 读取
# --------------------------------------------------------------------------- #
def load_panel_close(panel: Path) -> tuple[list[str], dict[str, pd.Series], dict[str, pd.Series]]:
    """读取面板的 [date, symbol, close/open]，pivot 成 (日期列表, {sym: close}, {sym: open})。
    symbol 统一 6 位补零。"""
    df = pd.read_csv(panel, usecols=["date", "symbol", "close", "open"],
                     dtype={"symbol": str}, low_memory=False)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.dropna(subset=["close"])
    wide = df.pivot(index="date", columns="symbol")
    wide = wide.sort_index()
    dates = list(wide.index)
    close_map = {sym: wide["close"][sym] for sym in wide["close"].columns}
    open_map = {sym: wide["open"][sym] for sym in wide["open"].columns}
    return dates, close_map, open_map


def load_benchmark(bench: Path):
    """读取 510300 基准 close（+open 若存在），供 next_open 口径 open-to-open 对齐。
    P1-4: 返回 (close_series, open_series|None)。"""
    if not bench.exists():
        return None, None
    cols = pd.read_csv(bench, nrows=1).columns
    use = ["date", "close"] if "open" not in cols else ["date", "open", "close"]
    df = pd.read_csv(bench, usecols=use)
    close_s = df.set_index("date")["close"]
    close_s = close_s[~close_s.index.duplicated(keep="last")].sort_index()
    open_s = None
    if "open" in cols:
        open_s = df.set_index("date")["open"]
        open_s = open_s[~open_s.index.duplicated(keep="last")].sort_index()
    return close_s, open_s


def _infer_variant(overlay: Path) -> str:
    """P1-4: variant 跟随数据源文件名，避免把 derisked 真实推荐误标成 blended。"""
    name = overlay.name.lower()
    if "blended" in name:
        return "blended"
    if "derisked" in name:
        return "control"
    return "legacy"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def panel_last_date(panel: Path) -> str:
    # 复用 load_panel_close 的日期列表末尾
    dates, _, _ = load_panel_close(panel)
    return dates[-1]


# --------------------------------------------------------------------------- #
# snapshot：把每日推荐清单写入账本
# --------------------------------------------------------------------------- #
def cmd_snapshot(args) -> None:
    overlay = Path(args.overlay)
    if not overlay.exists():
        raise SystemExit(f"缺失 overlay 报告: {overlay}")
    rec_date = args.rec_date or panel_last_date(Path(args.panel))
    df = pd.read_csv(overlay, dtype={"symbol": str})
    # 最终推荐清单 = 行为叠加后的 personal_selected（含 watchlist 配额补足的 20 只）。
    # 注意：overlay 里另有 selected(模型原始 top20) 列，其 personal_adjusted_target_weight 为 0
    # （被行为层替换），不是真实组合，绝不能用它。
    if "personal_selected" in df.columns:
        df = df[df["personal_selected"] == True]  # noqa: E712
    else:
        df = df[df["selected"] == True]  # noqa: E712
    if df.empty:
        raise SystemExit("overlay 中 final selected 为 0 行，无法快照")

    # P1-4: variant 跟随数据源（默认按文件名推断，显式传入优先），避免 A/B 误标
    variant = args.variant or _infer_variant(overlay)
    overlay_sha = _sha256_file(overlay)

    # 标记：是否来自方案B blended 顶入（仅 blended variant 有此列）
    has_blend_flag = "blend_inserted" in df.columns
    rows = []
    for _, r in df.iterrows():
        sym = str(r["symbol"]).zfill(6)
        # 真实部署权重：personal_selected 行上 target_weight 恒为 0，真正权重在
        # target_weight_after_behavior / personal_adjusted_target_weight（均=0.05 等权）。
        # 按优先级取非零值，最后兜底等权 0.05。
        for wc in ("target_weight_after_behavior", "personal_adjusted_target_weight", "target_weight"):
            v = r.get(wc)
            if pd.notna(v) and float(v) > 0:
                wval = float(v)
                break
        else:
            wval = 0.05  # 兜底等权
        rec = {
            "rec_date": rec_date,
            "variant": variant,
            "symbol": sym,
            "weight": wval,
            "score": float(r.get("score", float("nan"))),
            "adjusted_score": float(r.get("personal_adjusted_score", float("nan"))),
            "user_watchlist": bool(r.get("user_watchlist", False)),
            "rank": int(r.get("personal_rank", 0)),
            "blend_inserted": bool(r.get("blend_inserted", False)) if has_blend_flag else False,
            "overlay_sha": overlay_sha,  # P1-4: 溯源 overlay 版本
            "fwd_ret_1d": None, "fwd_ret_5d": None, "fwd_ret_20d": None,
            "bench_ret_1d": None, "bench_ret_5d": None, "bench_ret_20d": None,
            "status": "pending",
            "updated_at": "",
        }
        rows.append(rec)
    new = pd.DataFrame(rows)

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    # 合并前把数值列统一为 float、文本列统一为 str，避免 concat 因 all-NA 列触发 dtype FutureWarning
    _num = ["weight", "score", "adjusted_score", "rank",
            "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d",
            "bench_ret_1d", "bench_ret_5d", "bench_ret_20d"]
    _str = ["rec_date", "symbol", "variant", "overlay_sha", "updated_at", "status"]
    for c in _num:
        new[c] = pd.to_numeric(new[c], errors="coerce")
    for c in _str:
        new[c] = new[c].astype(str)
    if LEDGER.exists():
        old = pd.read_csv(LEDGER, dtype={"symbol": str, "rec_date": str})
        # 向后兼容：旧台账无 variant 列 → 回填 "legacy"（历史真实推荐，等同 blended 口径）
        if "variant" not in old.columns:
            old["variant"] = "legacy"
        if "blend_inserted" not in old.columns:
            old["blend_inserted"] = False
        if "overlay_sha" not in old.columns:
            old["overlay_sha"] = ""
        for c in _num:
            old[c] = pd.to_numeric(old[c], errors="coerce")
        for c in _str:
            old[c] = old[c].astype(str)
        # 去重键升级为 (rec_date, variant)，允许同日两个 variant 共存
        key = old["rec_date"].astype(str) + "|" + old["variant"].astype(str)
        if f"{rec_date}|{variant}" in set(key):
            print(f"[snapshot] {rec_date}/{variant} 已存在于台账，跳过（共 {len(old)} 行）")
            return
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged.to_csv(LEDGER, index=False)
    print(f"[snapshot] 写入 {rec_date}/{variant} 推荐 {len(new)} 只 → {LEDGER}（台账共 {len(merged)} 行）")


# --------------------------------------------------------------------------- #
# update：用最新面板重算前瞻收益
# --------------------------------------------------------------------------- #
def cmd_update(args) -> None:
    if not LEDGER.exists():
        raise SystemExit(f"台账不存在: {LEDGER}，请先 snapshot")
    ledger = pd.read_csv(LEDGER, dtype={"symbol": str, "rec_date": str})
    if "variant" not in ledger.columns:
        ledger["variant"] = "legacy"
    if "blend_inserted" not in ledger.columns:
        ledger["blend_inserted"] = False
    if "overlay_sha" not in ledger.columns:
        ledger["overlay_sha"] = ""
    ledger["updated_at"] = ledger["updated_at"].astype(object)  # 避免字符串写入 float 列触发 dtype 警告
    dates, close_map, open_map = load_panel_close(Path(args.panel))
    date_idx = {d: i for i, d in enumerate(dates)}
    bench_close, bench_open = load_benchmark(Path(args.bench)) if args.bench else (None, None)
    exec_mode = args.exec_mode

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_done = n_partial = n_pending = 0

    for idx, row in ledger.iterrows():
        rec = str(row["rec_date"])
        sym = str(row["symbol"]).zfill(6)
        if rec not in date_idx:
            ledger.at[idx, "status"] = "pending"
            continue
        ri = date_idx[rec]
        series = close_map.get(sym)
        oseries = open_map.get(sym)
        fwd = {}
        for h in HORIZONS:
            if exec_mode == "next_open":
                buy_i, sell_i = ri + 1, ri + 1 + h
            else:
                buy_i, sell_i = ri, ri + h
            if sell_i < len(dates) and series is not None:
                if exec_mode == "next_open" and oseries is not None:
                    p0 = oseries.get(dates[buy_i], None)
                    p1 = oseries.get(dates[sell_i], None)
                else:
                    p0 = series.get(rec, None) if exec_mode != "next_open" else series.get(dates[buy_i], None)
                    p1 = series.get(dates[sell_i], None)
                if pd.notna(p0) and pd.notna(p1) and p0 != 0:
                    gross = p1 / p0 - 1.0
                    # 扣成本 (next_open 模式用开盘滑点近似)
                    if args.no_cost:
                        net = gross
                    else:
                        net = cost_model.apply_costs(gross, weight=float(row.get("weight", 0.05) or 0.05))
                    fwd[h] = net
                else:
                    fwd[h] = None
            else:
                fwd[h] = None
        # 基准（510300）同质 horizon；P1-4: next_open 下基准与个股同口径 open-to-open
        bret = {}
        bd = ri + 1 if exec_mode == "next_open" else ri
        if bench_close is not None and rec in bench_close.index:
            for h in HORIZONS:
                tgt = ri + 1 + h if exec_mode == "next_open" else ri + h
                if tgt < len(dates):
                    if exec_mode == "next_open" and bench_open is not None:
                        b0 = bench_open.get(dates[bd], None)
                        b1 = bench_open.get(dates[tgt], None)
                    else:
                        b0 = bench_close.get(rec, None)
                        b1 = bench_close.get(dates[tgt], None)
                    if pd.notna(b0) and pd.notna(b1) and b0 != 0:
                        bret[h] = b1 / b0 - 1.0
                    else:
                        bret[h] = None
                else:
                    bret[h] = None

        for h in HORIZONS:
            ledger.at[idx, f"fwd_ret_{h}d"] = fwd.get(h)
            ledger.at[idx, f"bench_ret_{h}d"] = bret.get(h)

        avail = [h for h in HORIZONS if fwd.get(h) is not None]
        if len(avail) == len(HORIZONS):
            st = "complete"
            n_done += 1
        elif avail:
            st = "partial"
            n_partial += 1
        else:
            st = "pending"
            n_pending += 1
        ledger.at[idx, "status"] = st
        ledger.at[idx, "updated_at"] = now

    ledger.to_csv(LEDGER, index=False)
    print(f"[update] 台账 {len(ledger)} 行：complete={n_done} partial={n_partial} pending={n_pending}；面板末日={dates[-1]}")


# --------------------------------------------------------------------------- #
# report：逐期 + 累计
# --------------------------------------------------------------------------- #
def cmd_report(args) -> None:
    if not LEDGER.exists():
        raise SystemExit(f"台账不存在: {LEDGER}，请先 snapshot")
    ledger = pd.read_csv(LEDGER, dtype={"symbol": str, "rec_date": str})
    if "variant" not in ledger.columns:
        ledger["variant"] = "legacy"
    if "blend_inserted" not in ledger.columns:
        ledger["blend_inserted"] = False
    if "overlay_sha" not in ledger.columns:
        ledger["overlay_sha"] = ""

    def port_ret(sub: pd.DataFrame, h: int) -> float | None:
        w = sub["weight"].fillna(0.0)
        r = sub[f"fwd_ret_{h}d"]
        m = r.notna() & w.notna()
        if m.sum() == 0 or w[m].sum() == 0:
            return None
        return float((r[m] * w[m]).sum() / w[m].sum())

    def bench_ret(sub: pd.DataFrame, h: int) -> float | None:
        r = sub[f"bench_ret_{h}d"].dropna()
        return float(r.mean()) if len(r) else None

    lines = ["# Forward-test 跟踪报告", ""]
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"台账行数: {len(ledger)}，推荐期数: {ledger['rec_date'].nunique()}，"
                 f"variant: {sorted(ledger['variant'].unique().tolist())}")
    lines.append("")

    # 逐期（按 rec_date + variant 拆分行）
    lines.append("## 逐期组合前瞻收益（按权重加权，variant 分组）")
    lines.append("")
    lines.append("| 推荐日 | variant | 1日 | 5日 | 20日 | 基准1日 | 基准5日 | 基准20日 | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for (rec, variant), sub in ledger.groupby(["rec_date", "variant"]):
        pr = {h: port_ret(sub, h) for h in HORIZONS}
        br = {h: bench_ret(sub, h) for h in HORIZONS}
        st = "complete" if sub["status"].eq("complete").all() else (
            "partial" if sub["status"].eq("partial").any() else "pending")
        def fmt(x):
            return f"{x*100:+.2f}%" if x is not None else "—"
        lines.append(f"| {rec} | {variant} | {fmt(pr[1])} | {fmt(pr[5])} | {fmt(pr[20])} | "
                     f"{fmt(br[1])} | {fmt(br[5])} | {fmt(br[20])} | {st} |")
    lines.append("")

    # 累计 + variant 对比
    lines.append("## 累计统计 & 方案B增益（variant 对比）")
    lines.append("")
    completed = ledger.groupby(["rec_date", "variant"]).filter(
        lambda g: g["status"].eq("complete").all())
    n_done_dates = completed["rec_date"].nunique()
    n_total_dates = ledger["rec_date"].nunique()
    lines.append(f"- 已完整(20日可读)的推荐期数: {n_done_dates} / {n_total_dates}")
    # R11：20 日窗口冻结 —— 完整 20 日窗口 <20 前冻结增量结论（PRD R11, 2026-08-20）
    freeze_threshold = 20
    filled = min(10, int(round(n_done_dates / freeze_threshold * 10)))
    bar = "█" * filled + "░" * (10 - filled)
    if n_done_dates < freeze_threshold:
        lines.append(
            f"- ⚠️ **证据不足冻结**（R11）：完整 20 日窗口 {n_done_dates}/{freeze_threshold} [{bar}] "
            f"—— 增量结论冻结，仅作描述性记录，不对个人化组合下有效/无效定论"
        )
    else:
        lines.append(
            f"- ✅ 证据充足（R11）：完整 20 日窗口 {n_done_dates} [{bar}] ≥ {freeze_threshold}，可解锁增量结论"
        )

    # 每个 variant 单独统计
    for variant in sorted(ledger["variant"].unique()):
        vsub = ledger[ledger["variant"] == variant]
        lines.append("")
        lines.append(f"### variant = {variant}")
        for h in HORIZONS:
            vals = []
            for rec, sub in vsub.groupby("rec_date"):
                pr = port_ret(sub, h)
                if pr is not None:
                    vals.append(pr)
            if vals:
                mean = sum(vals) / len(vals)
                lines.append(f"- 组合 {h}日 平均前瞻收益: {mean*100:+.2f}%（{len(vals)} 期）")
        sym_obs = vsub[[f"fwd_ret_{h}d" for h in HORIZONS]].notna().sum().sum()
        if sym_obs:
            wins = sum((vsub[f"fwd_ret_{h}d"] > 0).sum() for h in HORIZONS)
            lines.append(f"- 个股-horizon 观测数: {int(sym_obs)}，正向占比(命中率): {wins/sym_obs*100:.1f}%")
            # blend 顶入个股的命中率（仅 blended variant 有意义）
            if "blend_inserted" in vsub.columns:
                be = vsub[vsub["blend_inserted"] == True]  # noqa: E712
                be_obs = be[[f"fwd_ret_{h}d" for h in HORIZONS]].notna().sum().sum()
                if be_obs:
                    be_wins = sum((be[f"fwd_ret_{h}d"] > 0).sum() for h in HORIZONS)
                    lines.append(f"- 其中 blend 顶入个股命中率: {be_wins/be_obs*100:.1f}%（{int(be_obs)} 观测）")

    # 方案B增益速览：blended vs control（取共同 horizon 均值差）
    variants = sorted(ledger["variant"].unique())
    if "blended" in variants and "control" in variants:
        lines.append("")
        lines.append("### 方案B增益速览（blended − control）")
        for h in HORIZONS:
            b_vals, c_vals = [], []
            for rec, sub in ledger[ledger["variant"] == "blended"].groupby("rec_date"):
                pr = port_ret(sub, h)
                if pr is not None:
                    b_vals.append(pr)
            for rec, sub in ledger[ledger["variant"] == "control"].groupby("rec_date"):
                pr = port_ret(sub, h)
                if pr is not None:
                    c_vals.append(pr)
            if b_vals and c_vals:
                bm = sum(b_vals) / len(b_vals)
                cm = sum(c_vals) / len(c_vals)
                lines.append(f"- {h}日 组合收益: blended {bm*100:+.2f}% vs control {cm*100:+.2f}% "
                             f"→ 增益 {(bm-cm)*100:+.2f}%（{len(b_vals)}/{len(c_vals)} 期）")
    lines.append("")
    lines.append("> 口径：默认 next_open（决策日收盘定候选，次日开盘买入，持有 h 个交易日至开盘卖出，"
                 "对齐策略名 next_open_rank）；可用 --exec-mode close 退回旧收盘口径。收益已扣交易成本"
                 "(印花税/佣金/滑点/冲击)。所有收益仅用已落地行情计算，无前视。"
                 "variant=blended 记方案B研究候选（当前隔离, 仅供 A/B 参考），variant=control 记降仓对照。"
                 "注：blended overlay 自 2026-08-23 起隔离为研究快照，不再作为实盘主名单。")

    txt = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n[report] 已写出 → {REPORT}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Forward-test 跟踪器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="把每日 overlay 报告的 20 只写入台账")
    s.add_argument("--overlay", default=str(OVERLAY),
                   help="overlay 候选池（默认=derisked 降仓版真实推荐；blended 已隔离仅作 A/B 参考）")
    s.add_argument("--variant", default=None,
                   help="台账 variant 标记；默认按 overlay 文件名推断（blended→blended / derisked→control / 其他→legacy）")
    s.add_argument("--rec-date", default=None, help="推荐日(YYYY-MM-DD)，默认=面板末日")
    s.add_argument("--panel", default=str(PANEL))
    s.set_defaults(func=cmd_snapshot)

    u = sub.add_parser("update", help="用最新面板重算前瞻收益")
    u.add_argument("--panel", default=str(PANEL))
    u.add_argument("--bench", default=str(BENCH))
    u.add_argument("--exec-mode", choices=("close", "next_open"), default="next_open",
                   help="执行时点: next_open=决策日收盘定候选,次日开盘买(对齐策略名,默认); close=旧口径")
    u.add_argument("--no-cost", action="store_true", help="不扣交易成本")
    u.set_defaults(func=cmd_update)

    r = sub.add_parser("report", help="输出统计报告")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
