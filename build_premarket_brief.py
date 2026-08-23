#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""盘前简报生成器（开盘前主动推送）。

读取最新一期的 overlay 报告（收盘后自动化步骤3 产出）与 regime_monitor，
生成开盘前可读的「今日关注 + 大盘状态 + 上一期 forward-test 表现」简报。

用法：
    python build_premarket_brief.py [--date YYYY-MM-DD] [--overlay <csv>] [--regime <json>]

--date 缺省取系统当天（即即将开盘的交易日）。overlay/regime 缺省自动找最新的。
输出：outputs/premarket/brief_<date>.md
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OVERLAY_DIR = ROOT / "outputs" / "watchlist_audit"
# 宽度降仓产出：收盘后自动化步骤3.5 用 apply_market_exposure.py 生成的非侵入式降仓版名单。
# 若缺失则回退到原始 full_overlay_calibrated.csv（不降仓，但仍是行为叠加口径）。
DERISKED_OVERLAY = OVERLAY_DIR / "full_overlay_calibrated_derisked.csv"
# 方案 B 并入版：merge_blend_into_overlay.py 产出。
# ⚠️ 隔离研究快照（2026-08-23 起）：未通过 run 一致性校验，禁止作为实盘主名单消费。
# 隔离态唯一真源 = .QUARANTINE 标记文件存在性（与 publish_blended_release.py G2 一致）：
# 标记存在 → 隔离生效，主名单回退降仓版→原始 overlay；解除隔离 = 删除该标记文件。
OVERLAY_Q = OVERLAY_DIR / "full_overlay_calibrated_blended.csv.QUARANTINE"
BLENDED_QUARANTINED = OVERLAY_Q.exists()
BLENDED_OVERLAY = OVERLAY_DIR / "full_overlay_calibrated_blended.csv"
SIGNAL_DIR = ROOT / "outputs" / "market_regime"
REGIME_DIR = ROOT / "outputs" / "ab_pit"
LEDGER = ROOT / "outputs" / "forward_test" / "ledger.csv"
OUT_DIR = ROOT / "outputs" / "premarket"
# 方案 B: trend_ignition × next_open_rank 加权混合候选 (produce_daily_blended_pool.py 产出)
BLENDED_POOL = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "blended_pool_latest.csv"


def _latest(pattern: str) -> str | None:
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def load_latest_overlay(specified: str | None) -> tuple[pd.DataFrame, str | None]:
    # ⚠️ 隔离期（BLENDED_QUARANTINED=True）：跳过 blended overlay，主名单直接回退降仓版→原始 overlay。
    if specified:
        path = specified
    else:
        if BLENDED_QUARANTINED:
            path = _latest(str(DERISKED_OVERLAY)) or (str(DERISKED_OVERLAY) if DERISKED_OVERLAY.exists() else None)
        else:
            blended = _latest(str(BLENDED_OVERLAY)) or (str(BLENDED_OVERLAY) if BLENDED_OVERLAY.exists() else None)
            path = blended or _latest(str(DERISKED_OVERLAY)) or (str(DERISKED_OVERLAY) if DERISKED_OVERLAY.exists() else None)
        path = path or _latest(str(OVERLAY_DIR / "full_overlay_calibrated.csv"))
    if not path or not os.path.exists(path):
        return pd.DataFrame(), None
    df = pd.read_csv(path, dtype={"symbol": str})
    asof = None
    if "rec_date" in df.columns and len(df):
        asof = str(df["rec_date"].iloc[0])
    return df, asof


def load_latest_regime(specified: str | None) -> dict | None:
    path = specified or _latest(str(REGIME_DIR / "regime_monitor_*.json"))
    if not path or not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def load_breadth_signal(specified: str | None) -> dict | None:
    """读取市场宽度信号（收盘后自动化步骤3.5 由 market_breadth_signal.py 产出）。

    与仪表盘 build_dashboard.load_breadth_signal 同口径，仅在盘前简报补充「自动降仓」提示。
    """
    path = specified or (str(SIGNAL_DIR / "breadth_signal.json"))
    if not os.path.exists(path):
        alt = _latest(str(SIGNAL_DIR / "breadth_signal_*.json"))
        path = alt or path
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def load_prev_forward_test() -> dict | None:
    if not LEDGER.exists():
        return None
    try:
        df = pd.read_csv(LEDGER)
    except Exception:
        return None
    if df.empty:
        return None
    latest = df["rec_date"].max()
    sub = df[df["rec_date"] == latest]
    out: dict = {"rec_date": latest}
    for h in (1, 5, 20):
        col = f"fwd_ret_{h}d"
        vals = sub[col].dropna()
        if len(vals):
            out[f"ret_{h}d"] = float(vals.mean())
        colb = f"bench_ret_{h}d"
        vb = sub[colb].dropna()
        if len(vb):
            out[f"bench_{h}d"] = float(vb.mean())
    return out


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x*100:+.2f}%"


def load_blended_pool() -> dict:
    """方案 B 研究信号面板。gated: 文件缺失/空返回空。"""
    if not BLENDED_POOL.exists():
        return {"asof": None, "rows": []}
    try:
        df = pd.read_csv(BLENDED_POOL, dtype={"symbol": str})
    except Exception:
        return {"asof": None, "rows": []}
    if df.empty:
        return {"asof": None, "rows": []}
    asof = str(df["asof"].iloc[0]) if "asof" in df.columns else None
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "symbol": str(r["symbol"]).zfill(6),
            "ti": (float(r["ti_score"]) if pd.notna(r.get("ti_score")) else None),
            "nor": (float(r["nor_score"]) if pd.notna(r.get("nor_score")) else None),
            "blended": (float(r["blended_score"]) if pd.notna(r.get("blended_score")) else None),
            "inter": (bool(r.get("in_intersection")) if "in_intersection" in df.columns else False),
        })
    return {"asof": asof, "rows": rows}


def build(brief_date: str, overlay_path: str | None, regime_path: str | None, signal_path: str | None = None) -> str:
    df, asof = load_latest_overlay(overlay_path)
    regime = load_latest_regime(regime_path)
    signal = load_breadth_signal(signal_path)
    prev = load_prev_forward_test()
    blended = load_blended_pool()
    # overlay 无 rec_date 列时，用 forward-test 台账的真实推荐日回填
    if asof is None and prev:
        asof = prev.get("rec_date")

    lines: list[str] = []
    lines.append(f"# 盘前简报 · {brief_date}")
    lines.append("")
    src = asof or "上一交易日"
    lines.append(f"> 数据截至 **{src}** 收盘（收盘后自动化产出），盘前推送，仅供开盘参考。")
    lines.append("")

    # ---- 大盘 regime ----
    lines.append("## 大盘 Regime")
    if regime:
        status = regime.get("status", "n/a")
        scale = regime.get("recommended_gross_scale", 1.0)
        dead = regime.get("alpha_dead_today", False)
        alert = regime.get("alert", False)
        ic = regime.get("trailing_ic_current")
        ic_s = f"{ic:.4f}" if isinstance(ic, (int, float)) else "n/a"
        lines.append(f"- 状态：**{status}** ｜ 建议仓位缩放：**{scale:.2f}x** ｜ 动作：{regime.get('recommended_action', 'maintain')}")
        lines.append(f"- Alpha 是否失效：**{'是' if dead else '否'}** ｜ 告警：{'有' if alert else '无'}")
        lines.append(f"- 近程 trailing IC（当前）：{ic_s}")
    else:
        lines.append("- 无 regime 数据（收盘后自动化步骤2 未产出）。")
    lines.append("")

    # ---- 市场宽度信号·自动降仓 ----
    lines.append("## 市场宽度信号 · 自动降仓")
    if signal:
        b = signal.get("breadth_above_ma60")
        exp = signal.get("exposure_target", 1.0)
        reg = signal.get("regime")
        seats = signal.get("recommended_seats")
        b_str = f"{b*100:.1f}%" if isinstance(b, (int, float)) else "—"
        reg_cn = {"normal": "常态满仓", "caution": "谨慎·宽度预警", "crash_defense": "崩溃防御"}.get(reg, "—")
        lines.append(f"- 站上 MA60 个股占比（宽度）：**{b_str}**")
        lines.append(f"- 宽度信号状态：**{reg_cn}** ｜ 建议整体暴露：**{exp:.2f}x** ｜ 今日推荐席位数：**{seats} 席**（满仓基准 20 席）")
        if reg in ("caution", "crash_defense"):
            lines.append("- ⚠️ 宽度塌陷，收盘后自动化已自动对名单降仓（减席位 + 缩权重），盘前名单即为降仓后结果，请勿自行加回仓位。")
        else:
            lines.append("- 宽度健康，名单维持满仓基准 20 席。")
    else:
        lines.append("- 无宽度信号（收盘后自动化步骤3.5 未产出）。名单按原始 overlay 口径，未做降仓。")
    lines.append("")

    # ---- 今日关注 ----
    lines.append("## 今日关注（行为叠加最终 20 席）")
    if df.empty or "personal_selected" not in df.columns:
        lines.append("- 无 overlay 推荐数据。")
    else:
        sel = df[df["personal_selected"] == True]  # noqa: E712
        if sel.empty:
            sel = df[df.get("selected") == True]  # noqa: E712
        n = len(sel)
        wl = int(sel.get("user_watchlist", pd.Series([False] * n)).fillna(False).sum()) if "user_watchlist" in sel else 0
        lines.append(f"共 **{n}** 席，等权各 5%。其中 **{wl}** 只为自选股（★）。")
        lines.append("")
        lines.append("| # | 代码 | 名称 | 动作 | 自选 | 备注 |")
        lines.append("|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(sel.iterrows(), 1):
            sym = str(r.get("symbol", "")).zfill(6)
            name = r.get("stock_name", "") or ""
            act = r.get("personal_action_cn", "") or r.get("personal_action", "") or ""
            star = "★" if bool(r.get("user_watchlist", False)) else ""
            note = r.get("personal_reasons_cn", "") or ""
            if isinstance(note, (list, tuple)):
                note = " ".join(map(str, note))
            note = str(note)[:28]
            lines.append(f"| {i} | {sym} | {name} | {act} | {star} | {note} |")
        lines.append("")

    # ---- 趋势点火×NOR 混合候选（方案B 隔离研究快照）----
    lines.append("## 趋势点火 × next_open_rank 混合候选（方案B·隔离研究快照，未并入实盘）")
    if blended["rows"]:
        lines.append(f"> as-of **{blended['asof']}** · <b>隔离研究快照</b>：方案 B 混合候选未通过 run 一致性校验（不同批次产物 / 并入排序 bug），已暂停实盘消费，下表仅为研究明细，不作为推荐依据。")
        lines.append("")
        lines.append("| # | 代码 | TI评分 | NOR评分 | 混合分 | 双覆盖 |")
        lines.append("|---|---|---|---|---|---|")
        for i, s in enumerate(blended["rows"][:20], 1):
            ti = f"{s['ti']:.3f}" if s["ti"] is not None else "—"
            nor = f"{s['nor']:.3f}" if s["nor"] is not None else "—"
            bl = f"{s['blended']:.3f}" if s["blended"] is not None else "—"
            star = "✓" if s["inter"] else ""
            lines.append(f"| {i} | {s['symbol']} | {ti} | {nor} | {bl} | {star} |")
        lines.append("")
    else:
        lines.append("- 无混合候选（blended_pool 未生成）。")
        lines.append("")

    # ---- 上一期 forward-test ----
    lines.append("## 上一期 Forward-test 表现")
    if prev:
        rd = prev.get("rec_date")
        r1, b1 = prev.get("ret_1d"), prev.get("bench_1d")
        extra = ""
        if r1 is not None and b1 is not None:
            extra = f" ｜ 基准 {fmt_pct(b1)} → 超额 {fmt_pct(r1 - b1)}"
        lines.append(f"- {rd} 推荐组合：1日 {fmt_pct(r1)}{extra}")
        lines.append(f"- 5日 {fmt_pct(prev.get('ret_5d'))} ｜ 20日 {fmt_pct(prev.get('ret_20d'))}（样本外滚动累计中）")
    else:
        lines.append("- 暂无 forward-test 数据（收盘后自动化步骤4 尚未产出）。")
    lines.append("")

    # ---- 风险提示 ----
    lines.append("## 风险提示")
    lines.append("- 本组合为**完全按用户自选股名单出票**（watchlist_quota=20），非模型自由选股；")
    lines.append("  历史诚实 PIT A/B 显示该宇宙显著跑输全市场（sharpe 约 −6.7 / 总收益约 −11.5%）。")
    lines.append("- 本简报为纪律性跟踪工具，不构成任何买卖建议。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="盘前简报生成器")
    ap.add_argument("--date", default=date.today().isoformat(), help="盘前交易日 YYYY-MM-DD（缺省=今天）")
    ap.add_argument("--overlay", default=None, help="指定 overlay CSV（缺省优先找降仓版，回退原始）")
    ap.add_argument("--regime", default=None, help="指定 regime_monitor JSON（缺省找最新）")
    ap.add_argument("--signal", default=None, help="指定 breadth_signal JSON（缺省找最新）")
    args = ap.parse_args()

    md = build(args.date, args.overlay, args.regime, args.signal)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"brief_{args.date}.md"
    out.write_text(md, encoding="utf-8")
    print(f"[premarket] 已写出 → {out}")
    print(md)


if __name__ == "__main__":
    main()
