#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dashboard.py — 生成每日量化自包含仪表盘（完全离线）

读取最新一期的：
  - overlay 报告（personal_selected 20 席）
  - regime_monitor_<token>.json（大盘状态 / 仓位缩放 / trailing IC）
  - forward-test 台账 + 报告
  - 健康巡检 alert_<date>.md
  - 生产基线（pit / full / static-survivor / watchlist 四档 Sharpe）

输出：outputs/dashboard/dashboard.html
  - **完全离线自包含**：图表由 Python 直接生成内联 SVG，不依赖任何 CDN / 外部 JS。
  - 红涨绿跌（A 股惯例），可直接双击打开或推前端。

用法：python build_dashboard.py [--date YYYY-MM-DD]
不传 --date 时用面板末日作为锚点。
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OVERLAY = ROOT / "outputs" / "watchlist_audit" / "full_overlay_calibrated.csv"
DERISKED_OVERLAY = ROOT / "outputs" / "watchlist_audit" / "full_overlay_calibrated_derisked.csv"
# 方案 B 并入版：merge_blend_into_overlay.py 产出。
# ⚠️ 隔离研究快照（2026-08-23 起）：经审查发现 blended overlay 与主名单/报告不同批次、
# 且并入脚本存在确定性排序 bug（P0-2），未通过 run 一致性校验，禁止作为实盘主名单消费。
# 隔离态唯一真源 = .QUARANTINE 标记文件存在性（与 publish_blended_release.py G2 一致）：
# 标记存在 → 隔离生效，dashboard 不把 blended overlay 当主名单来源，仅作隔离标注；
# 解除隔离 = 删除该标记文件（见报告§五前置条件），无需再改本脚本常量。
OVERLAY_Q = ROOT / "outputs" / "watchlist_audit" / "full_overlay_calibrated_blended.csv.QUARANTINE"
BLENDED_QUARANTINED = OVERLAY_Q.exists()
BLENDED_OVERLAY = ROOT / "outputs" / "watchlist_audit" / "full_overlay_calibrated_blended.csv"
SIGNAL = ROOT / "outputs" / "market_regime" / "breadth_signal.json"
REGIME_GLOB = str(ROOT / "outputs" / "ab_pit" / "regime_monitor_*.json")
LEDGER = ROOT / "outputs" / "forward_test" / "ledger.csv"
PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
ALERT_DIR = ROOT / "outputs" / "alerts"
DASH_DIR = ROOT / "outputs" / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)
# 方案 B: trend_ignition × next_open_rank 加权混合候选 (produce_daily_blended_pool.py 产出)
BLENDED_POOL = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "blended_pool_latest.csv"

# 生产基线（来自 2026-08-05 公平 A/B 研究，完整面板）
BASELINE = {
    "pit":             {"label": "诚实PIT宇宙",  "sharpe": 0.792, "color": "#2563eb"},
    "full":            {"label": "旧上界full",   "sharpe": 0.998, "color": "#94a3b8"},
    "static_survivor": {"label": "前视反例",     "sharpe": 0.219, "color": "#dc2626"},
    "watchlist_window":{"label": "自选股PIT窗口","sharpe": -6.706,"color": "#e11d48"},
}


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def _latest_date_sorted(globpat: str) -> str | None:
    files = glob.glob(globpat, recursive=True)
    if not files:
        return None
    def key(f):
        toks = re.findall(r"(\d{8})", os.path.basename(f))
        return int(toks[-1]) if toks else 0
    return sorted(files, key=key)[-1]


def load_regime() -> dict | None:
    f = _latest_date_sorted(REGIME_GLOB)
    return json.load(open(f, encoding="utf-8")) if f else None


def load_breadth_signal() -> dict | None:
    if not SIGNAL.exists():
        return None
    try:
        return json.load(open(SIGNAL, encoding="utf-8"))
    except Exception:
        return None


def load_overlay() -> pd.DataFrame:
    # ⚠️ 隔离期（BLENDED_QUARANTINED=True）：blended overlay 不接入实盘主名单，
    # 直接回退到降仓版（derisked）→ 原始 overlay，避免不同批次/排序 bug 污染实盘推荐。
    if BLENDED_QUARANTINED:
        src = (DERISKED_OVERLAY if DERISKED_OVERLAY.exists() else OVERLAY)
    else:
        src = (BLENDED_OVERLAY if BLENDED_OVERLAY.exists() else
               DERISKED_OVERLAY if DERISKED_OVERLAY.exists() else OVERLAY)
    if not src.exists():
        return pd.DataFrame()
    df = pd.read_csv(src)
    return df[df.get("personal_selected") == True]  # noqa: E712


def load_panel_last() -> str:
    return str(pd.read_csv(PANEL, usecols=["date"])["date"].max())


def load_alert(asof: str) -> dict:
    f = ALERT_DIR / f"alert_{asof}.md"
    if not f.exists():
        return {"conclusion": "无巡检文件", "items": []}
    txt = f.read_text(encoding="utf-8")
    conclusion = ""
    items = []
    for ln in txt.splitlines():
        if ln.startswith("**结论"):
            conclusion = ln.split(":", 1)[-1].strip().strip("*").strip()
        if ln.strip().startswith("|") and ("✅" in ln or "⚠️" in ln or "❌" in ln):
            parts = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(parts) >= 3:
                items.append({"name": parts[0], "status": parts[1], "detail": parts[2]})
    return {"conclusion": conclusion, "items": items}


def load_blended_pool() -> dict:
    """方案 B: trend_ignition × next_open_rank 加权混合候选（研究信号）。gated: 文件缺失/空返回空。"""
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
            "ti": float(r["ti_score"]) if pd.notna(r.get("ti_score")) else None,
            "nor": float(r["nor_score"]) if pd.notna(r.get("nor_score")) else None,
            "blended": float(r["blended_score"]) if pd.notna(r.get("blended_score")) else None,
            "rank": int(r["blended_rank"]) if pd.notna(r.get("blended_rank")) else None,
            "inter": bool(r.get("in_intersection")) if "in_intersection" in df.columns else False,
        })
    return {"asof": asof, "rows": rows}


def portfolio_returns(ledger: pd.DataFrame) -> list[dict]:
    out = []
    for rd, g in ledger.groupby("rec_date"):
        rec = {}
        for h in (1, 5, 20):
            w = g["weight"].fillna(0.0)
            r = g[f"fwd_ret_{h}d"]
            m = r.notna() & w.notna() & (w > 0)
            rec[f"r{h}"] = float((r[m] * w[m]).sum() / w[m].sum()) if m.sum() else None
            bw = g["weight"].fillna(0.0)
            br = g[f"bench_ret_{h}d"]
            bm = br.notna() & bw.notna() & (bw > 0)
            rec[f"b{h}"] = float((br[bm] * bw[bm]).sum() / bw[bm].sum()) if bm.sum() else None
        n = int((g["fwd_ret_1d"].notna() & (g["weight"] > 0)).sum())
        hit = int(((g["fwd_ret_1d"] > 0) & (g["weight"] > 0)).sum())
        out.append({
            "rec_date": str(rd), "r1": rec["r1"], "r5": rec["r5"], "r20": rec["r20"],
            "b1": rec["b1"], "b5": rec["b5"], "b20": rec["b20"],
            "hit_rate": round(hit / n, 4) if n else None, "n": n,
            "status": g["status"].iloc[0],
        })
    out.sort(key=lambda x: x["rec_date"])
    return out


def build_data(asof: str) -> dict:
    regime = load_regime() or {}
    overlay = load_overlay()
    ledger = pd.read_csv(LEDGER, dtype={"symbol": str, "rec_date": str}) if LEDGER.exists() else pd.DataFrame()
    alert = load_alert(asof)

    ic_series = regime.get("trailing_ic_series_last120d", {})
    ic_dates = sorted(ic_series.keys())
    ic_vals = [float(ic_series[d]) for d in ic_dates]

    seats = []
    if not overlay.empty:
        for _, r in overlay.iterrows():
            seats.append({
                "rank": int(r.get("personal_rank", 0) or 0),
                "symbol": str(r["symbol"]).zfill(6),
                "name": str(r.get("stock_name", "")),
                "action": str(r.get("personal_action_cn", "")),
                "weight": float(r.get("target_weight_after_behavior", 0.05) or 0.05),
                "score": round(float(r.get("personal_adjusted_score", float("nan")) or float("nan")), 4),
                "watch": bool(r.get("user_watchlist", False)),
                "reasons": str(r.get("personal_reasons_cn", "")),
                "blend_inserted": bool(r.get("blend_inserted", False)),
                "blend_boosted": bool(r.get("blend_boosted", False)),
                "ret20": (round(float(r["return_20d"]), 4) if pd.notna(r.get("return_20d")) else None),
            })

    ft = portfolio_returns(ledger) if not ledger.empty else []
    return {
        "asof": asof,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": {
            "asof": regime.get("asof"), "status": regime.get("status"),
            "scale": regime.get("recommended_gross_scale"), "alpha_dead": regime.get("alpha_dead_today"),
            "ic_current": regime.get("trailing_ic_current"), "ic_60d": regime.get("trailing_ic_recent_60d"),
            "dead_pct_60d": regime.get("recent_dead_pct_60d"), "universe": regime.get("universe"),
            "bt_sharpe": (regime.get("backtest_full", {}) or {}).get("sharpe_like"),
            "bt_total": (regime.get("backtest_full", {}) or {}).get("total_return"),
            "bt_dd": (regime.get("backtest_full", {}) or {}).get("max_drawdown"),
            "thr_hi": (regime.get("thresholds", {}) or {}).get("thr_hi"),
            "ic_dates": ic_dates, "ic_vals": ic_vals,
        },
        "baseline": BASELINE, "seats": seats, "forward_test": ft, "health": alert,
        "breadth": load_breadth_signal(), "blended": load_blended_pool(),
    }


# --------------------------------------------------------------------------- #
# 内联 SVG 图表生成（零外部依赖）
# --------------------------------------------------------------------------- #
def _svg_line(width, height, labels, values, color, value_fmt=None,
              y_min=None, y_max=None, thr_lines=None):
    """单序列折线图 + 网格 + 可选阈值线。"""
    pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 22
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    vals = [v for v in values if v is not None] or [0]
    y_min = min(vals) if y_min is None else y_min
    y_max = max(vals) if y_max is None else y_max
    rng = (y_max - y_min) or 1e-9
    y_min, y_max = y_min - rng * 0.12, y_max + rng * 0.12
    rng = y_max - y_min
    n = len(values)

    def X(i): return pad_l + (pw * i / (n - 1) if n > 1 else pw / 2)
    def Y(v): return pad_t + ph * (1 - (v - y_min) / rng)

    grid = []
    for k in range(5):
        v = y_min + rng * k / 4
        y = Y(v)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#eef0f3"/>'
                    f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" font-size="9" fill="#9aa0a6">{v:.3f}</text>')
    zero = ""
    if y_min < 0 < y_max:
        yz = Y(0)
        zero = f'<line x1="{pad_l}" y1="{yz:.1f}" x2="{width-pad_r}" y2="{yz:.1f}" stroke="#cbd5e1" stroke-dasharray="3 3"/>'
    thr = ""
    if thr_lines:
        for tv, tc in thr_lines:
            if y_min <= tv <= y_max:
                yt = Y(tv)
                thr += f'<line x1="{pad_l}" y1="{yt:.1f}" x2="{width-pad_r}" y2="{yt:.1f}" stroke="{tc}" stroke-dasharray="4 2" stroke-width="1"/>'
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(values) if v is not None)
    area = f"{pad_l},{Y(y_min):.1f} " + pts + f" {width-pad_r},{Y(y_min):.1f}"
    xl = ""
    if n > 1:
        step = max(1, n // 6)
        for i in range(0, n, step):
            lab = labels[i][5:] if labels and len(labels[i]) >= 10 else (labels[i] if labels else "")
            xl += f'<text x="{X(i):.1f}" y="{height-6}" text-anchor="middle" font-size="9" fill="#9aa0a6">{lab}</text>'
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="font-family:sans-serif">'
            f'{"".join(grid)}{zero}{thr}'
            f'<polygon points="{area}" fill="{color}14" stroke="none"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>{xl}</svg>')


def _svg_bars_single(width, height, labels, values, colors, val_fmt=None):
    pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 30
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    vmax = max([v for v in values if v is not None] + [0.001])
    vmin = min([v for v in values if v is not None] + [0])
    rng = (vmax - vmin) or 1
    def Y(v): return pad_t + ph * (1 - (v - vmin) / rng)
    zero_y = Y(0)
    n = len(values)
    gap = pw / n
    bw = gap * 0.55
    blocks = [f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width-pad_r}" y2="{zero_y:.1f}" stroke="#cbd5e1"/>']
    for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
        if v is None:
            continue
        x = pad_l + gap * i + (gap - bw) / 2
        y = Y(v)
        h = abs(zero_y - y)
        yy = min(y, zero_y)
        vf = val_fmt(v) if val_fmt else f"{v:.2f}"
        blocks.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{c}" rx="2"/>')
        blocks.append(f'<text x="{x+bw/2:.1f}" y="{y-4 if v>=0 else y+11:.1f}" text-anchor="middle" font-size="9" fill="#374151">{vf}</text>')
        blocks.append(f'<text x="{x+bw/2:.1f}" y="{height-12}" text-anchor="middle" font-size="8.5" fill="#9aa0a6">{lab}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" style="font-family:sans-serif">{"".join(blocks)}</svg>'


def _svg_bars_grouped(width, height, labels, series, colors, val_fmt=None):
    """series: [(name, [v_per_label]), ...]"""
    pad_l, pad_r, pad_t, pad_b = 46, 12, 26, 30
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    allv = [v for _, vs in series for v in vs if v is not None]
    vmax = max(allv + [0.001]); vmin = min(allv + [0])
    rng = (vmax - vmin) or 1
    def Y(v): return pad_t + ph * (1 - (v - vmin) / rng)
    zero_y = Y(0)
    n = len(labels)
    gap = pw / n
    ng = len(series)
    bw = gap * 0.62 / ng
    blocks = [f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width-pad_r}" y2="{zero_y:.1f}" stroke="#cbd5e1"/>']
    # legend
    lx = pad_l
    for (name, _), c in zip(series, colors):
        blocks.append(f'<rect x="{lx}" y="6" width="10" height="10" fill="{c}" rx="2"/>'
                      f'<text x="{lx+14}" y="15" font-size="9" fill="#374151">{name}</text>')
        lx += 14 + len(name) * 9 + 14
    for i, lab in enumerate(labels):
        gx = pad_l + gap * i + gap * 0.19
        for si, (_, vs) in enumerate(series):
            v = vs[i] if i < len(vs) else None
            if v is None:
                continue
            x = gx + si * bw
            y = Y(v); h = abs(zero_y - y); yy = min(y, zero_y)
            vf = val_fmt(v) if val_fmt else f"{v:.2f}"
            blocks.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[si]}" rx="2"/>')
            blocks.append(f'<text x="{x+bw/2:.1f}" y="{y-3 if v>=0 else y+10:.1f}" text-anchor="middle" font-size="8" fill="#374151">{vf}</text>')
        blocks.append(f'<text x="{pad_l+gap*i+gap/2:.1f}" y="{height-12}" text-anchor="middle" font-size="8.5" fill="#9aa0a6">{lab[5:] if len(lab)>=10 else lab}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" style="font-family:sans-serif">{"".join(blocks)}</svg>'


# --------------------------------------------------------------------------- #
# HTML 组装
# --------------------------------------------------------------------------- #
def fmt_p(v):
    return "—" if v is None else ("+" if v >= 0 else "") + f"{v*100:.2f}%"


def build_html(data: dict) -> str:
    r = data["regime"]
    ft = data["forward_test"]
    last = ft[-1] if ft else None
    h = data["health"]
    hcls = "b-healthy" if "全绿" in h["conclusion"] else ("b-err" if "ERROR" in h["conclusion"] else "b-warn")
    rcls = "b-healthy" if r["status"] == "healthy" else ("b-warn" if r["status"] == "warn" else "b-err")

    # —— 预计算安全字符串，避免在 f-string 格式说明符里写条件表达式 —— #
    last_ft_str = fmt_p(last["r1"]) if (last and last["r1"] is not None) else "—"
    last_ft_cls = ("up" if last["r1"] >= 0 else "down") if (last and last["r1"] is not None) else ""
    hit_str = f"{last['hit_rate']*100:.0f}%" if (last and last["hit_rate"] is not None) else "—"
    ic_cur = f"{r['ic_current']:.4f}" if r["ic_current"] is not None else "—"
    ic_cur_cls = "up" if (r["ic_current"] is not None and r["ic_current"] >= 0) else "down"
    ic_60 = f"{r['ic_60d']:.4f}" if r["ic_60d"] is not None else "—"
    dead = f"{r['dead_pct_60d']*100:.1f}%" if r["dead_pct_60d"] is not None else "—"
    bt_sharpe = f"{r['bt_sharpe']:.3f}" if r["bt_sharpe"] is not None else "—"
    bt_total = r["bt_total"] or "—"
    bt_dd = r["bt_dd"] or "—"

    # 市场宽度信号（固化 next_open_rank 宽度闸门）
    bsig = data.get("breadth")
    if bsig:
        b_exp = bsig.get("exposure_target", 1.0)
        b_breadth = bsig.get("breadth_above_ma60")
        b_regime = bsig.get("regime")
        b_seats = bsig.get("recommended_seats")
        bcls = "b-err" if b_regime == "crash_defense" else ("b-warn" if b_regime == "caution" else "b-healthy")
        b_exp_str = f"{b_exp*100:.0f}%"
        b_breadth_str = f"{b_breadth*100:.1f}%" if b_breadth is not None else "—"
        b_regime_cn = {"normal": "常态满仓", "caution": "谨慎·宽度预警", "crash_defense": "崩溃防御"}.get(b_regime, "—")
        b_seats_str = f"{b_seats} 席" if b_seats is not None else "—"
        b_note = "（已自动降仓）" if b_exp < 1.0 else ""
    else:
        bcls = "b-healthy"; b_exp_str = "—"; b_breadth_str = "—"; b_regime_cn = "无信号"; b_seats_str = "—"; b_note = ""

    kpis = f"""
    <div class="card kpi"><div class="v"><span class="badge {rcls}">{r['status'] or '—'}</span></div><div class="l">大盘 Regime</div></div>
    <div class="card kpi"><div class="v">{r['scale']*100:.0f}%</div><div class="l">建议仓位缩放</div></div>
    <div class="card kpi"><div class="v {last_ft_cls}">{last_ft_str}</div><div class="l">最新一期 1日前瞻</div></div>
    <div class="card kpi"><div class="v">{hit_str}</div><div class="l">命中率(1日)</div></div>
    <div class="card kpi"><div class="v"><span class="badge {hcls}">{(h['conclusion'] or '—')[:6]}</span></div><div class="l">管线健康</div></div>
    <div class="card kpi"><div class="v">{data['baseline']['pit']['sharpe']}</div><div class="l">生产基线 Sharpe(PIT)</div></div>
    <div class="card kpi"><div class="v"><span class="badge {bcls}">{b_exp_str}</span></div><div class="l">宽度信号·建议暴露{b_note}</div></div>
    <div class="card kpi"><div class="v">{b_breadth_str}</div><div class="l">宽度(站上MA60占比)</div></div>
    """

    regime_detail = f"""
      <div class="muted" style="font-size:13px;line-height:1.9">
        asof: <b>{r['asof'] or '—'}</b> · 宇宙: {r['universe'] or '—'}<br/>
        trailing IC(当前): <b class="{ic_cur_cls}">{ic_cur}</b> · 近60日: {ic_60}<br/>
        alpha失效今日: <b>{'是' if r['alpha_dead'] else '否'}</b> · 近60日失效占比: {dead}<br/>
        回测(full): Sharpe <b>{bt_sharpe}</b> · 总收益 {bt_total} · 回撤 {bt_dd}
      </div>"""

    breadth_detail = f"""
      <div class="muted" style="font-size:13px;line-height:1.9">
        状态: <b><span class="badge {bcls}">{b_regime_cn}</span></b>{b_note}<br/>
        宽度(站上MA60个股占比): <b>{b_breadth_str}</b> · 建议暴露: <b>{b_exp_str}</b><br/>
        建议席位: <b>{b_seats_str}</b>（常态 {20} 席）<br/>
        规则：宽度&lt;45%→0.55；宽度&lt;32%→0.20（与 next_open_rank 模型共用口径）
      </div>"""

    # charts
    thr = [(r["thr_hi"], "#dc2626")] if r.get("thr_hi") is not None else None
    ic_svg = (_svg_line(600, 230, r["ic_dates"], r["ic_vals"], "#2563eb", thr_lines=thr)
              if r["ic_dates"] else '<div class="muted">无 trailing IC 数据</div>')
    bk = list(data["baseline"].values())
    base_svg = _svg_bars_single(600, 230, [b["label"] for b in bk], [b["sharpe"] for b in bk],
                                [b["color"] for b in bk], val_fmt=lambda v: f"{v:.3f}")
    ft_labels = [f["rec_date"] for f in ft]
    ft_svg = (_svg_bars_grouped(600, 220, ft_labels,
                                [("组合1日", [f["r1"] for f in ft]), ("基准1日", [f["b1"] for f in ft])],
                                ["#2563eb", "#94a3b8"], val_fmt=lambda v: fmt_p(v))
              if ft else '<div class="muted">暂无 forward-test 数据</div>')

    ft_rows = ""
    for f in ft:
        r1c = ("up" if f["r1"] >= 0 else "down") if f["r1"] is not None else ""
        b1c = ("up" if f["b1"] >= 0 else "down") if f["b1"] is not None else ""
        exc = fmt_p(f["r1"] - f["b1"]) if (f["r1"] is not None and f["b1"] is not None) else "—"
        exc_c = (("up" if (f["r1"] - f["b1"]) >= 0 else "down")
                 if (f["r1"] is not None and f["b1"] is not None) else "")
        hr = f"{f['hit_rate']*100:.0f}%" if f["hit_rate"] is not None else "—"
        ft_rows += (f"<tr><td>{f['rec_date']}</td>"
                    f"<td class=\"{r1c}\">{fmt_p(f['r1'])}</td>"
                    f"<td class=\"{b1c}\">{fmt_p(f['b1'])}</td>"
                    f"<td class=\"{exc_c}\">{exc}</td>"
                    f"<td>{hr}</td><td>{f['n']}</td><td>{f['status']}</td></tr>")
    if not ft_rows:
        ft_rows = '<tr><td colspan="7" class="muted">暂无数据</td></tr>'

    seat_rows = ""
    for s in data["seats"]:
        rc = ("up" if s["ret20"] >= 0 else "down") if s["ret20"] is not None else ""
        r20 = fmt_p(s["ret20"]) if s["ret20"] is not None else "—"
        star = '<span class="star">★</span>' if s["watch"] else ""
        if s["blend_inserted"]:
            badge = ' <span class="badge" style="background:#ede9fe;color:#6d28d9">方案B·新顶入</span>'
        elif s["blend_boosted"]:
            badge = ' <span class="badge" style="background:#fef3c7;color:#92400e">方案B·强化</span>'
        else:
            badge = ""
        seat_rows += (f"<tr><td>{s['rank'] or ''}</td><td>{s['symbol']}{badge}</td><td>{s['name']}</td>"
                      f"<td>{s['action']}</td><td>{s['weight']*100:.0f}%</td><td>{s['score']}</td>"
                      f"<td class=\"{rc}\">{r20}</td><td>{star}</td>"
                      f"<td class=\"muted\">{s['reasons']}</td></tr>")
    if not seat_rows:
        seat_rows = '<tr><td colspan="9" class="muted">暂无 overlay 数据</td></tr>'

    # 方案 B: 趋势点火 × NOR 加权混合候选（研究信号面板，非侵入、gated）
    blended = data.get("blended", {"asof": None, "rows": []})
    blended_rows = ""
    for s in blended["rows"][:20]:
        ti_s = f"{s['ti']:.3f}" if s["ti"] is not None else "—"
        nor_s = f"{s['nor']:.3f}" if s["nor"] is not None else "—"
        bl_s = f"{s['blended']:.3f}" if s["blended"] is not None else "—"
        star = " ✓" if s["inter"] else ""
        blended_rows += (f"<tr><td>{s['rank'] or ''}</td><td>{s['symbol']}</td>"
                         f"<td>{ti_s}</td><td>{nor_s}</td><td>{bl_s}</td><td>{star}</td></tr>")
    if not blended_rows:
        blended_rows = '<tr><td colspan="6" class="muted">无混合候选（blended_pool 未生成或为空）</td></tr>'
    blended_asof = blended.get("asof") or "—"

    health_rows = "".join(
        f"<tr><td>{i['name']}</td><td>{i['status']}</td><td class=\"muted\">{i['detail']}</td></tr>"
        for i in h["items"]) or f'<tr><td colspan="3" class="muted">{h["conclusion"] or "无巡检数据"}</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>量化每日仪表盘 · {data['asof']}</title>
<style>
:root{{--bg:#f5f7fa;--card:#fff;--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;
--red:#e11d48;--green:#059669;--blue:#2563eb;--amber:#d97706;--ok:#059669;--warn:#d97706;--err:#dc2626;}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:24px;}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--muted);font-size:13px;margin-bottom:20px}}
.grid{{display:grid;gap:16px;}} .kpis{{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin-bottom:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.kpi .v{{font-size:24px;font-weight:700;display:flex;align-items:center}} .kpi .l{{color:var(--muted);font-size:12px;margin-top:6px}}
.up{{color:var(--red)}} .down{{color:var(--green)}}
.badge{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}}
.b-healthy{{background:#dcfce7;color:var(--ok)}} .b-warn{{background:#fef3c7;color:var(--warn)}} .b-err{{background:#fee2e2;color:var(--err)}}
section{{margin-top:16px}} h2{{font-size:16px;margin:0 0 10px;border-left:4px solid var(--blue);padding-left:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--muted);font-weight:600;background:#fafafa}} .star{{color:var(--amber)}} .muted{{color:var(--muted)}}
.flex{{display:flex;gap:16px;flex-wrap:wrap}} .flex>div{{flex:1;min-width:320px}}
.chart-box{{width:100%;overflow:hidden}} footer{{margin-top:24px;color:var(--muted);font-size:12px;text-align:center}}
</style></head><body>
<h1>量化每日仪表盘</h1>
<div class="sub">数据锚点 {data['asof']} · 生成于 {data['generated']} · 红涨绿跌（A股惯例）· 完全离线自包含</div>

<div class="grid kpis">{kpis}</div>

<div class="flex">
  <div class="card"><h2>大盘 Regime 状态</h2>{regime_detail}
    <div class="chart-box">{ic_svg}</div></div>
  <div class="card"><h2>市场宽度信号 · 自动降仓</h2>{breadth_detail}</div>
  <div class="card"><h2>生产基线（四档宇宙 Sharpe）</h2>
    <div class="chart-box">{base_svg}</div>
    <div class="muted" style="font-size:12px;margin-top:8px">pit=诚实PIT宇宙(真实OOS上界) · full=旧上界 · static-survivor=前视反例 · watchlist=自选股PIT(窗口)</div>
  </div>
</div>

<section class="card"><h2>Forward-test 样本外跟踪</h2>
  <div class="chart-box">{ft_svg}</div>
  <table><tr><th>推荐日</th><th>组合1日</th><th>基准1日</th><th>超额</th><th>命中率</th><th>样本</th><th>状态</th></tr>{ft_rows}</table>
</section>

<section class="card"><h2>今日关注 20 席（行为叠加后最终组合）</h2>
  <table><tr><th>#</th><th>代码</th><th>名称</th><th>动作</th><th>权重</th><th>调整分</th><th>20日</th><th>自选</th><th>备注</th></tr>{seat_rows}</table>
</section>

<section class="card"><h2>管线健康巡检 <span class="badge {hcls}">{h['conclusion'] or '—'}</span></h2>
  <table><tr><th>检查项</th><th>状态</th><th>详情</th></tr>{health_rows}</table>
</section>

<section class="card"><h2>趋势点火 × NOR 混合候选（方案B·研究信号）</h2>
  <div class="muted" style="font-size:12px;margin-bottom:8px;color:#b00">as-of {blended_asof} · <b>隔离研究快照（未并入实盘主名单）</b>：方案 B 混合候选目前未通过 run 一致性校验（不同批次产物 / 并入排序 bug），暂停实盘消费。此表为研究明细，不作为推荐依据。</div>
  <table><tr><th>#</th><th>代码</th><th>TI评分</th><th>NOR评分</th><th>混合分</th><th>双覆盖</th></tr>{blended_rows}</table>
</section>

<footer>本仪表盘由 build_dashboard.py 自动生成（内联 SVG，无外部依赖），汇总 overlay / regime / forward-test / 健康巡检 产物。仅供研究参考，非投资建议。</footer>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="锚点日期 YYYY-MM-DD，缺省取面板末日")
    args = ap.parse_args()
    asof = args.date or load_panel_last()
    data = build_data(asof)
    html = build_html(data)
    out = DASH_DIR / f"dashboard_{asof}.html"
    out.write_text(html, encoding="utf-8")
    (DASH_DIR / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"[dashboard] 已生成(离线SVG) → {out}（{len(data['seats'])} 席 / {len(data['forward_test'])} 期 / regime={data['regime']['status']}）")


if __name__ == "__main__":
    main()
