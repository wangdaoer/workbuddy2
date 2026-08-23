"""Pipeline health check — 每日量化流程完整性巡检。

设计目标：把"管线/数据异常从静默跳过变成主动报警"。
每日 19:00 自动化末尾调用一次，也可被盘前简报/手动调用。

锚点：以 data_panel.csv 的末日（数据真实覆盖到的最后交易日）为"期望完整流程末日"。
逐项检查各产物是否新鲜/完整，分级输出 OK / WARN / ERROR：
  - npz   : linear_mlp_scores_pit.npz 存在且 shape[0] == 面板交易日数（内容寻址对应面板）
  - regime: 最新 regime_monitor_*.json 的 asof == 锚点；status=dead 或 alert=True => WARN（流程健康但信号失效）
  - overlay完整性: full_overlay_calibrated.csv 存在且 personal_selected==True 恰为 20 席
  - overlay新鲜度: 其 asof（从 forward-test 台账 rec_date 代理，overlay CSV 无 date 列）相对锚点落后天数
  - ledger新鲜度 : ledger.csv 最新 rec_date 相对锚点落后天数
  - benchmark    : 510300.csv 末日 >= 锚点 - 2（基准允许晚补 2 天）

退出码：全 OK => 0；仅 WARN => 1；有 ERROR => 2。
产物：outputs/alerts/alert_<锚点日期>.md（含状态表 + 摘要）。
"""
from __future__ import annotations
import glob
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
NPZ_GLOB = str(HERE / "outputs" / "**" / "linear_mlp_scores_pit.npz")
REGIME_GLOB = str(HERE / "outputs" / "**" / "regime_monitor_*.json")
OVERLAY = HERE / "outputs" / "watchlist_audit" / "full_overlay_calibrated.csv"
LEDGER = HERE / "outputs" / "forward_test" / "ledger.csv"
BENCH = HERE / "external_data" / "benchmarks" / "510300.csv"
ALERT_DIR = HERE / "outputs" / "alerts"


def _load_panel_last() -> tuple[str, int]:
    p = pd.read_csv(PANEL, usecols=["date"])
    last = str(p["date"].max())
    ndays = int(p["date"].nunique())
    return last, ndays


def _latest(globpat: str, by: str = "mtime") -> str | None:
    fs = glob.glob(globpat, recursive=True)
    if not fs:
        return None
    if by == "name_date":
        def key(f: str) -> int:
            m = __import__("re").search(r"(\d{8})", os.path.basename(f))
            return int(m.group(1)) if m else 0
        return sorted(fs, key=key)[-1]
    return sorted(fs, key=lambda f: os.path.getmtime(f))[-1]


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def run(panel_path: Path = PANEL, expect: str | None = None) -> tuple[list[dict], str]:
    checks: list[dict] = []
    try:
        plast, ndays = _load_panel_last()
    except Exception as e:
        return [_check("PANEL", "ERROR", f"读取面板失败: {e}")], expect or "?"
    anchored = expect or plast
    checks.append(_check("PANEL", "OK", f"面板末日 {plast} / {ndays} 交易日"))

    # --- npz ---
    npz = _latest(NPZ_GLOB)
    if not npz:
        checks.append(_check("SCORES_NPZ", "ERROR", "未找到 linear_mlp_scores_pit.npz"))
    else:
        try:
            d = np.load(npz, allow_pickle=True)
            shape0 = max((getattr(d[k], "shape", (0,))[0] for k in d.files if getattr(d[k], "ndim", 0) == 2), default=0)
            if shape0 == ndays:
                checks.append(_check("SCORES_NPZ", "OK", f"{Path(npz).name} shape[0]={shape0} 对应面板"))
            else:
                checks.append(_check("SCORES_NPZ", "ERROR",
                                     f"{Path(npz).name} shape[0]={shape0} != 面板交易日 {ndays}（缓存未随面板刷新重建）"))
        except Exception as e:
            checks.append(_check("SCORES_NPZ", "ERROR", f"读取 npz 失败: {e}"))

    # --- regime ---
    rj = _latest(REGIME_GLOB, by="name_date")
    if not rj:
        checks.append(_check("REGIME", "ERROR", "未找到 regime_monitor_*.json"))
    else:
        try:
            j = json.load(open(rj))
            asof = str(j.get("asof"))
            status = j.get("status")
            alert = j.get("alert")
            if asof != anchored:
                checks.append(_check("REGIME", "ERROR",
                                     f"{Path(rj).name} asof={asof} != 锚点 {anchored}"))
            elif status == "dead" or alert:
                checks.append(_check("REGIME", "WARN",
                                     f"asof={asof} 但 status={status}/alert={alert}（信号失效，流程仍健康）"))
            else:
                checks.append(_check("REGIME", "OK", f"{Path(rj).name} asof={asof} status={status}"))
        except Exception as e:
            checks.append(_check("REGIME", "ERROR", f"读取 regime 失败: {e}"))

    # --- overlay 完整性 ---
    if not OVERLAY.exists():
        checks.append(_check("OVERLAY_COMPLETE", "ERROR", "未找到 full_overlay_calibrated.csv"))
    else:
        try:
            df = pd.read_csv(OVERLAY)
            n = int((df.get("personal_selected") == True).sum()) if "personal_selected" in df.columns else -1
            if n == 20:
                checks.append(_check("OVERLAY_COMPLETE", "OK", "personal_selected 恰为 20 席"))
            else:
                checks.append(_check("OVERLAY_COMPLETE", "ERROR",
                                     f"personal_selected={n} 席（期望 20，overlay 未正常产出）"))
        except Exception as e:
            checks.append(_check("OVERLAY_COMPLETE", "ERROR", f"读取 overlay 失败: {e}"))

    # --- overlay / ledger 新鲜度（按交易日落后天数）---
    def _trading_lag(d: str) -> int:
        a = datetime.strptime(anchored, "%Y-%m-%d").date()
        b = datetime.strptime(d, "%Y-%m-%d").date()
        return (a - b).days  # 正数=产物落后锚点天数

    # overlay asof 代理 = 台账最新 rec_date（overlay CSV 无 date 列）
    ov_asof = None
    if LEDGER.exists():
        try:
            l = pd.read_csv(LEDGER)
            if "rec_date" in l.columns and len(l):
                ov_asof = str(l["rec_date"].max())
        except Exception:
            pass
    if ov_asof is None:
        checks.append(_check("OVERLAY_FRESH", "WARN", "无法推断 overlay asof（台账为空/未跑 snapshot）"))
    else:
        lag = _trading_lag(ov_asof)
        if lag <= 0:
            checks.append(_check("OVERLAY_FRESH", "OK", f"overlay asof={ov_asof} 与锚点一致"))
        elif lag == 1:
            checks.append(_check("OVERLAY_FRESH", "WARN",
                                 f"overlay asof={ov_asof} 落后锚点 {lag} 天（当日流程可能未完整跑完/周末）"))
        else:
            checks.append(_check("OVERLAY_FRESH", "ERROR",
                                 f"overlay asof={ov_asof} 落后锚点 {lag} 天（明确漏跑，需补 overlay+forward-test snapshot）"))

    # ledger 新鲜度
    if not LEDGER.exists():
        checks.append(_check("LEDGER_FRESH", "WARN", "未找到 ledger.csv（尚未跑过 forward-test）"))
    else:
        try:
            l = pd.read_csv(LEDGER)
            lmax = str(l["rec_date"].max())
            lag = _trading_lag(lmax)
            if lag <= 0:
                checks.append(_check("LEDGER_FRESH", "OK", f"ledger 最新 rec_date={lmax} 与锚点一致"))
            elif lag == 1:
                checks.append(_check("LEDGER_FRESH", "WARN",
                                     f"ledger 最新 {lmax} 落后锚点 {lag} 天（forward-test snapshot 未补当日推荐）"))
            else:
                checks.append(_check("LEDGER_FRESH", "ERROR",
                                     f"ledger 最新 {lmax} 落后锚点 {lag} 天（forward-test 明显滞后）"))
        except Exception as e:
            checks.append(_check("LEDGER_FRESH", "ERROR", f"读取 ledger 失败: {e}"))

    # --- benchmark ---
    if not BENCH.exists():
        checks.append(_check("BENCH", "WARN", "未找到 510300.csv 基准"))
    else:
        try:
            b = pd.read_csv(BENCH)
            blast = str(b["date"].max())
            lag = _trading_lag(blast)
            if lag <= 2:
                checks.append(_check("BENCH", "OK", f"基准末日 {blast}（落后锚点 {max(lag,0)} 天，允许≤2）"))
            else:
                checks.append(_check("BENCH", "WARN", f"基准末日 {blast} 落后锚点 {lag} 天（forward-test 对比将缺最新 horizon）"))
        except Exception as e:
            checks.append(_check("BENCH", "WARN", f"读取基准失败: {e}"))

    return checks, anchored


def _render(checks: list[dict], anchored: str) -> str:
    icon = {"OK": "✅", "WARN": "⚠️", "ERROR": "🛑"}
    lines = [f"# 管线健康巡检 — 锚点 {anchored}", ""]
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    n_err = sum(1 for c in checks if c["status"] == "ERROR")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    if n_err:
        verdict = f"🛑 ERROR（{n_err} 项）— 管线存在明确故障，需人工介入"
    elif n_warn:
        verdict = f"⚠️ WARN（{n_warn} 项）— 流程有缺口/信号失效，建议核查"
    else:
        verdict = "✅ 全绿 — 完整流程已覆盖到锚点日"
    lines.append(f"**结论: {verdict}**")
    lines.append("")
    lines.append("| 检查项 | 状态 | 详情 |")
    lines.append("|---|---|---|")
    for c in checks:
        lines.append(f"| {c['name']} | {icon[c['status']]} {c['status']} | {c['detail']} |")
    lines.append("")
    lines.append("---")
    lines.append("锚点 = data_panel.csv 末日（数据真实覆盖到的最后交易日）。")
    lines.append("ERROR=退出码2(故障) / WARN=退出码1(缺口) / OK=退出码0。")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--expect-date", default=None, help="期望完整流程末日（默认=面板末日）")
    ap.add_argument("--json", action="store_true", help="额外打印 JSON")
    args = ap.parse_args()

    checks, anchored = run(Path(args.panel), args.expect_date)
    md = _render(checks, anchored)
    print(md)
    if args.json:
        print("\nJSON:", json.dumps(
            {"anchored": anchored,
             "checks": [{k: c[k] for k in ("name", "status", "detail")} for c in checks]},
            ensure_ascii=False))

    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    out = ALERT_DIR / f"alert_{anchored}.md"
    out.write_text(md, encoding="utf-8")
    print(f"\n[health] 告警文件: {out}")

    n_err = sum(1 for c in checks if c["status"] == "ERROR")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    sys.exit(2 if n_err else (1 if n_warn else 0))


if __name__ == "__main__":
    main()
