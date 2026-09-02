"""策略体检（断路器 + 数据健康 + 决策台账）。

每日收盘后由 run_daily_overlay.py 作为 [3/3] 步调用（也可独立运行）。
只读面板 / npz / overlay / 名单目录，所有输出落 outputs/watchlist_audit/strategy_health/。

体检维度（LLM 交易代理安全模式移植，AI 无下单权、只做监控告警，不干预用户拍板）：
  A. 数据健康   面板末日 vs asof、名单源最新日期、blended 隔离状态、分数覆盖
  B. 出票体检   入选数、名单覆盖、行为动作分布、分数/权重分布
  C. 断路器     基于 ledger 累积的次日开盘收益（next_open_return，一日滞后口径）：
                连续亏损天数、20 日滚动胜率、单日最差 → GREEN/YELLOW/RED 状态

ledger（决策台账）：每次运行将当日出票 upsert 进 outputs/watchlist_audit/strategy_health/ledger.csv，
面板新增数据后自动回填 next_open_return（T1.open / asof.close - 1）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "watchlist_audit"
HEALTH_DIR = OUT / "strategy_health"
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
# 2026-08-31 修复：生产文件为 _broker 后缀（broker 因子并入生产默认），旧 _pit.npz 已停更。
# 2026-09-02 修复：MLP 配置参与内容寻址后缓存名带参数后缀，运行时动态解析最新 broker npz。
def _npz_default() -> Path:
    from production_soft_score import latest_pit_broker_npz
    try:
        return latest_pit_broker_npz()
    except SystemExit:
        return HERE / "outputs" / "p10e_regime_gated" / "linear_mlp_scores_pit_broker.npz"


NPZ_DEFAULT = _npz_default()
OVERLAY_DEFAULT = OUT / "full_overlay_calibrated.csv"
CAND_DEFAULT = OUT / "full_candidates.csv"
THS_DIR = Path("D:/codex/outputs/stock-analysis-dashboard/input")
QUARANTINE_MARKER = OUT / "full_overlay_calibrated_blended.csv.QUARANTINE"
SCORE_COLUMN = "mlp"

# 断路器参数（可配，默认参考 LLM 交易安全模式：连续 3 亏 halt / 单日 -3% halt）
DEFAULT_MAX_CONSEC_LOSS_DAYS = 3
DEFAULT_MAX_DAILY_LOSS_PCT = -3.0
DEFAULT_WARN_WIN_RATE_20D = 35.0
ROLLING_WINDOW_DAYS = 20


def load_panel_wide(panel_path: Path):
    """只读 date/symbol/open/close，返回 (close_wide, open_wide, dates)。"""
    df = pd.read_csv(panel_path, usecols=["date", "symbol", "open", "close"],
                     dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    close_wide = df.pivot(index="date", columns="symbol", values="close")
    open_wide = df.pivot(index="date", columns="symbol", values="open")
    dates = close_wide.index.sort_values()
    return close_wide, open_wide, dates


def infer_asof(npz_path: Path, dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    """npz 最后一个有分数的日期（与 build_watchlist_candidates 同口径）。

    优先级：meta.first_date/last_date 对齐面板日期序列 → 定位最后有分行；
    对齐失败回退面板末日。
    """
    try:
        d = np.load(npz_path, allow_pickle=True)
        if SCORE_COLUMN not in d.files:
            return None
        mat = d[SCORE_COLUMN]
        good = ~np.isnan(mat).all(axis=1)
        if not good.any():
            return None
        idx_last = int(np.where(good)[0][-1])
        meta = {}
        if "meta" in d.files:
            try:
                meta = json.loads(str(d["meta"]))
            except Exception:
                pass
        last = meta.get("last_date")
        if last:
            sub = dates[dates <= pd.Timestamp(last)]
            if len(sub) == mat.shape[0]:
                return pd.Timestamp(sub[idx_last])
        if len(dates) == mat.shape[0]:
            return pd.Timestamp(dates[idx_last])
        if last:
            return pd.Timestamp(last)
        return None
    except Exception:
        return None


def load_overlay(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})


def next_open_return(row, close_wide, open_wide, dates):
    """T1.open / asof.close - 1；T1 为 asof 之后面板第一个交易日；无则 None(pending)。"""
    asof = row["asof"]
    sym = row["symbol"]
    after = dates[dates > asof]
    if len(after) == 0:
        return None
    t1 = after[0]
    c0 = close_wide.loc[asof, sym] if asof in close_wide.index else np.nan
    o1 = open_wide.loc[t1, sym] if t1 in open_wide.index else np.nan
    if pd.isna(c0) or pd.isna(o1) or c0 == 0:
        return np.nan
    return float(o1 / c0 - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="策略体检：数据健康 + 出票体检 + 断路器")
    parser.add_argument("--overlay-csv", default=str(OVERLAY_DEFAULT))
    parser.add_argument("--candidates-csv", default=str(CAND_DEFAULT))
    parser.add_argument("--panel", default=str(PANEL))
    parser.add_argument("--scores-npz", default=str(NPZ_DEFAULT))
    parser.add_argument("--ths-dir", default=str(THS_DIR))
    parser.add_argument("--max-consec-loss-days", type=int, default=DEFAULT_MAX_CONSEC_LOSS_DAYS)
    parser.add_argument("--max-daily-loss-pct", type=float, default=DEFAULT_MAX_DAILY_LOSS_PCT)
    parser.add_argument("--warn-win-rate-20d", type=float, default=DEFAULT_WARN_WIN_RATE_20D)
    args = parser.parse_args()

    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = HEALTH_DIR / "ledger.csv"
    report_path = HEALTH_DIR / "health_report.md"
    status_path = HEALTH_DIR / "health_status.json"

    # ---------- 读取 ----------
    overlay = load_overlay(Path(args.overlay_csv))
    if overlay.empty:
        print("[health] overlay 为空，跳过体检")
        return
    overlay = overlay.rename(columns=lambda c: c.strip().lstrip("\ufeff"))
    if "selected" in overlay.columns:
        sel = overlay[overlay["selected"].astype(str).isin(["True", "1", "true"])].copy()
    else:
        sel = overlay.copy()
    if sel.empty:
        print("[health] overlay 无 selected 行，跳过体检")
        return

    print("[health] loading panel ...")
    close_wide, open_wide, dates = load_panel_wide(Path(args.panel))
    panel_last = dates[-1]
    asof = infer_asof(Path(args.scores_npz), dates)
    if asof is None:
        asof = panel_last
        print(f"[health] npz 无有效分数，asof 回退面板末日 {asof.date()}")

    # ---------- A. 数据健康 ----------
    ths_files = sorted(Path(args.ths_dir).glob("ths_money_flow_*.xls")) if Path(args.ths_dir).exists() else []
    ths_last = None
    if ths_files:
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2})", ths_files[-1].name)
        ths_last = m.group(1) if m else ths_files[-1].stem
    quarantined = QUARANTINE_MARKER.exists()
    asof_gap = (panel_last - asof).days
    npz_exists = Path(args.scores_npz).exists()

    # ---------- B. 出票体检 ----------
    n_sel = len(sel)
    wl_col = "user_watchlist" if "user_watchlist" in sel.columns else None
    n_wl = int(sel[wl_col].astype(str).isin(["True", "1", "true"]).sum()) if wl_col else None
    act_col = "personal_action" if "personal_action" in sel.columns else None
    act_dist = sel[act_col].value_counts().to_dict() if act_col else {}
    score_col = "score" if "score" in sel.columns else None
    score_stats = {}
    if score_col:
        s = pd.to_numeric(sel[score_col], errors="coerce").dropna()
        if len(s):
            score_stats = {"mean": round(float(s.mean()), 4), "max": round(float(s.max()), 4),
                           "min": round(float(s.min()), 4), "n": int(len(s))}
    wt_col = "target_weight" if "target_weight" in sel.columns else None
    wt_stats = {}
    if wt_col:
        w = pd.to_numeric(sel[wt_col], errors="coerce").dropna()
        if len(w):
            wt_stats = {"nunique": int(w.nunique()), "max": round(float(w.max()), 4),
                        "sum": round(float(w.sum()), 4)}

    # ---------- C. 断路器（ledger upsert + 回填） ----------
    now = datetime.now().isoformat(timespec="seconds")
    new_rows = []
    for _, r in sel.iterrows():
        new_rows.append({
            "asof": asof.date().isoformat(),
            "symbol": str(r.get("symbol", "")).strip(),
            "stock_name": str(r.get("stock_name", "")),
            "user_watchlist": bool(r.get(wl_col, False)) if wl_col else False,
            "score": float(r[score_col]) if score_col and pd.notna(r.get(score_col)) else np.nan,
            "rank": int(r.get("rank", np.nan)) if pd.notna(r.get("rank")) else np.nan,
            "target_weight": float(r[wt_col]) if wt_col and pd.notna(r.get(wt_col)) else np.nan,
            "personal_action": str(r.get(act_col, "")) if act_col else "",
            "run_date": now[:10],
        })
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        print("[health] 无出票行，跳过")
        return

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path, dtype={"symbol": str})
    else:
        ledger = pd.DataFrame()

    # upsert by (asof, symbol)
    if not ledger.empty:
        key = list(new_df.columns)
        merged = ledger.copy()
        for _, nr in new_df.iterrows():
            hit = (merged["asof"] == nr["asof"]) & (merged["symbol"] == nr["symbol"])
            if hit.any():
                merged.loc[hit, "score"] = nr["score"]
                merged.loc[hit, "target_weight"] = nr["target_weight"]
                merged.loc[hit, "personal_action"] = nr["personal_action"]
                merged.loc[hit, "user_watchlist"] = nr["user_watchlist"]
                merged.loc[hit, "run_date"] = nr["run_date"]
            else:
                merged = pd.concat([merged, pd.DataFrame([nr])], ignore_index=True)
        ledger = merged
    else:
        ledger = new_df

    # 回填收益（只回填未填/已填但面板新增了后续数据的行）
    if "next_open_return" not in ledger.columns:
        ledger["next_open_return"] = np.nan
    for i, row in ledger.iterrows():
        if pd.isna(row["asof"]):
            continue
        asof_ts = pd.Timestamp(row["asof"])
        val = next_open_return({"asof": asof_ts, "symbol": row["symbol"]}, close_wide, open_wide, dates)
        if val is not None:
            ledger.at[i, "next_open_return"] = val

    # 组合层面指标（按 asof 日）
    combo = ledger.dropna(subset=["next_open_return"]).groupby("asof")["next_open_return"].mean()
    n_days = len(combo)
    consecutive_loss = 0
    for v in combo.iloc[::-1]:
        if v < 0:
            consecutive_loss += 1
        else:
            break
    daily_worst = float(combo.min()) if n_days else None
    # 20 日滚动胜率（标的口径，最近 ROLLING_WINDOW_DAYS 个出票日的全部标的）
    filled = ledger.dropna(subset=["next_open_return"])
    if len(filled):
        recent = filled.tail(ROLLING_WINDOW_DAYS * 25)
        win_rate = float((recent["next_open_return"] > 0).mean() * 100.0)
        recent_n = int((recent["next_open_return"] > 0).sum())
        recent_total = int(len(recent))
    else:
        win_rate, recent_n, recent_total = None, 0, 0

    # 状态机
    flags = []
    if n_days == 0:
        status = "GREEN"
        flags.append("无历史出票收益，断路器数据累积中（首日运行）")
    else:
        status = "GREEN"
        if consecutive_loss >= args.max_consec_loss_days:
            status = "RED"
            flags.append(f"连续亏损 {consecutive_loss} 天 ≥ {args.max_consec_loss_days}（halt）")
        if daily_worst is not None and daily_worst * 100.0 <= args.max_daily_loss_pct:
            status = "RED"
            flags.append(f"单日组合均收益 {daily_worst*100:.2f}% ≤ {args.max_daily_loss_pct}%（halt）")
        if status == "GREEN" and consecutive_loss >= 2:
            status = "YELLOW"
            flags.append(f"连续亏损 {consecutive_loss} 天（接近 halt 阈值）")
        if status == "GREEN" and win_rate is not None and win_rate < args.warn_win_rate_20d:
            status = "YELLOW"
            flags.append(f"近窗口胜率 {win_rate:.1f}% < {args.warn_win_rate_20d}%")

    # ---------- 输出 ----------
    lines = []
    lines.append("# 策略体检报告（断路器 + 数据健康）")
    lines.append("")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 信号日 asof: {asof.date()} | 面板末日: {panel_last.date()}（间隔 {asof_gap} 天）")
    lines.append(f"- 断路器状态: **{status}**")
    for f in flags:
        lines.append(f"  - ⚠️ {f}")
    lines.append("")
    lines.append("## A. 数据健康")
    lines.append(f"- 面板: {len(dates)} 交易日，末日 {panel_last.date()}")
    lines.append(f"- 分数 npz: {'存在' if npz_exists else '缺失'}（asof={asof.date()}）")
    lines.append(f"- 名单源 ths_money_flow: {len(ths_files)} 个文件，最新 {ths_last}")
    lines.append(f"- blended 隔离: {'是（QUARANTINE 标记存在，主名单已回退原始 overlay）' if quarantined else '否'}")
    lines.append("")
    lines.append("## B. 出票体检")
    lines.append(f"- 入选数: {n_sel}")
    if n_wl is not None:
        lines.append(f"- 自选名单覆盖: {n_wl}/{n_sel}（{n_wl/n_sel*100:.0f}%）")
    if act_dist:
        lines.append(f"- 行为动作分布: {json.dumps(act_dist, ensure_ascii=False)}")
    if score_stats:
        lines.append(f"- 分数: mean={score_stats['mean']} max={score_stats['max']} min={score_stats['min']}（n={score_stats['n']}）")
    if wt_stats:
        lines.append(f"- 权重: {wt_stats['nunique']} 档，max={wt_stats['max']}，sum={wt_stats['sum']}")
    lines.append("")
    lines.append("## C. 断路器（出票组合次日开盘收益，next_open_return）")
    if n_days:
        lines.append(f"- 已回填出票日: {n_days} 天")
        lines.append(f"- 连续亏损: {consecutive_loss} 天（阈值 {args.max_consec_loss_days}）")
        lines.append(f"- 单日最差组合均收益: {daily_worst*100:.2f}%（阈值 {args.max_daily_loss_pct}%）")
        if win_rate is not None:
            lines.append(f"- 近窗口胜率: {win_rate:.1f}%（{recent_n}/{recent_total}）")
        lines.append("")
        lines.append("| asof | 组合均收益 |")
        lines.append("|------|----------|")
        for d, v in combo.iloc[-10:].items():
            lines.append(f"| {d} | {v*100:+.2f}% |")
    else:
        lines.append("- 无已回填数据：ledger 已初始化，明日面板更新后自动回填。")
    lines.append("")
    lines.append("> 说明：体检只做监控告警，不干预出票决策。RED 状态请人工审视策略。")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")

    status = {
        "generated_at": now,
        "asof": asof.date().isoformat(),
        "panel_last": panel_last.date().isoformat(),
        "circuit_breaker": status,
        "flags": flags,
        "metrics": {
            "n_selected": n_sel,
            "n_watchlist": n_wl,
            "n_backfilled_days": n_days,
            "consecutive_loss_days": consecutive_loss,
            "daily_worst_pct": round(daily_worst * 100.0, 2) if daily_worst is not None else None,
            "win_rate_20d_pct": round(win_rate, 1) if win_rate is not None else None,
            "ths_files": len(ths_files),
            "ths_last": ths_last,
            "blended_quarantined": quarantined,
        },
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[health] 断路器状态: {status} | 回填出票日: {n_days} | 报告 -> {report_path}")


if __name__ == "__main__":
    main()
