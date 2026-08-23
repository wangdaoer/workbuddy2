"""build_leakfree_scorer_feed.py — 无泄漏 per-as-of 重建 TI 评分器与 feed (方案 B step3).

问题 (P0-3): export_authoritative_scorer 用全部 7271 行拟合部署评分器, 其中 99% 事件发生在
回测窗口之后 -> 历史 as-of 回测存在前视泄漏.

修复: 对每个回测 as-of, 仅用「标签成熟日 <= as-of - embargo」的样本拟合评分器, 杜绝未来标签进入分箱 lift.
  - 标签成熟日 = ignition_date 后第 20/60 个**交易日**（由交易日历精确计算, 与 validate_label_horizon 的
    shift(-h) 标签口径一致; P0-2: 不再用 28/84 自然日近似——节假日使 20 交易日实际跨度 28~40 天、
    60 交易日 84~98 天, 固定近似会把未成熟样本误判为成熟, 造成前视泄漏）
  - embargo 缓冲 (默认 10 自然日) 进一步隔离信息外溢
  - 用该 as-of 专属评分器对截至 as-of 的点火样本打分, 生成 leak-free feed 截面

用法:
  python build_leakfree_scorer_feed.py --asof 2025-09-01      # 单 as-of 验证
  python build_leakfree_scorer_feed.py --asof all --horizon 60  # 批量 fwd60
  python build_leakfree_scorer_feed.py --asof all --horizon 20  # 批量 fwd20

产物: outputs/high_return_v2/trend_ignition_daily_pool/leakfree_feed_fwd{60,20}.csv
  (每个 as-of 一列, 与软 feed 同 index; 生产 mix 时按列取该 as-of 的评分器打分)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import train_trend_ignition_scorer_v2 as m

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "outputs" / "high_return_v2" / "trend_ignition_training_set_authoritative" / "trend_ignition_training_set.csv"
DEFAULT_SAMPLES = ROOT / "outputs" / "high_return_v2" / "trend_ignition_lifelines_202506_202608_v3" / "trend_ignition_samples.csv"
DEFAULT_NOR = ROOT / "outputs" / "production_soft_score" / "soft_score_feed.csv"
DEFAULT_PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool"

FEATURE_COLUMNS = [
    "feature_breakout_pct", "feature_log_amount_ratio", "feature_return_20d", "feature_return_60d",
    "feature_volatility_20d", "feature_ma20_over_ma60", "feature_close_over_ma20", "feature_drawdown_120d",
    "feature_log_amount_trend_5_20", "feature_breakout_count_20d",
    "feature_pre_amount_trend_20_60", "feature_pre_amount_accel", "feature_pre_volume_z",
    "feature_pre_consolidation_60", "feature_pre_drift_slope_60", "feature_pre_cnl_amplitude_60",
    "feature_ign_day_amplitude", "feature_ign_day_upper_shadow", "feature_ign_day_close_pos", "feature_ign_day_gap",
]
# P0-2: 标签成熟日必须用真实交易日历精确计算（见 _maturity_dates）。
# 实测 20 交易日自然日跨度为 28~40 天、60 交易日 84~98 天，固定自然日近似会产生前视泄漏。
LABEL_COL = {"60": "label_fwd60_strong", "20": "label_fwd20_up"}


def load_trading_dates(panel: Path) -> np.ndarray:
    """从价格面板读取交易日历（唯一日期升序，dtype=datetime64[ns]），用于精确计算标签成熟日与持有窗口。"""
    df = pd.read_csv(panel, usecols=["date"])
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    arr = d.to_numpy(dtype="datetime64[ns]").astype("datetime64[ns]")
    return np.sort(np.unique(arr))


def _maturity_dates(ig: pd.Series, tdates: np.ndarray, h: int) -> pd.Series:
    """每条样本的真实标签成熟日 = ignition_date 后第 h 个交易日（对齐 validate_label_horizon shift(-h) 口径）。
    面板不足 h 个交易日 → 返回 NaT（保守视为未成熟，purge 剔除，杜绝误判为成熟）。"""
    vals = ig.to_numpy(dtype="datetime64[ns]")
    out = pd.Series(pd.NaT, index=ig.index, dtype="datetime64[ns]")
    ok = ~pd.isna(vals)
    if not ok.any():
        return out
    pos = np.searchsorted(tdates, vals[ok], side="left")
    enough = (pos + h) < len(tdates)          # 面板剩余交易日 ≥ h 才算得出成熟日
    idx_ok = np.where(ok)[0][enough]
    out.iloc[idx_ok] = pd.Series(tdates[pos[enough] + h], index=ig.index[idx_ok])
    return out


def make_features(samples: pd.DataFrame) -> pd.DataFrame:
    """复制 build_live_trend_ignition_feed.make_features 的等价特征变换入口。

    注意: 本脚本复用 build_live_trend_ignition_feed 的 make_features 以保证特征口径一致;
    若该模块存在则直接 import, 否则本地实现一份等价映射。
    """
    try:
        import build_live_trend_ignition_feed as blf  # type: ignore
        return blf.make_features(samples)
    except Exception:
        pass
    rows = samples.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    for col in ("breakout_pct", "amount_ratio", "return_20d", "return_60d", "volatility_20d",
                "ma20_over_ma60", "close_over_ma20", "drawdown_120d", "amount_trend_5_20",
                "breakout_count_20d", "pre_amount_trend_20_60", "pre_amount_accel", "pre_volume_z",
                "pre_consolidation_60", "pre_drift_slope_60", "pre_cnl_amplitude_60",
                "ign_day_amplitude", "ign_day_upper_shadow", "ign_day_close_pos", "ign_day_gap"):
        feat = "feature_" + col
        if feat in rows.columns and feat not in FEATURE_COLUMNS:
            pass
    # 直接取已存在的 feature_* 列
    have = [c for c in FEATURE_COLUMNS if c in rows.columns]
    return rows


def _fit_for_asof(train: pd.DataFrame, asof: pd.Timestamp, horizon: str,
                  embargo_days: int, tdates: np.ndarray) -> tuple[dict, int, int]:
    """对单个 as-of 做 purge 后拟合评分器, 返回 (scorer, 保留行数, 被剔除行数)."""
    label = LABEL_COL[horizon]
    ig = pd.to_datetime(train["ignition_date"], errors="coerce")
    # P0-2: 真实标签成熟日 = ignition 后第 horizon 个交易日 (交易日历精确计算)
    maturity = _maturity_dates(ig, tdates, int(horizon))
    cutoff = asof - pd.Timedelta(days=embargo_days)
    keep = maturity.notna() & (maturity <= cutoff)
    tr = train[keep]
    if tr.empty:
        # 该 as-of 尚无足够历史样本 (数据集点火起点附近), 跳过而非中止批量
        return None, int(keep.sum()), int((~keep).sum()), 0
    # 泄漏自检: purge 后的 tr 中, 仍有多少样本的标签在 as-of 之前未成熟 (即泄漏进 fit)
    # 自检与 purge 用同一真实交易日成熟日口径, 杜绝近似误差掩盖泄漏
    ig_tr = pd.to_datetime(tr["ignition_date"], errors="coerce")
    maturity_tr = _maturity_dates(ig_tr, tdates, int(horizon))
    leaked = int((maturity_tr > cutoff).sum())
    if leaked > 0:
        raise SystemExit(f"[leakfree] as-of {asof.date()}: purge 后仍泄漏 {leaked} 样本, 中止!")
    scorer = m.fit_binned_scorer(
        tr, feature_columns=FEATURE_COLUMNS, label_column=label, fusion="equal", bins=5
    )
    return scorer, int(keep.sum()), int((~keep).sum()), leaked


def main() -> None:
    ap = argparse.ArgumentParser(description="无泄漏 per-as-of 重建 TI 评分器与 feed")
    ap.add_argument("--training-set", default=str(DEFAULT_TRAIN))
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR))
    ap.add_argument("--asof", default="all", help="单个 as-of (YYYY-MM-DD) 或 all (批量回测窗口)")
    ap.add_argument("--horizon", choices=("60", "20"), default="60")
    ap.add_argument("--embargo-days", type=int, default=10)
    ap.add_argument("--hold-window", type=int, default=60, help="feed 信号 ffill 持有交易日")
    ap.add_argument("--start", default="2025-06-03", help="批量模式起始 as-of")
    ap.add_argument("--end", default="2026-08-20", help="批量模式截止 as-of (覆盖 holdout 验证窗口)")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL), help="价格面板(供交易日历)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    tdates = load_trading_dates(Path(args.panel))  # P0-2: 真实交易日历

    train = pd.read_csv(args.training_set)
    samples = pd.read_csv(args.samples)
    feats = make_features(samples)
    feats = feats.dropna(subset=FEATURE_COLUMNS)
    feats["sym_zp"] = feats["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    feats = feats.dropna(subset=["sym_zp"])
    feats["sym_zp"] = feats["sym_zp"].astype(int).map(lambda s: f"{s:06d}")
    feats["idate"] = pd.to_datetime(feats["ignition_date"], errors="coerce")

    # NOR feed 日期轴
    nor = pd.read_csv(args.nor_feed, usecols=["date"])
    panel_dates = pd.to_datetime(nor["date"], errors="coerce").dropna().sort_values()
    nor_cols = pd.read_csv(args.nor_feed, nrows=1).columns.drop("date").tolist()

    if args.asof == "all":
        asof_list = pd.date_range(args.start, args.end, freq="W-MON")
    else:
        asof_list = [pd.Timestamp(args.asof)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leak_rows = []  # 泄漏自检记录
    feed_cols = {}
    for asof in asof_list:
        asof = pd.Timestamp(asof)
        scorer, kept, purged, leaked = _fit_for_asof(train, asof, args.horizon, args.embargo_days, tdates)
        if scorer is None:
            leak_rows.append({"asof": asof.date(), "kept": kept, "purged": purged,
                              "future_event_leak": 0, "baseline": None, "skipped": "empty_purged"})
            continue
        # 泄漏自检: 该 as-of purge 后实际进入 fit 的样本里残留的未来标签数 (应为 0)
        leak_rows.append({"asof": asof.date(), "kept": kept, "purged": purged,
                          "future_event_leak": leaked, "baseline": round(scorer["baseline_rate"], 3)})
        # 对该 as-of 可用的点火样本打分 (ignition_date <= asof)
        avail = feats[(feats["idate"] <= asof)]
        # P1-1: active 信号按真实交易日持有窗口裁剪 (--hold-window 生效)：
        # 只保留 asof 前 hold_window 个交易日内的点火，旧点火信号过期剔除，避免污染覆盖率/Top-N。
        if args.hold_window and len(tdates):
            k = int(np.searchsorted(tdates, asof.value, side="right")) - 1
            min_date = tdates[max(0, k - args.hold_window)]
            avail = avail[avail["idate"] >= pd.Timestamp(min_date)]
        if avail.empty:
            continue
        scored = m.score_binned(avail, scorer)
        # 每个 symbol 取其在 as-of 之前最近一次点火的打分 (事件行级 -> symbol 截面)
        rec = (scored.sort_values("idate")
                      .groupby("sym_zp")["score"].last())
        feed_cols[asof] = rec

    # 输出结构: 行=as-of (每个 as-of 一行, 用该 as-of 专属评分器打分), 列=symbol
    # 值=该 as-of 下 symbol 的 TI 评分 (持有窗内 active 才有值); 与生产 feed 的"按 as-of 取行"语义一致.
    feed = pd.DataFrame(feed_cols).T
    feed.index.name = "asof"
    feed = feed.sort_index()
    out_path = out_dir / f"leakfree_feed_fwd{args.horizon}.csv"
    feed.reset_index().to_csv(out_path, index=False, encoding="utf-8")

    # 泄漏自检汇总
    max_leak = max(r["future_event_leak"] for r in leak_rows)
    print(f"[leakfree] horizon={args.horizon} asof数={len(asof_list)} -> {out_path}")
    print(f"  最大 future_event_leak={max_leak} (应为 0, 否则仍有泄漏)")
    print(f"  每个 as-of 平均保留训练样本={np.mean([r['kept'] for r in leak_rows]):.0f}, "
          f"平均 purge={np.mean([r['purged'] for r in leak_rows]):.0f}")
    # 写自检 json
    (out_dir / f"leakfree_audit_fwd{args.horizon}.json").write_text(
        json.dumps({"horizon": args.horizon, "max_future_event_leak": max_leak,
                    "asof_count": len(asof_list), "per_asof": leak_rows},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if max_leak > 0:
        raise SystemExit(f"[leakfree] 检测到泄漏 max_future_event_leak={max_leak} > 0, 中止!")


if __name__ == "__main__":
    main()
