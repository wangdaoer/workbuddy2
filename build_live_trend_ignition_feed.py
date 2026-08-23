"""build_live_trend_ignition_feed.py — 把 TI feed 延伸到 live (生产接入前置).

问题: 原 trend_ignition_daily_pool.py 用权威训练集(点火止于 2026-04-22)建 feed,
到 live(2026-07-23) 已归零 -> 混合在 live 为空。本脚本改用**全量点火样本**
(trend_ignition_lifelines_202506_202608_v3, 点火到 2026-08-20) 套用与权威训练集
完全相同的 20 特征变换, 用 authoritative_scorer 打分, 重建延伸到 live 的 TI feed.

变换严格复制 build_trend_ignition_training_set.build_training_rows 的 feature_* 段
(仅特征映射, 不做标签/followup 过滤 -> 评分不需要标签).

输入:
  --samples  全量点火样本 (默认 .../trend_ignition_lifelines_202506_202608_v3/trend_ignition_samples.csv)
  --scorer   authoritative_scorer.json
  --nor-feed next_open_rank soft feed (对齐日期/列)
  --hold-window 点火后持有信号交易日 (默认 60)
  --output-dir 默认 outputs/high_return_v2/trend_ignition_daily_pool (覆盖原 feed 文件)

产物: trend_ignition_pool.csv + trend_ignition_feed.csv (延伸到 live).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import train_trend_ignition_scorer_v2 as m

ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES = ROOT / "outputs" / "high_return_v2" / "trend_ignition_lifelines_202506_202608_v3" / "trend_ignition_samples.csv"
DEFAULT_SCORER = ROOT / "outputs" / "high_return_v2" / "trend_ignition_scorer_authoritative" / "authoritative_scorer.json"
DEFAULT_SCORER_FWD20 = ROOT / "outputs" / "high_return_v2" / "trend_ignition_scorer_authoritative" / "authoritative_scorer_fwd20.json"
DEFAULT_NOR = ROOT / "outputs" / "production_soft_score" / "soft_score_feed.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool"

FEATURE_COLUMNS = [
    "feature_breakout_pct", "feature_log_amount_ratio", "feature_return_20d", "feature_return_60d",
    "feature_volatility_20d", "feature_ma20_over_ma60", "feature_close_over_ma20", "feature_drawdown_120d",
    "feature_log_amount_trend_5_20", "feature_breakout_count_20d",
    "feature_pre_amount_trend_20_60", "feature_pre_amount_accel", "feature_pre_volume_z",
    "feature_pre_consolidation_60", "feature_pre_drift_slope_60", "feature_pre_cnl_amplitude_60",
    "feature_ign_day_amplitude", "feature_ign_day_upper_shadow", "feature_ign_day_close_pos", "feature_ign_day_gap",
]


def make_features(samples: pd.DataFrame) -> pd.DataFrame:
    """严格复制 build_trend_ignition_training_set.build_training_rows 的 feature_* 段."""
    rows = samples.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    for col in ("breakout_pct", "amount_ratio", "return_20d", "return_60d", "volatility_20d",
                "ma20_over_ma60", "close_over_ma20", "drawdown_120d", "amount_trend_5_20",
                "breakout_count_20d", "pre_amount_trend_20_60", "pre_amount_accel", "pre_volume_z",
                "pre_consolidation_60", "pre_drift_slope_60", "pre_cnl_amplitude_60",
                "ign_day_amplitude", "ign_day_upper_shadow", "ign_day_close_pos", "ign_day_gap"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["feature_breakout_pct"] = rows["breakout_pct"]
    rows["feature_log_amount_ratio"] = np.log1p(rows["amount_ratio"].clip(lower=0))
    rows["feature_return_20d"] = rows["return_20d"]
    rows["feature_return_60d"] = rows["return_60d"]
    rows["feature_volatility_20d"] = rows["volatility_20d"]
    rows["feature_ma20_over_ma60"] = rows["ma20_over_ma60"]
    rows["feature_close_over_ma20"] = rows["close_over_ma20"]
    rows["feature_drawdown_120d"] = rows["drawdown_120d"]
    rows["feature_log_amount_trend_5_20"] = np.log1p(rows["amount_trend_5_20"].clip(lower=0))
    rows["feature_breakout_count_20d"] = rows["breakout_count_20d"]
    rows["feature_pre_amount_trend_20_60"] = rows["pre_amount_trend_20_60"]
    rows["feature_pre_amount_accel"] = rows["pre_amount_accel"]
    rows["feature_pre_volume_z"] = rows["pre_volume_z"]
    rows["feature_pre_consolidation_60"] = rows["pre_consolidation_60"]
    rows["feature_pre_drift_slope_60"] = rows["pre_drift_slope_60"]
    rows["feature_pre_cnl_amplitude_60"] = rows["pre_cnl_amplitude_60"]
    rows["feature_ign_day_amplitude"] = rows["ign_day_amplitude"]
    rows["feature_ign_day_upper_shadow"] = rows["ign_day_upper_shadow"]
    rows["feature_ign_day_close_pos"] = rows["ign_day_close_pos"]
    rows["feature_ign_day_gap"] = rows["ign_day_gap"]
    rows = rows.dropna(subset=["symbol"])
    rows["symbol"] = rows["symbol"].astype(str).str.zfill(6)
    return rows


def _build_feed(samples: pd.DataFrame, scorer_path: str, hold_window: int, panel_dates: pd.DatetimeIndex):
    """用给定评分器对全量点火样本打分, 生成延伸到 live 的 TI feed.

    返回 (feed_df, scored_df):
      feed_df   — 以 panel_dates 为索引, 列为符号, 值为 ti_score (ffill 到 hold_window)
      scored_df — 含 sym_zp/idate/score/active/td_dist_to_live 的点火级明细
    """
    scorer = json.load(open(scorer_path, encoding="utf-8"))
    feats = make_features(samples)
    feats = feats.dropna(subset=FEATURE_COLUMNS)
    scored = m.score_binned(feats, scorer)
    scored["sym_zp"] = scored["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    scored = scored.dropna(subset=["sym_zp"])
    scored["sym_zp"] = scored["sym_zp"].astype(int).map(lambda s: f"{s:06d}")
    scored["idate"] = pd.to_datetime(scored["ignition_date"], errors="coerce")

    pos = pd.Series(range(len(panel_dates)), index=panel_dates)
    td_dist = scored["idate"].map(
        lambda d: (len(panel_dates) - 1 - int(pos.loc[d])) if d in pos.index else np.nan)
    scored["td_dist_to_live"] = td_dist.astype("float")
    scored["active"] = scored["td_dist_to_live"] <= hold_window

    nor_cols = list(panel_dates) and [c for c in scored["sym_zp"].unique()]
    feed = pd.DataFrame(index=panel_dates, columns=nor_cols, dtype=float)
    feed.index.name = "date"
    for sym_zp, grp in scored.groupby("sym_zp"):
        for _, r in grp.iterrows():
            d = r["idate"]
            if d in feed.index:
                feed.loc[d, sym_zp] = r["score"]
        feed[sym_zp] = feed[sym_zp].ffill(limit=hold_window)
    return feed, scored


def main() -> None:
    ap = argparse.ArgumentParser(description="重建延伸到 live 的 TI feed")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--scorer", default=str(DEFAULT_SCORER))
    ap.add_argument("--scorer-fwd20", default=str(DEFAULT_SCORER_FWD20))
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR))
    ap.add_argument("--hold-window", type=int, default=60)
    ap.add_argument("--hold-window-fwd20", type=int, default=20)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    samples = pd.read_csv(args.samples)
    nor = pd.read_csv(args.nor_feed, usecols=["date"])
    panel_dates = pd.to_datetime(nor["date"], errors="coerce").dropna().sort_values()
    live = panel_dates.max()

    # 主 fwd60 feed (保持原 schema, 覆盖 trend_ignition_feed.csv)
    feed60, _ = _build_feed(samples, args.scorer, args.hold_window, panel_dates)
    # fwd20 feed (独立文件, 避免列名冲突)
    feed20, _ = _build_feed(samples, args.scorer_fwd20, args.hold_window_fwd20, panel_dates)

    # 主 fwd60 pool (用于 trend_ignition_pool.csv 排名展示)
    scored60 = _build_feed(samples, args.scorer, args.hold_window, panel_dates)[1]
    pool = scored60.sort_values("score", ascending=False).reset_index(drop=True)
    pool["rank"] = range(1, len(pool) + 1)
    pool_out = pool[["rank", "sym_zp", "ignition_date", "score", "active", "td_dist_to_live"]].copy()
    pool_out = pool_out.rename(columns={"sym_zp": "symbol", "score": "ti_score"})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_out.to_csv(out_dir / "trend_ignition_pool.csv", index=False, encoding="utf-8")
    feed60.reset_index().to_csv(out_dir / "trend_ignition_feed.csv", index=False, encoding="utf-8")
    feed20.reset_index().to_csv(out_dir / "trend_ignition_feed_fwd20.csv", index=False, encoding="utf-8")

    print(f"输入样本 {len(samples)} -> 特征齐备 {len(scored60)} 点火 (fwd60)")
    print(f"点火日期范围: {scored60['idate'].min().date()} .. {scored60['idate'].max().date()}")
    print(f"live={live.date()}: fwd60 活跃(<= {args.hold_window}td) 点火 {int(pool_out['active'].sum())}")
    print(f"Feed: fwd60 {feed60.shape[0]} 日 x {feed60.shape[1]} 符号; fwd20 {feed20.shape[0]} 日 x {feed20.shape[1]} 符号")
    print(f"  -> live 日 {live.date()} fwd60 非NaN 符号数: {int(feed60.loc[live].notna().sum())}; fwd20: {int(feed20.loc[live].notna().sum())}")
    print(f"saved -> {out_dir}/trend_ignition_pool.csv , trend_ignition_feed.csv , trend_ignition_feed_fwd20.csv")


if __name__ == "__main__":
    main()
