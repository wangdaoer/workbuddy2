"""trend_ignition_daily_pool.py — 独立候选池 B 生成器.

把 trend_ignition 表达为与 next_open_rank (soft_score_feed) 并列的独立候选池:
  - Pool B : 按 ti_score 排序的"活跃点火"候选名单 (独立可用).
  - Feed   : (date x symbol) 信号, 对齐到 next_open_rank 的 soft feed 日期/列,
             点火后 hold_window 交易日内持有 ti_score (ffill), 供交集/加权混合.

输入:
  --training-set  含 20 个 feature_* 列 + symbol + ignition_date 的点火训练集
                  (默认 = 权威训练集 outputs/high_return_v2/trend_ignition_training_set_authoritative/)
  --scorer        authoritative_scorer.json (export_authoritative_scorer.py 产出)
  --nor-feed      next_open_rank 的 soft feed, 用于对齐日期/列
                  (默认 outputs/production_soft_score/soft_score_feed.csv)
  --hold-window   点火后持有信号的交易日天数 (默认 60, 对应 fwd60 标签 horizon)
  --output-dir    默认 outputs/high_return_v2/trend_ignition_daily_pool/

产物:
  trend_ignition_pool.csv     独立候选池 B (rank, symbol, ignition_date, ti_score, active, ...)
  trend_ignition_feed.csv     (date x symbol) 信号, 直接对齐 soft feed, 供 blend_candidate_pools.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import train_trend_ignition_scorer_v2 as m

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "outputs" / "high_return_v2" / "trend_ignition_training_set_authoritative" / "trend_ignition_training_set.csv"
DEFAULT_SCORER = ROOT / "outputs" / "high_return_v2" / "trend_ignition_scorer_authoritative" / "authoritative_scorer.json"
DEFAULT_SCORER_FWD20 = ROOT / "outputs" / "high_return_v2" / "trend_ignition_scorer_authoritative" / "authoritative_scorer_fwd20.json"
DEFAULT_NOR = ROOT / "outputs" / "production_soft_score" / "soft_score_feed.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool"


def _scored_with_scorer(training_set_path: str, scorer_path: str, score_col: str) -> pd.DataFrame:
    """用指定评分器给点火训练集打分，返回含 score_col 列的 DataFrame（按 symbol/ignition_date 索引）。"""
    scorer = json.load(open(scorer_path, encoding="utf-8"))
    df = pd.read_csv(training_set_path)
    scored = m.score_binned(df, scorer)
    scored["sym_zp"] = scored["symbol"].astype(int).map(lambda s: f"{s:06d}")
    scored["idate"] = pd.to_datetime(scored["ignition_date"], errors="coerce")
    scored = scored.rename(columns={"score": score_col})
    return scored


def _build_feed(scored: pd.DataFrame, panel_dates: pd.Series, nor_cols: list[str],
                score_col: str, hold_window: int) -> pd.DataFrame:
    """对单个评分维度构建 (date x symbol) ffill feed。"""
    feed = pd.DataFrame(index=panel_dates, columns=nor_cols, dtype=float)
    feed.index.name = "date"
    for sym_zp, grp in scored.groupby("sym_zp"):
        if sym_zp not in feed.columns:
            continue
        for _, r in grp.iterrows():
            d = r["idate"]
            if d in feed.index:
                feed.loc[d, sym_zp] = r[score_col]
        feed[sym_zp] = feed[sym_zp].ffill(limit=hold_window)
    return feed


def main() -> None:
    ap = argparse.ArgumentParser(description="trend_ignition 独立候选池 B 生成器（支持 horizon-aware 双评分）")
    ap.add_argument("--training-set", default=str(DEFAULT_TRAIN))
    ap.add_argument("--scorer", default=str(DEFAULT_SCORER), help="主评分器（默认 fwd60 horizon）")
    ap.add_argument("--scorer-fwd20", default=str(DEFAULT_SCORER_FWD20),
                    help="短 horizon 评分器（label_fwd20_up）；传空字符串可禁用第二维度")
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR), help="next_open_rank soft feed (对齐用)")
    ap.add_argument("--hold-window", type=int, default=60, help="点火后持有信号的交易日天数（主 fwd60 维度）")
    ap.add_argument("--hold-window-fwd20", type=int, default=20, help="fwd20 维度的信号持有窗口")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    use_fwd20 = bool(args.scorer_fwd20)

    # 1) 打分（主维度 fwd60）
    scored_main = _scored_with_scorer(args.training_set, args.scorer, "score")
    # 2) 对齐到 next_open_rank 的 soft feed 日期/列
    nor = pd.read_csv(args.nor_feed, usecols=["date"])
    panel_dates = pd.to_datetime(nor["date"], errors="coerce").dropna().sort_values()
    nor_cols = pd.read_csv(args.nor_feed, nrows=1).columns.drop("date").tolist()
    live = panel_dates.max()

    # 3) Pool B (活跃点火, 按主 ti_score 排序)
    pos = pd.Series(range(len(panel_dates)), index=panel_dates)
    td_dist = scored_main["idate"].map(lambda d: (len(panel_dates) - 1 - int(pos.loc[d])) if d in pos.index else pd.NA)
    scored_main["td_dist_to_live"] = td_dist
    scored_main["active"] = scored_main["td_dist_to_live"].astype("float") <= args.hold_window
    pool = scored_main.sort_values("score", ascending=False).reset_index(drop=True)
    pool["rank"] = range(1, len(pool) + 1)
    pool_out = pool[["rank", "sym_zp", "ignition_date", "score", "active", "td_dist_to_live"]].copy()
    pool_out = pool_out.rename(columns={"sym_zp": "symbol", "score": "ti_score"})

    # 4) Feed: (date x symbol)，主维度 fwd60；若启用 fwd20 维度则另写一份 fwd20 feed
    #    （保持 trend_ignition_feed.csv 原 schema 不变，新增 trend_ignition_feed_fwd20.csv 避免列冲突）
    feed = _build_feed(scored_main, panel_dates, nor_cols, "score", args.hold_window)
    feed_fwd20 = None
    if use_fwd20:
        scored_fwd20 = _scored_with_scorer(args.training_set, args.scorer_fwd20, "score_fwd20")
        feed_fwd20 = _build_feed(scored_fwd20, panel_dates, nor_cols, "score_fwd20", args.hold_window_fwd20)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_out.to_csv(out_dir / "trend_ignition_pool.csv", index=False, encoding="utf-8")
    feed.reset_index().to_csv(out_dir / "trend_ignition_feed.csv", index=False, encoding="utf-8")
    if feed_fwd20 is not None:
        feed_fwd20.reset_index().to_csv(out_dir / "trend_ignition_feed_fwd20.csv", index=False, encoding="utf-8")

    n_active = int(pool_out["active"].sum())
    print(f"Pool B: {len(pool_out)} 点火, 活跃(<= {args.hold_window}td of live {live.date()}): {n_active}")
    print(f"  top10: {pool_out.head(10)['symbol'].tolist()}")
    print(f"Feed(fwd60): {feed.shape[0]} 日 x {feed.shape[1]} 符号, 对齐 soft feed")
    if feed_fwd20 is not None:
        print(f"Feed(fwd20): {feed_fwd20.shape[0]} 日 x {feed_fwd20.shape[1]} 符号")
    print(f"  活跃信号覆盖天数: {(feed.notna().any(axis=1)).sum()} / {feed.shape[0]}")
    print(f"saved -> {out_dir}/trend_ignition_pool.csv , trend_ignition_feed.csv"
          f"{' , trend_ignition_feed_fwd20.csv' if feed_fwd20 is not None else ''}")


if __name__ == "__main__":
    main()
