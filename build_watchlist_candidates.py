"""Build the watchlist overlay candidates CSV.

Candidates = 修复后 PIT 自选宇宙(掩码) ∩ 模型分数，asof 取最后一个有分数的日期。

分数源（2026-08-05 实测）：watchlist 宇宙 npz 尾部覆盖仅 ~10 只/天（宇宙训练样本少），
PIT 宇宙 npz 到 08-03 仍有 ~1160 只有效 —— 故演示默认用 **PIT npz 分数**（真实分数，
清晰标注来源），按 watchlist 掩码取候选。

列契约（apply_personal_trade_overlay 消费）：
    symbol, score, stock_name, selected, target_weight, trend_state, close, return_20d, close_position

用法：
    python build_watchlist_candidates.py [--scores-npz outputs/p10e_regime_gated/linear_mlp_scores_pit.npz]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from production_soft_score import P10E, build_panel  # noqa: E402
from watchlist_leak_audit import load_watchlist_mask  # noqa: E402

PANEL = Path("external_data/daily-market-data/data_panel.csv")
OUT = Path("outputs/watchlist_audit")
TOP_N = 20
SCORE_COLUMN = "mlp"  # 用 mlp 因果分作候选分


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-npz", default=str(P10E / "linear_mlp_scores_pit.npz"))
    parser.add_argument("--mode", choices=("watchlist", "full"), default="watchlist",
                        help="watchlist=自选宇宙∩分数(小集)；full=全市场有分候选(生产用法：全市场出票+行为叠加)")
    parser.add_argument("--out", default=None, help="输出路径覆盖")
    args = parser.parse_args()
    npz = Path(args.scores_npz)
    if not npz.exists():
        raise SystemExit(f"缺失 {npz}；请先跑 production_soft_score 重建后再执行")
    d = np.load(npz, allow_pickle=True)
    if "meta" not in d.files:
        print("[warn] npz 无 meta（旧缓存），分数可能基于旧宇宙；建议重建后重跑")

    print("[build] loading panel ...")
    P = build_panel(PANEL)
    cols = list(P["symbols"])
    mask = load_watchlist_mask(P["close"])

    score_df = pd.DataFrame(d[SCORE_COLUMN], index=P["label"].index, columns=cols)
    good = score_df.notna().any(axis=1)
    if not good.any():
        raise SystemExit("npz 无任何有效分数")
    asof = score_df.index[good][-1]  # 最后一个有分数的日期（分数覆盖尾部天然衰减）
    print(f"[build] 分数源={npz.name}  asof={asof.date()}（最后有分日期）")

    if args.mode == "watchlist":
        row = mask.loc[asof]
        cands = [c for c in cols if bool(row.get(c, False)) and pd.notna(score_df.loc[asof, c])]
        print(f"[build] 候选 {len(cands)} 只（修复后 PIT 自选宇宙 ∩ 有效分数, 掩码末日={mask.index.max().date()})")
    else:  # full — 全市场出票（生产用法：全市场候选 + 行为叠加）
        cands = [c for c in cols if pd.notna(score_df.loc[asof, c])]
        print(f"[build] 候选 {len(cands)} 只（全市场有分，asof={asof.date()}）")

    scores = score_df.loc[asof, cands]
    names = json.loads((OUT / "watchlist_names.json").read_text(encoding="utf-8"))

    df = pd.DataFrame({"symbol": cands, "score": scores.values.astype(float)})
    df["stock_name"] = df["symbol"].map(names).fillna("")
    # user_watchlist：该票是否在用户自选名单（掩码末日成员），供 overlay 偏好加分
    wl_last = mask.loc[asof]
    df["user_watchlist"] = df["symbol"].map(lambda c: bool(wl_last.get(c, False))).astype(bool)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["selected"] = df["rank"] <= TOP_N
    df["target_weight"] = np.where(df["selected"], 1.0 / TOP_N, 0.0)
    df["trend_state"] = ""
    df["close"] = np.nan
    df["return_20d"] = np.nan
    df["close_position"] = np.nan

    out_csv = Path(args.out) if args.out else OUT / "watchlist_candidates.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[build] -> {out_csv} ({len(df)} 行, top {TOP_N} selected, score col={SCORE_COLUMN}, asof={asof.date()})")


if __name__ == "__main__":
    main()
