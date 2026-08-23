"""把方案 B 混合候选名单（trend_ignition × next_open_rank 加权混合）真正并入推荐 overlay。

设计（非侵入、可回退）：
  - 输入：宽度降仓后的权威 overlay（full_overlay_calibrated_derisked.csv）
          + 方案 B 混合名单（blended_pool_latest.csv，top-N 按 blended_rank 排序）
  - 输出：full_overlay_calibrated_blended.csv（与 derisked 同 schema + 若干 blend 标记列）
          **原始 derisked / full_overlay_calibrated.csv 完全不动**；展示层优先读 blended，
          缺失时回退 derisked → 删除 blended 文件即无损还原。

合并语义（"并入" = 混合候选成为推荐席位的优先来源）：
  1. seat_cap = derisked 当前 selected 席位数（已由宽度信号裁剪）。
  2. blend 优先占席：取混合名单 top-min(seat_cap, --max-blend-seats, top_n) 作为优先席位。
  3. overlay 补余：剩余席位由 overlay 原 selected（按 final_score）中未被 blend 顶替者补足。
     → 混合名单候选多到超过席位数时，会顶替纯 overlay 候选（这是"参与仓位"的本意）。
  4. 权重：selected 席位等权，总权重 = derisked selected 原总权重（暴露档位不变）。
  5. 标记列：in_blend_pool / blend_rank / blended_score / ti_score / nor_score /
            blend_inserted（新顶入）/ blend_boosted（原已在列中被强化）。

用法：
  python merge_blend_into_overlay.py
  python merge_blend_into_overlay.py --top-n 20 --max-blend-seats 11
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "outputs" / "watchlist_audit" / "full_overlay_calibrated_derisked.csv"
BLEND = HERE / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "blended_pool_latest.csv"
OUT = HERE / "outputs" / "watchlist_audit" / "full_overlay_calibrated_blended.csv"

WEIGHT_COLS = ["target_weight_after_behavior", "personal_adjusted_target_weight", "target_weight"]
SEL_FLAG = "personal_selected"


def _as_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(6)


def main() -> None:
    parser = argparse.ArgumentParser(description="把方案B混合名单并入推荐 overlay")
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--blend", default=str(BLEND))
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--top-n", type=int, default=20, help="混合名单取前 N 并入")
    parser.add_argument("--max-blend-seats", type=int, default=None,
                        help="blend 最多占多少席位（绝对上限；不传则由 --blend-share 决定）")
    parser.add_argument("--blend-share", type=float, default=0.5,
                        help="blend 占席位的比例（相对 seat_cap，默认 0.5=均衡混合）；"
                             "仅在未传 --max-blend-seats 时生效")
    parser.add_argument("--full-seats", type=int, default=20)
    args = parser.parse_args()

    if not Path(args.overlay).exists():
        raise SystemExit(f"缺失 overlay: {args.overlay}")
    if not Path(args.blend).exists():
        raise SystemExit(f"缺失混合名单: {args.blend}（请先跑 produce_daily_blended_pool.py）")

    df = pd.read_csv(args.overlay, dtype=str)
    df["symbol"] = _as_str_series(df["symbol"])
    blend = pd.read_csv(args.blend, dtype=str)
    blend["symbol"] = _as_str_series(blend["symbol"])
    # P0-2 修复：blended_rank 必须为数值，否则字符串排序会让 "10" < "2" 跳号。
    if "blended_rank" in blend.columns:
        blend["blended_rank"] = pd.to_numeric(blend["blended_rank"], errors="coerce")
        if blend["blended_rank"].isna().any():
            bad = blend.loc[blend["blended_rank"].isna(), "symbol"].tolist()
            raise SystemExit(f"[merge] blended_rank 解析失败(非数值): {bad[:10]} — 检查 blend 输入")
    else:
        raise SystemExit("[merge] blend 输入缺少 blended_rank 列，无法确定性排序")

    for c in WEIGHT_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[SEL_FLAG] = df.get(SEL_FLAG, pd.Series(False, index=df.index)).astype(str).str.lower().eq("true")

    score_col = "final_score" if "final_score" in df.columns else (
        "personal_adjusted_score" if "personal_adjusted_score" in df.columns else None
    )

    seat_cap = int(df[SEL_FLAG].sum())
    wcol = WEIGHT_COLS[0]
    selected_weight_sum = float(df.loc[df[SEL_FLAG], wcol].sum())
    exp = selected_weight_sum / (args.full_seats * 0.05) if (args.full_seats * 0.05) > 0 else 1.0

    # P0-2 修复：数值稳定排序（mergesort 保证同分保序），不再受 dtype=str 影响。
    blend_top = (blend.dropna(subset=["blended_rank"])
                     .sort_values("blended_rank", kind="mergesort")
                     .head(args.top_n)["symbol"].tolist())
    overlay_sel = df[df[SEL_FLAG]].copy()
    if score_col:
        overlay_sel = overlay_sel.sort_values(score_col, ascending=False)
    overlay_sel_syms = overlay_sel["symbol"].tolist()

    max_blend = min(seat_cap, len(blend_top))
    if args.max_blend_seats is not None:
        max_blend = min(max_blend, args.max_blend_seats)
    else:
        max_blend = min(max_blend, int(round(seat_cap * args.blend_share)))
    blend_keep = blend_top[:max_blend]
    overlay_fill = [s for s in overlay_sel_syms if s not in set(blend_keep)]
    overlay_fill = overlay_fill[: max(0, seat_cap - len(blend_keep))]
    new_selected = set(blend_keep) | set(overlay_fill)

    # 重置选中/权重
    df[SEL_FLAG] = False
    for c in WEIGHT_COLS:
        df[c] = 0.0

    n_sel = len(new_selected)
    equal_w = (selected_weight_sum / n_sel) if n_sel > 0 else 0.0
    sel_mask = df["symbol"].isin(new_selected)
    df.loc[sel_mask, SEL_FLAG] = True
    for c in WEIGHT_COLS:
        df.loc[sel_mask, c] = equal_w

    # 标记列（blend 维度，挂在全集上便于展示层取用）
    blend_lookup = blend.set_index("symbol")
    df["in_blend_pool"] = df["symbol"].isin(set(blend["symbol"]))
    orig_selected = set(overlay_sel_syms)
    df["blend_inserted"] = df["symbol"].isin(set(blend_keep)) & ~df["symbol"].isin(orig_selected)
    df["blend_boosted"] = df["symbol"].isin(set(blend_keep)) & df["symbol"].isin(orig_selected)
    df["blend_rank"] = df["symbol"].map(blend_lookup["blended_rank"])
    df["blended_score"] = df["symbol"].map(blend_lookup["blended_score"])
    df["ti_score"] = df["symbol"].map(blend_lookup["ti_score"])
    df["nor_score"] = df["symbol"].map(blend_lookup["nor_score"])

    displaced = len(orig_selected - new_selected)
    # ---- 并入校验 (P0-2 加固; P1-3: 校验全部通过后才写盘，失败不留坏产物) ----
    keep_ranks = (blend.set_index("symbol").loc[blend_keep, "blended_rank"]
                  .dropna().sort_values().tolist()) if blend_keep else []
    # rank 应连续 1..N（无字符串跳号遗留）
    if keep_ranks:
        expected = list(range(1, len(keep_ranks) + 1))
        if keep_ranks != expected:
            raise SystemExit(
                f"[merge] 顶入 rank 不连续(疑似排序bug): got {keep_ranks} expected {expected}")
    # 总权重必须等于 derisked 原 selected 总权重（暴露档位不变）
    final_sel = df[df[SEL_FLAG]]
    final_w = float(final_sel[WEIGHT_COLS[0]].sum())
    if abs(final_w - selected_weight_sum) > 1e-6:
        raise SystemExit(
            f"[merge] 总权重漂移: 原 {selected_weight_sum:.4f} != 现 {final_w:.4f}（暴露档位被破坏）")
    if n_sel != seat_cap:
        raise SystemExit(f"[merge] 席位数不匹配: 期望 {seat_cap} 实际 {n_sel}")

    # 全部校验通过后才落盘
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"[merge] 方案B并入完成 -> {args.output}")
    print(f"  席位数(seat_cap)={seat_cap}  暴露≈{exp:.2f}  总权重={selected_weight_sum:.3f}")
    print(f"  blend 顶入席位数={len(blend_keep)}  overlay 保留补余={len(overlay_fill)}")
    print(f"  新 selected 总数={n_sel}  等权={equal_w:.4f}/席")
    print(f"  顶入 rank 序列(连续校验通过)={keep_ranks}")
    print(f"  被 blend 顶替的纯 overlay 候选={displaced}")
    print(f"  blend 新顶入符号: {sorted(df.loc[df['blend_inserted'],'symbol'].tolist())}")


if __name__ == "__main__":
    main()
