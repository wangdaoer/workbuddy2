"""按市场宽度信号对当日 overlay 降仓（衍生层，不改动原始叠加产物）。

输入：
  - 当日 overlay：outputs/watchlist_audit/full_overlay_calibrated.csv
  - 宽度信号：outputs/market_regime/breadth_signal.json

输出：
  - outputs/watchlist_audit/full_overlay_calibrated_derisked.csv
    （在原始列基础上，对 personal_selected 行按比例缩放权重并裁剪席位；
     原始 full_overlay_calibrated.csv 保持不动，可随时回退）

降仓逻辑（与 next_open_rank 宽度闸门口径一致）：
  exposure_target ∈ {1.0, 0.55, 0.20}
  - 席位：保留得分最高的 round(20 * exposure_target) 席（其余 personal_selected 置 False、权重置 0）
  - 权重：选中席位的 target_weight_after_behavior / personal_adjusted_target_weight
          乘以 exposure_target，并在选中席位间重新归一化到该档暴露总权重。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "outputs" / "watchlist_audit" / "full_overlay_calibrated.csv"
SIGNAL = HERE / "outputs" / "market_regime" / "breadth_signal.json"
OUT = HERE / "outputs" / "watchlist_audit" / "full_overlay_calibrated_derisked.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="按宽度信号对 overlay 降仓")
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--signal", default=str(SIGNAL))
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--full-seats", type=int, default=20)
    args = parser.parse_args()

    if not Path(args.overlay).exists():
        raise SystemExit(f"缺失 overlay: {args.overlay}")
    if not Path(args.signal).exists():
        raise SystemExit(f"缺失信号: {args.signal}（请先跑 market_breadth_signal.py）")

    df = pd.read_csv(args.overlay)
    signal = json.loads(Path(args.signal).read_text(encoding="utf-8"))
    exp = float(signal["exposure_target"])
    asof = signal["date"]

    weight_cols = ["target_weight_after_behavior", "personal_adjusted_target_weight"]
    for c in weight_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    sel_flag = "personal_selected"
    if sel_flag not in df.columns:
        # 兜底：按原 target_weight>0 判定
        df[sel_flag] = df.get("target_weight", pd.Series(0.0, index=df.index)).gt(0.0)
    df[sel_flag] = df[sel_flag].astype(bool)

    base = df[df[sel_flag]].copy()
    # 按最终得分排序选前 round(full*exp) 席
    score_col = "final_score" if "final_score" in df.columns else (
        "personal_adjusted_score" if "personal_adjusted_score" in df.columns else None
    )
    if score_col:
        base = base.sort_values(score_col, ascending=False)

    keep_n = max(0, round(args.full_seats * exp))
    keep_idx = base.index[:keep_n]

    # 新选中标记
    df[sel_flag] = False
    df.loc[keep_idx, sel_flag] = True

    # 缩放权重 + 重新归一化到该档总暴露
    total_target = args.full_seats * 0.05 * exp  # 原每席 0.05
    if keep_n > 0:
        raw_w = df.loc[keep_idx, weight_cols[0]].clip(lower=0.0)
        s = raw_w.sum()
        if s > 0:
            scaled = raw_w / s * total_target
        else:
            scaled = pd.Series(total_target / keep_n, index=keep_idx)
        for c in weight_cols:
            df.loc[keep_idx, c] = scaled
        # 其余行权重清零
        df.loc[~df[sel_flag], weight_cols] = 0.0
    else:
        df[weight_cols] = 0.0

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"[derisk] {asof} 暴露={exp:.2f} 保留席位={keep_n}/{args.full_seats} "
          f"总权重={df[weight_cols[0]].sum():.3f} -> {args.output}")


if __name__ == "__main__":
    main()
