"""build_authoritative_trend_ignition_set.py

重建权威 trend_ignition 训练集: 以短 horizon 标签 label_fwd60_strong 为新研究目标,
按 ignition_date 切成多个 chronological fold, 供严格 past->future walk-forward.

做法(可逆, 独立 output-dir, 不改源):
  1. 从短 horizon 训练集(含 20 point-in-time 特征 + fwd 标签)加载.
  2. 剔除 fwd60 标签缺失行(点火过晚/无完整 60 交易日前向).
  3. 按 ignition_date 切 4 个 chronological period (名称按 end-date 命名, 字典序=时间序).
  4. 落到 outputs/high_return_v2/trend_ignition_training_set_authoritative/.

walk-forward 时:
  - 训练 = 该 fold 之前所有 fold; 验证 = 该 fold.
  - 4 fold -> 3 个验证折 (p2 / p3 / p4).
标签语义: 点火后 60 交易日内收益 >= 15% (中短期续涨排序信号, 非 4 月长牛识别).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "outputs" / "high_return_v2" / "trend_ignition_training_set_shorthorizon" / "trend_ignition_training_set.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_training_set_authoritative"
LABEL = "label_fwd60_strong"

# (period 名称, ignition_date 截止) — 名称字典序与时间序一致, 保证 walk-forward 正确排序
FOLDS = [
    ("p1_20250930", "2025-09-30"),
    ("p2_20251231", "2025-12-31"),
    ("p3_20260331", "2026-03-31"),
    ("p4_20260630", "2026-06-30"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build authoritative trend_ignition training set (multi-period).")
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--label", default=LABEL)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    d = pd.to_datetime(df["ignition_date"], errors="coerce")

    before = len(df)
    df = df[df[args.label].notna()].copy()
    after = len(df)

    period = pd.Series("pX_unassigned", index=df.index, dtype=object)
    # 互斥分配: 仅赋给尚未分配的行, 且按 cut 升序 -> 每行落入最早满足的 bucket
    for name, cut in FOLDS:
        m = (d <= pd.Timestamp(cut)) & (period == "pX_unassigned")
        period[m] = name
    # 兜底: 任何未落入的 (点火晚于最后 cut, 理论上不会, fwd60 缺失已剔除) -> 并入最后一折
    period[period == "pX_unassigned"] = FOLDS[-1][0]
    df["period"] = period

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "trend_ignition_training_set.csv", index=False, encoding="utf-8-sig")

    rep = {
        "rows_in": int(before),
        "rows_out": int(after),
        "dropped_no_label": int(before - after),
        "label": args.label,
        "n_folds": len(FOLDS),
        "folds": {name: int((period == name).sum()) for name, _ in FOLDS},
        "fold_positive_rate": {name: round(float(df.loc[period == name, args.label].mean()) * 100, 1) for name, _ in FOLDS},
    }
    (out_dir / "build_summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("saved ->", out_dir / "trend_ignition_training_set.csv")


if __name__ == "__main__":
    main()
