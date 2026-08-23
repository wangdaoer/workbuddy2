"""validate_label_horizon.py — 验证假设: trend_ignition 的 ~0.11 corr 天花板来自标签 horizon 太长.

做法(纯诊断, 不改源):
  1. 从 data_panel.csv 构建 close pivot (date x symbol).
  2. 对合并训练集(7322行, 含 20 特征 + 双周期)每条 (symbol, ignition_date):
       基准 = pivot 在点火日的收盘
       fwd_ret_H = pivot.shift(-H).loc[ignition_date, symbol] / 基准 - 1   (H=20/60/120 交易日)
  3. 定义短 horizon 标签: label_fwd{H}_up = (fwd_ret_H >= 0); label_fwd60_strong = (fwd_ret_60 >= 0.15)
  4. 打印各标签下 20 特征的 per-feature |IC| 与均值, 验证"horizon 越短越可预测".
  5. 保存带新标签的训练集 CSV 供评分器复用.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
TRAIN = ROOT / "outputs" / "high_return_v2" / "trend_ignition_training_set_v3_combined" / "trend_ignition_training_set.csv"
OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_training_set_shorthorizon"


def build_close_pivot(panel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(panel_path, usecols=["date", "symbol", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    piv = df.pivot(index="date", columns="symbol", values="close").sort_index()
    return piv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--output-dir", default=str(OUT))
    ap.add_argument("--horizons", default="20,60,120")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    print(f"[1/4] loading close pivot from {args.panel} ...")
    piv = build_close_pivot(Path(args.panel))
    print(f"    pivot shape={piv.shape} ({piv.index.min().date()}..{piv.index.max().date()})")

    print(f"[2/4] loading training set {args.train} ...")
    tr = pd.read_csv(args.train)
    feats = [c for c in tr.columns if c.startswith("feature_")]
    print(f"    rows={len(tr)}, features={len(feats)}")

    # 基准收盘 + 各 horizon 前向收盘 (pandas>=2: 用 reindex 矩阵, pivot.lookup 已移除)
    # 注意: 面板 symbol 为 int64, 保持 int 不要转 str, 否则 reindex 列不匹配 -> 全 NaN
    idx = pd.to_datetime(tr["ignition_date"], errors="coerce")
    sym = tr["symbol"].astype("int64")
    base = piv.reindex(index=idx, columns=sym).to_numpy().ravel()
    out = {f"fwd_ret_{h}": np.full(len(tr), np.nan) for h in horizons}
    for h in horizons:
        fwd = piv.shift(-h).reindex(index=idx, columns=sym).to_numpy().ravel()
        out[f"fwd_ret_{h}"] = fwd / base - 1.0
    fwd_df = pd.DataFrame(out)
    for h in horizons:
        r = fwd_df[f"fwd_ret_{h}"]
        tr[f"fwd_ret_{h}"] = r
        # 前向价格缺失 -> 标签 NaN (排除), 而非误标为 0(down)
        lab = pd.Series(np.nan, index=tr.index)
        lab[r.notna()] = (r[r.notna()] >= 0).astype(int)
        tr[f"label_fwd{h}_up"] = lab
    r60 = fwd_df["fwd_ret_60"]
    strong = pd.Series(np.nan, index=tr.index)
    strong[r60.notna()] = (r60[r60.notna()] >= 0.15).astype(int)
    tr["label_fwd60_strong"] = strong

    # 覆盖率
    print("[3/4] coverage + per-feature |IC| diagnostic")
    labels = [f"label_fwd{h}_up" for h in horizons] + ["label_fwd60_strong", "label_high_trend"]
    print(f"{'label':22s} {'cov%':>6s} {'pos%':>6s} |", end="")
    for c in feats[:6]:
        print(f" {c[8:18]:>11s}", end="")
    print("  mean|IC|")
    rows = []
    for lab in labels:
        sub = tr.dropna(subset=[lab])
        cov = len(sub) / len(tr)
        pos = sub[lab].mean()
        ics = [abs(sub[f].corr(sub[lab], method="spearman")) for f in feats]
        mean_ic = float(np.nanmean(ics))
        rows.append((lab, cov, pos, mean_ic))
        print(f"{lab:22s} {cov*100:5.1f}% {pos*100:5.1f}% |", end="")
        for c in feats[:6]:
            print(f" {abs(sub[c].corr(sub[lab],method='spearman')):11.4f}", end="")
        print(f"  {mean_ic:.4f}")

    print("    full per-feature |IC| table (mean across horizon labels):")
    for c in feats:
        line = f"    {c[8:]:24s}"
        for lab in labels:
            sub = tr.dropna(subset=[lab])
            line += f" {abs(sub[c].corr(sub[lab],method='spearman')):7.4f}"
        print(line + f"   <- {labels[0][:14]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out_dir / "trend_ignition_training_set.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "rows": int(len(tr)),
        "horizons": horizons,
        "labels": {lab: {"coverage": float(cov), "positive_rate": float(pos), "mean_abs_ic": float(mic)}
                   for (lab, cov, pos, mic) in rows},
        "feature_count": len(feats),
    }
    (out_dir / "validate_horizon_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] saved -> {out_dir / 'trend_ignition_training_set.csv'}")
    print(json.dumps(metrics["labels"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
