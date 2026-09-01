"""诊断：生产 soft_score 的「非线性(MLP)分支」到底有没有在出力。

背景
----
生产合成逻辑（production_soft_score.causal_soft_blend）：

    mlp_ic   = daily_ic({mlp: mlp_score}, label)
    trailing = mlp_ic.shift(2).rolling(IC_WIN=60, min_periods=IC_MIN=40).mean()
    adv      = ((THR_HI - trailing) / (THR_HI - THR_LO)).clip(0, 1).fillna(1.0)
    soft     = adv * linear + (1 - adv) * mlp

即：**MLP 只有在 trailing IC 抬到 THR_HI=0.03 时才能拿到权重**；
trailing <= THR_LO=0 时 adv=1，完全退化为纯线性模型。
而 MLP 本体（p10c_ensemble.MLP）配置为 hid=16 / lr=0.05 / **iters=150** /
**cap=8000**（训练样本硬截断），相对可用训练量（成熟窗口 252 日 × 数千 live 票）
是严重欠训练的。

本脚本量化回答：
  1. MLP 的逐日 IC 与线性分相比处于什么水平？
  2. trailing IC 有多少比例的日子越过 THR_HI / 落在 THR_LO 之下？
  3. adv 的分布 —— 有多少日子 adv==1（纯线性，非线性分支完全空闲）？
  4. 若把 MLP 权重强制打开（或关闭），soft 分的 IC 会怎样变化？（理论上下界探测）

用法：
    python diagnose_mlp_branch.py [--scores-npz PATH] [--out DIR]

产出：
    outputs/diagnose_mlp_branch/report.json
    outputs/diagnose_mlp_branch/adv_series.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_next_open_rank_model import (
    load_prices, pivot_prices, clean_matrix, next_open_return_label,
)
from p10c_ensemble import daily_ic
from production_soft_score import (
    MAX_ABS, PANEL, IC_WIN, IC_MIN, THR_HI, THR_LO, P10E,
)

OUT = Path(__file__).resolve().parent / "outputs" / "diagnose_mlp_branch"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-npz", default=None,
                    help="分数缓存路径（默认 outputs/p10e_regime_gated/linear_mlp_scores_pit_broker.npz）")
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    npz_path = Path(args.scores_npz) if args.scores_npz else (
        P10E / "linear_mlp_scores_pit_broker.npz")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) 标签（只需 open，不构建任何特征，成本低） ---
    print("[1/4] 载入面板并构建标签 ...")
    raw = load_prices(Path(args.panel), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    symbols = list(close.columns)
    print(f"      面板 {close.shape[0]} 行 / {len(symbols)} 只；标签 {label.shape}")

    # --- 2) 载入缓存分数，按 npz 的日期区间对齐 ---
    print(f"[2/4] 载入分数缓存 {npz_path.name} ...")
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    first, last = pd.Timestamp(meta["first_date"]), pd.Timestamp(meta["last_date"])
    n_dates = int(meta["n_dates"])
    sub_label = label.loc[(label.index >= first) & (label.index <= last)]
    if len(sub_label) != n_dates:
        raise SystemExit(
            f"面板与缓存行数不一致：缓存 {n_dates} 行 / 面板对齐后 {len(sub_label)} 行 "
            f"（缓存已陈旧，请用 load_or_build 重建）")
    linear = pd.DataFrame(d["linear"], index=sub_label.index, columns=symbols)
    mlp = pd.DataFrame(d["mlp"], index=sub_label.index, columns=symbols)
    print(f"      对齐 {n_dates} 行 ({first.date()} ~ {last.date()})")

    # --- 3) 复算生产同款 IC / trailing / adv ---
    print("[3/4] 计算 IC / trailing / adv ...")
    ic = daily_ic({"mlp": mlp, "linear": linear}, sub_label)
    mlp_ic, lin_ic = ic["mlp"], ic["linear"]
    trailing = mlp_ic.shift(2).rolling(IC_WIN, min_periods=IC_MIN).mean()
    adv = ((THR_HI - trailing) / (THR_HI - THR_LO)).clip(0, 1).fillna(1.0)

    # --- 4) 汇总 ---
    print("[4/4] 汇总 ...")
    valid = trailing.dropna()
    n_all = len(adv)
    rep = {
        "config": {
            "scores_npz": str(npz_path),
            "dates": f"{first.date()} ~ {last.date()}",
            "n_dates": n_dates,
            "IC_WIN": IC_WIN, "IC_MIN": IC_MIN,
            "THR_HI": THR_HI, "THR_LO": THR_LO,
        },
        "ic": {
            "mlp_mean": float(mlp_ic.mean()),
            "mlp_median": float(mlp_ic.median()),
            "mlp_std": float(mlp_ic.std()),
            "linear_mean": float(lin_ic.mean()),
            "linear_median": float(lin_ic.median()),
            "linear_std": float(lin_ic.std()),
            "mlp_minus_linear": float(mlp_ic.mean() - lin_ic.mean()),
            "mlp_ic_ir": float(mlp_ic.mean() / mlp_ic.std()) if mlp_ic.std() else None,
            "linear_ic_ir": float(lin_ic.mean() / lin_ic.std()) if lin_ic.std() else None,
        },
        "trailing": {
            "n_valid": int(len(valid)),
            "mean": float(valid.mean()),
            "median": float(valid.median()),
            "p95": float(valid.quantile(0.95)),
            "max": float(valid.max()),
            "pct_days_at_or_above_thr_hi": float((valid >= THR_HI).mean() * 100),
            "pct_days_below_thr_lo": float((valid < THR_LO).mean() * 100),
        },
        "adv": {
            "mean": float(adv.mean()),
            "median": float(adv.median()),
            "pct_days_pure_linear_adv_ge_0_999": float((adv >= 0.999).mean() * 100),
            "pct_days_mlp_any_weight_adv_lt_0_999": float((adv < 0.999).mean() * 100),
            "pct_days_mlp_dominant_adv_lt_0_5": float((adv < 0.5).mean() * 100),
            "pct_days_mlp_full_adv_le_0_001": float((adv <= 0.001).mean() * 100),
            "n_days_total": int(n_all),
        },
    }

    # 理论探测：固定混合比下 soft 分的全样本 IC（仅作上下界参考，非 walk-forward）
    probe = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = w * linear.values + (1 - w) * mlp.values
        s = pd.DataFrame(s, index=sub_label.index, columns=symbols)
        probe[f"linear_w_{w:.2f}"] = float(daily_ic({"s": s}, sub_label)["s"].mean())
    rep["fixed_blend_ic_probe"] = probe

    (out_dir / "report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"mlp_ic": mlp_ic, "linear_ic": lin_ic,
                  "trailing": trailing, "adv": adv}).to_csv(out_dir / "adv_series.csv")

    print()
    print("=" * 66)
    print("MLP 非线性分支诊断")
    print("=" * 66)
    icr = rep["ic"]
    print(f"逐日 IC        MLP {icr['mlp_mean']:+.4f} (IR {icr['mlp_ic_ir']:.3f})   "
          f"线性 {icr['linear_mean']:+.4f} (IR {icr['linear_ic_ir']:.3f})   "
          f"差 {icr['mlp_minus_linear']:+.4f}")
    tr = rep["trailing"]
    print(f"trailing IC    均值 {tr['mean']:+.4f} / 中位 {tr['median']:+.4f} / "
          f"p95 {tr['p95']:+.4f} / max {tr['max']:+.4f}  (THR_HI={THR_HI})")
    print(f"               越过 THR_HI 的日子 {tr['pct_days_at_or_above_thr_hi']:.1f}%   "
          f"低于 THR_LO 的日子 {tr['pct_days_below_thr_lo']:.1f}%")
    av = rep["adv"]
    print(f"adv 分布       均值 {av['mean']:.3f} / 中位 {av['median']:.3f}")
    print(f"               adv>=0.999（纯线性，MLP 零权重）{av['pct_days_pure_linear_adv_ge_0_999']:.1f}%")
    print(f"               adv<0.5（MLP 主导）            {av['pct_days_mlp_dominant_adv_lt_0_5']:.1f}%")
    print()
    print("固定混合比 IC 探测（全样本，仅供参考）：")
    for k, v in probe.items():
        print(f"   {k}: {v:+.4f}")
    print()
    print(f"报告 -> {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
