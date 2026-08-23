"""export_authoritative_scorer.py

将权威 trend_ignition 评分器 (label_fwd60_strong, binned-equal, 20 合法特征)
在全部数据上拟合, 导出为可部署 JSON 产物, 供生产侧 json.load + score_binned 直接加载打分.

注意: 此文件是 *部署快照* (fit on all data), 泛化证据来自同目录 walk_forward_summary_cn.csv (3 折严格 walk-forward, mean corr=0.27).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import train_trend_ignition_scorer_v2 as m

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "outputs" / "high_return_v2" / "trend_ignition_training_set_authoritative" / "trend_ignition_training_set.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_scorer_authoritative" / "authoritative_scorer.json"
LABEL = "label_fwd60_strong"
FEATURES = [
    "feature_breakout_pct", "feature_log_amount_ratio", "feature_return_20d", "feature_return_60d",
    "feature_volatility_20d", "feature_ma20_over_ma60", "feature_close_over_ma20", "feature_drawdown_120d",
    "feature_log_amount_trend_5_20", "feature_breakout_count_20d",
    "feature_pre_amount_trend_20_60", "feature_pre_amount_accel", "feature_pre_volume_z",
    "feature_pre_consolidation_60", "feature_pre_drift_slope_60", "feature_pre_cnl_amplitude_60",
    "feature_ign_day_amplitude", "feature_ign_day_upper_shadow", "feature_ign_day_close_pos", "feature_ign_day_gap",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-set", default=str(DEFAULT_TRAIN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--label-column", default=LABEL)
    args = ap.parse_args()

    df = pd.read_csv(args.training_set)
    scorer = m.fit_binned_scorer(
        df, feature_columns=FEATURES, label_column=args.label_column, fusion="equal", bins=5
    )

    # 自洽校验: 部署快照在自己数据上打分 (in-sample, 仅确认产物可加载/打分, 非泛化证据)
    scored = m.score_binned(df, scorer)
    self_corr = float(scored["score"].corr(df[args.label_column], method="spearman"))

    # numpy -> python 原生, 保证 JSON 可序列化
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, int)) and hasattr(o, "item"):
            return o.item()
        return o

    out = clean(scorer)
    out["_export_meta"] = {
        "label_column": args.label_column,
        "label_meaning": "ignition 后 60 交易日内收益 >= 15% (中短期续涨排序信号)",
        "training_rows": int(len(df)),
        "feature_count": len(FEATURES),
        "self_score_label_corr": self_corr,
        "generalization_evidence": "outputs/high_return_v2/trend_ignition_scorer_authoritative/walk_forward_summary_cn.csv (3-fold strict walk-forward mean corr=0.268)",
        "how_to_score": "import json,pandas as pd; scorer=json.load(open('authoritative_scorer.json')); from train_trend_ignition_scorer_v2 import score_binned; out=score_binned(new_df, scorer)",
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"deployed scorer -> {args.output}")
    print(f"  training_rows={len(df)} features={len(FEATURES)} self_corr={self_corr:.4f} baseline={scorer['baseline_rate']:.3f}")
    print(f"  score_thresholds={scorer['score_thresholds']}")


if __name__ == "__main__":
    main()
