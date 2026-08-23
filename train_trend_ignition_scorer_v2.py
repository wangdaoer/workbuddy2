"""train_trend_ignition_scorer_v2.py — leak-safe打磨版趋势点火评分器.

相比 v1 (train_trend_ignition_scorer.py) 的改进:
  - fusion: equal (现状等权) / ic_weighted (按|IC|加权各特征 lift)
  - model:  binned (v1 增强) / lightgbm (时序 walk-forward 内 fit, scale_pos_weight 处理极端不均衡)
  - 严格 past->future walk-forward (train period 训练, 之后 period 验证)
  - 仅使用合法 point-in-time 特征, 严禁任何未来泄漏字段
    (peak_return / days_to_peak / return_to_end / max_drawdown_to_peak /
     *distance_to_lifeline / lifeline_ma 全部属于标签组成或含未来信息, 一律不可作特征)

默认特征集 = build_trend_ignition_training_set.FEATURE_COLUMNS (10 个合法字段).
输出到独立 --output-dir, 不破坏原产物.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_trend_ignition_training_set import FEATURE_COLUMNS
from train_trend_ignition_scorer import (
    _assign_bin,
    _feature_edges,
    _period_summary,
    passes_research_gate,
)


# --------------------------------------------------------------------------
# binned (equal / ic_weighted)
# --------------------------------------------------------------------------
def fit_binned_scorer(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    fusion: str = "equal",
    bins: int = 5,
) -> dict:
    if train.empty:
        raise ValueError("Cannot fit scorer on empty training set")
    baseline = float(pd.to_numeric(train[label_column], errors="coerce").mean())
    features: dict[str, object] = {}
    for feature in feature_columns:
        edges = _feature_edges(train[feature], bins)
        assigned = _assign_bin(train[feature], edges)
        rates = train.groupby(assigned)[label_column].mean().to_dict()
        lift = {str(int(b)): float(rates.get(b, np.nan) - baseline) for b in rates}
        ic = train[feature].corr(train[label_column], method="spearman")
        ic = 0.0 if pd.isna(ic) else float(ic)
        features[feature] = {"edges": edges, "bin_lift": lift, "ic": ic}
    scorer = {
        "schema_version": 3,
        "model": "binned",
        "fusion": fusion,
        "label_column": label_column,
        "feature_columns": feature_columns,
        "bins": bins,
        "baseline_rate": baseline,
        "features": features,
        "training_end_date": (
            pd.to_datetime(train["ignition_date"], errors="coerce").max().strftime("%Y-%m-%d")
            if "ignition_date" in train and pd.to_datetime(train["ignition_date"], errors="coerce").notna().any()
            else None
        ),
    }
    tr_scored = score_binned(train, scorer)
    scorer["score_thresholds"] = {
        "low_max": float(tr_scored["score"].quantile(1 / 3)),
        "high_min": float(tr_scored["score"].quantile(2 / 3)),
    }
    return scorer


def score_binned(frame: pd.DataFrame, scorer: dict) -> pd.DataFrame:
    out = frame.copy()
    baseline = float(scorer["baseline_rate"])
    contributions = []
    weights = []
    for feature in scorer["feature_columns"]:
        spec = scorer["features"][feature]
        assigned = _assign_bin(out[feature], list(spec["edges"]))
        lift = assigned.astype(str).map(spec["bin_lift"]).fillna(0.0).astype(float)
        w = abs(float(spec["ic"])) if scorer.get("fusion") == "ic_weighted" else 1.0
        contributions.append(lift * w)
        weights.append(w)
    if contributions:
        score = baseline + pd.concat(contributions, axis=1).sum(axis=1) / sum(weights)
    else:
        score = baseline
    out["score"] = score
    return out


# --------------------------------------------------------------------------
# lightgbm (strict walk-forward fit)
# --------------------------------------------------------------------------
def fit_lightgbm(train: pd.DataFrame, *, feature_columns: list[str], label_column: str) -> dict:
    import lightgbm as lgb

    y = train[label_column].astype(int)
    pos = max(1, int(y.sum()))
    neg = max(1, int((y == 0).sum()))
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        random_state=42,
        verbose=-1,
    )
    model.fit(train[feature_columns], y)
    tr_score = model.predict_proba(train[feature_columns])[:, 1]
    return {
        "model": "lightgbm",
        "feature_columns": feature_columns,
        "estimator": model,
        "score_thresholds": {
            "low_max": float(pd.Series(tr_score).quantile(1 / 3)),
            "high_min": float(pd.Series(tr_score).quantile(2 / 3)),
        },
        "pos_weight": neg / pos,
    }


def score_lightgbm(frame: pd.DataFrame, scorer: dict) -> pd.DataFrame:
    out = frame.copy()
    cols = scorer["feature_columns"]
    out["score"] = scorer["estimator"].predict_proba(out[cols])[:, 1]
    return out


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------
def run_walk_forward(
    training: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    model: str = "binned",
    fusion: str = "equal",
    bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = sorted(str(p) for p in training["period"].dropna().unique())
    if len(periods) < 2:
        raise ValueError("Walk-forward validation requires at least two chronological periods")

    scored_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for index, period in enumerate(periods[1:], start=1):
        train = training[training["period"].isin(periods[:index])]
        valid = training[training["period"].eq(period)]
        if model == "lightgbm":
            scorer = fit_lightgbm(train, feature_columns=feature_columns, label_column=label_column)
            scored = score_lightgbm(valid, scorer)
        else:
            scorer = fit_binned_scorer(
                train, feature_columns=feature_columns, label_column=label_column, fusion=fusion, bins=bins
            )
            scored = score_binned(valid, scorer)
        scored["validation_period"] = period
        scored["training_periods"] = "|".join(periods[:index])
        scored_frames.append(scored)
        summary = _period_summary(scored, label_column, str(period), dict(scorer["score_thresholds"]))
        summary["training_periods"] = "|".join(periods[:index])
        summary["training_rows"] = int(len(train))
        summary_rows.append(summary)

    all_scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    return all_scored, pd.DataFrame(summary_rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-safe打磨版 trend ignition 评分器.")
    p.add_argument("--training-set", default="outputs/high_return_v2/trend_ignition_training_set/trend_ignition_training_set.csv")
    p.add_argument("--output-dir", default="outputs/high_return_v2/trend_ignition_scorer_v2")
    p.add_argument("--label-column", default="label_high_trend")
    p.add_argument("--model", choices=("binned", "lightgbm"), default="binned")
    p.add_argument("--fusion", choices=("equal", "ic_weighted"), default="equal")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--feature-columns", default=None, help="可选特征子集 (默认=合法 FEATURE_COLUMNS)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    training = pd.read_csv(args.training_set)
    feature_columns = (
        [c.strip() for c in args.feature_columns.split(",") if c.strip()]
        if args.feature_columns
        else list(FEATURE_COLUMNS)
    )
    # 防泄漏护栏: 拒绝任何已知未来/标签组成字段
    LEAK_FIELDS = {
        "peak_return", "days_to_peak", "return_to_end_from_ignition", "period_return_to_peak",
        "max_drawdown_to_peak", "min_distance_to_lifeline", "avg_distance_to_lifeline",
        "lifeline_ma", "lifeline_status", "lifeline_break_date", "breach_days_to_peak",
    }
    bad = [c for c in feature_columns if c in LEAK_FIELDS]
    if bad:
        raise ValueError(f"Refusing label-leakage fields as features: {bad}")

    scored, summary = run_walk_forward(
        training,
        feature_columns=feature_columns,
        label_column=args.label_column,
        model=args.model,
        fusion=args.fusion,
        bins=args.bins,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "walk_forward_scored.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "walk_forward_summary_cn.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "rows": int(len(training)),
        "label_column": args.label_column,
        "model": args.model,
        "fusion": args.fusion if args.model == "binned" else None,
        "feature_columns": feature_columns,
        "validation_mode": "strict_past_to_future_walk_forward",
        "validation_periods": summary["validation_period"].tolist() if not summary.empty else [],
        "mean_top_quantile_positive_rate": float(summary["top_quantile_positive_rate"].mean()) if not summary.empty else None,
        "mean_bottom_quantile_positive_rate": float(summary["bottom_quantile_positive_rate"].mean()) if not summary.empty else None,
        "mean_score_label_corr": float(summary["score_label_corr"].mean()) if not summary.empty else None,
        "mean_top_bottom_spread": float(summary["fixed_bucket_spread"].mean()) if not summary.empty else None,
        "min_fixed_bucket_spread": float(summary["fixed_bucket_spread"].min()) if not summary.empty else None,
        "min_score_label_corr": float(summary["score_label_corr"].min()) if not summary.empty else None,
        "passes_research_gate": passes_research_gate(summary) if not summary.empty else False,
    }
    metrics["deployment_status"] = "research_only"
    (out_dir / "scorer_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not summary.empty:
        print(summary.to_markdown(index=False, floatfmt=".4f"))
    print(f"v2 scorer saved to: {out_dir}")


if __name__ == "__main__":
    main()
