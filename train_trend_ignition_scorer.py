from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_trend_ignition_training_set import FEATURE_COLUMNS


def _feature_edges(values: pd.Series, bins: int) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return []
    quantiles = np.linspace(0, 1, bins + 1)
    edges = sorted(set(float(value) for value in clean.quantile(quantiles).to_numpy()))
    return edges


def _assign_bin(values: pd.Series, edges: list[float]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna()
    if len(edges) <= 1:
        assigned = pd.Series(0, index=values.index, dtype=int)
        return assigned.mask(missing, -1)
    clipped = numeric.clip(lower=edges[0], upper=edges[-1])
    assigned = pd.cut(
        clipped,
        bins=edges,
        include_lowest=True,
        labels=False,
        duplicates="drop",
    ).fillna(-1).astype(int)
    return assigned.mask(missing, -1)


def validate_feature_contract(training: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in training.columns]
    if missing:
        raise ValueError(f"Training set is missing required point-in-time features: {missing}")


def fit_binned_feature_scorer(
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    bins: int = 5,
) -> dict[str, object]:
    if train.empty:
        raise ValueError("Cannot fit trend ignition scorer on an empty training set")
    missing = [column for column in [label_column, *feature_columns] if column not in train.columns]
    if missing:
        raise ValueError(f"Training set is missing required columns: {missing}")
    baseline_rate = float(pd.to_numeric(train[label_column], errors="coerce").mean())
    features: dict[str, object] = {}
    for feature in feature_columns:
        edges = _feature_edges(train[feature], bins)
        assigned = _assign_bin(train[feature], edges)
        rates = train.groupby(assigned)[label_column].mean().to_dict()
        features[feature] = {
            "edges": edges,
            "missing_bin": -1,
            "bin_lift": {str(int(bin_id)): float(rate - baseline_rate) for bin_id, rate in rates.items()},
        }
    scorer = {
        "schema_version": 2,
        "feature_contract": "ignition_close_point_in_time_v2",
        "label_column": label_column,
        "feature_columns": feature_columns,
        "bins": bins,
        "baseline_rate": baseline_rate,
        "features": features,
        "training_end_date": (
            pd.to_datetime(train["ignition_date"], errors="coerce").max().strftime("%Y-%m-%d")
            if "ignition_date" in train and pd.to_datetime(train["ignition_date"], errors="coerce").notna().any()
            else None
        ),
        "training_periods": sorted(str(period) for period in train.get("period", pd.Series(dtype=str)).dropna().unique()),
    }
    training_scores = score_with_binned_feature_scorer(train, scorer)["score"].dropna()
    scorer["score_thresholds"] = {
        "low_max": float(training_scores.quantile(1 / 3)),
        "high_min": float(training_scores.quantile(2 / 3)),
    }
    return scorer


def score_with_binned_feature_scorer(frame: pd.DataFrame, scorer: dict[str, object]) -> pd.DataFrame:
    scored = frame.copy()
    contributions = []
    baseline = float(scorer["baseline_rate"])
    for feature in scorer["feature_columns"]:
        spec = scorer["features"][feature]
        assigned = _assign_bin(scored[feature], list(spec["edges"]))
        lift = assigned.astype(str).map(spec["bin_lift"]).fillna(0.0).astype(float)
        scored[f"score_lift_{feature}"] = lift
        contributions.append(lift)
    if contributions:
        scored["score"] = baseline + pd.concat(contributions, axis=1).mean(axis=1)
    else:
        scored["score"] = baseline
    return scored


def _fixed_score_buckets(score: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    numeric = pd.to_numeric(score, errors="coerce")
    buckets = pd.Series("middle", index=score.index, dtype="object")
    buckets.loc[numeric.le(float(thresholds["low_max"]))] = "low"
    buckets.loc[numeric.ge(float(thresholds["high_min"]))] = "high"
    buckets.loc[numeric.isna()] = "missing"
    return buckets


def _period_summary(
    scored: pd.DataFrame,
    label_column: str,
    period: str,
    score_thresholds: dict[str, float],
) -> dict[str, object]:
    scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
    if scored.empty:
        return {
            "validation_period": period,
            "rows": 0,
            "positive_rate": None,
            "top_quantile_positive_rate": None,
            "bottom_quantile_positive_rate": None,
            "score_label_corr": None,
        }
    bucket = max(1, int(np.ceil(len(scored) * 0.2)))
    score = pd.to_numeric(scored["score"], errors="coerce")
    label = pd.to_numeric(scored[label_column], errors="coerce")
    buckets = _fixed_score_buckets(score, score_thresholds)
    bucket_counts = buckets.value_counts()
    bucket_rates = label.groupby(buckets).mean()
    correlation = (
        float(score.corr(label, method="spearman"))
        if score.nunique(dropna=True) > 1 and label.nunique(dropna=True) > 1
        else None
    )
    return {
        "validation_period": period,
        "rows": int(len(scored)),
        "positive_rate": float(scored[label_column].mean()),
        "top_quantile_positive_rate": float(scored.head(bucket)[label_column].mean()),
        "bottom_quantile_positive_rate": float(scored.tail(bucket)[label_column].mean()),
        "score_label_corr": correlation,
        "fixed_high_count": int(bucket_counts.get("high", 0)),
        "fixed_middle_count": int(bucket_counts.get("middle", 0)),
        "fixed_low_count": int(bucket_counts.get("low", 0)),
        "fixed_high_positive_rate": float(bucket_rates["high"]) if "high" in bucket_rates else None,
        "fixed_middle_positive_rate": float(bucket_rates["middle"]) if "middle" in bucket_rates else None,
        "fixed_low_positive_rate": float(bucket_rates["low"]) if "low" in bucket_rates else None,
        "fixed_bucket_spread": (
            float(bucket_rates["high"] - bucket_rates["low"])
            if "high" in bucket_rates and "low" in bucket_rates
            else None
        ),
    }


def walk_forward_period_validation(
    training: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = sorted(str(period) for period in training["period"].dropna().unique())
    if len(periods) < 2:
        raise ValueError("Walk-forward validation requires at least two chronological periods")
    scored_frames = []
    summary_rows = []
    normalized_period = training["period"].astype(str)
    for index, period in enumerate(periods[1:], start=1):
        training_periods = periods[:index]
        train = training[normalized_period.isin(training_periods)]
        valid = training[normalized_period.eq(period)]
        scorer = fit_binned_feature_scorer(
            train,
            feature_columns=feature_columns,
            label_column=label_column,
            bins=bins,
        )
        scored = score_with_binned_feature_scorer(valid, scorer)
        scored["score_threshold_low_max"] = float(scorer["score_thresholds"]["low_max"])
        scored["score_threshold_high_min"] = float(scorer["score_thresholds"]["high_min"])
        scored["validation_period"] = period
        scored["training_periods"] = "|".join(training_periods)
        scored_frames.append(scored)
        summary = _period_summary(
            scored,
            label_column,
            str(period),
            dict(scorer["score_thresholds"]),
        )
        summary["training_periods"] = "|".join(training_periods)
        summary["training_rows"] = int(len(train))
        summary_rows.append(summary)
    all_scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    return all_scored, pd.DataFrame(summary_rows)


def walk_forward_feature_diagnostics(
    training: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames = []
    for feature in feature_columns:
        _, summary = walk_forward_period_validation(
            training,
            feature_columns=[feature],
            label_column=label_column,
            bins=bins,
        )
        summary.insert(0, "feature", feature)
        summary["top_bottom_spread"] = summary["fixed_bucket_spread"]
        detail_frames.append(summary)
    detail_records = [record for frame in detail_frames for record in frame.to_dict("records")]
    details = pd.DataFrame(detail_records)
    if details.empty:
        return details, pd.DataFrame()
    aggregate = details.groupby("feature", as_index=False).agg(
        validation_periods=("validation_period", "nunique"),
        mean_top_bottom_spread=("top_bottom_spread", "mean"),
        min_top_bottom_spread=("top_bottom_spread", "min"),
        mean_score_label_corr=("score_label_corr", "mean"),
        min_score_label_corr=("score_label_corr", "min"),
        min_high_count=("fixed_high_count", "min"),
        min_middle_count=("fixed_middle_count", "min"),
        min_low_count=("fixed_low_count", "min"),
    )
    aggregate["passes_exploratory_gate"] = (
        aggregate["min_top_bottom_spread"].gt(0)
        & aggregate["min_score_label_corr"].gt(0)
        & aggregate["mean_top_bottom_spread"].ge(0.005)
        & aggregate["mean_score_label_corr"].ge(0.01)
        & aggregate[["min_high_count", "min_middle_count", "min_low_count"]].gt(0).all(axis=1)
    )
    return details, aggregate.sort_values(
        ["passes_exploratory_gate", "mean_top_bottom_spread", "mean_score_label_corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def passes_research_gate(summary: pd.DataFrame) -> bool:
    if summary.empty:
        return False
    spread = pd.to_numeric(summary["fixed_bucket_spread"], errors="coerce")
    correlation = pd.to_numeric(summary["score_label_corr"], errors="coerce")
    return bool(
        summary[["fixed_high_count", "fixed_middle_count", "fixed_low_count"]].gt(0).all(axis=None)
        and spread.notna().all()
        and spread.min() >= 0.005
        and spread.mean() >= 0.01
        and correlation.notna().all()
        and correlation.min() >= 0.01
        and correlation.mean() >= 0.02
    )


def leave_one_period_validation(
    training: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility alias; validation is strictly past-to-future."""
    return walk_forward_period_validation(
        training,
        feature_columns=feature_columns,
        label_column=label_column,
        bins=bins,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binned scorer for trend ignition labels.")
    parser.add_argument("--training-set", default="outputs/high_return_v2/trend_ignition_training_set/trend_ignition_training_set.csv")
    parser.add_argument("--output-dir", default="outputs/high_return_v2/trend_ignition_scorer")
    parser.add_argument("--label-column", default="label_high_trend")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Optional comma-separated point-in-time feature subset for a preregistered run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    training = pd.read_csv(args.training_set)
    feature_columns = (
        [column.strip() for column in args.feature_columns.split(",") if column.strip()]
        if args.feature_columns
        else list(FEATURE_COLUMNS)
    )
    unknown = [column for column in feature_columns if column not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown point-in-time features: {unknown}")
    validate_feature_contract(training, feature_columns)
    scored, summary = walk_forward_period_validation(
        training,
        feature_columns=feature_columns,
        label_column=args.label_column,
        bins=args.bins,
    )
    feature_details, feature_summary = walk_forward_feature_diagnostics(
        training,
        feature_columns=feature_columns,
        label_column=args.label_column,
        bins=args.bins,
    )
    scorer = fit_binned_feature_scorer(
        training,
        feature_columns=feature_columns,
        label_column=args.label_column,
        bins=args.bins,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "walk_forward_scored.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "walk_forward_summary_cn.csv", index=False, encoding="utf-8-sig")
    feature_details.to_csv(
        output_dir / "walk_forward_feature_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    feature_summary.to_csv(
        output_dir / "walk_forward_feature_diagnostics_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "binned_scorer.json").write_text(
        json.dumps(scorer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "rows": int(len(training)),
        "label_column": args.label_column,
        "feature_columns": feature_columns,
        "validation_mode": "strict_past_to_future_walk_forward",
        "validation_periods": summary["validation_period"].tolist() if not summary.empty else [],
        "mean_top_quantile_positive_rate": float(summary["top_quantile_positive_rate"].mean()) if not summary.empty else None,
        "mean_bottom_quantile_positive_rate": float(summary["bottom_quantile_positive_rate"].mean()) if not summary.empty else None,
        "mean_score_label_corr": float(summary["score_label_corr"].mean()) if not summary.empty else None,
        "exploratory_gate_features": feature_summary.loc[
            feature_summary["passes_exploratory_gate"], "feature"
        ].tolist(),
    }
    if not summary.empty:
        spread = pd.to_numeric(summary["fixed_bucket_spread"], errors="coerce")
        metrics["mean_top_bottom_spread"] = float(spread.mean())
        metrics["min_fixed_bucket_spread"] = float(spread.min())
        metrics["min_score_label_corr"] = float(summary["score_label_corr"].min())
        metrics["passes_research_gate"] = passes_research_gate(summary)
    else:
        metrics["mean_top_bottom_spread"] = None
        metrics["min_fixed_bucket_spread"] = None
        metrics["min_score_label_corr"] = None
        metrics["passes_research_gate"] = False
    metrics["deployment_status"] = "research_only"
    (output_dir / "scorer_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print(f"Trend ignition scorer saved to: {output_dir}")


if __name__ == "__main__":
    main()
