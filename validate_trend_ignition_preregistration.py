from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


TOP_LEVEL_KEYS = {
    "schema_version",
    "registration_id",
    "status",
    "registered_at",
    "source",
    "research_contract",
    "unseen_validation",
    "promotion_controls",
}
SOURCE_KEYS = {"draft_pr", "source_commit", "model_path", "model_sha256"}
CONTRACT_KEYS = {
    "feature_contract",
    "feature_columns",
    "bins",
    "label_column",
    "label_peak_return_at_least",
    "label_days_to_peak_at_least",
    "outcome_horizon_days",
    "minimum_followup_days",
    "ignition_cooldown_bars",
    "training_end_date",
    "training_periods",
}
VALIDATION_KEYS = {
    "signal_date_must_be_after",
    "minimum_completed_samples",
    "required_buckets",
    "minimum_bucket_count",
    "minimum_each_period_bucket_spread",
    "minimum_mean_bucket_spread",
    "minimum_each_period_score_label_corr",
    "minimum_mean_score_label_corr",
}
PROMOTION_KEYS = {
    "allow_daily_ranking",
    "allow_position_sizing",
    "allow_live_trading",
    "require_new_unseen_period",
    "require_manual_review",
}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(mapping: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{name} keys do not match contract; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _positive_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if number < 0 if allow_zero else number <= 0:
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preregistration(path: Path) -> dict[str, object]:
    registration = yaml.safe_load(path.read_text(encoding="utf-8"))
    registration = _mapping(registration, "preregistration")
    _require_exact_keys(registration, TOP_LEVEL_KEYS, "preregistration")
    if registration["schema_version"] != 1:
        raise ValueError("Unsupported preregistration schema_version")
    if registration["status"] != "preregistered_research_only":
        raise ValueError("Preregistration status must remain preregistered_research_only")
    return registration


def validate_preregistration(path: Path, *, repo_root: Path | None = None) -> dict[str, object]:
    registration = load_preregistration(path)
    source = _mapping(registration["source"], "source")
    contract = _mapping(registration["research_contract"], "research_contract")
    validation = _mapping(registration["unseen_validation"], "unseen_validation")
    promotion = _mapping(registration["promotion_controls"], "promotion_controls")
    _require_exact_keys(source, SOURCE_KEYS, "source")
    _require_exact_keys(contract, CONTRACT_KEYS, "research_contract")
    _require_exact_keys(validation, VALIDATION_KEYS, "unseen_validation")
    _require_exact_keys(promotion, PROMOTION_KEYS, "promotion_controls")

    if not re.fullmatch(r"[0-9a-f]{7,40}", str(source["source_commit"])):
        raise ValueError("source_commit must be a lowercase Git commit id")
    expected_sha = str(source["model_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("model_sha256 must be a lowercase SHA-256 digest")

    root = repo_root or path.resolve().parent.parent
    model_path = (root / str(source["model_path"])).resolve()
    if not model_path.is_file():
        raise ValueError(f"Frozen model does not exist: {model_path}")
    actual_sha = _sha256(model_path)
    if actual_sha != expected_sha:
        raise ValueError(f"Frozen model SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    model = _mapping(json.loads(model_path.read_text(encoding="utf-8")), "frozen model")

    feature_columns = contract["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be a non-empty unique list")
    if model.get("feature_columns") != feature_columns:
        raise ValueError("Frozen model feature_columns differ from preregistration")
    for field in ("feature_contract", "label_column", "bins", "training_end_date", "training_periods"):
        if model.get(field) != contract[field]:
            raise ValueError(f"Frozen model {field} differs from preregistration")
    if set(_mapping(model.get("features"), "frozen model features")) != set(feature_columns):
        raise ValueError("Frozen model feature definitions differ from feature_columns")

    thresholds = _mapping(model.get("score_thresholds"), "score_thresholds")
    low_max = float(thresholds.get("low_max", math.nan))
    high_min = float(thresholds.get("high_min", math.nan))
    if not math.isfinite(low_max) or not math.isfinite(high_min) or low_max >= high_min:
        raise ValueError("Frozen model score thresholds must be finite and ordered")

    training_end = pd.Timestamp(contract["training_end_date"])
    registered_at = pd.Timestamp(registration["registered_at"])
    validation_cutoff = pd.Timestamp(validation["signal_date_must_be_after"])
    if registered_at <= training_end:
        raise ValueError("registered_at must be later than training_end_date")
    if validation_cutoff != training_end:
        raise ValueError("signal_date_must_be_after must equal training_end_date")

    for field in (
        "bins",
        "label_peak_return_at_least",
        "label_days_to_peak_at_least",
        "outcome_horizon_days",
        "minimum_followup_days",
        "ignition_cooldown_bars",
    ):
        _positive_number(contract[field], f"research_contract.{field}")
    for field in (
        "minimum_completed_samples",
        "minimum_bucket_count",
        "minimum_each_period_bucket_spread",
        "minimum_mean_bucket_spread",
        "minimum_each_period_score_label_corr",
        "minimum_mean_score_label_corr",
    ):
        _positive_number(validation[field], f"unseen_validation.{field}")

    if validation["required_buckets"] != ["high", "middle", "low"]:
        raise ValueError("required_buckets must remain [high, middle, low]")
    expected_controls = {
        "allow_daily_ranking": False,
        "allow_position_sizing": False,
        "allow_live_trading": False,
        "require_new_unseen_period": True,
        "require_manual_review": True,
    }
    if promotion != expected_controls:
        raise ValueError("promotion_controls cannot be relaxed in this preregistration")

    return {
        "ok": True,
        "registration_id": registration["registration_id"],
        "status": registration["status"],
        "model_path": str(model_path),
        "model_sha256": actual_sha,
        "training_end_date": contract["training_end_date"],
        "first_eligible_signal_date": (training_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }


def validate_unseen_evidence(
    evidence: Mapping[str, object],
    registration: Mapping[str, object],
) -> dict[str, object]:
    source = _mapping(registration["source"], "source")
    validation = _mapping(registration["unseen_validation"], "unseen_validation")
    if evidence.get("registration_id") != registration["registration_id"]:
        raise ValueError("Evidence registration_id does not match preregistration")
    if evidence.get("model_sha256") != source["model_sha256"]:
        raise ValueError("Evidence model_sha256 does not match frozen model")
    if pd.Timestamp(evidence.get("signal_start_date")) <= pd.Timestamp(validation["signal_date_must_be_after"]):
        raise ValueError("Evidence signal_start_date is not strictly unseen")
    completed = int(evidence.get("completed_samples", 0))
    if completed < int(validation["minimum_completed_samples"]):
        raise ValueError("Evidence has too few completed samples")
    periods = evidence.get("periods")
    if not isinstance(periods, list) or not periods:
        raise ValueError("Evidence must contain at least one unseen period")

    spreads: list[float] = []
    correlations: list[float] = []
    for period in periods:
        period = _mapping(period, "evidence period")
        counts = _mapping(period.get("bucket_counts"), "evidence bucket_counts")
        for bucket in validation["required_buckets"]:
            if int(counts.get(bucket, 0)) < int(validation["minimum_bucket_count"]):
                raise ValueError(f"Evidence period {period.get('period')} has too few {bucket} samples")
        spread = float(period.get("fixed_bucket_spread", math.nan))
        correlation = float(period.get("score_label_corr", math.nan))
        if not math.isfinite(spread) or spread < float(validation["minimum_each_period_bucket_spread"]):
            raise ValueError(f"Evidence period {period.get('period')} fails bucket-spread gate")
        if not math.isfinite(correlation) or correlation < float(validation["minimum_each_period_score_label_corr"]):
            raise ValueError(f"Evidence period {period.get('period')} fails correlation gate")
        spreads.append(spread)
        correlations.append(correlation)

    if sum(spreads) / len(spreads) < float(validation["minimum_mean_bucket_spread"]):
        raise ValueError("Evidence fails mean bucket-spread gate")
    if sum(correlations) / len(correlations) < float(validation["minimum_mean_score_label_corr"]):
        raise ValueError("Evidence fails mean correlation gate")
    return {
        "ok": True,
        "completed_samples": completed,
        "period_count": len(periods),
        "mean_fixed_bucket_spread": sum(spreads) / len(spreads),
        "mean_score_label_corr": sum(correlations) / len(correlations),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a frozen trend-ignition preregistration.")
    parser.add_argument("--config", default="configs/trend_ignition_shortlist_preregistered.yaml")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--evidence", default=None, help="Optional unseen-evidence JSON to validate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    report = validate_preregistration(
        config_path,
        repo_root=Path(args.repo_root) if args.repo_root else None,
    )
    if args.evidence:
        registration = load_preregistration(config_path)
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        report["unseen_evidence"] = validate_unseen_evidence(evidence, registration)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
