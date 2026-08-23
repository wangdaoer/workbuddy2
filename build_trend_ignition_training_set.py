from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "feature_breakout_pct",
    "feature_log_amount_ratio",
    "feature_return_20d",
    "feature_return_60d",
    "feature_volatility_20d",
    "feature_ma20_over_ma60",
    "feature_close_over_ma20",
    "feature_drawdown_120d",
    "feature_log_amount_trend_5_20",
    "feature_breakout_count_20d",
    "feature_pre_amount_trend_20_60",
    "feature_pre_amount_accel",
    "feature_pre_volume_z",
    "feature_pre_consolidation_60",
    "feature_pre_drift_slope_60",
    "feature_pre_cnl_amplitude_60",
    "feature_ign_day_amplitude",
    "feature_ign_day_upper_shadow",
    "feature_ign_day_close_pos",
    "feature_ign_day_gap",
]
LABEL_COLUMNS = ["label_high_trend", "label_lifeline_survives"]


def classify_lifeline_quality(status: object, break_date: object) -> int:
    if str(status) == "alive_or_unbroken":
        return 1
    if pd.notna(break_date) and str(break_date):
        return 0
    return 0


def build_training_rows(
    samples: pd.DataFrame,
    *,
    high_return_threshold: float = 1.5,
    min_days_to_peak: int = 120,
    min_followup_days: int = 365,
) -> pd.DataFrame:
    rows = samples.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    rows["period"] = rows.get("period", "").astype(str)
    for column in (
        "peak_return",
        "return_to_end_from_ignition",
        "days_to_peak",
        "max_drawdown_to_peak",
        "breakout_pct",
        "amount_ratio",
        "return_20d",
        "return_60d",
        "volatility_20d",
        "ma20_over_ma60",
        "close_over_ma20",
        "drawdown_120d",
        "amount_trend_5_20",
        "breakout_count_20d",
        "pre_amount_trend_20_60",
        "pre_amount_accel",
        "pre_volume_z",
        "pre_consolidation_60",
        "pre_drift_slope_60",
        "pre_cnl_amplitude_60",
        "ign_day_amplitude",
        "ign_day_upper_shadow",
        "ign_day_close_pos",
        "ign_day_gap",
        "lifeline_ma",
        "min_distance_to_lifeline",
        "avg_distance_to_lifeline",
    ):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    if "analysis_end_date" not in rows.columns:
        raise ValueError("Training samples require analysis_end_date to remove censored outcomes")
    ignition_dates = pd.to_datetime(rows["ignition_date"], errors="coerce")
    analysis_end_dates = pd.to_datetime(rows["analysis_end_date"], errors="coerce")
    followup_days = (analysis_end_dates - ignition_dates).dt.days
    rows = rows.loc[followup_days.ge(min_followup_days)].copy()

    rows["label_high_trend"] = (
        rows["peak_return"].ge(high_return_threshold) & rows["days_to_peak"].ge(min_days_to_peak)
    ).astype(int)
    rows["label_lifeline_survives"] = [
        classify_lifeline_quality(status, break_date)
        for status, break_date in zip(rows["lifeline_status"], rows["lifeline_break_date"])
    ]
    rows["feature_breakout_pct"] = rows["breakout_pct"]
    rows["feature_log_amount_ratio"] = np.log1p(rows["amount_ratio"].clip(lower=0))
    rows["feature_return_20d"] = rows["return_20d"]
    rows["feature_return_60d"] = rows["return_60d"]
    rows["feature_volatility_20d"] = rows["volatility_20d"]
    rows["feature_ma20_over_ma60"] = rows["ma20_over_ma60"]
    rows["feature_close_over_ma20"] = rows["close_over_ma20"]
    rows["feature_drawdown_120d"] = rows["drawdown_120d"]
    rows["feature_log_amount_trend_5_20"] = np.log1p(rows["amount_trend_5_20"].clip(lower=0))
    rows["feature_breakout_count_20d"] = rows["breakout_count_20d"]
    rows["feature_pre_amount_trend_20_60"] = rows["pre_amount_trend_20_60"]
    rows["feature_pre_amount_accel"] = rows["pre_amount_accel"]
    rows["feature_pre_volume_z"] = rows["pre_volume_z"]
    rows["feature_pre_consolidation_60"] = rows["pre_consolidation_60"]
    rows["feature_pre_drift_slope_60"] = rows["pre_drift_slope_60"]
    rows["feature_pre_cnl_amplitude_60"] = rows["pre_cnl_amplitude_60"]
    rows["feature_ign_day_amplitude"] = rows["ign_day_amplitude"]
    rows["feature_ign_day_upper_shadow"] = rows["ign_day_upper_shadow"]
    rows["feature_ign_day_close_pos"] = rows["ign_day_close_pos"]
    rows["feature_ign_day_gap"] = rows["ign_day_gap"]

    columns = [
        "period",
        "symbol",
        "ignition_date",
        "peak_date",
        "peak_return",
        "return_to_end_from_ignition",
        "analysis_end_date",
        *LABEL_COLUMNS,
        *FEATURE_COLUMNS,
    ]
    return rows[columns].dropna(subset=["symbol", "ignition_date", "peak_date"]).reset_index(drop=True)


def load_samples(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "period" not in frame.columns or frame["period"].astype(str).str.strip().eq("").any():
            raise ValueError(f"Sample file must contain a non-empty period column: {path}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training rows from historical trend ignition samples.")
    parser.add_argument("--sample-csv", action="append", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/trend_ignition_training_set")
    parser.add_argument("--high-return-threshold", type=float, default=1.5)
    parser.add_argument("--min-days-to-peak", type=int, default=120)
    parser.add_argument("--min-followup-days", type=int, default=365)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples([Path(path) for path in args.sample_csv])
    training = build_training_rows(
        samples,
        high_return_threshold=args.high_return_threshold,
        min_days_to_peak=args.min_days_to_peak,
        min_followup_days=args.min_followup_days,
    )
    training.to_csv(output_dir / "trend_ignition_training_set.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "rows": int(len(training)),
        "symbols": int(training["symbol"].nunique()) if not training.empty else 0,
        "high_trend_positive_rate": float(training["label_high_trend"].mean()) if not training.empty else None,
        "lifeline_survival_rate": float(training["label_lifeline_survives"].mean()) if not training.empty else None,
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
    }
    (output_dir / "training_set_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Trend ignition training set saved to: {output_dir}")


if __name__ == "__main__":
    main()
