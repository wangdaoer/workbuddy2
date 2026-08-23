from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from analyze_trend_ignition_lifelines import find_ignition_candidates
from build_trend_ignition_training_set import FEATURE_COLUMNS
from evaluate_strategy_family_forward_returns import load_strategy_watchlists
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix
from train_trend_ignition_scorer import score_with_binned_feature_scorer


EXPECTED_SCORER_SCHEMA = 2
EXPECTED_FEATURE_CONTRACT = "ignition_close_point_in_time_v2"
SELECTION_STATUSES = ("preregistered", "exploratory_posthoc")


def clean_symbol(value: object) -> str | None:
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) < 6:
        return None
    return digits[-6:]


def load_scorer_bundle(
    scorer_path: Path,
    summary_path: Path,
    *,
    selection_status: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if selection_status not in SELECTION_STATUSES:
        raise ValueError(f"Unknown scorer selection status: {selection_status}")
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if scorer.get("schema_version") != EXPECTED_SCORER_SCHEMA:
        raise ValueError("Trend ignition shadow scoring requires scorer schema_version=2")
    if scorer.get("feature_contract") != EXPECTED_FEATURE_CONTRACT:
        raise ValueError("Trend ignition scorer has an unsupported point-in-time feature contract")
    training_end = scorer.get("training_end_date")
    if not training_end or pd.isna(pd.to_datetime(training_end, errors="coerce")):
        raise ValueError("Trend ignition scorer must contain a valid training_end_date")

    features = scorer.get("feature_columns")
    if not isinstance(features, list) or not features:
        raise ValueError("Trend ignition scorer must contain non-empty feature_columns")
    unknown_features = [feature for feature in features if feature not in FEATURE_COLUMNS]
    if unknown_features:
        raise ValueError(f"Trend ignition scorer contains unknown features: {unknown_features}")
    feature_specs = scorer.get("features")
    if not isinstance(feature_specs, dict) or any(feature not in feature_specs for feature in features):
        raise ValueError("Trend ignition scorer is missing one or more feature specifications")

    thresholds = scorer.get("score_thresholds")
    if not isinstance(thresholds, dict) or not {"low_max", "high_min"}.issubset(thresholds):
        raise ValueError("Trend ignition scorer must contain fixed score_thresholds")
    low_max = float(thresholds["low_max"])
    high_min = float(thresholds["high_min"])
    if not np.isfinite(low_max) or not np.isfinite(high_min) or low_max > high_min:
        raise ValueError("Trend ignition scorer contains invalid fixed score thresholds")

    summary_features = summary.get("feature_columns")
    if summary_features != features:
        raise ValueError("Scorer summary feature_columns do not match the frozen scorer")
    if summary.get("deployment_status") != "research_only":
        raise ValueError("Trend ignition shadow scoring only accepts research_only scorers")

    metadata = {
        "schema_version": scorer["schema_version"],
        "feature_contract": scorer["feature_contract"],
        "training_end_date": str(training_end),
        "training_periods": scorer.get("training_periods", []),
        "feature_columns": list(features),
        "score_thresholds": {"low_max": low_max, "high_min": high_min},
        "passes_research_gate": bool(summary.get("passes_research_gate", False)),
        "deployment_status": "research_only",
        "selection_status": selection_status,
    }
    return scorer, metadata


def validate_shadow_dates(watchlists: pd.DataFrame, scorer_metadata: Mapping[str, object]) -> None:
    if watchlists.empty:
        return
    if "asof_date" not in watchlists.columns:
        raise ValueError("Trend ignition shadow watchlists require asof_date")
    score_dates = pd.to_datetime(watchlists["asof_date"], errors="coerce")
    if score_dates.isna().any():
        raise ValueError("Trend ignition shadow watchlists contain invalid asof_date values")
    training_end = pd.Timestamp(str(scorer_metadata["training_end_date"]))
    if score_dates.le(training_end).any():
        raise ValueError("Shadow score dates must be later than scorer training_end_date")


def _normalize_matrix_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    columns = [clean_symbol(column) for column in normalized.columns]
    if any(column is None for column in columns):
        raise ValueError("Historical price matrix contains an invalid A-share symbol")
    normalized.columns = columns
    if normalized.columns.duplicated().any():
        raise ValueError("Historical price matrix contains duplicate normalized symbols")
    return normalized


def _feature_rows(candidates: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rows = pd.DataFrame(index=candidates.index)
    rows["symbol"] = symbol
    rows["asof_date"] = pd.to_datetime(candidates.index).normalize()
    rows["feature_breakout_pct"] = pd.to_numeric(candidates["breakout_pct"], errors="coerce")
    rows["feature_log_amount_ratio"] = np.log1p(
        pd.to_numeric(candidates["amount_ratio"], errors="coerce").clip(lower=0)
    )
    rows["feature_return_20d"] = pd.to_numeric(candidates["return_20d"], errors="coerce")
    rows["feature_return_60d"] = pd.to_numeric(candidates["return_60d"], errors="coerce")
    rows["feature_volatility_20d"] = pd.to_numeric(candidates["volatility_20d"], errors="coerce")
    rows["feature_ma20_over_ma60"] = pd.to_numeric(candidates["ma20_over_ma60"], errors="coerce")
    rows["feature_close_over_ma20"] = pd.to_numeric(candidates["close_over_ma20"], errors="coerce")
    rows["feature_drawdown_120d"] = pd.to_numeric(candidates["drawdown_120d"], errors="coerce")
    rows["feature_log_amount_trend_5_20"] = np.log1p(
        pd.to_numeric(candidates["amount_trend_5_20"], errors="coerce").clip(lower=0)
    )
    rows["feature_breakout_count_20d"] = pd.to_numeric(
        candidates["breakout_count_20d"], errors="coerce"
    )
    return rows.reset_index(drop=True)


def score_shadow_watchlists(
    watchlists: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    amount: pd.DataFrame,
    scorer: dict[str, object],
    scorer_metadata: Mapping[str, object],
    *,
    breakout_window: int = 60,
    amount_window: int = 20,
    amount_multiplier: float = 1.5,
    breakout_buffer: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_shadow_dates(watchlists, scorer_metadata)
    source = watchlists.copy()
    if source.empty:
        return source, {
            "source_rows": 0,
            "source_dates": 0,
            "source_symbols": 0,
            "eligible_rows": 0,
            "eligible_dates": 0,
            "eligible_symbols": 0,
            "eligibility_ratio": 0.0,
            "missing_history_symbols": [],
            "score_bucket_counts": {},
        }

    source["symbol"] = source["symbol"].map(clean_symbol)
    if source["symbol"].isna().any():
        raise ValueError("Trend ignition shadow watchlists contain invalid A-share symbols")
    source["asof_date"] = pd.to_datetime(source["asof_date"], errors="raise").dt.normalize()
    close = _normalize_matrix_columns(close)
    high = _normalize_matrix_columns(high).reindex_like(close)
    amount = _normalize_matrix_columns(amount).reindex_like(close)

    feature_frames: list[pd.DataFrame] = []
    missing_history_symbols: list[str] = []
    for symbol in sorted(source["symbol"].unique()):
        if symbol not in close.columns:
            missing_history_symbols.append(symbol)
            continue
        candidates = find_ignition_candidates(
            close[symbol],
            high[symbol],
            amount[symbol],
            breakout_window=breakout_window,
            amount_window=amount_window,
            amount_multiplier=amount_multiplier,
            breakout_buffer=breakout_buffer,
        )
        if candidates.empty:
            continue
        wanted_dates = set(source.loc[source["symbol"].eq(symbol), "asof_date"])
        candidate_dates = pd.to_datetime(candidates.index).normalize()
        candidates = candidates.loc[candidate_dates.isin(wanted_dates)]
        if not candidates.empty:
            feature_frames.append(_feature_rows(candidates, symbol))

    feature_rows = (
        pd.concat(feature_frames, ignore_index=True)
        if feature_frames
        else pd.DataFrame(columns=["symbol", "asof_date", *FEATURE_COLUMNS])
    )
    eligible = source.merge(
        feature_rows,
        on=["symbol", "asof_date"],
        how="inner",
        validate="many_to_one",
    )
    if eligible.empty:
        scored = eligible.copy()
        scored["trend_ignition_score"] = pd.Series(dtype=float)
        scored["trend_ignition_score_bucket"] = pd.Series(dtype=str)
    else:
        scored = score_with_binned_feature_scorer(eligible, scorer).rename(
            columns={"score": "trend_ignition_score"}
        )
        thresholds = scorer_metadata["score_thresholds"]
        score = pd.to_numeric(scored["trend_ignition_score"], errors="coerce")
        scored["trend_ignition_score_bucket"] = "middle"
        scored.loc[score.le(float(thresholds["low_max"])), "trend_ignition_score_bucket"] = "low"
        scored.loc[score.ge(float(thresholds["high_min"])), "trend_ignition_score_bucket"] = "high"
        scored.loc[score.isna(), "trend_ignition_score_bucket"] = "missing"

    scored["trend_ignition_training_end_date"] = scorer_metadata["training_end_date"]
    scored["trend_ignition_selection_status"] = scorer_metadata["selection_status"]
    scored["trend_ignition_research_gate_passed"] = scorer_metadata["passes_research_gate"]
    scored["trend_ignition_deployment_status"] = "research_only"
    scored["trend_ignition_ranking_modified"] = False
    scored["research_only"] = True
    scored["trade_instruction"] = False
    scored = scored.sort_values(
        ["asof_date", "trend_ignition_score", "symbol"], ascending=[True, False, True]
    ).reset_index(drop=True)

    coverage = {
        "source_rows": int(len(source)),
        "source_dates": int(source["asof_date"].nunique()),
        "source_symbols": int(source["symbol"].nunique()),
        "eligible_rows": int(len(scored)),
        "eligible_dates": int(scored["asof_date"].nunique()) if not scored.empty else 0,
        "eligible_symbols": int(scored["symbol"].nunique()) if not scored.empty else 0,
        "eligibility_ratio": float(len(scored) / len(source)) if len(source) else 0.0,
        "missing_history_symbols": missing_history_symbols,
        "score_bucket_counts": {
            str(key): int(value)
            for key, value in scored.get(
                "trend_ignition_score_bucket", pd.Series(dtype=str)
            ).value_counts().items()
        },
    }
    return scored, coverage


def write_shadow_outputs(
    scored: pd.DataFrame,
    coverage: Mapping[str, object],
    output_dir: Path,
    scorer_metadata: Mapping[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "trend_ignition_shadow_scores.csv"
    report_path = output_dir / "trend_ignition_shadow_report.md"
    manifest_path = output_dir / "manifest.json"
    scored.to_csv(score_path, index=False, encoding="utf-8-sig")

    daily_files: list[str] = []
    if not scored.empty:
        for asof_date, group in scored.groupby("asof_date", sort=True):
            token = pd.Timestamp(asof_date).strftime("%Y%m%d")
            daily_path = output_dir / f"merged_priority_watchlist_{token}.csv"
            group.to_csv(daily_path, index=False, encoding="utf-8-sig")
            daily_files.append(daily_path.name)

    manifest = {
        "research_only": True,
        "trade_instruction": False,
        "ranking_modified": False,
        "source_watchlists_modified": False,
        "scorer": dict(scorer_metadata),
        "coverage": dict(coverage),
        "outputs": {
            "scores": score_path.name,
            "report": report_path.name,
            "daily_watchlists": daily_files,
        },
    }
    report = [
        "# Trend ignition shadow scoring",
        "",
        "Research/simulation only. Source watchlists and their rankings are unchanged.",
        "",
        f"- Selection status: `{scorer_metadata['selection_status']}`",
        f"- Training end: `{scorer_metadata['training_end_date']}`",
        f"- Research gate passed: `{str(scorer_metadata['passes_research_gate']).lower()}`",
        f"- Source rows: {coverage['source_rows']}",
        f"- Ignition-eligible rows: {coverage['eligible_rows']}",
        f"- Eligibility ratio: {float(coverage['eligibility_ratio']):.2%}",
        f"- Score buckets: `{json.dumps(coverage['score_bucket_counts'], ensure_ascii=False)}`",
        "",
        "Only rows satisfying the frozen point-in-time ignition contract are scored.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_shadow_scoring(
    data: Path,
    watchlist_dir: Path,
    scorer_path: Path,
    summary_path: Path,
    output_dir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    selection_status: str = "exploratory_posthoc",
    max_abs_daily_return: float = 0.22,
    breakout_window: int = 60,
    amount_window: int = 20,
    amount_multiplier: float = 1.5,
    breakout_buffer: float = 0.01,
) -> dict[str, object]:
    scorer, metadata = load_scorer_bundle(
        scorer_path,
        summary_path,
        selection_status=selection_status,
    )
    watchlists = load_strategy_watchlists(watchlist_dir, start=start, end=end)
    validate_shadow_dates(watchlists, metadata)
    if watchlists.empty:
        scored, coverage = score_shadow_watchlists(
            watchlists,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            scorer,
            metadata,
        )
    else:
        score_dates = pd.to_datetime(watchlists["asof_date"], errors="raise")
        history_start = (score_dates.min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        history_end = score_dates.max().strftime("%Y-%m-%d")
        wanted_symbols = {
            symbol
            for symbol in watchlists["symbol"].map(clean_symbol)
            if symbol is not None
        }
        raw = load_prices(data, history_start, history_end)
        if wanted_symbols and "symbol" in raw.columns:
            raw_symbols = raw["symbol"].map(clean_symbol)
            raw = raw.loc[raw_symbols.isin(wanted_symbols)].copy()
            raw["symbol"] = raw_symbols.loc[raw.index]
        close = clean_matrix(pivot_prices(raw, "close"), max_abs_daily_return)
        high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), max_abs_daily_return)
        amount = pivot_prices(raw, "amount").reindex_like(close)
        scored, coverage = score_shadow_watchlists(
            watchlists,
            close,
            high,
            amount,
            scorer,
            metadata,
            breakout_window=breakout_window,
            amount_window=amount_window,
            amount_multiplier=amount_multiplier,
            breakout_buffer=breakout_buffer,
        )
    manifest = write_shadow_outputs(scored, coverage, output_dir, metadata)
    manifest["inputs"] = {
        "data": str(data),
        "watchlist_dir": str(watchlist_dir),
        "scorer": str(scorer_path),
        "scorer_summary": str(summary_path),
        "start": start,
        "end": end,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score post-training trend ignition signals without modifying daily rankings."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--watchlist-dir", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--scorer-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--selection-status", choices=SELECTION_STATUSES, default="exploratory_posthoc")
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--breakout-window", type=int, default=60)
    parser.add_argument("--amount-window", type=int, default=20)
    parser.add_argument("--amount-multiplier", type=float, default=1.5)
    parser.add_argument("--breakout-buffer", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_shadow_scoring(
        Path(args.data),
        Path(args.watchlist_dir),
        Path(args.scorer),
        Path(args.scorer_summary),
        Path(args.output_dir),
        start=args.start,
        end=args.end,
        selection_status=args.selection_status,
        max_abs_daily_return=args.max_abs_daily_return,
        breakout_window=args.breakout_window,
        amount_window=args.amount_window,
        amount_multiplier=args.amount_multiplier,
        breakout_buffer=args.breakout_buffer,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
