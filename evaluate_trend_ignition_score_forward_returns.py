from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from evaluate_strategy_family_forward_returns import (
    attach_strategy_family_forward_returns,
    clean_symbol,
    load_strategy_watchlists,
    parse_horizons,
)
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix


def assign_score_bucket(
    frame: pd.DataFrame,
    score_thresholds: Mapping[str, float] | None = None,
    *,
    allow_dynamic_thresholds: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    if "trend_ignition_score" not in result.columns:
        raise ValueError("Forward evaluation requires trend_ignition_score; daily scoring is not integrated yet")
    raw_score = result["trend_ignition_score"]
    score = pd.to_numeric(raw_score, errors="coerce")
    valid = score.dropna()
    if valid.empty:
        raise ValueError("Forward evaluation has no valid trend_ignition_score values")
    if score_thresholds:
        low_cut = float(score_thresholds["low_max"])
        high_cut = float(score_thresholds["high_min"])
    elif allow_dynamic_thresholds:
        low_cut = float(valid.quantile(1 / 3))
        high_cut = float(valid.quantile(2 / 3))
    else:
        raise ValueError("Provide a time-valid scorer or explicitly enable dynamic research thresholds")
    result["trend_ignition_score_bucket"] = "middle"
    result.loc[score.le(low_cut), "trend_ignition_score_bucket"] = "low"
    result.loc[score.ge(high_cut), "trend_ignition_score_bucket"] = "high"
    result.loc[score.isna(), "trend_ignition_score_bucket"] = "missing"
    if score.isna().mean() > 0.05:
        raise ValueError("More than 5% of forward-evaluation rows are missing trend_ignition_score")
    return result


def validate_scorer_temporal_contract(
    watchlists: pd.DataFrame,
    scorer_metadata: Mapping[str, object],
) -> None:
    training_end = scorer_metadata.get("training_end_date")
    if not training_end:
        raise ValueError("Scorer metadata must contain training_end_date")
    if watchlists.empty:
        return
    score_dates = pd.to_datetime(watchlists["asof_date"], errors="coerce")
    if score_dates.isna().any():
        raise ValueError("Forward evaluation contains invalid asof_date values")
    if score_dates.le(pd.Timestamp(training_end)).any():
        raise ValueError("Historical score dates must be later than scorer training_end_date")


def summarize_score_forward_returns(
    samples: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if samples.empty:
        return pd.DataFrame()
    for bucket, group in samples.groupby("trend_ignition_score_bucket", dropna=False):
        for horizon in horizons:
            ret = pd.to_numeric(group.get(f"forward_return_{int(horizon)}d"), errors="coerce")
            adverse = pd.to_numeric(group.get(f"max_adverse_return_{int(horizon)}d"), errors="coerce")
            completed = ret.dropna()
            adverse_completed = adverse.dropna()
            rows.append(
                {
                    "trend_ignition_score_bucket": bucket,
                    "horizon_days": int(horizon),
                    "signal_count": int(len(group)),
                    "completed_count": int(len(completed)),
                    "pending_count": int(len(group) - len(completed)),
                    "avg_score": float(pd.to_numeric(group.get("trend_ignition_score"), errors="coerce").mean()),
                    "avg_return": float(completed.mean()) if len(completed) else np.nan,
                    "median_return": float(completed.median()) if len(completed) else np.nan,
                    "win_rate": float(completed.gt(0).mean()) if len(completed) else np.nan,
                    "worst_return": float(completed.min()) if len(completed) else np.nan,
                    "avg_adverse_return": float(adverse_completed.mean()) if len(adverse_completed) else np.nan,
                    "worst_adverse_return": float(adverse_completed.min()) if len(adverse_completed) else np.nan,
                }
            )
    order = {"high": 0, "middle": 1, "low": 2, "missing": 3}
    summary = pd.DataFrame(rows)
    summary["_order"] = summary["trend_ignition_score_bucket"].map(order).fillna(9)
    return summary.sort_values(["_order", "horizon_days"]).drop(columns=["_order"]).reset_index(drop=True)


def run_score_forward_evaluation(
    data: Path,
    watchlist_dir: Path,
    output_dir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    token: str | None = None,
    max_abs_daily_return: float = 0.22,
    score_thresholds: Mapping[str, float] | None = None,
    scorer_metadata: Mapping[str, object] | None = None,
    allow_dynamic_thresholds: bool = False,
) -> dict[str, Path]:
    watchlists = load_strategy_watchlists(watchlist_dir, start=start, end=end)
    if score_thresholds:
        validate_scorer_temporal_contract(watchlists, scorer_metadata or {})
    watchlists = assign_score_bucket(
        watchlists,
        score_thresholds,
        allow_dynamic_thresholds=allow_dynamic_thresholds,
    )
    if not watchlists.empty:
        bucket_counts = watchlists["trend_ignition_score_bucket"].value_counts()
        empty_buckets = [bucket for bucket in ("high", "middle", "low") if bucket_counts.get(bucket, 0) == 0]
        if empty_buckets:
            raise ValueError(f"Forward evaluation has empty score buckets: {empty_buckets}")
    raw = load_prices(data, None, None)
    open_px = clean_matrix(pivot_prices(raw, "open"), max_abs_daily_return)
    low_px = clean_matrix(pivot_prices(raw, "low").reindex_like(open_px), max_abs_daily_return)
    samples = attach_strategy_family_forward_returns(watchlists, open_px, low_px, horizons=horizons)
    summary = summarize_score_forward_returns(samples, horizons=horizons)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_token = token or (pd.Timestamp(end).strftime("%Y%m%d") if end else pd.Timestamp(open_px.index.max()).strftime("%Y%m%d"))
    paths = {
        "samples": output_dir / f"trend_ignition_score_forward_returns_{output_token}.csv",
        "summary": output_dir / f"trend_ignition_score_forward_summary_{output_token}.csv",
        "report": output_dir / f"trend_ignition_score_forward_report_{output_token}.md",
    }
    samples.to_csv(paths["samples"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    report = [
        f"# 强趋势起爆评分分层前瞻收益 {output_token}",
        "",
        "口径：收盘后生成观察清单，下一交易日开盘进入，按指定交易日后的开盘价退出。该报告只用于研究，不改变每日候选排序。",
        "分层：" + ("使用训练期固定阈值。" if score_thresholds else "显式启用当批动态分位研究。"),
        "评分器：" + (
            f"schema={scorer_metadata.get('schema_version')}, contract={scorer_metadata.get('feature_contract')}, "
            f"training_end={scorer_metadata.get('training_end_date')}"
            if scorer_metadata
            else "dynamic-research"
        ),
        "",
        summary.to_markdown(index=False, floatfmt=".4f") if not summary.empty else "_暂无样本。_",
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    return paths


# 研究性前向评估在以下情况下不应拖垮主流程：评分尚未集成、当日无符合条件样本、
# 分桶为空或缺失比例过高。这些属于"研究链路未就绪"，应优雅跳过而非抛错。
_RESEARCH_SKIP_PATTERNS = (
    "requires trend_ignition_score",
    "no valid trend_ignition_score",
    "Forward evaluation has empty score buckets",
    "More than 5% of forward-evaluation rows are missing",
)


def _is_research_skip(exc: ValueError) -> bool:
    message = str(exc)
    return any(pattern in message for pattern in _RESEARCH_SKIP_PATTERNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forward returns by trend ignition score buckets.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--watchlist-dir", default="outputs/high_return_v2")
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizons", type=parse_horizons, default=(1, 3, 5, 10, 20))
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--scorer", default=None, help="Optional scorer JSON providing fixed score_thresholds.")
    parser.add_argument("--allow-dynamic-thresholds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_thresholds = None
    scorer_metadata = None
    if args.scorer:
        scorer = json.loads(Path(args.scorer).read_text(encoding="utf-8"))
        score_thresholds = scorer.get("score_thresholds")
        if not score_thresholds:
            raise ValueError("Scorer JSON does not contain score_thresholds")
        scorer_metadata = {
            "schema_version": scorer.get("schema_version"),
            "feature_contract": scorer.get("feature_contract"),
            "training_end_date": scorer.get("training_end_date"),
        }
    try:
        paths = run_score_forward_evaluation(
            Path(args.data),
            Path(args.watchlist_dir),
            Path(args.output_dir),
            start=args.start,
            end=args.end,
            horizons=args.horizons,
            token=args.token,
            max_abs_daily_return=args.max_abs_daily_return,
            score_thresholds=score_thresholds,
            scorer_metadata=scorer_metadata,
            allow_dynamic_thresholds=args.allow_dynamic_thresholds,
        )
    except ValueError as exc:
        # 研究性前向评估：链路未就绪（评分未集成/无样本/分桶为空）时优雅跳过，
        # 不抛出非零退出码，避免拖垮主流程。其余异常如实上抛。
        if _is_research_skip(exc):
            print(f"RESEARCH_SKIP: trend ignition forward evaluation skipped: {exc}")
            return
        raise
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
