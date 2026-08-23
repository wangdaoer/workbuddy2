"""Diagnose stock-level losses for a multifactor walk-forward fold."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from multifactor_observation_evolution import (
    DEFAULT_PARAMETERS,
    FACTOR_NAMES,
    FACTOR_RANK_COLUMNS,
    WEIGHT_KEYS,
    EvolutionPeriods,
    ParameterEvaluation,
    evaluate_parameter_set,
    load_evolution_config,
    prepare_factor_panel,
    validate_parameters,
)
from run_multifactor_observation_evolution import load_benchmark, load_market_data


DEFAULT_CANDIDATES = ("baseline", "ma120_portfolio_stop_30d")
FORWARD_HORIZONS = (5, 10, 20)


def normalize_symbol(value: object) -> str:
    text = str(value).strip().upper()
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else text


def _first_matching_column(
    columns: Sequence[str], exact: Sequence[str], contains: Sequence[str] = ()
) -> str | None:
    for name in exact:
        if name in columns:
            return name
    for column in columns:
        if any(token in column for token in contains):
            return column
    return None


def _numeric_series(series: pd.Series, *, percent: bool = False) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    multipliers = pd.Series(1.0, index=series.index)
    multipliers.loc[text.str.endswith("亿", na=False)] = 100_000_000.0
    multipliers.loc[text.str.endswith("万", na=False)] = 10_000.0
    text = text.str.replace(r"[+%亿万]", "", regex=True)
    values = pd.to_numeric(text, errors="coerce") * multipliers
    return values.div(100.0) if percent else values


def load_current_snapshot_metadata(path: Path | None) -> pd.DataFrame:
    columns = [
        "symbol",
        "security_name_current",
        "industry_current",
        "subindustry_current",
        "market_cap_current",
        "turnover_rate_current",
        "snapshot_asof",
        "snapshot_is_point_in_time_for_fold",
    ]
    if path is None:
        return pd.DataFrame(columns=columns)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, low_memory=False)
    else:
        frame = pd.read_csv(
            path,
            sep="\t",
            encoding="gb18030",
            dtype=str,
            low_memory=False,
        )
    frame.columns = [str(column).strip() for column in frame.columns]
    code_column = _first_matching_column(
        frame.columns, ("代码", "证券代码", "symbol", "security_code")
    )
    if code_column is None:
        raise ValueError("snapshot metadata does not contain a security-code column")
    name_column = _first_matching_column(
        frame.columns, ("名称", "证券名称", "security_name", "name")
    )
    industry_column = _first_matching_column(
        frame.columns, ("所属行业", "industry"), ("所属行业",)
    )
    subindustry_column = _first_matching_column(
        frame.columns, ("细分行业", "subindustry"), ("细分行业",)
    )
    cap_column = _first_matching_column(
        frame.columns, ("总市值", "market_cap"), ("总市值",)
    )
    turnover_column = _first_matching_column(
        frame.columns, ("换手", "换手率", "turnover_rate"), ("换手",)
    )
    result = pd.DataFrame({"symbol": frame[code_column].map(normalize_symbol)})
    result["security_name_current"] = (
        frame[name_column].astype("string").str.strip() if name_column else pd.NA
    )
    result["industry_current"] = (
        frame[industry_column].astype("string").str.strip()
        if industry_column
        else pd.NA
    )
    result["subindustry_current"] = (
        frame[subindustry_column].astype("string").str.strip()
        if subindustry_column
        else pd.NA
    )
    result["market_cap_current"] = (
        _numeric_series(frame[cap_column]) if cap_column else np.nan
    )
    result["turnover_rate_current"] = (
        _numeric_series(frame[turnover_column], percent=True)
        if turnover_column
        else np.nan
    )
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    result["snapshot_asof"] = match.group(1) if match else pd.NA
    result["snapshot_is_point_in_time_for_fold"] = False
    return (
        result.loc[result["symbol"].str.fullmatch(r"\d{6}", na=False), columns]
        .drop_duplicates("symbol", keep="last")
        .reset_index(drop=True)
    )


def load_candidate_parameters(
    run_dir: Path, candidate_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    scores_path = Path(run_dir) / "candidate_scores.csv"
    frame = pd.read_csv(scores_path, dtype={"candidate_id": str}, low_memory=False)
    result: dict[str, dict[str, object]] = {}
    for candidate_id in candidate_ids:
        rows = frame.loc[frame["candidate_id"].eq(candidate_id)]
        if rows.empty:
            raise ValueError(f"candidate not found in {scores_path}: {candidate_id}")
        parameters = dict(DEFAULT_PARAMETERS)
        parameters.update(json.loads(str(rows.iloc[0]["parameters"])))
        result[candidate_id] = validate_parameters(parameters)
    return result


def _fold_row(evaluation: ParameterEvaluation, fold_id: str) -> dict[str, object]:
    for row in evaluation.fold_rows:
        if row["fold_id"] == fold_id:
            return dict(row)
    available = [str(row["fold_id"]) for row in evaluation.fold_rows]
    raise ValueError(f"fold {fold_id!r} not found; available: {available}")


def _factor_state_for_entries(
    entries: pd.DataFrame,
    factor_panel: pd.DataFrame,
    parameters: Mapping[str, object],
) -> pd.DataFrame:
    if entries.empty:
        return entries.copy()
    factor_columns = [
        "date",
        "symbol",
        *FACTOR_NAMES,
        *FACTOR_RANK_COLUMNS.values(),
    ]
    available = [column for column in factor_columns if column in factor_panel]
    signal_dates = entries["signal_date"].dropna().unique()
    symbols = entries["symbol"].unique()
    states = factor_panel.loc[
        factor_panel["date"].isin(signal_dates)
        & factor_panel["symbol"].isin(symbols),
        available,
    ].rename(columns={"date": "signal_date"})
    result = entries.merge(states, on=["signal_date", "symbol"], how="left")
    numerator = pd.Series(0.0, index=result.index)
    complete = pd.Series(True, index=result.index)
    weight_total = 0.0
    for factor, weight_key in zip(FACTOR_NAMES, WEIGHT_KEYS):
        rank_column = FACTOR_RANK_COLUMNS[factor]
        weight = float(parameters[weight_key])
        ranks = pd.to_numeric(result.get(rank_column), errors="coerce")
        complete &= ranks.notna()
        numerator = numerator.add(ranks.fillna(0.0) * weight)
        weight_total += weight
    result["evolution_score"] = numerator.div(weight_total).where(complete)
    result["median_amount_20"] = np.expm1(
        pd.to_numeric(result.get("liquidity_20"), errors="coerce")
    )
    return result


def _add_forward_open_returns(
    entries: pd.DataFrame, factor_panel: pd.DataFrame
) -> pd.DataFrame:
    if entries.empty:
        return entries.copy()
    symbols = entries["symbol"].unique()
    prices = factor_panel.loc[
        factor_panel["symbol"].isin(symbols), ["date", "symbol", "open"]
    ].copy()
    prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
    prices = prices.sort_values(["symbol", "date"], kind="mergesort")
    grouped = prices.groupby("symbol", sort=False)["open"]
    return_columns: list[str] = []
    for horizon in FORWARD_HORIZONS:
        column = f"forward_open_return_{horizon}d"
        prices[column] = grouped.shift(-horizon).div(prices["open"]).sub(1.0)
        return_columns.append(column)
    prices = prices.rename(columns={"date": "execution_date"})
    return entries.merge(
        prices[["execution_date", "symbol", *return_columns]],
        on=["execution_date", "symbol"],
        how="left",
    )


def build_entry_events(
    evaluation: ParameterEvaluation,
    factor_panel: pd.DataFrame,
    parameters: Mapping[str, object],
    candidate_id: str,
    fold_id: str,
) -> pd.DataFrame:
    positions = evaluation.backtest.position_weights
    if positions is None:
        raise ValueError("position recording is required for fold diagnostics")
    row = _fold_row(evaluation, fold_id)
    start = pd.Timestamp(row["start"])
    end = pd.Timestamp(row["end"])
    changes = positions.diff().fillna(positions)
    increases = changes.where(changes.gt(1e-12))
    stacked = (
        increases.stack(future_stack=True)
        .dropna()
        .rename("weight_increase")
        .reset_index()
    )
    stacked.columns = ["execution_date", "symbol", "weight_increase"]
    stacked = stacked.loc[
        stacked["execution_date"].between(start, end, inclusive="both")
    ].copy()
    previous_dates = pd.Series(positions.index, index=positions.index).shift(1)
    stacked["signal_date"] = stacked["execution_date"].map(previous_dates)
    stacked.insert(0, "candidate_id", candidate_id)
    stacked.insert(1, "fold_id", fold_id)
    stacked = _factor_state_for_entries(stacked, factor_panel, parameters)
    return _add_forward_open_returns(stacked, factor_panel)


def _historical_liquidity_bucket(value: object) -> str:
    if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(
        float(value)
    ):
        return "unknown"
    amount = float(value)
    if amount < 49_999_999.5:
        return "lt_50m"
    if amount < 99_999_999.5:
        return "50m_to_100m"
    if amount < 299_999_999.5:
        return "100m_to_300m"
    return "gte_300m"


def _current_cap_bucket(value: object) -> str:
    if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(
        float(value)
    ):
        return "unknown"
    cap = float(value)
    if cap < 5_000_000_000.0:
        return "lt_5bn"
    if cap < 10_000_000_000.0:
        return "5bn_to_10bn"
    if cap < 30_000_000_000.0:
        return "10bn_to_30bn"
    return "gte_30bn"


def build_symbol_contributions(
    evaluation: ParameterEvaluation,
    entries: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate_id: str,
    fold_id: str,
) -> pd.DataFrame:
    positions = evaluation.backtest.position_weights
    if positions is None:
        raise ValueError("position recording is required for fold diagnostics")
    row = _fold_row(evaluation, fold_id)
    start = pd.Timestamp(row["start"])
    end = pd.Timestamp(row["end"])
    held = positions.shift(1).loc[start:end].fillna(0.0)
    pnl = evaluation.backtest.symbol_pnl.loc[start:end].sum(axis=0)
    result = pd.DataFrame(
        {
            "symbol": held.columns,
            "exposure_days": held.abs().gt(1e-12).sum(axis=0).to_numpy(),
            "mean_abs_weight": held.abs().mean(axis=0).to_numpy(),
            "gross_contribution": pnl.reindex(held.columns).fillna(0.0).to_numpy(),
        }
    )
    result = result.loc[
        result["exposure_days"].gt(0) | result["gross_contribution"].ne(0.0)
    ].copy()
    if not entries.empty:
        aggregations: dict[str, tuple[str, str]] = {
            "entry_events": ("weight_increase", "size"),
            "entry_weight_total": ("weight_increase", "sum"),
            "entry_score_mean": ("evolution_score", "mean"),
            "entry_momentum_20_mean": ("momentum_20", "mean"),
            "entry_momentum_60_mean": ("momentum_60", "mean"),
            "entry_breakout_mean": ("breakout_distance_20", "mean"),
            "entry_trend_acceleration_mean": ("trend_acceleration", "mean"),
            "entry_median_amount_20": ("median_amount_20", "median"),
            "entry_momentum_20_rank_mean": (
                FACTOR_RANK_COLUMNS["momentum_20"],
                "mean",
            ),
            "entry_breakout_rank_mean": (
                FACTOR_RANK_COLUMNS["breakout_distance_20"],
                "mean",
            ),
            "entry_forward_5d_mean": ("forward_open_return_5d", "mean"),
            "entry_forward_10d_mean": ("forward_open_return_10d", "mean"),
            "entry_forward_20d_mean": ("forward_open_return_20d", "mean"),
        }
        entry_summary = entries.groupby("symbol", sort=False).agg(**aggregations)
        result = result.merge(entry_summary, on="symbol", how="left")
    result.insert(0, "candidate_id", candidate_id)
    result.insert(1, "fold_id", fold_id)
    result["gross_loss"] = result["gross_contribution"].clip(upper=0.0).abs()
    result["gross_gain"] = result["gross_contribution"].clip(lower=0.0)
    if not metadata.empty:
        result = result.merge(metadata, on="symbol", how="left")
    result["historical_liquidity_bucket"] = result.get(
        "entry_median_amount_20", pd.Series(np.nan, index=result.index)
    ).map(_historical_liquidity_bucket)
    result["current_cap_bucket"] = result.get(
        "market_cap_current", pd.Series(np.nan, index=result.index)
    ).map(_current_cap_bucket)
    return result.sort_values("gross_contribution", kind="mergesort").reset_index(
        drop=True
    )


def summarize_buckets(symbols: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    total_loss = float(symbols["gross_loss"].sum())
    summary = (
        symbols.groupby(["candidate_id", "fold_id", bucket_column], dropna=False)
        .agg(
            symbols=("symbol", "nunique"),
            exposure_days=("exposure_days", "sum"),
            gross_contribution=("gross_contribution", "sum"),
            gross_loss=("gross_loss", "sum"),
            gross_gain=("gross_gain", "sum"),
        )
        .reset_index()
    )
    summary["share_of_candidate_gross_loss"] = (
        summary["gross_loss"].div(total_loss) if total_loss > 1e-12 else 0.0
    )
    summary["metadata_is_point_in_time_for_fold"] = (
        bucket_column == "historical_liquidity_bucket"
    )
    return summary


def _weighted_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["weight_increase"], errors="coerce")
    valid = values.notna() & weights.gt(0.0)
    if not valid.any() or float(weights.loc[valid].sum()) <= 0.0:
        return None
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def summarize_entries(entries: pd.DataFrame, candidate_id: str, fold_id: str) -> dict[str, object]:
    result: dict[str, object] = {
        "candidate_id": candidate_id,
        "fold_id": fold_id,
        "entry_events": int(len(entries)),
        "entry_symbols": int(entries["symbol"].nunique()) if not entries.empty else 0,
    }
    metrics = (
        "momentum_20",
        "momentum_60",
        "breakout_distance_20",
        "trend_acceleration",
        "evolution_score",
        "median_amount_20",
        FACTOR_RANK_COLUMNS["momentum_20"],
        FACTOR_RANK_COLUMNS["breakout_distance_20"],
        "forward_open_return_5d",
        "forward_open_return_10d",
        "forward_open_return_20d",
    )
    for column in metrics:
        result[f"weighted_mean_{column}"] = (
            _weighted_mean(entries, column) if not entries.empty else None
        )
    if entries.empty:
        result["share_momentum_20_top_decile"] = None
        result["share_breakout_top_decile"] = None
        result["share_forward_10d_positive"] = None
    else:
        result["share_momentum_20_top_decile"] = float(
            pd.to_numeric(
                entries[FACTOR_RANK_COLUMNS["momentum_20"]], errors="coerce"
            ).ge(0.90).mean()
        )
        result["share_breakout_top_decile"] = float(
            pd.to_numeric(
                entries[FACTOR_RANK_COLUMNS["breakout_distance_20"]], errors="coerce"
            ).ge(0.90).mean()
        )
        result["share_forward_10d_positive"] = float(
            pd.to_numeric(entries["forward_open_return_10d"], errors="coerce")
            .dropna()
            .gt(0.0)
            .mean()
        )
    return result


def summarize_candidate(
    evaluation: ParameterEvaluation,
    symbols: pd.DataFrame,
    candidate_id: str,
    fold_id: str,
) -> dict[str, object]:
    row = _fold_row(evaluation, fold_id)
    losses = symbols.loc[symbols["gross_loss"].gt(0.0), "gross_loss"].sort_values(
        ascending=False
    )
    total_loss = float(losses.sum())
    return {
        "candidate_id": candidate_id,
        "fold_id": fold_id,
        "start": row["start"],
        "end": row["end"],
        "total_return_net": row["total_return"],
        "max_drawdown_net": row["max_drawdown"],
        "benchmark_return": row["benchmark_return"],
        "excess_return": row["excess_return"],
        "average_market_exposure": row["average_market_exposure"],
        "gross_symbol_contribution_sum": float(symbols["gross_contribution"].sum()),
        "gross_loss_sum": total_loss,
        "loss_symbols": int(losses.size),
        "exposed_symbols": int(symbols["symbol"].nunique()),
        "top_5_loss_share": (
            float(losses.head(5).sum() / total_loss) if total_loss > 1e-12 else 0.0
        ),
        "top_10_loss_share": (
            float(losses.head(10).sum() / total_loss) if total_loss > 1e-12 else 0.0
        ),
        "portfolio_stop_triggers_full_selection": int(
            evaluation.backtest.execution_counts.get("portfolio_stop_triggers", 0)
        ),
    }


def _format_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.2%}"


def _display_text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def build_markdown_report(
    candidate_summary: pd.DataFrame,
    entry_summary: pd.DataFrame,
    symbols: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
    cap_summary: pd.DataFrame,
    fold_id: str,
    snapshot_path: Path | None,
) -> str:
    lines = [
        f"# {fold_id} multifactor failure diagnosis",
        "",
        "Research/simulation only. This report does not promise returns and does not authorize trading.",
        "",
        "Historical liquidity and signal factors are point-in-time. Current market cap, name, and industry are annotations from the latest snapshot only and are not valid historical selection inputs.",
        "",
        "## Candidate comparison",
        "",
    ]
    comparison_rows = []
    for row in candidate_summary.to_dict("records"):
        comparison_rows.append(
            (
                row["candidate_id"],
                _format_pct(row["total_return_net"]),
                _format_pct(row["max_drawdown_net"]),
                _format_pct(row["excess_return"]),
                _format_pct(row["top_5_loss_share"]),
                row["loss_symbols"],
            )
        )
    lines.extend(
        _markdown_table(
            ("Candidate", "Net return", "Max drawdown", "Excess", "Top-5 loss share", "Loss symbols"),
            comparison_rows,
        )
    )
    lines.extend(["", "## Entry-state evidence", ""])
    entry_rows = []
    for row in entry_summary.to_dict("records"):
        entry_rows.append(
            (
                row["candidate_id"],
                row["entry_events"],
                _format_pct(row.get("share_momentum_20_top_decile")),
                _format_pct(row.get("share_breakout_top_decile")),
                _format_pct(row.get("weighted_mean_forward_open_return_10d")),
                _format_pct(row.get("share_forward_10d_positive")),
            )
        )
    lines.extend(
        _markdown_table(
            ("Candidate", "Buy/increase events", "Momentum top decile", "Breakout top decile", "Weighted fwd 10d", "Positive fwd 10d"),
            entry_rows,
        )
    )
    for candidate_id in candidate_summary["candidate_id"]:
        lines.extend(["", f"## Largest gross losses: {candidate_id}", ""])
        subset = symbols.loc[symbols["candidate_id"].eq(candidate_id)].head(10)
        loser_rows = []
        for row in subset.to_dict("records"):
            cap = row.get("market_cap_current")
            cap_text = (
                f"{float(cap) / 100_000_000:.1f} yi"
                if pd.notna(cap)
                else "n/a"
            )
            loser_rows.append(
                (
                    row["symbol"],
                    _display_text(row.get("security_name_current")),
                    _format_pct(row["gross_contribution"]),
                    row["exposure_days"],
                    row.get("historical_liquidity_bucket", "unknown"),
                    cap_text,
                    _display_text(row.get("industry_current")),
                )
            )
        lines.extend(
            _markdown_table(
                ("Symbol", "Current name", "Gross contribution", "Exposure days", "PIT liquidity", "Current cap", "Current industry"),
                loser_rows,
            )
        )
    lines.extend(["", "## Automated audit conclusion", ""])
    for row in candidate_summary.to_dict("records"):
        candidate_id = row["candidate_id"]
        entry_row = entry_summary.loc[
            entry_summary["candidate_id"].eq(candidate_id)
        ].iloc[0]
        liquidity = liquidity_summary.loc[
            liquidity_summary["candidate_id"].eq(candidate_id)
        ]
        low_loss_share = float(
            liquidity.loc[
                liquidity["historical_liquidity_bucket"].isin(
                    ["lt_50m", "50m_to_100m"]
                ),
                "share_of_candidate_gross_loss",
            ].sum()
        )
        findings = []
        findings.append(
            "losses are concentrated in a small group"
            if float(row["top_5_loss_share"]) >= 0.50
            else "losses are broad rather than dominated by five names"
        )
        findings.append(
            "low-liquidity names dominate losses"
            if low_loss_share >= 0.50
            else "low-liquidity names do not explain most losses"
        )
        crowded_share = max(
            float(entry_row.get("share_momentum_20_top_decile") or 0.0),
            float(entry_row.get("share_breakout_top_decile") or 0.0),
        )
        forward_10d = entry_row.get("weighted_mean_forward_open_return_10d")
        if crowded_share >= 0.50 and pd.notna(forward_10d) and float(forward_10d) < 0.0:
            findings.append("high-rank entries reversed over the following 10 sessions")
        else:
            findings.append("the evidence does not isolate a high-rank 10-session reversal")
        lines.append(f"- `{candidate_id}`: " + "; ".join(findings) + ".")
    lines.extend(
        [
            "",
            "Next-factor rule: use only point-in-time evidence. A current-cap or current-industry pattern may guide a data-quality question, but it must not become a backtest filter until historical point-in-time membership/metadata exist.",
            "",
            f"Current snapshot: `{snapshot_path}`" if snapshot_path else "Current snapshot: not supplied.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_failure_diagnosis(
    panel: pd.DataFrame,
    benchmark: pd.Series | None,
    periods: EvolutionPeriods,
    candidate_parameters: Mapping[str, Mapping[str, object]],
    fold_id: str,
    metadata: pd.DataFrame,
    output_dir: Path,
    snapshot_path: Path | None = None,
) -> dict[str, object]:
    factor_panel, factor_metadata = prepare_factor_panel(panel)
    all_entries: list[pd.DataFrame] = []
    all_symbols: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    for candidate_id, raw_parameters in candidate_parameters.items():
        parameters = validate_parameters(raw_parameters)
        evaluation = evaluate_parameter_set(
            factor_panel,
            parameters,
            periods,
            benchmark=benchmark,
            record_positions=True,
        )
        entries = build_entry_events(
            evaluation, factor_panel, parameters, candidate_id, fold_id
        )
        symbols = build_symbol_contributions(
            evaluation, entries, metadata, candidate_id, fold_id
        )
        all_entries.append(entries)
        all_symbols.append(symbols)
        candidate_rows.append(
            summarize_candidate(evaluation, symbols, candidate_id, fold_id)
        )
        entry_rows.append(summarize_entries(entries, candidate_id, fold_id))
    entries_frame = pd.concat(all_entries, ignore_index=True, sort=False)
    symbols_frame = pd.concat(all_symbols, ignore_index=True, sort=False)
    candidate_summary = pd.DataFrame(candidate_rows)
    entry_summary = pd.DataFrame(entry_rows)
    liquidity_summary = pd.concat(
        [
            summarize_buckets(group, "historical_liquidity_bucket")
            for _, group in symbols_frame.groupby("candidate_id", sort=False)
        ],
        ignore_index=True,
    )
    cap_summary = pd.concat(
        [
            summarize_buckets(group, "current_cap_bucket")
            for _, group in symbols_frame.groupby("candidate_id", sort=False)
        ],
        ignore_index=True,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    entry_summary.to_csv(output_dir / "entry_signal_summary.csv", index=False)
    entries_frame.to_csv(output_dir / "entry_events.csv", index=False)
    symbols_frame.to_csv(output_dir / "symbol_contributions.csv", index=False)
    liquidity_summary.to_csv(
        output_dir / "historical_liquidity_bucket_summary.csv", index=False
    )
    cap_summary.to_csv(output_dir / "current_cap_bucket_summary.csv", index=False)
    report = build_markdown_report(
        candidate_summary,
        entry_summary,
        symbols_frame,
        liquidity_summary,
        cap_summary,
        fold_id,
        snapshot_path,
    )
    (output_dir / "failure_diagnosis.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "fold_id": fold_id,
        "candidate_ids": list(candidate_parameters),
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "snapshot_metadata_is_point_in_time_for_fold": False,
        "factor_metadata": factor_metadata,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {**manifest, "output_dir": str(output_dir)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose stock-pool and entry-state losses for one walk-forward fold."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--config", default="configs/evolution_multifactor_observation.yaml"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--fold", default="wf_08")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    candidates = tuple(args.candidates or DEFAULT_CANDIDATES)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / "diagnostics" / f"{args.fold}_failure_diagnosis"
    )
    snapshot_path = Path(args.snapshot) if args.snapshot else None
    config = load_evolution_config(Path(args.config))
    outcome = run_failure_diagnosis(
        panel=load_market_data([Path(value) for value in args.data]),
        benchmark=load_benchmark(Path(args.benchmark) if args.benchmark else None),
        periods=config.periods,
        candidate_parameters=load_candidate_parameters(run_dir, candidates),
        fold_id=args.fold,
        metadata=load_current_snapshot_metadata(snapshot_path),
        output_dir=output_dir,
        snapshot_path=snapshot_path,
    )
    print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
