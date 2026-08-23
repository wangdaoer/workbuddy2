"""Monitor point-in-time factor decay without changing model parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from analyze_factor_selection_alpha import (
    DEFAULT_HORIZONS,
    attach_forward_open_returns,
    compute_healthy_breadth_state,
)
from multifactor_observation_evolution import prepare_factor_panel
from panel_io import is_parquet_path, read_panel
from pipeline_cache import file_sha256
from run_multifactor_observation_evolution import load_market_data


FACTOR = "liquidity_stability_20"
# 闸门触发状态：因子方向性反转是“必须评估替换并暂停新增资本”的最严重衰减信号。
# 仅此状态触发告警（与 production_soft_score.regime_alert 的 dead 区同构），
# “weakened”为软观察信号、不直接阻断（见 MOS 规则 15）。
FACTOR_DECAY_GATE_STATUS = "direction_reversal"
FACTOR_DIRECTIONS: dict[str, float] = {
    FACTOR: 1.0,
    "momentum_20": 1.0,
    "momentum_60": 1.0,
    "breakout_distance_20": 1.0,
    "trend_acceleration": 1.0,
    "liquidity_20": 1.0,
    "flow_persistence": 1.0,
    "volatility_20": -1.0,
}
REPLACEMENT_TARGET_MATURE_DAYS = 20
REPLACEMENT_SHORTLIST_SIZE = 3
INCREMENTAL_CHECKPOINT_SCHEMA_VERSION = 1
INCREMENTAL_FACTOR_WARMUP_SESSIONS = 60
INCREMENTAL_BREADTH_WARMUP_SESSIONS = 20
CALCULATION_SOURCE_FILES = (
    "monitor_factor_decay.py",
    "analyze_factor_selection_alpha.py",
    "multifactor_observation_evolution.py",
    "legacy_observation_factors.py",
    "execution_rules.py",
)


def factor_monitor_calculation_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in CALCULATION_SOURCE_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    digest.update(json.dumps(list(DEFAULT_HORIZONS)).encode("ascii"))
    digest.update(json.dumps(FACTOR_DIRECTIONS, sort_keys=True).encode("ascii"))
    return digest.hexdigest()


def _checkpoint_candidates(output_dir: Path, asof_date: pd.Timestamp) -> list[Path]:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in output_dir.glob("factor_decay_monitor_*.json"):
        match = re.fullmatch(r"factor_decay_monitor_(\d{8})\.json", path.name)
        if not match:
            continue
        date = pd.Timestamp(datetime.strptime(match.group(1), "%Y%m%d").date())
        if date < asof_date.normalize():
            candidates.append((date, path))
    return [path for _, path in sorted(candidates, reverse=True)]


def load_incremental_checkpoint(
    *,
    output_dir: Path,
    asof_date: pd.Timestamp,
    base_panel: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    if base_panel is None or not base_panel.is_file():
        return None
    base_hash = file_sha256(base_panel)
    calculation_fingerprint = factor_monitor_calculation_fingerprint()
    for metadata_path in _checkpoint_candidates(output_dir, asof_date):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = payload.get("incremental_checkpoint")
        if not isinstance(checkpoint, dict):
            continue
        if checkpoint.get("schema_version") != INCREMENTAL_CHECKPOINT_SCHEMA_VERSION:
            continue
        if checkpoint.get("calculation_fingerprint") != calculation_fingerprint:
            continue
        if checkpoint.get("source_panel_sha256") != base_hash:
            continue
        daily_path_value = checkpoint.get("competition_daily_path")
        if not isinstance(daily_path_value, str):
            continue
        daily_path = Path(daily_path_value)
        if not daily_path.is_file():
            continue
        try:
            daily = pd.read_csv(daily_path, parse_dates=["date"])
        except (OSError, ValueError):
            continue
        required = {
            "date",
            "factor",
            "direction",
            "horizon",
            "observations",
            "rank_ic",
            "top_bottom_spread",
            "breadth_gap",
        }
        if not required.issubset(daily.columns):
            continue
        return daily, {
            "metadata_path": str(metadata_path),
            "source_asof_date": str(checkpoint.get("source_asof_date")),
            "base_panel": str(base_panel.resolve()),
            "base_panel_sha256": base_hash,
        }
    return None


def incremental_recompute_plan(
    date_symbol_frame: pd.DataFrame,
    previous_asof_date: pd.Timestamp,
) -> dict[str, object]:
    if not {"date", "symbol"}.issubset(date_symbol_frame.columns):
        raise ValueError("incremental planning requires date and symbol columns")
    index_frame = date_symbol_frame.loc[:, ["date", "symbol"]].copy()
    index_frame["date"] = pd.to_datetime(index_frame["date"], errors="raise")
    index_frame["symbol"] = index_frame["symbol"].astype(str)
    index_frame = index_frame.sort_values(["symbol", "date"], kind="mergesort")
    previous_rows = index_frame.loc[
        index_frame["date"].le(previous_asof_date.normalize())
    ]
    if previous_rows.empty or not index_frame["date"].gt(previous_asof_date).any():
        raise ValueError("incremental checkpoint date is not before the current panel end")
    affected = previous_rows.groupby("symbol", sort=False).tail(
        max(DEFAULT_HORIZONS) + 1
    )
    recompute_start = pd.Timestamp(affected["date"].min())
    available = pd.DatetimeIndex(sorted(index_frame["date"].unique()))
    affected_index = int(available.get_indexer([recompute_start])[0])
    breadth_index = max(
        affected_index - INCREMENTAL_BREADTH_WARMUP_SESSIONS,
        0,
    )
    raw_index = max(
        breadth_index - INCREMENTAL_FACTOR_WARMUP_SESSIONS,
        0,
    )
    breadth_start = pd.Timestamp(available[breadth_index])
    raw_start = pd.Timestamp(available[raw_index])

    supplemental_symbols: list[str] = []
    for symbol, group in index_frame.groupby("symbol", sort=False):
        values = group["date"].to_numpy()
        first_breadth_row = int(
            values.searchsorted(breadth_start.to_datetime64(), side="left")
        )
        if first_breadth_row >= len(group):
            continue
        required_start = max(
            first_breadth_row - INCREMENTAL_FACTOR_WARMUP_SESSIONS,
            0,
        )
        if pd.Timestamp(group.iloc[required_start]["date"]) < raw_start:
            supplemental_symbols.append(str(symbol))
    return {
        "recompute_start": recompute_start,
        "breadth_start": breadth_start,
        "raw_start": raw_start,
        "supplemental_symbols": supplemental_symbols,
    }


def load_incremental_panel(
    panel_path: Path,
    raw_start: pd.Timestamp,
    supplemental_symbols: Sequence[str] = (),
) -> pd.DataFrame:
    if not is_parquet_path(panel_path):
        return load_market_data([panel_path])
    tail = read_panel(
        panel_path,
        dtype={"symbol": str},
        filters=[("date", ">=", raw_start.date().isoformat())],
        low_memory=False,
    )
    if not supplemental_symbols:
        return tail
    supplemental = read_panel(
        panel_path,
        dtype={"symbol": str},
        filters=[
            ("symbol", "in", list(supplemental_symbols)),
            ("date", "<", raw_start.date().isoformat()),
        ],
        low_memory=False,
    )
    return pd.concat([supplemental, tail], ignore_index=True, sort=False)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(value) if isinstance(value, Integral) else number
    return value


def _rank_ic(group: pd.DataFrame, factor: str, return_column: str) -> float:
    values = pd.DataFrame(
        {
            "factor": pd.to_numeric(group[factor], errors="coerce"),
            "forward": pd.to_numeric(group[return_column], errors="coerce"),
        }
    ).dropna()
    if len(values) < 20 or values["factor"].nunique() < 2 or values["forward"].nunique() < 2:
        return math.nan
    return float(values["factor"].rank().corr(values["forward"].rank()))


def compute_daily_factor_observations(
    factor_panel: pd.DataFrame,
    breadth_state: pd.DataFrame,
    factor: str = FACTOR,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Compute matured daily RankIC only when the hard breadth gate is open."""
    if factor not in factor_panel:
        raise ValueError(f"missing monitored factor: {factor}")
    enriched = attach_forward_open_returns(factor_panel, horizons)
    enriched = enriched.merge(
        breadth_state.reset_index()[["date", "breadth_gap", "breadth_healthy"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    eligible = enriched.get("score_eligible", pd.Series(True, index=enriched.index))
    enriched = enriched.loc[
        eligible.fillna(False) & enriched["breadth_healthy"].fillna(False)
    ]
    rows: list[dict[str, object]] = []
    for date, group in enriched.groupby("date", sort=True):
        factor_rank = pd.to_numeric(group[factor], errors="coerce").rank(pct=True)
        for horizon in horizons:
            return_column = f"forward_open_return_{int(horizon)}d"
            forward = pd.to_numeric(group[return_column], errors="coerce")
            valid = factor_rank.notna() & forward.notna()
            top = valid & factor_rank.ge(0.8)
            bottom = valid & factor_rank.le(0.2)
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor,
                    "horizon": int(horizon),
                    "observations": int(valid.sum()),
                    "rank_ic": _rank_ic(group.loc[valid], factor, return_column),
                    "top_bottom_spread": (
                        float(forward.loc[top].mean() - forward.loc[bottom].mean())
                        if top.any() and bottom.any()
                        else math.nan
                    ),
                    "breadth_gap": float(group["breadth_gap"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def compute_daily_factor_competition(
    factor_panel: pd.DataFrame,
    breadth_state: pd.DataFrame,
    factor_directions: Mapping[str, float] = FACTOR_DIRECTIONS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Compare candidate factors under one PIT return and breadth calculation."""
    requested = {
        str(factor): float(direction)
        for factor, direction in factor_directions.items()
        if factor in factor_panel
    }
    if FACTOR not in requested:
        raise ValueError(f"missing incumbent factor: {FACTOR}")
    if any(direction not in (-1.0, 1.0) for direction in requested.values()):
        raise ValueError("factor directions must be either -1 or 1")

    enriched = attach_forward_open_returns(factor_panel, horizons)
    enriched = enriched.merge(
        breadth_state.reset_index()[["date", "breadth_gap", "breadth_healthy"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    eligible = enriched.get("score_eligible", pd.Series(True, index=enriched.index))
    enriched = enriched.loc[
        eligible.fillna(False) & enriched["breadth_healthy"].fillna(False)
    ]

    dates = pd.DatetimeIndex(sorted(pd.to_datetime(enriched["date"]).unique()))
    date_values = pd.to_datetime(enriched["date"])
    breadth_gap = (
        enriched.assign(date=date_values)
        .groupby("date", sort=True)["breadth_gap"]
        .first()
        .reindex(dates)
    )
    frames: list[pd.DataFrame] = []
    factor_order = {factor: index for index, factor in enumerate(requested)}
    for factor, direction in requested.items():
        signal = pd.to_numeric(enriched[factor], errors="coerce") * direction
        signal_rank = signal.groupby(date_values, sort=False).rank(
            method="average", pct=True
        )
        for horizon in horizons:
            forward = pd.to_numeric(
                enriched[f"forward_open_return_{int(horizon)}d"], errors="coerce"
            )
            valid = signal.notna() & forward.notna()
            observations = valid.groupby(date_values, sort=True).sum().reindex(
                dates, fill_value=0
            ).astype(int)

            valid_signal_rank = signal.where(valid).groupby(
                date_values, sort=False
            ).rank(method="average")
            valid_forward_rank = forward.where(valid).groupby(
                date_values, sort=False
            ).rank(method="average")
            rank_frame = pd.DataFrame(
                {
                    "date": date_values,
                    "signal_rank": valid_signal_rank,
                    "forward_rank": valid_forward_rank,
                }
            ).dropna(subset=["signal_rank", "forward_rank"])
            rank_frame["product"] = (
                rank_frame["signal_rank"] * rank_frame["forward_rank"]
            )
            rank_frame["signal_square"] = rank_frame["signal_rank"].pow(2)
            rank_frame["forward_square"] = rank_frame["forward_rank"].pow(2)
            grouped = rank_frame.groupby("date", sort=True)
            count = grouped.size().astype(float)
            sum_signal = grouped["signal_rank"].sum()
            sum_forward = grouped["forward_rank"].sum()
            sums = grouped[["product", "signal_square", "forward_square"]].sum()
            sum_product = sums["product"]
            sum_signal_sq = sums["signal_square"]
            sum_forward_sq = sums["forward_square"]
            covariance = sum_product - sum_signal.mul(sum_forward).div(count)
            signal_variance = sum_signal_sq - sum_signal.pow(2).div(count)
            forward_variance = sum_forward_sq - sum_forward.pow(2).div(count)
            denominator = signal_variance.mul(forward_variance).pow(0.5)
            rank_ic = covariance.div(denominator).where(
                count.ge(20) & denominator.gt(0.0)
            ).reindex(dates)

            top_mean = forward.where(valid & signal_rank.ge(0.8)).groupby(
                date_values, sort=True
            ).mean()
            bottom_mean = forward.where(valid & signal_rank.le(0.2)).groupby(
                date_values, sort=True
            ).mean()
            spread = top_mean.sub(bottom_mean).reindex(dates)
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "factor": factor,
                        "direction": direction,
                        "horizon": int(horizon),
                        "observations": observations.to_numpy(),
                        "rank_ic": rank_ic.to_numpy(),
                        "top_bottom_spread": spread.to_numpy(),
                        "breadth_gap": breadth_gap.to_numpy(),
                        "_factor_order": factor_order[factor],
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "_factor_order", "horizon"], kind="mergesort")
        .drop(columns="_factor_order")
        .reset_index(drop=True)
    )


def summarize_factor_decay(
    daily: pd.DataFrame,
    *,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    asof_date: pd.Timestamp,
    recent_signal_days: int,
) -> pd.DataFrame:
    """Compare a fixed selection reference with the latest matured observations."""
    if recent_signal_days < 1:
        raise ValueError("recent_signal_days must be positive")
    rows: list[dict[str, object]] = []
    for horizon, group in daily.groupby("horizon", sort=True):
        matured = group.loc[group["rank_ic"].notna() & group["date"].le(asof_date)]
        reference = matured.loc[
            matured["date"].between(selection_start, selection_end)
        ]
        recent = matured.loc[matured["date"].gt(selection_end)].tail(recent_signal_days)
        reference_ic = float(reference["rank_ic"].mean()) if not reference.empty else math.nan
        recent_ic = float(recent["rank_ic"].mean()) if not recent.empty else math.nan
        if len(recent) < min(10, recent_signal_days):
            status = "insufficient_history"
        elif reference_ic > 0.0 and recent_ic < 0.0:
            status = "direction_reversal"
        elif reference_ic > 0.0 and recent_ic < reference_ic * 0.5:
            status = "weakened"
        else:
            status = "stable"
        rows.append(
            {
                "factor": str(group["factor"].iloc[0]),
                "horizon": int(horizon),
                "selection_start": selection_start.date().isoformat(),
                "selection_end": selection_end.date().isoformat(),
                "reference_days": int(len(reference)),
                "reference_mean_rank_ic": reference_ic,
                "reference_positive_ic_ratio": (
                    float(reference["rank_ic"].gt(0.0).mean())
                    if not reference.empty
                    else math.nan
                ),
                "recent_days": int(len(recent)),
                "recent_start": (
                    pd.Timestamp(recent["date"].min()).date().isoformat()
                    if not recent.empty
                    else None
                ),
                "recent_end": (
                    pd.Timestamp(recent["date"].max()).date().isoformat()
                    if not recent.empty
                    else None
                ),
                "recent_mean_rank_ic": recent_ic,
                "recent_positive_ic_ratio": (
                    float(recent["rank_ic"].gt(0.0).mean())
                    if not recent.empty
                    else math.nan
                ),
                "recent_mean_top_bottom_spread": (
                    float(recent["top_bottom_spread"].mean())
                    if not recent.empty
                    else math.nan
                ),
                "rank_ic_delta": recent_ic - reference_ic,
                "status": status,
                "latest_mature_signal_date": (
                    pd.Timestamp(matured["date"].max()).date().isoformat()
                    if not matured.empty
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_overall_status(summary: pd.DataFrame) -> str:
    statuses = summary.get("status", pd.Series(dtype="string")).astype("string")
    if statuses.empty or statuses.eq("insufficient_history").all():
        return "insufficient_history"
    reversals = int(statuses.eq("direction_reversal").sum())
    if reversals >= 2:
        return "direction_reversal"
    if reversals == 1 or statuses.eq("weakened").any():
        return "weakened"
    return "stable"


def summarize_factor_competition(
    daily: pd.DataFrame,
    *,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    asof_date: pd.Timestamp,
    recent_signal_days: int,
    factor_directions: Mapping[str, float] = FACTOR_DIRECTIONS,
    factor_availability: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Build a cross-factor scoreboard without selecting on unseen observations."""
    minimum_recent_days = min(10, recent_signal_days)
    rows: list[dict[str, object]] = []
    for factor, direction in factor_directions.items():
        factor_daily = daily.loc[daily["factor"].eq(factor)]
        if factor_daily.empty:
            rows.append(
                {
                    "factor": factor,
                    "direction": float(direction),
                    "available": False,
                    "valid_horizons": 0,
                    "reference_mean_rank_ic": math.nan,
                    "recent_mean_rank_ic": math.nan,
                    "rank_ic_delta": math.nan,
                    "recent_positive_horizons": 0,
                    "recent_positive_ic_ratio": math.nan,
                    "recent_mean_top_bottom_spread": math.nan,
                    "support_status": "unavailable",
                    "horizon_statuses": "",
                }
            )
            continue
        horizon_summary = summarize_factor_decay(
            factor_daily,
            selection_start=selection_start,
            selection_end=selection_end,
            asof_date=asof_date,
            recent_signal_days=recent_signal_days,
        )
        valid = horizon_summary.loc[
            horizon_summary["recent_days"].ge(minimum_recent_days)
            & horizon_summary["recent_mean_rank_ic"].notna()
        ]
        source_available = (
            bool(factor_availability.get(factor, False))
            if factor_availability is not None
            else True
        )
        reference_ic = pd.to_numeric(
            valid["reference_mean_rank_ic"], errors="coerce"
        ).mean()
        recent_ic = pd.to_numeric(valid["recent_mean_rank_ic"], errors="coerce").mean()
        recent_positive_ratio = pd.to_numeric(
            valid["recent_positive_ic_ratio"], errors="coerce"
        ).mean()
        recent_spread = pd.to_numeric(
            valid["recent_mean_top_bottom_spread"], errors="coerce"
        ).mean()
        positive_horizons = int(
            pd.to_numeric(valid["recent_mean_rank_ic"], errors="coerce").gt(0.0).sum()
        )
        if not source_available:
            support_status = "unavailable"
        elif len(valid) < 2:
            support_status = "insufficient_history"
        elif (
            positive_horizons >= 2
            and pd.notna(recent_ic)
            and float(recent_ic) > 0.0
            and pd.notna(recent_positive_ratio)
            and float(recent_positive_ratio) >= 0.5
        ):
            support_status = "supported"
        else:
            support_status = "not_supported"
        rows.append(
            {
                "factor": factor,
                "direction": float(direction),
                "available": source_available,
                "valid_horizons": int(len(valid)),
                "reference_mean_rank_ic": reference_ic,
                "recent_mean_rank_ic": recent_ic,
                "rank_ic_delta": recent_ic - reference_ic,
                "recent_positive_horizons": positive_horizons,
                "recent_positive_ic_ratio": recent_positive_ratio,
                "recent_mean_top_bottom_spread": recent_spread,
                "support_status": support_status,
                "horizon_statuses": ",".join(
                    f"{int(row.horizon)}d:{row.status}"
                    for row in horizon_summary.itertuples(index=False)
                ),
            }
        )
    result = pd.DataFrame(rows)
    incumbent = result.loc[result["factor"].eq(FACTOR)]
    incumbent_recent_ic = (
        float(incumbent.iloc[0]["recent_mean_rank_ic"])
        if not incumbent.empty and pd.notna(incumbent.iloc[0]["recent_mean_rank_ic"])
        else math.nan
    )
    result["recent_ic_advantage_vs_incumbent"] = (
        pd.to_numeric(result["recent_mean_rank_ic"], errors="coerce")
        - incumbent_recent_ic
    )
    result["candidate_status"] = "not_shortlisted"
    result.loc[result["factor"].eq(FACTOR), "candidate_status"] = "incumbent"
    shortlist = (
        result["factor"].ne(FACTOR)
        & result["support_status"].eq("supported")
        & result["recent_ic_advantage_vs_incumbent"].gt(0.0)
    )
    result.loc[shortlist, "candidate_status"] = "shortlist"
    return result.sort_values(
        ["candidate_status", "recent_mean_rank_ic", "factor"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def update_factor_replacement_preregistration(
    path: Path,
    competition: pd.DataFrame,
    *,
    asof_date: pd.Timestamp,
    incumbent_status: str,
    shortlist_size: int = REPLACEMENT_SHORTLIST_SIZE,
) -> dict[str, object] | None:
    """Freeze a shortlist once; all later evidence must come from unseen dates."""
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected preregistration object: {path}")
        if payload.get("research_only") is not True or payload.get("trade_instruction") is not False:
            raise ValueError(f"invalid research boundary in preregistration: {path}")
        return payload
    if incumbent_status not in {"weakened", "direction_reversal"}:
        return None
    candidates = competition.loc[competition["candidate_status"].eq("shortlist")].copy()
    candidates = candidates.sort_values(
        ["recent_mean_rank_ic", "factor"],
        ascending=[False, True],
        kind="mergesort",
    ).head(shortlist_size)
    if candidates.empty:
        return None
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_from_asof_date": asof_date.date().isoformat(),
        "validation_start_date": (asof_date + timedelta(days=1)).date().isoformat(),
        "selection_status": "exploratory_posthoc_shortlist_preregistered_forward",
        "incumbent_factor": FACTOR,
        "candidate_factors": candidates["factor"].astype(str).tolist(),
        "horizons": [int(value) for value in DEFAULT_HORIZONS],
        "target_mature_signal_days": REPLACEMENT_TARGET_MATURE_DAYS,
        "gates": {
            "min_mean_rank_ic": 0.01,
            "min_positive_ic_ratio": 0.55,
            "min_rank_ic_advantage_vs_incumbent": 0.0,
        },
        "selection_evidence": candidates[
            [
                "factor",
                "recent_mean_rank_ic",
                "recent_positive_ic_ratio",
                "recent_ic_advantage_vs_incumbent",
            ]
        ].to_dict("records"),
        "research_only": True,
        "trade_instruction": False,
        "automatic_model_change": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def evaluate_preregistered_replacements(
    daily: pd.DataFrame,
    preregistration: Mapping[str, object] | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if preregistration is None:
        return pd.DataFrame(), {
            "status": "not_preregistered",
            "research_only": True,
            "trade_instruction": False,
            "automatic_model_change": False,
        }
    start = pd.Timestamp(str(preregistration["validation_start_date"]))
    incumbent = str(preregistration["incumbent_factor"])
    candidates = [str(value) for value in preregistration.get("candidate_factors", [])]
    horizons = [int(value) for value in preregistration.get("horizons", DEFAULT_HORIZONS)]
    target_days = int(
        preregistration.get("target_mature_signal_days", REPLACEMENT_TARGET_MATURE_DAYS)
    )
    gates = preregistration.get("gates")
    gates = gates if isinstance(gates, Mapping) else {}
    tracked = daily.loc[
        daily["date"].ge(start)
        & daily["factor"].isin([incumbent, *candidates])
        & daily["horizon"].isin(horizons)
    ].copy()
    rows: list[dict[str, object]] = []
    factor_summaries: dict[str, dict[str, object]] = {}
    for factor in [incumbent, *candidates]:
        horizon_rows: list[dict[str, object]] = []
        for horizon in horizons:
            group = tracked.loc[
                tracked["factor"].eq(factor)
                & tracked["horizon"].eq(horizon)
                & tracked["rank_ic"].notna()
            ]
            rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
            row = {
                "factor": factor,
                "horizon": horizon,
                "mature_signal_days": int(len(rank_ic)),
                "mean_rank_ic": float(rank_ic.mean()) if not rank_ic.empty else math.nan,
                "positive_ic_ratio": (
                    float(rank_ic.gt(0.0).mean()) if not rank_ic.empty else math.nan
                ),
                "latest_mature_signal_date": (
                    pd.Timestamp(group["date"].max()).date().isoformat()
                    if not group.empty
                    else None
                ),
            }
            horizon_rows.append(row)
            rows.append(row)
        horizon_frame = pd.DataFrame(horizon_rows)
        factor_summaries[factor] = {
            "factor": factor,
            "minimum_mature_signal_days": int(horizon_frame["mature_signal_days"].min()),
            "mean_rank_ic": pd.to_numeric(
                horizon_frame["mean_rank_ic"], errors="coerce"
            ).mean(),
            "positive_ic_ratio": pd.to_numeric(
                horizon_frame["positive_ic_ratio"], errors="coerce"
            ).mean(),
        }
    incumbent_mean = pd.to_numeric(
        pd.Series([factor_summaries[incumbent]["mean_rank_ic"]]), errors="coerce"
    ).iloc[0]
    factor_results: list[dict[str, object]] = []
    for factor in [incumbent, *candidates]:
        result = dict(factor_summaries[factor])
        factor_mean = pd.to_numeric(
            pd.Series([result["mean_rank_ic"]]), errors="coerce"
        ).iloc[0]
        positive_ratio = pd.to_numeric(
            pd.Series([result["positive_ic_ratio"]]), errors="coerce"
        ).iloc[0]
        advantage = factor_mean - incumbent_mean
        result["rank_ic_advantage_vs_incumbent"] = advantage
        if int(result["minimum_mature_signal_days"]) < target_days:
            decision = "collecting"
        elif factor == incumbent:
            decision = "benchmark_ready"
        elif (
            pd.notna(factor_mean)
            and float(factor_mean) >= float(gates.get("min_mean_rank_ic", 0.01))
            and pd.notna(positive_ratio)
            and float(positive_ratio) >= float(gates.get("min_positive_ic_ratio", 0.55))
            and pd.notna(advantage)
            and float(advantage)
            >= float(gates.get("min_rank_ic_advantage_vs_incumbent", 0.0))
        ):
            decision = "ready_for_research_review"
        else:
            decision = "rejected"
        result["decision"] = decision
        factor_results.append(result)
    candidate_decisions = [
        str(row["decision"]) for row in factor_results if row["factor"] != incumbent
    ]
    if not candidate_decisions:
        status = "no_candidates"
    elif "collecting" in candidate_decisions:
        status = "collecting"
    elif "ready_for_research_review" in candidate_decisions:
        status = "ready_for_research_review"
    else:
        status = "rejected"
    payload = {
        "status": status,
        "validation_start_date": start.date().isoformat(),
        "target_mature_signal_days": target_days,
        "factors": factor_results,
        "research_only": True,
        "trade_instruction": False,
        "automatic_model_change": False,
    }
    return pd.DataFrame(rows), payload


def build_current_observation(
    factor_panel: pd.DataFrame,
    asof_date: pd.Timestamp,
    top_n: int,
    factor: str = FACTOR,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rows = factor_panel.loc[factor_panel["date"].eq(asof_date)].copy()
    eligible = rows.get("score_eligible", pd.Series(True, index=rows.index))
    rows = rows.loc[eligible.fillna(False)]
    rows["factor_pct_rank"] = pd.to_numeric(rows[factor], errors="coerce").rank(
        method="average", pct=True
    )
    rows["median_amount_20"] = pd.to_numeric(
        rows.get("liquidity_20"), errors="coerce"
    ).map(lambda value: math.expm1(value) if pd.notna(value) else math.nan)
    columns = [
        name
        for name in (
            "date",
            "symbol",
            "stock_name",
            "close",
            factor,
            "factor_pct_rank",
            "median_amount_20",
            "momentum_20",
            "momentum_60",
            "breakout_distance_20",
        )
        if name in rows
    ]
    out = rows.sort_values(
        ["factor_pct_rank", "symbol"], ascending=[False, True], kind="mergesort"
    ).head(top_n)[columns]
    out.insert(len(out.columns), "observation_only", True)
    out.insert(len(out.columns), "trade_instruction", False)
    return out.reset_index(drop=True)


def update_monitor_history(
    history_path: Path,
    asof_date: pd.Timestamp,
    summary: pd.DataFrame,
    overall_status: str,
) -> pd.DataFrame:
    current = summary.copy()
    current.insert(0, "asof_date", asof_date.date().isoformat())
    current.insert(1, "overall_status", overall_status)
    if history_path.exists():
        history = pd.read_csv(history_path, dtype={"asof_date": "string"})
        current = pd.concat([history, current], ignore_index=True, sort=False)
    current = current.drop_duplicates(
        subset=["asof_date", "factor", "horizon"], keep="last"
    ).sort_values(["asof_date", "factor", "horizon"], kind="mergesort")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    current.to_csv(temporary, index=False)
    temporary.replace(history_path)
    return current


def _pct(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(number) else f"{float(number):.4f}"


def build_report(
    summary: pd.DataFrame,
    overall_status: str,
    asof_date: pd.Timestamp,
    breadth_row: pd.Series,
    observation: pd.DataFrame,
) -> str:
    lines = [
        "# 选股因子衰减观察",
        "",
        "> 研究/模拟盘用途，不构成交易指令。本报告不会自动修改模型参数。",
        "",
        f"- 数据截至：{asof_date.date().isoformat()}",
        f"- 监测因子：`{FACTOR}`",
        f"- 总体状态：`{overall_status}`",
        f"- 当日动量广度：{_pct(breadth_row.get('breadth_20'))}",
        f"- 广度相对 20 日均线：{_pct(breadth_row.get('breadth_gap'))}",
        f"- 硬广度门控：{'开放' if bool(breadth_row.get('breadth_healthy')) else '关闭'}",
        "",
        "## 前瞻 RankIC",
        "",
        "| 周期 | 固定参考期 | 最近成熟样本 | 变化 | 最近正IC比例 | 状态 | 最新成熟信号日 |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {int(row['horizon'])}日 | {_number(row['reference_mean_rank_ic'])} | "
            f"{_number(row['recent_mean_rank_ic'])} | {_number(row['rank_ic_delta'])} | "
            f"{_pct(row['recent_positive_ic_ratio'])} | {row['status']} | "
            f"{row['latest_mature_signal_date'] or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## 当日高稳定性观察",
            "",
            "以下仅按因子排序，不是买入名单。",
            "",
            "| 代码 | 名称 | 因子分位 | 20日动量 | 60日动量 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in observation.head(10).to_dict("records"):
        lines.append(
            f"| {row.get('symbol', '')} | {row.get('stock_name', '') or ''} | "
            f"{_pct(row.get('factor_pct_rank'))} | {_pct(row.get('momentum_20'))} | "
            f"{_pct(row.get('momentum_60'))} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 状态只用于观察因子是否失效，不触发调仓、训练、候选晋级或模拟盘写入。",
            "- 前瞻收益按信号日后的下一开盘进入，达到对应持有期后的开盘退出。",
            "- 最近样本只使用已经成熟的信号日，尾部未实现收益不会被填充。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_factor_competition_report(
    competition: pd.DataFrame,
    tracking: Mapping[str, object],
    preregistration: Mapping[str, object] | None,
    asof_date: pd.Timestamp,
) -> str:
    lines = [
        "# 衰减因子替代候选竞赛",
        "",
        "> 研究/模拟盘用途，不构成交易指令。历史竞赛只产生候选，只有冻结后的未来样本可用于研究复核。",
        "",
        f"- 数据截至：{asof_date.date().isoformat()}",
        f"- 当前因子：`{FACTOR}`",
        "- 负向因子已统一转换为‘数值越高越有利’后计算 RankIC。",
        "- 自动改权重：关闭",
        "",
        "## 历史竞赛",
        "",
        "| 因子 | 方向 | 参考RankIC | 最近RankIC | 相对当前 | 正向周期 | 正IC比例 | 支持状态 | 候选状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    ordered = competition.sort_values(
        ["candidate_status", "recent_mean_rank_ic", "factor"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    for row in ordered.to_dict("records"):
        lines.append(
            f"| {row['factor']} | {int(row['direction']):+d} | "
            f"{_number(row['reference_mean_rank_ic'])} | {_number(row['recent_mean_rank_ic'])} | "
            f"{_number(row['recent_ic_advantage_vs_incumbent'])} | "
            f"{int(row['recent_positive_horizons'])}/{int(row['valid_horizons'])} | "
            f"{_pct(row['recent_positive_ic_ratio'])} | {row['support_status']} | "
            f"{row['candidate_status']} |"
        )
    lines.extend(["", "## 前瞻冻结", ""])
    if preregistration is None:
        lines.append("- 当前没有同时满足门槛的替代候选，未创建前瞻验证名单。")
    else:
        candidates = "、".join(
            f"`{value}`" for value in preregistration.get("candidate_factors", [])
        )
        lines.extend(
            [
                f"- 候选：{candidates or '无'}",
                f"- 未见样本起点：{preregistration.get('validation_start_date')}",
                f"- 每个周期目标成熟信号日：{preregistration.get('target_mature_signal_days')}",
                f"- 当前跟踪状态：`{tracking.get('status')}`",
            ]
        )
        for row in tracking.get("factors", []):
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"- `{row.get('factor')}`：最少成熟样本 "
                f"{row.get('minimum_mature_signal_days', 0)}，平均 RankIC "
                f"{_number(row.get('mean_rank_ic'))}，结论 `{row.get('decision')}`。"
            )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 候选名单一旦写入预注册文件便不会随每日排行榜变化。",
            "- `ready_for_research_review` 只允许人工复核和独立回测，不能直接进入正式排名。",
            "- 所有收益标签继续采用下一开盘进入、对应持有期后开盘退出。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_factor_decay_monitor(
    panel: pd.DataFrame,
    *,
    asof_date: pd.Timestamp,
    output_dir: Path,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    recent_signal_days: int = 20,
    top_n: int = 40,
    replacement_preregistration: Path | None = None,
    previous_competition_daily: pd.DataFrame | None = None,
    recompute_start: pd.Timestamp | None = None,
    breadth_start: pd.Timestamp | None = None,
    checkpoint_context: Mapping[str, object] | None = None,
    source_panel: Path | None = None,
) -> dict[str, object]:
    panel = panel.loc[pd.to_datetime(panel["date"]).le(asof_date)].copy()
    factor_panel, metadata = prepare_factor_panel(panel)
    calculation_panel = (
        factor_panel.loc[factor_panel["date"].ge(breadth_start)].copy()
        if breadth_start is not None
        else factor_panel
    )
    breadth = compute_healthy_breadth_state(calculation_panel, ma_window=20)
    recalculated_daily = compute_daily_factor_competition(calculation_panel, breadth)
    if previous_competition_daily is not None:
        if recompute_start is None:
            raise ValueError("recompute_start is required with an incremental checkpoint")
        previous = previous_competition_daily.copy()
        previous["date"] = pd.to_datetime(previous["date"], errors="raise")
        competition_daily = pd.concat(
            [
                previous.loc[previous["date"].lt(recompute_start)],
                recalculated_daily.loc[recalculated_daily["date"].ge(recompute_start)],
            ],
            ignore_index=True,
            sort=False,
        ).sort_values(["date", "factor", "horizon"], kind="mergesort")
        competition_daily = competition_daily.reset_index(drop=True)
        calculation_mode = "incremental"
    else:
        competition_daily = recalculated_daily
        calculation_mode = "full"
    daily = competition_daily.loc[competition_daily["factor"].eq(FACTOR)].copy()
    summary = summarize_factor_decay(
        daily,
        selection_start=selection_start,
        selection_end=selection_end,
        asof_date=asof_date,
        recent_signal_days=recent_signal_days,
    )
    overall_status = classify_overall_status(summary)
    competition = summarize_factor_competition(
        competition_daily,
        selection_start=selection_start,
        selection_end=selection_end,
        asof_date=asof_date,
        recent_signal_days=recent_signal_days,
        factor_availability=metadata.get("factor_availability"),
    )
    observation = build_current_observation(calculation_panel, asof_date, top_n)
    breadth_row = breadth.loc[asof_date]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = asof_date.strftime("%Y%m%d")
    csv_path = output_dir / f"factor_decay_monitor_{token}.csv"
    json_path = output_dir / f"factor_decay_monitor_{token}.json"
    report_path = output_dir / f"factor_decay_monitor_{token}.md"
    observation_path = output_dir / f"liquidity_stability_observation_{token}.csv"
    history_path = output_dir / "factor_decay_monitor_history.csv"
    competition_daily_path = output_dir / f"factor_replacement_daily_ic_{token}.csv"
    competition_path = output_dir / f"factor_replacement_competition_{token}.csv"
    competition_report_path = output_dir / f"factor_replacement_competition_{token}.md"
    preregistration_path = (
        Path(replacement_preregistration)
        if replacement_preregistration is not None
        else output_dir / "factor_replacement_preregistration.json"
    )
    tracking_path = output_dir / f"factor_replacement_tracking_{token}.csv"
    tracking_json_path = output_dir / f"factor_replacement_tracking_{token}.json"
    summary.to_csv(csv_path, index=False)
    observation.to_csv(observation_path, index=False)
    competition_daily.to_csv(competition_daily_path, index=False)
    competition.to_csv(competition_path, index=False)
    history = update_monitor_history(history_path, asof_date, summary, overall_status)
    preregistration = update_factor_replacement_preregistration(
        preregistration_path,
        competition,
        asof_date=asof_date,
        incumbent_status=overall_status,
    )
    tracking_rows, tracking = evaluate_preregistered_replacements(
        competition_daily, preregistration
    )
    tracking_rows.to_csv(tracking_path, index=False)
    tracking_json_path.write_text(
        json.dumps(
            _json_safe(tracking),
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        build_report(summary, overall_status, asof_date, breadth_row, observation),
        encoding="utf-8",
    )
    competition_report_path.write_text(
        build_factor_competition_report(
            competition, tracking, preregistration, asof_date
        ),
        encoding="utf-8",
    )
    shortlist = competition.loc[
        competition["candidate_status"].eq("shortlist"), "factor"
    ].astype(str).tolist()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asof_date": asof_date.date().isoformat(),
        "factor": FACTOR,
        "overall_status": overall_status,
        "research_only": True,
        "trade_instruction": False,
        "automatic_model_change": False,
        "calculation_mode": calculation_mode,
        "incremental_checkpoint": {
            "schema_version": INCREMENTAL_CHECKPOINT_SCHEMA_VERSION,
            "calculation_fingerprint": factor_monitor_calculation_fingerprint(),
            "source_asof_date": asof_date.date().isoformat(),
            "source_panel": str(source_panel.resolve()) if source_panel else None,
            "source_panel_sha256": (
                file_sha256(source_panel)
                if source_panel is not None and source_panel.is_file()
                else None
            ),
            "competition_daily_path": str(competition_daily_path.resolve()),
            "recompute_start": (
                recompute_start.date().isoformat()
                if recompute_start is not None
                else None
            ),
            "breadth_start": (
                breadth_start.date().isoformat()
                if breadth_start is not None
                else None
            ),
            "parent": dict(checkpoint_context or {}),
        },
        "selection_reference": {
            "start": selection_start.date().isoformat(),
            "end": selection_end.date().isoformat(),
        },
        "breadth": {
            "value": breadth_row.get("breadth_20"),
            "moving_average": breadth_row.get("breadth_20_ma"),
            "gap": breadth_row.get("breadth_gap"),
            "healthy": bool(breadth_row.get("breadth_healthy")),
        },
        "horizons": summary.to_dict("records"),
        "observation_count": int(len(observation)),
        "history_rows": int(len(history)),
        "factor_metadata": metadata,
        "replacement_competition": {
            "status": "complete",
            "selection_status": "exploratory_posthoc",
            "shortlist": shortlist,
            "preregistered_candidates": (
                list(preregistration.get("candidate_factors", []))
                if preregistration is not None
                else []
            ),
            "validation_start_date": (
                preregistration.get("validation_start_date")
                if preregistration is not None
                else None
            ),
            "tracking_status": tracking.get("status"),
            "research_only": True,
            "trade_instruction": False,
            "automatic_model_change": False,
        },
        "artifacts": {
            "summary_csv": str(csv_path),
            "report": str(report_path),
            "observation": str(observation_path),
            "history": str(history_path),
            "replacement_daily_ic": str(competition_daily_path),
            "replacement_competition": str(competition_path),
            "replacement_report": str(competition_report_path),
            "replacement_preregistration": (
                str(preregistration_path) if preregistration is not None else None
            ),
            "replacement_tracking": str(tracking_path),
            "replacement_tracking_json": str(tracking_json_path),
        },
    }
    json_path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return {**payload, "json": str(json_path)}


def factor_decay_alert(payload: dict[str, object]) -> dict[str, object]:
    """把因子衰减监控从“纯报告”升级为可被执行层/运行卡消费的“闸门”信号。

    与 production_soft_score.regime_alert 同构：返回含 alert 布尔、status、factor、
    recommended_action、severity 的告警 dict。整体仍为 research_only（不改模型参数），
    但 alert=True 时要求“评估因子替换并暂停新增资本部署”，由流水线 monitoring_alerts
    与可选的 --block-on-alert 阻断消费（见 MOS 规则 15）。
    """
    overall = str(payload.get("overall_status", "unknown"))
    alert = overall == FACTOR_DECAY_GATE_STATUS
    return {
        "model": "factor_decay_monitor",
        "asof": payload.get("asof_date"),
        "status": overall,
        "alert": alert,
        "factor": payload.get("factor"),
        "severity": "high" if alert else "low",
        "recommended_action": (
            "evaluate_factor_replacement_and_hold_new_capital"
            if alert else "maintain"
        ),
        "research_only": payload.get("research_only", True),
        "trade_instruction": payload.get("trade_instruction", False),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor liquidity-stability factor decay without model changes."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument(
        "--base-panel",
        default=None,
        help=(
            "Exact prior panel used to validate an incremental checkpoint. "
            "A hash mismatch falls back to full calculation."
        ),
    )
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--selection-start", default="2023-01-01")
    parser.add_argument("--selection-end", default="2025-12-31")
    parser.add_argument("--recent-signal-days", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument(
        "--replacement-preregistration",
        default="configs/factor_replacement_preregistration.json",
    )
    parser.add_argument(
        "--block-on-alert",
        action="store_true",
        help="当因子衰减闸门触发（direction_reversal）时阻断退出（code=2），供流水线闸门消费。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    asof_date = pd.Timestamp(args.asof_date)
    output_dir = Path(args.output_dir)
    data_paths = [Path(value) for value in args.data]
    source_panel = data_paths[0] if len(data_paths) == 1 and data_paths[0].is_file() else None
    checkpoint = load_incremental_checkpoint(
        output_dir=output_dir,
        asof_date=asof_date,
        base_panel=Path(args.base_panel) if args.base_panel else None,
    )
    previous_daily: pd.DataFrame | None = None
    recompute_start: pd.Timestamp | None = None
    breadth_start: pd.Timestamp | None = None
    checkpoint_context: dict[str, object] | None = None
    if checkpoint is not None and source_panel is not None:
        try:
            previous_daily, checkpoint_context = checkpoint
            previous_asof = pd.Timestamp(str(checkpoint_context["source_asof_date"]))
            date_frame = read_panel(
                source_panel,
                columns=["date", "symbol"],
                dtype={"symbol": str},
                low_memory=False,
            )
            plan = incremental_recompute_plan(date_frame, previous_asof)
            recompute_start = pd.Timestamp(plan["recompute_start"])
            breadth_start = pd.Timestamp(plan["breadth_start"])
            panel = load_incremental_panel(
                source_panel,
                pd.Timestamp(plan["raw_start"]),
                list(plan["supplemental_symbols"]),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            print(f"Incremental checkpoint rejected; running full calculation: {exc}")
            previous_daily = None
            recompute_start = None
            breadth_start = None
            checkpoint_context = None
            panel = load_market_data(data_paths)
    else:
        panel = load_market_data(data_paths)
    result = run_factor_decay_monitor(
        panel,
        asof_date=asof_date,
        output_dir=output_dir,
        selection_start=pd.Timestamp(args.selection_start),
        selection_end=pd.Timestamp(args.selection_end),
        recent_signal_days=args.recent_signal_days,
        top_n=args.top_n,
        replacement_preregistration=Path(args.replacement_preregistration),
        previous_competition_daily=previous_daily,
        recompute_start=recompute_start,
        breadth_start=breadth_start,
        checkpoint_context=checkpoint_context,
        source_panel=source_panel,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    alert = factor_decay_alert(result)
    token = asof_date.strftime("%Y%m%d")
    alert_path = output_dir / f"factor_decay_alert_{token}.json"
    alert_path.write_text(
        json.dumps(alert, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "factor_decay_alert.json").write_text(
        json.dumps(alert, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[monitor] factor={result.get('factor')} status={result.get('overall_status')} "
        f"alert={alert['alert']} action={alert['recommended_action']} -> {alert_path.name}"
    )
    if alert["alert"] and args.block_on_alert:
        print("[monitor] ALERT=True 且 --block-on-alert：阻断退出（code=2）。")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
