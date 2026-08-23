"""Run guarded parameter evolution for the observation-factor strategy."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from panel_io import read_panel

from multifactor_observation_evolution import (
    EvolutionConfig,
    ParameterEvaluation,
    benchmark_total_return,
    combine_panel_frames,
    evaluate_holdout,
    evaluate_parameter_set,
    load_evolution_config,
    prepare_factor_panel,
    validate_parameters,
)
from strategy_evolution_core import (
    EvolutionState,
    StateWriteExpectation,
    evaluate_candidate,
    fingerprint_payload,
    generate_parameter_candidates,
    load_evolution_state,
    promote_to_shadow,
    write_evolution_state_atomic,
)


def _json_safe(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _data_fingerprint(panel: pd.DataFrame) -> str:
    columns = [
        name
        for name in ("date", "symbol", "open", "close", "amount")
        if name in panel
    ]
    stable = panel.loc[:, columns].sort_values(["date", "symbol"], kind="mergesort")
    values = pd.util.hash_pandas_object(stable, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256(values.tobytes())
    digest.update("|".join(columns).encode("utf-8"))
    return digest.hexdigest()


def _group_seed(seed: int, group_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _mean_metric(evaluation: ParameterEvaluation, name: str) -> float:
    values = [float(getattr(fold, name)) for fold in evaluation.folds]
    return float(np.mean(values)) if values else float("-inf")


def _positive_excess_ratio(evaluation: ParameterEvaluation) -> float | None:
    values = [row["excess_return"] for row in evaluation.fold_rows]
    if not values or any(value is None for value in values):
        return None
    return float(np.mean([float(value) > 0.0 for value in values]))


def _mean_row_metric(evaluation: ParameterEvaluation, name: str) -> float:
    values = [float(row[name]) for row in evaluation.fold_rows]
    return float(np.mean(values)) if values else float("nan")


def _candidate_parameters(
    incumbent: Mapping[str, object],
    candidates: Sequence[object],
    max_candidates: int,
    seed: int,
) -> tuple[tuple[str, dict[str, object]], ...]:
    lookup: dict[str, str] = {}
    overrides: list[Mapping[str, object]] = []
    for candidate in candidates:
        parameters = dict(incumbent)
        parameters.update(candidate.overrides)
        parameters = validate_parameters(parameters)
        fingerprint = fingerprint_payload(parameters)
        if fingerprint in lookup:
            raise ValueError("candidate overrides produce duplicate parameter sets")
        lookup[fingerprint] = candidate.candidate_id
        overrides.append(candidate.overrides)
    generated = generate_parameter_candidates(
        incumbent, overrides, max_candidates=max_candidates, seed=seed
    )
    result: list[tuple[str, dict[str, object]]] = []
    for parameters in generated:
        validated = validate_parameters(parameters)
        result.append((lookup[fingerprint_payload(validated)], validated))
    return tuple(result)


def _holdout_gates(
    config: EvolutionConfig,
    champion_row: Mapping[str, object],
    challenger_row: Mapping[str, object],
    champion_stress_row: Mapping[str, object],
    challenger_stress_row: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    policy = config.holdout
    benchmark_available = challenger_row.get("excess_return") is not None
    return_delta = float(challenger_row["total_return"]) - float(
        champion_row["total_return"]
    )
    stress_delta = float(challenger_stress_row["total_return"]) - float(
        champion_stress_row["total_return"]
    )
    return {
        "min_sessions": {
            "passed": int(challenger_row["sessions"]) >= policy.min_sessions,
            "value": int(challenger_row["sessions"]),
            "threshold": policy.min_sessions,
        },
        "benchmark_available": {
            "passed": benchmark_available,
            "value": benchmark_available,
            "threshold": True,
        },
        "min_return_delta": {
            "passed": return_delta >= policy.min_return_delta,
            "value": return_delta,
            "threshold": policy.min_return_delta,
        },
        "min_excess_return": {
            "passed": benchmark_available
            and float(challenger_row["excess_return"]) >= policy.min_excess_return,
            "value": challenger_row.get("excess_return"),
            "threshold": policy.min_excess_return,
        },
        "max_drawdown": {
            "passed": float(challenger_row["max_drawdown"])
            >= policy.max_drawdown_floor,
            "value": float(challenger_row["max_drawdown"]),
            "threshold": policy.max_drawdown_floor,
        },
        "double_cost_return": {
            "passed": float(challenger_stress_row["total_return"])
            >= policy.min_double_cost_return,
            "value": float(challenger_stress_row["total_return"]),
            "threshold": policy.min_double_cost_return,
        },
        "double_cost_return_delta": {
            "passed": stress_delta >= policy.min_double_cost_return_delta,
            "value": stress_delta,
            "threshold": policy.min_double_cost_return_delta,
        },
    }


def _write_summary(
    run_dir: Path,
    manifest: Mapping[str, object],
    holdout: Mapping[str, object] | None,
) -> None:
    lines = [
        "# Multifactor Observation Evolution",
        "",
        "Research/simulation only. This output is not a trade instruction or return promise.",
        "",
        f"- Run: {manifest['run_id']}",
        f"- Status: {manifest['status']}",
        f"- Data as of: {manifest['data_asof_date']}",
        f"- Benchmark as of: {manifest.get('benchmark_asof_date') or 'missing'}",
        f"- Holdout benchmark covered: {manifest.get('holdout_benchmark_covered')}",
        f"- Starting champion: {manifest['starting_champion_id']}",
        f"- Locked candidate: {manifest.get('locked_candidate_id') or 'none'}",
        f"- Shadow state write: {manifest['shadow_write_status']}",
        "",
        "Hard constraints: lag-1 signal, next-open execution, long-only exposure, "
        "historical universe, limit-up/down blocking, liquidity capacity and explicit costs.",
    ]
    if holdout is not None:
        challenger = holdout["challenger"]
        lines.extend(
            [
                "",
                "## Holdout",
                "",
                f"- Total return: {float(challenger['total_return']):.2%}",
                f"- Excess return: {challenger.get('excess_return')}",
                f"- Max drawdown: {float(challenger['max_drawdown']):.2%}",
                f"- Double-cost return: {float(holdout['challenger_double_cost']['total_return']):.2%}",
            ]
        )
    (run_dir / "evolution_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_evolution(
    *,
    panel: pd.DataFrame,
    benchmark: pd.Series | None,
    config: EvolutionConfig,
    output_root: Path,
    state_path: Path,
    run_id: str,
    dry_run: bool = True,
    promote_shadow: bool = False,
) -> dict[str, object]:
    output_root = Path(output_root)
    state_path = Path(state_path)
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    factors, factor_metadata = prepare_factor_panel(panel)
    factor_metadata = dict(factor_metadata)
    quarantined_zero_rows = int(
        panel.attrs.get("quarantined_all_zero_market_rows", 0)
    )
    suspended_quote_rows = int(panel.attrs.get("suspended_no_open_quote_rows", 0))
    factor_metadata["input_quarantined_all_zero_market_rows"] = quarantined_zero_rows
    factor_metadata["input_suspended_no_open_quote_rows"] = suspended_quote_rows
    data_fingerprint = _data_fingerprint(factors)
    data_asof = pd.Timestamp(factors["date"].max()).date().isoformat()
    state_exists = state_path.exists()
    default_state = EvolutionState.initial("baseline", config.baseline)
    state = load_evolution_state(state_path, default_state)
    champion_parameters = validate_parameters(state.champion_parameters)
    run_dir.mkdir(parents=True)

    evaluation_cache: dict[str, ParameterEvaluation] = {}

    def selection_evaluation(parameters: Mapping[str, object]) -> ParameterEvaluation:
        fingerprint = fingerprint_payload(parameters)
        if fingerprint not in evaluation_cache:
            evaluation_cache[fingerprint] = evaluate_parameter_set(
                factors,
                parameters,
                config.periods,
                benchmark=benchmark,
            )
        return evaluation_cache[fingerprint]

    incumbent_id = state.champion_version
    incumbent_parameters = dict(champion_parameters)
    starting_fingerprint = fingerprint_payload(incumbent_parameters)
    score_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    starting_evaluation = selection_evaluation(incumbent_parameters)
    score_rows.append(
        {
            "group_id": "__baseline__",
            "incumbent_id": incumbent_id,
            "candidate_id": incumbent_id,
            "status": "incumbent",
            "generic_status": "incumbent",
            "failed_gates": "[]",
            "positive_excess_fold_ratio": _positive_excess_ratio(
                starting_evaluation
            ),
            "mean_total_return": _mean_metric(
                starting_evaluation, "total_return"
            ),
            "mean_max_drawdown": _mean_metric(
                starting_evaluation, "max_drawdown"
            ),
            "mean_sharpe": _mean_metric(starting_evaluation, "sharpe"),
            "mean_turnover": _mean_metric(
                starting_evaluation, "average_turnover"
            ),
            "mean_market_exposure": _mean_row_metric(
                starting_evaluation, "average_market_exposure"
            ),
            "portfolio_stop_triggers": int(
                starting_evaluation.backtest.execution_counts.get(
                    "portfolio_stop_triggers", 0
                )
            ),
            "parameters": json.dumps(incumbent_parameters, sort_keys=True),
            "gates": "{}",
        }
    )
    for row in starting_evaluation.fold_rows:
        fold_rows.append(
            {
                "group_id": "__baseline__",
                "candidate_id": incumbent_id,
                **row,
            }
        )

    for group in config.search_groups:
        incumbent_evaluation = selection_evaluation(incumbent_parameters)
        eligible: list[tuple[str, dict[str, object], ParameterEvaluation]] = []
        generated = _candidate_parameters(
            incumbent_parameters,
            group.candidates,
            config.max_candidates_per_group,
            _group_seed(config.random_seed, group.group_id),
        )
        for candidate_id, candidate_parameters in generated:
            evaluation = selection_evaluation(candidate_parameters)
            decision = evaluate_candidate(
                evaluation.folds, incumbent_evaluation.folds, config.promotion
            )
            positive_excess_ratio = _positive_excess_ratio(evaluation)
            excess_gate = (
                positive_excess_ratio is not None
                and positive_excess_ratio
                >= config.holdout.min_positive_excess_fold_ratio
            )
            failed_gates = list(decision.failed_gates)
            if not excess_gate:
                failed_gates.append("positive_excess_fold_ratio")
            accepted = decision.status == "eligible_for_shadow" and excess_gate
            score_rows.append(
                {
                    "group_id": group.group_id,
                    "incumbent_id": incumbent_id,
                    "candidate_id": candidate_id,
                    "status": "eligible" if accepted else "rejected",
                    "generic_status": decision.status,
                    "failed_gates": json.dumps(failed_gates, ensure_ascii=True),
                    "positive_excess_fold_ratio": positive_excess_ratio,
                    "mean_total_return": _mean_metric(evaluation, "total_return"),
                    "mean_max_drawdown": _mean_metric(evaluation, "max_drawdown"),
                    "mean_sharpe": _mean_metric(evaluation, "sharpe"),
                    "mean_turnover": _mean_metric(evaluation, "average_turnover"),
                    "mean_market_exposure": _mean_row_metric(
                        evaluation, "average_market_exposure"
                    ),
                    "portfolio_stop_triggers": int(
                        evaluation.backtest.execution_counts.get(
                            "portfolio_stop_triggers", 0
                        )
                    ),
                    "parameters": json.dumps(candidate_parameters, sort_keys=True),
                    "gates": json.dumps(_json_safe(decision.gates), sort_keys=True),
                }
            )
            for row in evaluation.fold_rows:
                fold_rows.append(
                    {"group_id": group.group_id, "candidate_id": candidate_id, **row}
                )
            if accepted:
                eligible.append((candidate_id, candidate_parameters, evaluation))
        if eligible:
            winner = max(
                eligible,
                key=lambda item: (
                    _mean_metric(item[2], "total_return"),
                    _mean_metric(item[2], "sharpe"),
                    item[0],
                ),
            )
            incumbent_id, incumbent_parameters, _ = winner

    locked = fingerprint_payload(incumbent_parameters) != starting_fingerprint
    holdout_payload: dict[str, object] | None = None
    gates: dict[str, dict[str, object]] = {}
    status = "no_eligible_challenger"
    eligible_for_shadow = False
    available_dates = pd.DatetimeIndex(sorted(factors["date"].unique()))
    configured_holdout_end = config.periods.holdout_end or available_dates[-1]
    holdout_dates = available_dates[
        (available_dates >= config.periods.holdout_start)
        & (available_dates <= configured_holdout_end)
    ]
    actual_holdout_end = holdout_dates[-1] if len(holdout_dates) else None
    holdout_benchmark_return = (
        benchmark_total_return(
            benchmark,
            config.periods.holdout_start,
            actual_holdout_end,
        )
        if actual_holdout_end is not None
        else None
    )
    holdout_benchmark_covered = holdout_benchmark_return is not None
    if locked:
        if not holdout_benchmark_covered:
            gates = {
                "benchmark_coverage": {
                    "passed": False,
                    "value": (
                        None
                        if benchmark is None or benchmark.dropna().empty
                        else pd.Timestamp(benchmark.dropna().index.max())
                        .date()
                        .isoformat()
                    ),
                    "threshold": (
                        actual_holdout_end.date().isoformat()
                        if actual_holdout_end is not None
                        else "available holdout period"
                    ),
                }
            }
            status = "holdout_blocked_benchmark_coverage"
        else:
            _, champion_holdout, _ = evaluate_holdout(
                factors, champion_parameters, config.periods, benchmark=benchmark
            )
            _, challenger_holdout, _ = evaluate_holdout(
                factors, incumbent_parameters, config.periods, benchmark=benchmark
            )
            _, champion_stress, _ = evaluate_holdout(
                factors,
                champion_parameters,
                config.periods,
                benchmark=benchmark,
                cost_multiplier=2.0,
            )
            _, challenger_stress, challenger_backtest = evaluate_holdout(
                factors,
                incumbent_parameters,
                config.periods,
                benchmark=benchmark,
                cost_multiplier=2.0,
            )
            gates = _holdout_gates(
                config,
                champion_holdout,
                challenger_holdout,
                champion_stress,
                challenger_stress,
            )
            eligible_for_shadow = all(
                bool(gate["passed"]) for gate in gates.values()
            )
            status = (
                "eligible_for_shadow" if eligible_for_shadow else "holdout_rejected"
            )
            holdout_payload = {
                "champion": champion_holdout,
                "challenger": challenger_holdout,
                "champion_double_cost": champion_stress,
                "challenger_double_cost": challenger_stress,
                "gates": gates,
                "execution_counts_double_cost": challenger_backtest.execution_counts,
            }

    shadow_write_status = "not_authorized"
    if eligible_for_shadow and state.shadow_status == "shadow":
        shadow_write_status = "blocked_existing_shadow"
        status = "eligible_but_existing_shadow_requires_review"
    elif eligible_for_shadow and not dry_run and promote_shadow:
        new_state = promote_to_shadow(
            state,
            challenger_version=incumbent_id,
            challenger_parameters=incumbent_parameters,
            experiment_id=fingerprint_payload(incumbent_parameters),
            run_id=run_id,
            data_fingerprint=data_fingerprint,
            data_asof_date=data_asof,
        )
        expected = (
            fingerprint_payload(state)
            if state_exists
            else StateWriteExpectation.ABSENT
        )
        write_evolution_state_atomic(
            state_path, new_state, expected_previous_fingerprint=expected
        )
        shadow_write_status = "committed"
    elif eligible_for_shadow:
        shadow_write_status = "dry_run"

    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "strategy": config.strategy,
        "target_mode": config.target_mode,
        "annual_return_stretch": config.annual_return_stretch,
        "benchmark": config.benchmark,
        "benchmark_provided": benchmark is not None,
        "benchmark_asof_date": (
            None
            if benchmark is None or benchmark.dropna().empty
            else pd.Timestamp(benchmark.dropna().index.max()).date().isoformat()
        ),
        "holdout_benchmark_covered": holdout_benchmark_covered,
        "holdout_actual_end": (
            actual_holdout_end.date().isoformat()
            if actual_holdout_end is not None
            else None
        ),
        "data_asof_date": data_asof,
        "data_fingerprint": data_fingerprint,
        "starting_champion_id": state.champion_version,
        "locked_candidate_id": incumbent_id if locked else None,
        "locked_candidate_parameters": incumbent_parameters if locked else None,
        "holdout_gates": gates,
        "shadow_write_status": shadow_write_status,
        "state_path": str(state_path),
        "research_only": True,
        "trade_instruction": False,
        "return_promise": False,
        "locked_execution": config.locked_execution,
        "input_quality": {
            "quarantined_all_zero_market_rows": quarantined_zero_rows,
            "suspended_no_open_quote_rows": suspended_quote_rows,
        },
    }

    pd.DataFrame(score_rows).to_csv(run_dir / "candidate_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    _atomic_write_json(run_dir / "manifest.json", manifest)
    _atomic_write_json(run_dir / "resolved_config.json", config)
    _atomic_write_json(run_dir / "factor_metadata.json", factor_metadata)
    if holdout_payload is not None:
        _atomic_write_json(run_dir / "holdout_metrics.json", holdout_payload)
    if eligible_for_shadow:
        (run_dir / "champion_candidate.yaml").write_text(
            yaml.safe_dump(
                {
                    "strategy": config.strategy,
                    "candidate_id": incumbent_id,
                    "parameters": incumbent_parameters,
                    "research_only": True,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    _write_summary(run_dir, manifest, holdout_payload)

    latest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "data_asof_date": data_asof,
        "research_only": True,
    }
    _atomic_write_json(output_root / "latest.json", latest)
    registry_path = output_root / "evolution_registry.jsonl"
    with registry_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(latest, ensure_ascii=False, sort_keys=True) + "\n")
    return {**manifest, "run_dir": str(run_dir)}


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            raise ValueError(f"market-data file must be CSV or Parquet: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("ths_hs_a_share_*.csv")) + sorted(
        path.glob("ths_hs_a_share_*.parquet")
    )
    if not files:
        raise FileNotFoundError(
            f"no ths_hs_a_share_YYYY-MM-DD CSV or Parquet files found in {path}"
        )
    return files


def load_market_data(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in paths:
        for file_path in _source_files(Path(source)):
            frames.append(
                read_panel(file_path, dtype={"symbol": str}, low_memory=False)
            )
    return combine_panel_frames(frames)


def load_benchmark(path: Path | None) -> pd.Series | None:
    if path is None:
        return None
    frame = pd.read_csv(path, low_memory=False)
    if "date" not in frame or "close" not in frame:
        raise ValueError("benchmark CSV requires date and close columns")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError("benchmark CSV contains duplicate dates")
    if not frame["close"].gt(0.0).all():
        raise ValueError("benchmark close must be positive")
    return frame.set_index("date")["close"].sort_index()


def select_search_groups(
    config: EvolutionConfig, group_ids: Sequence[str] | None
) -> EvolutionConfig:
    if not group_ids:
        return config
    requested = set(group_ids)
    available = {group.group_id for group in config.search_groups}
    unknown = requested.difference(available)
    if unknown:
        raise ValueError(f"unknown search groups: {sorted(unknown)}")
    selected = tuple(
        group for group in config.search_groups if group.group_id in requested
    )
    return dataclasses.replace(config, search_groups=selected)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded self-evolution for multifactor observation research."
    )
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help=(
            "Historical CSV or normalized daily-data directory. Repeat to merge "
            "the old history with the normalized daily directory under QUANT_DATA_ROOT."
        ),
    )
    parser.add_argument("--benchmark", default=None, help="CSI1000 date/close CSV")
    parser.add_argument(
        "--config",
        default="configs/evolution_multifactor_observation.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/evolution_runs/multifactor_observation",
    )
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        help="Evaluate only the named search group. Repeat to select more groups.",
    )
    parser.add_argument(
        "--state-path",
        default="outputs/evolution_state/multifactor_observation.json",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--promote-shadow", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_id = args.run_id or (
        "multifactor-observation-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    config = select_search_groups(
        load_evolution_config(Path(args.config)), args.groups
    )
    outcome = run_evolution(
        panel=load_market_data([Path(value) for value in args.data]),
        benchmark=load_benchmark(Path(args.benchmark) if args.benchmark else None),
        config=config,
        output_root=Path(args.output_root),
        state_path=Path(args.state_path),
        run_id=run_id,
        dry_run=bool(args.dry_run),
        promote_shadow=bool(args.promote_shadow),
    )
    print(json.dumps(_json_safe(outcome), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
