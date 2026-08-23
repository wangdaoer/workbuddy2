"""Analyze point-in-time momentum breadth and leadership concentration."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from multifactor_observation_evolution import (
    EvolutionPeriods,
    ParameterEvaluation,
    evaluate_parameter_set,
    load_evolution_config,
    prepare_factor_panel,
)
from run_multifactor_observation_evolution import load_benchmark, load_market_data


BUCKETS: dict[str, tuple[float, ...]] = {
    "breadth_20": (0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 1.000001),
    "breadth_20_gap_ma20": (-1.0, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 1.0),
    "breadth_20_change_10": (-1.0, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 1.0),
    "leadership_gap_20": (0.0, 0.10, 0.20, 0.40, 0.80, 1.60, float("inf")),
}


def compute_daily_momentum_breadth(factor_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "momentum_20", "momentum_60"}
    missing = required.difference(factor_panel.columns)
    if missing:
        raise ValueError(f"missing breadth inputs: {sorted(missing)}")
    eligible = factor_panel.get(
        "score_eligible", pd.Series(True, index=factor_panel.index)
    ).fillna(False).astype(bool)
    working = factor_panel.loc[
        eligible,
        ["date", "symbol", "momentum_20", "momentum_60"],
    ].copy()
    working["momentum_20"] = pd.to_numeric(
        working["momentum_20"], errors="coerce"
    )
    working["momentum_60"] = pd.to_numeric(
        working["momentum_60"], errors="coerce"
    )
    working = working.dropna(subset=["momentum_20", "momentum_60"])
    grouped = working.groupby("date", sort=True)
    daily = grouped.agg(
        universe_size=("symbol", "nunique"),
        breadth_20=("momentum_20", lambda values: float(values.gt(0.0).mean())),
        breadth_60=("momentum_60", lambda values: float(values.gt(0.0).mean())),
        median_momentum_20=("momentum_20", "median"),
        p90_momentum_20=("momentum_20", lambda values: float(values.quantile(0.90))),
        median_momentum_60=("momentum_60", "median"),
        p90_momentum_60=("momentum_60", lambda values: float(values.quantile(0.90))),
    ).sort_index()
    daily["breadth_20_ma20"] = daily["breadth_20"].rolling(
        20, min_periods=20
    ).mean()
    daily["breadth_20_gap_ma20"] = (
        daily["breadth_20"] - daily["breadth_20_ma20"]
    )
    daily["breadth_20_change_10"] = daily["breadth_20"].diff(10)
    daily["leadership_gap_20"] = (
        daily["p90_momentum_20"] - daily["median_momentum_20"]
    )
    daily["leadership_gap_20_ma20"] = daily["leadership_gap_20"].rolling(
        20, min_periods=20
    ).mean()
    daily["leadership_gap_20_ratio_ma20"] = daily["leadership_gap_20"].div(
        daily["leadership_gap_20_ma20"].replace(0.0, np.nan)
    )
    daily.index.name = "date"
    return daily


def attach_forward_returns(
    daily: pd.DataFrame,
    evaluation: ParameterEvaluation,
    benchmark: pd.Series | None,
    periods: EvolutionPeriods,
) -> pd.DataFrame:
    out = daily.copy()
    equity = evaluation.backtest.equity.sort_index()
    aligned_equity = equity.reindex(out.index)
    aligned_benchmark = (
        benchmark.sort_index().reindex(out.index) if benchmark is not None else None
    )
    for horizon in (5, 10, 20):
        strategy_column = f"forward_strategy_return_{horizon}d"
        out[strategy_column] = aligned_equity.shift(-horizon).div(aligned_equity).sub(1.0)
        if aligned_benchmark is not None:
            benchmark_column = f"forward_benchmark_return_{horizon}d"
            excess_column = f"forward_excess_return_{horizon}d"
            out[benchmark_column] = (
                aligned_benchmark.shift(-horizon).div(aligned_benchmark).sub(1.0)
            )
            out[excess_column] = out[strategy_column] - out[benchmark_column]
    return out.loc[
        (out.index >= periods.selection_start)
        & (out.index <= periods.selection_end)
    ].copy()


def summarize_signal_buckets(
    daily: pd.DataFrame,
    signal: str,
    bins: Sequence[float],
) -> pd.DataFrame:
    if signal not in daily:
        raise ValueError(f"missing breadth signal: {signal}")
    working = daily.copy()
    working["bucket"] = pd.cut(
        pd.to_numeric(working[signal], errors="coerce"),
        bins=list(bins),
        right=False,
        include_lowest=True,
    )
    metrics: dict[str, tuple[str, object]] = {
        "sessions": (signal, "size"),
        "signal_mean": (signal, "mean"),
    }
    for horizon in (5, 10, 20):
        strategy = f"forward_strategy_return_{horizon}d"
        if strategy in working:
            metrics[f"mean_forward_strategy_{horizon}d"] = (strategy, "mean")
            metrics[f"positive_forward_strategy_{horizon}d"] = (
                strategy,
                lambda values: float(values.dropna().gt(0.0).mean()),
            )
        excess = f"forward_excess_return_{horizon}d"
        if excess in working:
            metrics[f"mean_forward_excess_{horizon}d"] = (excess, "mean")
            metrics[f"positive_forward_excess_{horizon}d"] = (
                excess,
                lambda values: float(values.dropna().gt(0.0).mean()),
            )
    summary = (
        working.dropna(subset=["bucket"])
        .groupby("bucket", observed=True)
        .agg(**metrics)
        .reset_index()
    )
    summary.insert(0, "signal", signal)
    summary["bucket"] = summary["bucket"].astype("string")
    return summary


def summarize_folds(
    daily: pd.DataFrame, evaluation: ParameterEvaluation
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in evaluation.fold_rows:
        start = pd.Timestamp(fold["start"])
        end = pd.Timestamp(fold["end"])
        window = daily.loc[start:end]
        rows.append(
            {
                "fold_id": fold["fold_id"],
                "start": fold["start"],
                "end": fold["end"],
                "total_return": fold["total_return"],
                "max_drawdown": fold["max_drawdown"],
                "benchmark_return": fold["benchmark_return"],
                "excess_return": fold["excess_return"],
                "mean_breadth_20": float(window["breadth_20"].mean()),
                "mean_breadth_60": float(window["breadth_60"].mean()),
                "share_breadth_below_ma20": float(
                    window["breadth_20_gap_ma20"].lt(0.0).mean()
                ),
                "share_breadth_drop_10d_10pp": float(
                    window["breadth_20_change_10"].le(-0.10).mean()
                ),
                "mean_leadership_gap_20": float(
                    window["leadership_gap_20"].mean()
                ),
                "mean_leadership_gap_ratio_ma20": float(
                    window["leadership_gap_20_ratio_ma20"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def signal_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal in BUCKETS:
        for horizon in (5, 10, 20):
            for target_type in ("strategy", "excess"):
                target = f"forward_{target_type}_return_{horizon}d"
                if target not in daily:
                    continue
                aligned = daily[[signal, target]].dropna()
                rows.append(
                    {
                        "signal": signal,
                        "target": target,
                        "observations": int(len(aligned)),
                        "spearman_correlation": float(
                            aligned[signal].corr(aligned[target], method="spearman")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.2%}"


def build_report(
    folds: pd.DataFrame,
    correlations: pd.DataFrame,
    data_asof: object,
) -> str:
    worst = folds.sort_values("total_return").head(3)
    wf08 = folds.loc[folds["fold_id"].eq("wf_08")]
    lines = [
        "# 动量广度与拥挤状态诊断",
        "",
        "仅用于研究/模拟盘，不构成交易指令或收益承诺。所有广度信号均在当日收盘后计算，最早用于下一交易日开盘。",
        "",
        f"- 数据截至：{data_asof}",
        f"- 选择期折数：{len(folds)}",
        "",
        "## 最差三个折",
        "",
        "| 折 | 收益 | 超额 | 20日广度均值 | 广度低于20日均线占比 | 10日广度下降10个百分点占比 | 头部差距均值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in worst.to_dict("records"):
        lines.append(
            "| {fold_id} | {total} | {excess} | {breadth} | {below} | {drop} | {gap} |".format(
                fold_id=row["fold_id"],
                total=_pct(row["total_return"]),
                excess=_pct(row["excess_return"]),
                breadth=_pct(row["mean_breadth_20"]),
                below=_pct(row["share_breadth_below_ma20"]),
                drop=_pct(row["share_breadth_drop_10d_10pp"]),
                gap=_pct(row["mean_leadership_gap_20"]),
            )
        )
    lines.extend(["", "## wf_08", ""])
    if wf08.empty:
        lines.append("`wf_08` 不在当前选择期内。")
    else:
        row = wf08.iloc[0]
        lines.extend(
            [
                f"- 收益：{_pct(row['total_return'])}",
                f"- 超额：{_pct(row['excess_return'])}",
                f"- 20日广度均值：{_pct(row['mean_breadth_20'])}",
                f"- 广度低于20日均线占比：{_pct(row['share_breadth_below_ma20'])}",
                f"- 10日广度下降至少10个百分点占比：{_pct(row['share_breadth_drop_10d_10pp'])}",
                f"- 头部动量与中位数差距均值：{_pct(row['mean_leadership_gap_20'])}",
            ]
        )
    lines.extend(["", "## 审核规则", ""])
    strong = correlations.loc[
        correlations["spearman_correlation"].abs().ge(0.10)
    ]
    if strong.empty:
        lines.append(
            "未发现绝对 Spearman 相关达到 0.10 的稳定信号；不应接入仓位控制。"
        )
    else:
        lines.append(
            "存在绝对 Spearman 相关达到 0.10 的候选关系，但仍须通过固定阈值的10折回放，不能直接据此晋级。"
        )
    lines.append(
        "当前行业、市值和 ST 名称仅为当前快照信息，不参与本诊断和历史过滤。"
    )
    return "\n".join(lines) + "\n"


def run_breadth_diagnosis(
    panel: pd.DataFrame,
    benchmark: pd.Series | None,
    periods: EvolutionPeriods,
    baseline: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    factor_panel, factor_metadata = prepare_factor_panel(panel)
    evaluation = evaluate_parameter_set(
        factor_panel, baseline, periods, benchmark=benchmark
    )
    daily = attach_forward_returns(
        compute_daily_momentum_breadth(factor_panel),
        evaluation,
        benchmark,
        periods,
    )
    folds = summarize_folds(daily, evaluation)
    correlations = signal_correlations(daily)
    buckets = pd.concat(
        [summarize_signal_buckets(daily, signal, bins) for signal, bins in BUCKETS.items()],
        ignore_index=True,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_csv(output_dir / "daily_momentum_breadth.csv", index=False)
    folds.to_csv(output_dir / "fold_breadth_summary.csv", index=False)
    correlations.to_csv(output_dir / "signal_correlations.csv", index=False)
    buckets.to_csv(output_dir / "signal_bucket_summary.csv", index=False)
    data_asof = panel["date"].max()
    (output_dir / "momentum_breadth_diagnosis.md").write_text(
        build_report(folds, correlations, data_asof), encoding="utf-8"
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "trade_instruction": False,
        "data_asof_date": pd.Timestamp(data_asof).date().isoformat(),
        "selection_start": periods.selection_start.date().isoformat(),
        "selection_end": periods.selection_end.date().isoformat(),
        "fold_count": int(len(folds)),
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
        description="Diagnose point-in-time momentum breadth and crowding regimes."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--config", default="configs/evolution_multifactor_observation.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evolution_runs/multifactor_observation/breadth_diagnostics/latest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_evolution_config(Path(args.config))
    result = run_breadth_diagnosis(
        panel=load_market_data([Path(value) for value in args.data]),
        benchmark=load_benchmark(Path(args.benchmark) if args.benchmark else None),
        periods=config.periods,
        baseline=config.baseline,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
