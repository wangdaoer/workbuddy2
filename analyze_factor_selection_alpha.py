"""Diagnose stock-selection factor alpha in healthy momentum-breadth regimes."""

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
    FACTOR_NAMES,
    FACTOR_RANK_COLUMNS,
    WEIGHT_KEYS,
    EvolutionPeriods,
    build_cross_sectional_momentum_breadth,
    evaluate_parameter_set,
    load_evolution_config,
    prepare_factor_panel,
    validate_parameters,
)
from run_multifactor_observation_evolution import load_benchmark, load_market_data


DEFAULT_HORIZONS = (5, 10, 20)


def load_reference_parameters(
    run_dir: Path,
    candidate_id: str,
    fallback: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load the exact parameter set recorded for a prior research candidate."""
    score_path = Path(run_dir) / "candidate_scores.csv"
    if not score_path.exists():
        if fallback is None:
            raise FileNotFoundError(score_path)
        return validate_parameters(fallback)
    scores = pd.read_csv(score_path, dtype={"candidate_id": "string"})
    matches = scores.loc[scores["candidate_id"].eq(candidate_id)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one candidate {candidate_id!r} in {score_path}, found {len(matches)}"
        )
    raw = json.loads(str(matches.iloc[0]["parameters"]))
    return validate_parameters(raw)


def compute_healthy_breadth_state(
    factor_panel: pd.DataFrame,
    ma_window: int,
) -> pd.DataFrame:
    """Build a point-in-time breadth state; healthy means breadth is at/above its MA."""
    if ma_window < 2:
        raise ValueError("ma_window must be at least 2")
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(factor_panel["date"]).unique()))
    breadth = build_cross_sectional_momentum_breadth(factor_panel, dates)
    breadth_ma = breadth.rolling(ma_window, min_periods=ma_window).mean()
    out = pd.DataFrame(
        {
            "breadth_20": breadth,
            "breadth_20_ma": breadth_ma,
            "breadth_gap": breadth - breadth_ma,
        }
    )
    out["breadth_healthy"] = out["breadth_gap"].ge(0.0) & breadth_ma.notna()
    out.index.name = "date"
    return out


def attach_forward_open_returns(
    factor_panel: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Attach signal-close to later-open returns using next open as entry."""
    requested = tuple(int(value) for value in horizons)
    if not requested or any(value < 1 for value in requested):
        raise ValueError("forward horizons must be positive")
    out = factor_panel.sort_values(["symbol", "date"], kind="mergesort").copy()
    opens = pd.to_numeric(out["open"], errors="coerce")
    grouped = opens.groupby(out["symbol"], sort=False)
    entry = grouped.shift(-1)
    for horizon in requested:
        exit_open = grouped.shift(-(horizon + 1))
        returns = exit_open.div(entry) - 1.0
        out[f"forward_open_return_{horizon}d"] = returns.where(
            entry.gt(0.0) & exit_open.gt(0.0)
        )
    return out


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(valid) < 20 or valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return math.nan
    return float(valid["left"].rank().corr(valid["right"].rank()))


def compute_healthy_factor_ic(
    factor_panel: pd.DataFrame,
    breadth_state: pd.DataFrame,
    periods: EvolutionPeriods,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily cross-sectional RankIC and quintile spread in healthy breadth."""
    enriched = attach_forward_open_returns(factor_panel, horizons)
    enriched = enriched.merge(
        breadth_state.reset_index()[["date", "breadth_gap", "breadth_healthy"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    eligible = enriched.get("score_eligible", pd.Series(True, index=enriched.index))
    enriched = enriched.loc[
        eligible.fillna(False)
        & enriched["breadth_healthy"].fillna(False)
        & enriched["date"].between(periods.selection_start, periods.selection_end)
    ]

    daily_rows: list[dict[str, object]] = []
    for date, group in enriched.groupby("date", sort=True):
        for factor in FACTOR_NAMES:
            factor_values = pd.to_numeric(group[factor], errors="coerce")
            factor_pct_rank = factor_values.rank(method="average", pct=True)
            for horizon in horizons:
                return_column = f"forward_open_return_{int(horizon)}d"
                returns = pd.to_numeric(group[return_column], errors="coerce")
                valid = factor_values.notna() & returns.notna()
                top = valid & factor_pct_rank.ge(0.8)
                bottom = valid & factor_pct_rank.le(0.2)
                spread = (
                    float(returns.loc[top].mean() - returns.loc[bottom].mean())
                    if top.any() and bottom.any()
                    else math.nan
                )
                daily_rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "horizon": int(horizon),
                        "observations": int(valid.sum()),
                        "rank_ic": _rank_correlation(
                            factor_values.loc[valid], returns.loc[valid]
                        ),
                        "top_bottom_spread": spread,
                        "breadth_gap": float(group["breadth_gap"].iloc[0]),
                    }
                )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("no healthy-breadth factor observations are available")

    summary_rows: list[dict[str, object]] = []
    for (factor, horizon), group in daily.groupby(["factor", "horizon"], sort=False):
        ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        spread = pd.to_numeric(group["top_bottom_spread"], errors="coerce").dropna()
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else math.nan
        summary_rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "days": int(len(ic)),
                "mean_rank_ic": float(ic.mean()) if not ic.empty else math.nan,
                "median_rank_ic": float(ic.median()) if not ic.empty else math.nan,
                "positive_ic_ratio": float(ic.gt(0.0).mean()) if not ic.empty else math.nan,
                "rank_ic_ir": (
                    float(ic.mean() / ic_std * math.sqrt(252.0))
                    if math.isfinite(ic_std) and ic_std > 0.0
                    else math.nan
                ),
                "mean_top_bottom_spread": (
                    float(spread.mean()) if not spread.empty else math.nan
                ),
                "positive_spread_ratio": (
                    float(spread.gt(0.0).mean()) if not spread.empty else math.nan
                ),
            }
        )
    return daily, pd.DataFrame(summary_rows)


def build_factor_variants(
    reference_parameters: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Create reference, leave-one-out, and single-factor parameter variants."""
    reference = validate_parameters(reference_parameters)
    variants = {"reference": dict(reference)}
    for factor, weight_key in zip(FACTOR_NAMES, WEIGHT_KEYS):
        dropped = dict(reference)
        dropped[weight_key] = 0.0
        variants[f"drop_{factor}"] = validate_parameters(dropped)
    for factor, weight_key in zip(FACTOR_NAMES, WEIGHT_KEYS):
        isolated = dict(reference)
        for key in WEIGHT_KEYS:
            isolated[key] = 0.0
        isolated[weight_key] = 1.0
        variants[f"only_{factor}"] = validate_parameters(isolated)
    return variants


def evaluate_factor_variants(
    factor_panel: pd.DataFrame,
    benchmark: pd.Series | None,
    periods: EvolutionPeriods,
    variants: Mapping[str, Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every factor variant under identical PIT execution assumptions."""
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for variant_id, parameters in variants.items():
        evaluation = evaluate_parameter_set(
            factor_panel,
            parameters,
            periods,
            benchmark=benchmark,
        )
        folds = pd.DataFrame(evaluation.fold_rows)
        mode, factor = (
            ("reference", "all")
            if variant_id == "reference"
            else variant_id.split("_", 1)
        )
        folds.insert(0, "factor", factor)
        folds.insert(0, "mode", mode)
        folds.insert(0, "variant_id", variant_id)
        fold_rows.extend(folds.to_dict("records"))
        excess = pd.to_numeric(folds["excess_return"], errors="coerce").dropna()
        summary_rows.append(
            {
                "variant_id": variant_id,
                "mode": mode,
                "factor": factor,
                "folds": int(len(folds)),
                "mean_total_return": float(folds["total_return"].mean()),
                "median_total_return": float(folds["total_return"].median()),
                "positive_fold_ratio": float(folds["total_return"].gt(0.0).mean()),
                "mean_excess_return": float(excess.mean()) if not excess.empty else math.nan,
                "positive_excess_fold_ratio": (
                    float(excess.gt(0.0).mean()) if not excess.empty else math.nan
                ),
                "mean_max_drawdown": float(folds["max_drawdown"].mean()),
                "worst_max_drawdown": float(folds["max_drawdown"].min()),
                "mean_sharpe": float(folds["sharpe"].mean()),
                "mean_turnover": float(folds["average_turnover"].mean()),
                "mean_market_exposure": float(folds["average_market_exposure"].mean()),
                "max_pnl_concentration": float(folds["pnl_concentration"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    reference = summary.loc[summary["variant_id"].eq("reference")].iloc[0]
    summary["mean_return_delta_vs_reference"] = (
        summary["mean_total_return"] - float(reference["mean_total_return"])
    )
    summary["positive_fold_delta_vs_reference"] = (
        summary["positive_fold_ratio"] - float(reference["positive_fold_ratio"])
    )
    summary["positive_excess_delta_vs_reference"] = (
        summary["positive_excess_fold_ratio"]
        - float(reference["positive_excess_fold_ratio"])
    )
    return summary, pd.DataFrame(fold_rows)


def _pct(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(number) else f"{float(number):.4f}"


def build_report(
    ic_summary: pd.DataFrame,
    variant_summary: pd.DataFrame,
    breadth_state: pd.DataFrame,
    data_asof: object,
    reference_candidate: str,
) -> str:
    reference = variant_summary.loc[variant_summary["variant_id"].eq("reference")].iloc[0]
    drops = variant_summary.loc[variant_summary["mode"].eq("drop")].sort_values(
        "mean_return_delta_vs_reference", ascending=False
    )
    singles = variant_summary.loc[variant_summary["mode"].eq("only")].sort_values(
        "mean_total_return", ascending=False
    )
    healthy_sessions = int(breadth_state["breadth_healthy"].sum())
    observable_sessions = int(breadth_state["breadth_20_ma"].notna().sum())
    harmful = drops.loc[
        drops["mean_return_delta_vs_reference"].gt(0.0)
        & drops["positive_fold_delta_vs_reference"].ge(0.0)
        & drops["positive_excess_delta_vs_reference"].ge(0.0)
    ]

    lines = [
        "# 健康市场广度下的选股因子诊断",
        "",
        "> 研究/模拟盘用途，不构成收益承诺或交易指令。所有信号在收盘形成，按下一交易日开盘执行假设评价。",
        "",
        "## 范围",
        "",
        f"- 数据截至：{pd.Timestamp(data_asof).date().isoformat()}",
        f"- 固定参考候选：`{reference_candidate}`",
        f"- 可观察广度交易日：{observable_sessions}",
        f"- 健康广度交易日：{healthy_sessions}",
        f"- 参考模型平均 126 日收益：{_pct(reference['mean_total_return'])}",
        f"- 参考模型正收益窗口：{_pct(reference['positive_fold_ratio'])}",
        f"- 参考模型跑赢基准窗口：{_pct(reference['positive_excess_fold_ratio'])}",
        "",
        "## 健康广度 RankIC",
        "",
        "| 因子 | 周期 | 平均 RankIC | 正 IC 比例 | 多空分位差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ic_summary.sort_values(["horizon", "mean_rank_ic"], ascending=[True, False]).to_dict("records"):
        lines.append(
            f"| {row['factor']} | {int(row['horizon'])}日 | {_number(row['mean_rank_ic'])} | "
            f"{_pct(row['positive_ic_ratio'])} | {_pct(row['mean_top_bottom_spread'])} |"
        )
    lines.extend(
        [
            "",
            "## 逐项删除",
            "",
            "| 删除因子 | 平均收益 | 相对参考 | 正收益窗口 | 跑赢窗口 | 平均回撤 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in drops.to_dict("records"):
        lines.append(
            f"| {row['factor']} | {_pct(row['mean_total_return'])} | "
            f"{_pct(row['mean_return_delta_vs_reference'])} | "
            f"{_pct(row['positive_fold_ratio'])} | {_pct(row['positive_excess_fold_ratio'])} | "
            f"{_pct(row['mean_max_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "## 单因子",
            "",
            "| 单独保留 | 平均收益 | 正收益窗口 | 跑赢窗口 | 平均回撤 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in singles.to_dict("records"):
        lines.append(
            f"| {row['factor']} | {_pct(row['mean_total_return'])} | "
            f"{_pct(row['positive_fold_ratio'])} | {_pct(row['positive_excess_fold_ratio'])} | "
            f"{_pct(row['mean_max_drawdown'])} |"
        )
    lines.extend(["", "## 结论", ""])
    if harmful.empty:
        lines.append("- 没有因子同时满足删除后收益提升、正收益窗口不下降、跑赢窗口不下降；暂不建议机械删除。")
    else:
        names = "、".join(f"`{value}`" for value in harmful["factor"])
        lines.append(f"- 初步负贡献候选：{names}。仅允许进入下一轮受控组合实验，不直接改生产基线。")
    best_single = singles.iloc[0]
    lines.append(
        f"- 最强单因子是 `{best_single['factor']}`，平均 126 日收益 {_pct(best_single['mean_total_return'])}；"
        "单因子结果只用于辨认信号来源，不能视为已通过策略。"
    )
    lines.append("- 下一步只组合跨周期 RankIC 与滚动剥离结果方向一致的因子，并继续执行原成本、涨跌停和下一开盘约束。")
    return "\n".join(lines) + "\n"


def run_factor_selection_diagnosis(
    panel: pd.DataFrame,
    benchmark: pd.Series | None,
    periods: EvolutionPeriods,
    reference_parameters: Mapping[str, object],
    reference_candidate: str,
    output_dir: Path,
) -> dict[str, object]:
    factor_panel, factor_metadata = prepare_factor_panel(panel)
    breadth_state = compute_healthy_breadth_state(
        factor_panel, int(reference_parameters["breadth_ma_window"])
    )
    daily_ic, ic_summary = compute_healthy_factor_ic(
        factor_panel, breadth_state, periods
    )
    variants = build_factor_variants(reference_parameters)
    variant_summary, fold_results = evaluate_factor_variants(
        factor_panel, benchmark, periods, variants
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    breadth_state.reset_index().to_csv(output_dir / "breadth_state.csv", index=False)
    daily_ic.to_csv(output_dir / "daily_factor_ic.csv", index=False)
    ic_summary.to_csv(output_dir / "factor_ic_summary.csv", index=False)
    variant_summary.to_csv(output_dir / "factor_variant_summary.csv", index=False)
    fold_results.to_csv(output_dir / "factor_variant_folds.csv", index=False)
    (output_dir / "factor_variants.json").write_text(
        json.dumps(variants, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    data_asof = pd.Timestamp(panel["date"].max())
    (output_dir / "factor_selection_diagnosis.md").write_text(
        build_report(
            ic_summary,
            variant_summary,
            breadth_state,
            data_asof,
            reference_candidate,
        ),
        encoding="utf-8",
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "trade_instruction": False,
        "data_asof_date": data_asof.date().isoformat(),
        "reference_candidate": reference_candidate,
        "selection_start": periods.selection_start.date().isoformat(),
        "selection_end": periods.selection_end.date().isoformat(),
        "factor_count": len(FACTOR_NAMES),
        "variant_count": len(variants),
        "healthy_breadth_sessions": int(breadth_state["breadth_healthy"].sum()),
        "factor_metadata": factor_metadata,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {**manifest, "output_dir": str(output_dir)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose factor selection alpha in healthy momentum breadth."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--config", default="configs/evolution_multifactor_observation.yaml"
    )
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument(
        "--reference-candidate", default="breadth_hard_without_portfolio_stop"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evolution_runs/multifactor_observation/factor_selection/latest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_evolution_config(Path(args.config))
    reference = load_reference_parameters(
        Path(args.reference_run_dir), args.reference_candidate, config.baseline
    )
    result = run_factor_selection_diagnosis(
        panel=load_market_data([Path(value) for value in args.data]),
        benchmark=load_benchmark(Path(args.benchmark) if args.benchmark else None),
        periods=config.periods,
        reference_parameters=reference,
        reference_candidate=args.reference_candidate,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
