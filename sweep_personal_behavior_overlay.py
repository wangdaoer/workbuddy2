"""Sweep personal behavior overlay parameters against a fixed rank model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apply_personal_trade_overlay import _load_rules, apply_overlay, load_symbol_history
from execution_rules import apply_open_constraints, next_open_return_label
from run_backtest import load_prices, pivot_prices
from run_personal_behavior_overlay_backtest import (
    base_targets_from_candidates,
    build_candidate_table,
    build_metric_frames,
    metric_snapshot_from_frames,
    score_from_weights,
    strategy_metrics,
    targets_from_overlay_table,
    weights_for_signal_date,
)
from train_next_open_rank_model import (
    build_breadth_exposure,
    build_features,
    clean_matrix,
    load_market_exposure,
)


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _weak_label(value: float) -> str:
    return f"weak{int(round(value * 100)):03d}"


def _damage_label(value: float) -> str:
    return f"damage{int(round(value * 100)):+d}"


def build_rule_variants(
    base_rules: dict[str, Any],
    weak_multipliers: list[float],
    damaged_thresholds: list[float],
    selection_modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    modes = selection_modes or [str(base_rules.get("selection_mode", "conservative_fill"))]
    for mode in modes:
        for weak_multiplier in weak_multipliers:
            for damaged_threshold in damaged_thresholds:
                rules = copy.deepcopy(base_rules)
                rules["selection_mode"] = str(mode)
                rules["weak_20d_weight_multiplier"] = float(weak_multiplier)
                rules["damaged_20d_return_max"] = float(damaged_threshold)
                variants.append(
                    {
                        "variant": f"{mode}_{_weak_label(weak_multiplier)}_{_damage_label(damaged_threshold)}",
                        "selection_mode": str(mode),
                        "weak_20d_weight_multiplier": float(weak_multiplier),
                        "damaged_20d_return_max": float(damaged_threshold),
                        "rules": rules,
                    }
                )
    return variants


def rank_sweep_results(results: pd.DataFrame, max_drawdown_floor: float) -> pd.DataFrame:
    ranked = results.copy()
    ranked["passes_drawdown"] = pd.to_numeric(
        ranked["max_drawdown"],
        errors="coerce",
    ).ge(float(max_drawdown_floor))
    ranked = ranked.sort_values(
        ["passes_drawdown", "total_return", "sharpe_like"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "sweep_rank", range(1, len(ranked) + 1))
    return ranked


def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    features = build_features(close, open_px, high, low, amount)
    metric_frames = build_metric_frames(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=args.max_abs_daily_return)

    market_exposure = load_market_exposure(
        args.benchmark,
        close.index,
        ma_window=args.market_ma_window,
        risk_off_drawdown_20d=args.market_risk_off_drawdown_20d,
        below_ma_exposure=args.market_below_ma_exposure,
        crash_exposure=args.market_crash_exposure,
    )
    if args.breadth_filter:
        breadth_exposure = build_breadth_exposure(
            close,
            ma_window=args.breadth_ma_window,
            threshold=args.breadth_threshold,
            below_exposure=args.breadth_below_exposure,
            crash_threshold=args.breadth_crash_threshold,
            crash_exposure=args.breadth_crash_exposure,
        )
        market_exposure = pd.concat([market_exposure, breadth_exposure], axis=1).min(axis=1)

    return {
        "close": close,
        "open_px": open_px,
        "features": features,
        "metric_frames": metric_frames,
        "label": label,
        "market_exposure": market_exposure,
    }


def prepare_candidate_tables(
    prepared: dict[str, Any],
    weights: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[int, pd.DataFrame]:
    close: pd.DataFrame = prepared["close"]
    features: dict[str, pd.DataFrame] = prepared["features"]
    metric_frames: dict[str, pd.DataFrame] = prepared["metric_frames"]
    raw_weight = min(args.max_position_weight, args.leverage / max(args.top_n, 1))
    candidates_by_index: dict[int, pd.DataFrame] = {}
    for i in range(args.train_days, len(close.index) - 2):
        if (i - args.train_days) % args.rebalance_frequency != 0:
            continue
        signal_date = close.index[i]
        current_weights = weights_for_signal_date(weights, signal_date)
        score = score_from_weights(features, current_weights, i)
        metrics = metric_snapshot_from_frames(metric_frames, i)
        candidates_by_index[i] = build_candidate_table(
            score,
            metrics,
            top_n=args.top_n,
            candidate_pool_n=args.candidate_pool_n,
            raw_weight=raw_weight,
        )
    return candidates_by_index


def simulate_base(
    prepared: dict[str, Any],
    candidates_by_index: dict[int, pd.DataFrame],
    args: argparse.Namespace,
) -> dict[str, Any]:
    close: pd.DataFrame = prepared["close"]
    open_px: pd.DataFrame = prepared["open_px"]
    label: pd.DataFrame = prepared["label"]
    market_exposure: pd.Series = prepared["market_exposure"]
    equity = float(args.initial_capital)
    positions = pd.Series(0.0, index=close.columns)
    rows: list[dict[str, object]] = []
    for i in range(args.train_days, len(close.index) - 2):
        signal_date = close.index[i]
        if i in candidates_by_index:
            exposure = float(market_exposure.reindex([signal_date]).iloc[0])
            target = base_targets_from_candidates(candidates_by_index[i], close.columns, exposure)
        else:
            exposure = float(market_exposure.reindex([signal_date]).iloc[0])
            target = positions.copy()
        target = apply_open_constraints(
            positions,
            target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=args.max_buy_open_gap,
            limit_buffer=args.limit_buffer,
        )
        turnover = float((target - positions).abs().sum())
        cost = turnover * (args.commission_bps + args.impact_bps) / 1e4
        realized = label.iloc[i].reindex(close.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross_return = float((target * realized).sum())
        equity *= 1.0 + gross_return - cost
        positions = target
        rows.append(
            {
                "date": close.index[i + 2].strftime("%Y-%m-%d"),
                "equity": equity,
                "turnover": turnover,
                "gross_exposure": float(positions.abs().sum()),
                "market_exposure": exposure,
            }
        )
    nav = pd.Series([row["equity"] for row in rows], index=pd.to_datetime([row["date"] for row in rows]))
    metrics = strategy_metrics(nav, args.initial_capital)
    metrics["avg_turnover"] = float(pd.DataFrame(rows)["turnover"].mean())
    metrics["avg_gross_exposure"] = float(pd.DataFrame(rows)["gross_exposure"].mean())
    return {"rows": pd.DataFrame(rows), "metrics": metrics}


def simulate_overlay_variant(
    prepared: dict[str, Any],
    candidates_by_index: dict[int, pd.DataFrame],
    rules: dict[str, Any],
    symbol_history: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    close: pd.DataFrame = prepared["close"]
    open_px: pd.DataFrame = prepared["open_px"]
    label: pd.DataFrame = prepared["label"]
    market_exposure: pd.Series = prepared["market_exposure"]
    equity = float(args.initial_capital)
    positions = pd.Series(0.0, index=close.columns)
    rows: list[dict[str, object]] = []
    action_counts: dict[str, int] = {}
    selected_action_counts: dict[str, int] = {}

    for i in range(args.train_days, len(close.index) - 2):
        signal_date = close.index[i]
        exposure = float(market_exposure.reindex([signal_date]).iloc[0])
        if i in candidates_by_index:
            candidates = candidates_by_index[i]
            positive_weight = pd.to_numeric(candidates["target_weight"], errors="coerce")
            base_weight = float(positive_weight[positive_weight.gt(0)].median())
            overlay = apply_overlay(
                candidates,
                symbol_history,
                rules,
                reselect_top_n=args.top_n,
                base_target_weight=base_weight,
            )
            for key, value in overlay["personal_action"].value_counts().to_dict().items():
                action_counts[key] = action_counts.get(key, 0) + int(value)
            selected = overlay[overlay["personal_selected"].astype(bool)]
            for key, value in selected["personal_action"].value_counts().to_dict().items():
                selected_action_counts[key] = selected_action_counts.get(key, 0) + int(value)
            target = targets_from_overlay_table(overlay, close.columns, exposure)
        else:
            target = positions.copy()
        target = apply_open_constraints(
            positions,
            target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=args.max_buy_open_gap,
            limit_buffer=args.limit_buffer,
        )
        turnover = float((target - positions).abs().sum())
        cost = turnover * (args.commission_bps + args.impact_bps) / 1e4
        realized = label.iloc[i].reindex(close.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross_return = float((target * realized).sum())
        equity *= 1.0 + gross_return - cost
        positions = target
        rows.append(
            {
                "date": close.index[i + 2].strftime("%Y-%m-%d"),
                "equity": equity,
                "turnover": turnover,
                "gross_exposure": float(positions.abs().sum()),
                "market_exposure": exposure,
            }
        )
    frame = pd.DataFrame(rows)
    nav = pd.Series(frame["equity"].to_numpy(), index=pd.to_datetime(frame["date"]))
    metrics = strategy_metrics(nav, args.initial_capital)
    metrics["avg_turnover"] = float(frame["turnover"].mean())
    metrics["avg_gross_exposure"] = float(frame["gross_exposure"].mean())
    metrics["action_counts"] = action_counts
    metrics["selected_action_counts"] = selected_action_counts
    return {"rows": frame, "metrics": metrics}


def write_report(output_dir: Path, ranked: pd.DataFrame, base_metrics: dict[str, Any], args: argparse.Namespace) -> Path:
    top = ranked.head(10)
    best = ranked.iloc[0]
    lines = [
        "# 个人交易习惯参数扫描",
        "",
        "目标：在最大回撤不低于设定底线的前提下，优先选择收益率更高的个人习惯层参数。",
        "",
        "## 原模型基线",
        f"- 收益率: {base_metrics['total_return']:.2%}",
        f"- 最大回撤: {base_metrics['max_drawdown']:.2%}",
        f"- Sharpe-like: {base_metrics['sharpe_like']:.3f}",
        f"- 平均换手: {base_metrics['avg_turnover']:.2%}",
        "",
        "## 最优参数",
        f"- 参数名: {best['variant']}",
        f"- 选择模式: {best['selection_mode']}",
        f"- 弱20日降仓倍数: {best['weak_20d_weight_multiplier']:.2f}",
        f"- 20日趋势受损阈值: {best['damaged_20d_return_max']:.2%}",
        f"- 收益率: {best['total_return']:.2%}",
        f"- 最大回撤: {best['max_drawdown']:.2%}",
        f"- Sharpe-like: {best['sharpe_like']:.3f}",
        f"- 相对原模型收益差: {best['total_return_diff_vs_base']:.2%}",
        f"- 相对原模型回撤改善: {best['max_drawdown_diff_vs_base']:.2%}",
        "",
        "## 前10组参数",
        top.to_markdown(index=False),
        "",
        "## 扫描设置",
        f"- 回撤底线: {args.max_drawdown_floor:.2%}",
        f"- 弱20日降仓倍数候选: {args.weak_multipliers}",
        f"- 20日趋势受损阈值候选: {args.damaged_thresholds}",
        f"- 选择模式候选: {args.selection_modes}",
    ]
    report = output_dir / "sweep_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep personal behavior overlay parameters.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/personal_behavior_sweep")
    parser.add_argument("--rules", default="configs/personal_trade_habit_overlay.yaml")
    parser.add_argument("--symbol-history", default=None)
    parser.add_argument("--weak-multipliers", default="0.55,0.70,0.85,1.00")
    parser.add_argument("--damaged-thresholds", default="-0.15,-0.20,-0.25")
    parser.add_argument("--selection-modes", default="conservative_fill")
    parser.add_argument("--max-drawdown-floor", type=float, default=-0.30)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--candidate-pool-n", type=int, default=120)
    parser.add_argument("--rebalance-frequency", type=int, default=5)
    parser.add_argument("--max-position-weight", type=float, default=0.029)
    parser.add_argument("--leverage", type=float, default=0.93)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--impact-bps", type=float, default=0.7)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--max-buy-open-gap", type=float, default=0.05)
    parser.add_argument("--limit-buffer", type=float, default=0.995)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--market-ma-window", type=int, default=120)
    parser.add_argument("--market-risk-off-drawdown-20d", type=float, default=-0.08)
    parser.add_argument("--market-below-ma-exposure", type=float, default=0.65)
    parser.add_argument("--market-crash-exposure", type=float, default=0.10)
    parser.add_argument("--breadth-filter", action="store_true")
    parser.add_argument("--breadth-ma-window", type=int, default=60)
    parser.add_argument("--breadth-threshold", type=float, default=0.45)
    parser.add_argument("--breadth-below-exposure", type=float, default=0.55)
    parser.add_argument("--breadth-crash-threshold", type=float, default=0.32)
    parser.add_argument("--breadth-crash-exposure", type=float, default=0.20)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rules = _load_rules(Path(args.rules) if args.rules else None)
    symbol_history = load_symbol_history(Path(args.symbol_history) if args.symbol_history else None)
    weak_multipliers = _parse_float_list(args.weak_multipliers)
    damaged_thresholds = _parse_float_list(args.damaged_thresholds)
    selection_modes = [item.strip() for item in args.selection_modes.split(",") if item.strip()]
    variants = build_rule_variants(
        base_rules,
        weak_multipliers,
        damaged_thresholds,
        selection_modes=selection_modes,
    )

    prepared = prepare_data(args)
    weights = pd.read_csv(args.weights)
    candidates_by_index = prepare_candidate_tables(prepared, weights, args)
    base = simulate_base(prepared, candidates_by_index, args)
    base_metrics = base["metrics"]

    rows: list[dict[str, object]] = []
    for variant in variants:
        result = simulate_overlay_variant(
            prepared,
            candidates_by_index,
            variant["rules"],
            symbol_history,
            args,
        )
        metrics = result["metrics"]
        rows.append(
            {
                "variant": variant["variant"],
                "selection_mode": variant["selection_mode"],
                "weak_20d_weight_multiplier": variant["weak_20d_weight_multiplier"],
                "damaged_20d_return_max": variant["damaged_20d_return_max"],
                "final_equity": metrics["final_equity"],
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_like": metrics["sharpe_like"],
                "avg_turnover": metrics["avg_turnover"],
                "avg_gross_exposure": metrics["avg_gross_exposure"],
                "total_return_diff_vs_base": metrics["total_return"] - base_metrics["total_return"],
                "max_drawdown_diff_vs_base": metrics["max_drawdown"] - base_metrics["max_drawdown"],
                "sharpe_like_diff_vs_base": metrics["sharpe_like"] - base_metrics["sharpe_like"],
                "reduce_count": metrics["action_counts"].get("reduce", 0),
                "watch_only_count": metrics["action_counts"].get("watch_only", 0),
                "allow_count": metrics["action_counts"].get("allow", 0),
                "selected_reduce_count": metrics["selected_action_counts"].get("reduce", 0),
                "selected_allow_count": metrics["selected_action_counts"].get("allow", 0),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_sweep_results(pd.DataFrame(rows), args.max_drawdown_floor)
    ranked.to_csv(output_dir / "sweep_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "base_metrics.json").write_text(
        json.dumps(base_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = write_report(output_dir, ranked, base_metrics, args)
    print(f"Sweep variants={len(ranked)} output={output_dir / 'sweep_results.csv'}")
    print(f"Report={report}")
    print(ranked.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
