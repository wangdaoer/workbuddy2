"""Run preregistered point-in-time amount-capacity stress for the incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_prices, pivot_prices
from trading_calendar import validate_trading_sessions
from train_next_open_rank_model import (
    build_features,
    build_multi_horizon_ic,
    calculate_walk_forward_metrics,
    clean_matrix,
    load_market_exposure,
    next_open_return_label,
    run_walk_forward,
    select_feature_sleeve,
)


def evaluate_capacity_gates(
    metrics: dict[str, float | int],
    baseline: dict[str, float | int],
    gates: dict[str, float],
) -> dict[str, object]:
    baseline_return = float(baseline["total_return"])
    baseline_exposure = float(baseline["avg_gross_exposure"])
    return_retention = (
        float(metrics["total_return"]) / baseline_return if baseline_return > 0.0 else 0.0
    )
    exposure_retention = (
        float(metrics["avg_gross_exposure"]) / baseline_exposure
        if baseline_exposure > 0.0
        else 0.0
    )
    checks = {
        "total_return": float(metrics["total_return"]) >= float(gates["min_total_return"]),
        "annualized_return": float(metrics["annualized_return"])
        >= float(gates["min_annualized_return"]),
        "max_drawdown": float(metrics["max_drawdown"])
        >= float(gates["max_drawdown_floor"]),
        "total_return_retention": return_retention
        >= float(gates["min_total_return_retention_vs_baseline"]),
        "average_exposure_retention": exposure_retention
        >= float(gates["min_average_exposure_retention_vs_baseline"]),
    }
    return {
        "return_retention": return_retention,
        "exposure_retention": exposure_retention,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_control_metrics(
    observed: dict[str, float | int],
    registered: dict[str, float | int],
) -> None:
    keys = (
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "avg_gross_exposure",
    )
    for key in keys:
        if key not in observed or key not in registered:
            raise ValueError(f"control metric missing: {key}")
        if not np.isclose(
            float(observed[key]), float(registered[key]), rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(
                f"no-capacity control mismatch for {key}: "
                f"observed={observed[key]} registered={registered[key]}"
            )


def load_preregistration(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "baseline",
        "capacity_contract",
        "fixed_strategy",
        "scenarios",
        "capacity_acceptance_gates",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"capacity preregistration missing fields: {missing}")
    scenarios = raw["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("capacity scenarios must be a non-empty list")
    ids = [str(item["id"]) for item in scenarios]
    capitals = [float(item["initial_capital"]) for item in scenarios]
    if len(ids) != len(set(ids)) or len(capitals) != len(set(capitals)):
        raise ValueError("capacity scenarios must have unique ids and capital values")
    if any(capital <= 0.0 for capital in capitals):
        raise ValueError("capacity scenario capital must be positive")
    if capitals != sorted(capitals):
        raise ValueError("capacity scenarios must be ordered by initial capital")
    return raw


def _pct(value: object) -> str:
    return f"{float(value):.2%}"


def build_report(
    registration: dict[str, object],
    comparison: pd.DataFrame,
    largest_acceptable: dict[str, object] | None,
    control_metrics: dict[str, float | int],
) -> str:
    capacity = registration["capacity_contract"]
    lines = [
        "# 冠军策略点时容量压力",
        "",
        "状态：研究/模拟盘容量审计，不改变生产排序、仓位或执行配置。",
        "",
        f"- 成交额口径：信号日可见的 {int(capacity['amount_lookback_sessions'])} 日中位数",
        f"- 单票每日参与率上限：{_pct(capacity['max_daily_amount_participation'])}",
        "- 买卖双边均允许部分成交；缺失或非正成交额按零容量处理。",
        f"- 无容量 control 已对账：总收益 {_pct(control_metrics['total_return'])}，年化 {_pct(control_metrics['annualized_return'])}。",
        "",
        "## 结果",
        "",
        comparison.to_markdown(index=False),
        "",
        "## 结论",
        "",
    ]
    if largest_acceptable is None:
        lines.append("- 没有预登记资金规模通过全部容量门槛。")
    else:
        lines.append(
            f"- 预登记场景中的最大可接受资金规模为 {float(largest_acceptable['initial_capital']):,.0f} 元（{largest_acceptable['scenario_id']}）。"
        )
    lines.extend(
        [
            "- 该数字是历史成交额约束下的研究上限，不是实盘可承诺容量。",
            "- 容量压力只评估执行可行性，不能作为新 alpha 或收益提升证据。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incumbent amount-capacity stress.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registration = load_preregistration(Path(args.config))
    strategy = registration["fixed_strategy"]
    capacity = registration["capacity_contract"]

    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    validate_trading_sessions(
        close.index, Path(args.benchmark), context="capacity stress input panel"
    )
    open_px = clean_matrix(
        pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return
    )
    high = clean_matrix(
        pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return
    )
    low = clean_matrix(
        pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return
    )
    amount = pivot_prices(raw, "amount").reindex_like(close)

    all_features = build_features(close, open_px, high, low, amount)
    features, feature_directions = select_feature_sleeve(
        all_features, str(strategy["sleeve"])
    )
    horizons = tuple(int(value) for value in strategy["training_horizons"])
    realized_label = next_open_return_label(
        open_px, max_abs_daily_return=args.max_abs_daily_return
    )
    ic, _ = build_multi_horizon_ic(
        features, open_px, horizons, args.max_abs_daily_return
    )
    market_exposure = load_market_exposure(
        args.benchmark,
        close.index,
        ma_window=int(strategy["market_ma_window"]),
        risk_off_drawdown_20d=float(strategy["market_risk_off_drawdown_20d"]),
        below_ma_exposure=float(strategy["market_below_ma_exposure"]),
        crash_exposure=float(strategy["market_crash_exposure"]),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    control_equity, control_weights, control_trades = run_walk_forward(
        close=close,
        open_px=open_px,
        features=features,
        label=realized_label,
        ic=ic,
        train_days=int(strategy["train_days"]),
        retrain_frequency=int(strategy["retrain_frequency"]),
        top_n=int(strategy["top_n"]),
        rebalance_frequency=int(strategy["rebalance_frequency"]),
        max_position_weight=float(strategy["max_position_weight"]),
        leverage=float(strategy["leverage"]),
        commission_bps=float(strategy["commission_bps"]),
        impact_bps=float(strategy["impact_bps"]),
        max_buy_open_gap=0.06,
        limit_buffer=0.995,
        market_exposure=market_exposure,
        initial_capital=float(registration["baseline"]["initial_capital"]),
        max_training_horizon=max(horizons),
        feature_directions=feature_directions,
    )
    control_metrics = calculate_walk_forward_metrics(
        control_equity, float(registration["baseline"]["initial_capital"])
    )
    validate_control_metrics(control_metrics, registration["baseline"])
    control_dir = output_dir / "control_no_capacity"
    control_dir.mkdir(parents=True, exist_ok=True)
    control_equity.to_csv(
        control_dir / "equity_curve.csv", index=False, encoding="utf-8"
    )
    control_trades.to_csv(
        control_dir / "trade_audit.csv", index=False, encoding="utf-8"
    )
    (control_dir / "metrics.json").write_text(
        json.dumps(control_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    comparison_rows = []
    first_weights = control_weights
    for scenario in registration["scenarios"]:
        scenario_id = str(scenario["id"])
        initial_capital = float(scenario["initial_capital"])
        equity, weights, trades = run_walk_forward(
            close=close,
            open_px=open_px,
            features=features,
            label=realized_label,
            ic=ic,
            train_days=int(strategy["train_days"]),
            retrain_frequency=int(strategy["retrain_frequency"]),
            top_n=int(strategy["top_n"]),
            rebalance_frequency=int(strategy["rebalance_frequency"]),
            max_position_weight=float(strategy["max_position_weight"]),
            leverage=float(strategy["leverage"]),
            commission_bps=float(strategy["commission_bps"]),
            impact_bps=float(strategy["impact_bps"]),
            max_buy_open_gap=0.06,
            limit_buffer=0.995,
            market_exposure=market_exposure,
            initial_capital=initial_capital,
            max_training_horizon=max(horizons),
            feature_directions=feature_directions,
            amount=amount,
            capacity_lookback=int(capacity["amount_lookback_sessions"]),
            max_daily_amount_participation=float(
                capacity["max_daily_amount_participation"]
            ),
        )
        metrics = calculate_walk_forward_metrics(equity, initial_capital)
        gate = evaluate_capacity_gates(
            metrics,
            registration["baseline"],
            registration["capacity_acceptance_gates"],
        )
        requested_capacity_turnover = float(equity["turnover"].sum()) + float(
            equity["capacity_blocked_buy_weight"].sum()
            + equity["capacity_blocked_sell_weight"].sum()
        )
        fill_ratio = (
            float(equity["turnover"].sum()) / requested_capacity_turnover
            if requested_capacity_turnover > 0.0
            else 1.0
        )
        metrics.update(
            {
                "scenario_id": scenario_id,
                "capacity_fill_ratio": fill_ratio,
                "return_retention": gate["return_retention"],
                "exposure_retention": gate["exposure_retention"],
                "capacity_gate_checks": gate["checks"],
                "capacity_gate_passed": gate["passed"],
                "research_only": True,
                "trade_instruction": False,
            }
        )
        scenario_dir = output_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(scenario_dir / "equity_curve.csv", index=False, encoding="utf-8")
        trades.to_csv(scenario_dir / "trade_audit.csv", index=False, encoding="utf-8")
        (scenario_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        comparison_rows.append(
            {
                "scenario_id": scenario_id,
                "initial_capital": initial_capital,
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_like": metrics["sharpe_like"],
                "avg_gross_exposure": metrics["avg_gross_exposure"],
                "return_retention": gate["return_retention"],
                "exposure_retention": gate["exposure_retention"],
                "capacity_fill_ratio": fill_ratio,
                "capacity_limited_sessions": metrics.get(
                    "capacity_limited_symbols_sessions", 0
                ),
                "capacity_gate_passed": gate["passed"],
            }
        )

    if first_weights is not None:
        first_weights.to_csv(
            output_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8"
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    passed = comparison[comparison["capacity_gate_passed"]]
    largest = None if passed.empty else passed.sort_values("initial_capital").iloc[-1].to_dict()
    summary = {
        "schema_version": 1,
        "experiment_id": registration["experiment_id"],
        "status": "completed_research_only",
        "control_metrics": control_metrics,
        "largest_acceptable_scenario": largest,
        "scenario_count": int(len(comparison)),
        "research_only": True,
        "trade_instruction": False,
        "production_effect": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(registration, comparison, largest, control_metrics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
