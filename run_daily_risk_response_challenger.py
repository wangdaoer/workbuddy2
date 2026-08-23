"""Run the preregistered daily decrease-only risk-response challenger."""

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


def load_preregistration(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "baseline",
        "candidate_contract",
        "fixed_strategy",
        "sequential_acceptance_gates",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"daily risk preregistration missing fields: {missing}")
    if raw["candidate_contract"].get("risk_rebalance_policy") != "daily_decrease_only":
        raise ValueError("candidate risk policy must be daily_decrease_only")
    return raw


def validate_control_metrics(
    observed: dict[str, float | int],
    registered: dict[str, float | int],
) -> None:
    keys = (
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "avg_turnover",
        "avg_gross_exposure",
        "risk_zero_target_non_rebalance_with_positions_sessions",
    )
    for key in keys:
        if key not in observed or key not in registered:
            raise ValueError(f"daily risk control metric missing: {key}")
        if not np.isclose(
            float(observed[key]), float(registered[key]), rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(
                f"daily risk control mismatch for {key}: "
                f"observed={observed[key]} registered={registered[key]}"
            )


def evaluate_daily_risk_gates(
    candidate: dict[str, float | int],
    baseline: dict[str, float | int],
    gates: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    operational = gates["operational"]
    performance = gates["performance"]
    return_retention = float(candidate["total_return"]) / float(
        baseline["total_return"]
    )
    sharpe_retention = float(candidate["sharpe_like"]) / float(
        baseline["sharpe_like"]
    )
    turnover_increase = float(candidate["avg_turnover"]) - float(
        baseline["avg_turnover"]
    )
    drawdown_degradation = max(
        float(baseline["max_drawdown"]) - float(candidate["max_drawdown"]),
        0.0,
    )
    checks = {
        "risk_rebalance_attempts": int(candidate["risk_rebalance_attempt_sessions"])
        >= int(operational["min_risk_rebalance_attempt_sessions"]),
        "zero_target_unattempted": int(
            candidate["risk_zero_target_unattempted_sessions"]
        )
        <= int(operational["max_unattempted_zero_target_sessions"]),
        "zero_target_unexplained_residual": int(
            candidate["risk_zero_target_unexplained_residual_sessions"]
        )
        <= int(operational["max_unexplained_zero_target_residual_sessions"]),
        "total_return_retention": return_retention
        >= float(performance["min_total_return_retention_vs_baseline"]),
        "annualized_return": float(candidate["annualized_return"])
        >= float(performance["min_annualized_return"]),
        "max_drawdown_degradation": drawdown_degradation
        <= float(performance["max_drawdown_degradation"]),
        "sharpe_retention": sharpe_retention
        >= float(performance["min_sharpe_retention_vs_baseline"]),
        "average_turnover_increase": turnover_increase
        <= float(performance["max_average_turnover_increase"]),
    }
    return {
        "return_retention": return_retention,
        "sharpe_retention": sharpe_retention,
        "turnover_increase": turnover_increase,
        "drawdown_degradation": drawdown_degradation,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_relative_phase_attribution(
    control_equity: pd.DataFrame,
    candidate_equity: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "date",
        "gross_return",
        "cost",
        "scheduled_rebalance_due",
        "risk_rebalance_due",
    }
    if not required.issubset(control_equity) or not required.issubset(candidate_equity):
        raise ValueError("relative phase attribution is missing required columns")
    if control_equity["date"].astype(str).tolist() != candidate_equity[
        "date"
    ].astype(str).tolist():
        raise ValueError("control and candidate equity dates do not align")

    control_net = control_equity["gross_return"] - control_equity["cost"]
    candidate_net = candidate_equity["gross_return"] - candidate_equity["cost"]
    relative_log_return = np.log1p(candidate_net) - np.log1p(control_net)
    waiting = False
    phases = []
    for scheduled, attempted in zip(
        candidate_equity["scheduled_rebalance_due"].astype(bool),
        candidate_equity["risk_rebalance_due"].astype(bool),
    ):
        if scheduled:
            waiting = False
        if attempted:
            waiting = True
        phases.append(
            "risk_execution_attempt"
            if attempted
            else ("post_attempt_wait" if waiting else "normal")
        )
    return (
        pd.DataFrame({"phase": phases, "relative_log_return": relative_log_return})
        .groupby("phase", sort=False)
        .agg(
            sessions=("relative_log_return", "size"),
            relative_log_return=("relative_log_return", "sum"),
        )
        .reset_index()
    )


def _pct(value: object) -> str:
    return f"{float(value):.2%}"


def build_report(
    control: dict[str, float | int],
    candidate: dict[str, float | int],
    gate: dict[str, object],
) -> str:
    verdict = "通过历史门槛，可进入前向影子观察" if gate["passed"] else "未通过历史门槛，拒绝"
    disposition = (
        "- 该结果只允许进入前向影子观察，不能自动替换生产策略。"
        if gate["passed"]
        else "- 候选不得进入影子池或生产配置；不得在本样本内调参救回。"
    )
    checks = pd.DataFrame(
        [
            {"gate": name, "passed": passed}
            for name, passed in gate["checks"].items()
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "variant": "scheduled_only_control",
                "total_return": control["total_return"],
                "annualized_return": control["annualized_return"],
                "max_drawdown": control["max_drawdown"],
                "sharpe_like": control["sharpe_like"],
                "avg_turnover": control["avg_turnover"],
                "risk_attempt_sessions": control.get(
                    "risk_rebalance_attempt_sessions", 0
                ),
                "risk_trigger_sessions": control.get(
                    "risk_exposure_decrease_sessions", 0
                ),
                "zero_target_unattempted": control.get(
                    "risk_zero_target_unattempted_sessions", 0
                ),
            },
            {
                "variant": "daily_decrease_only",
                "total_return": candidate["total_return"],
                "annualized_return": candidate["annualized_return"],
                "max_drawdown": candidate["max_drawdown"],
                "sharpe_like": candidate["sharpe_like"],
                "avg_turnover": candidate["avg_turnover"],
                "risk_attempt_sessions": candidate[
                    "risk_rebalance_attempt_sessions"
                ],
                "risk_trigger_sessions": candidate.get(
                    "risk_exposure_decrease_sessions", 0
                ),
                "zero_target_unattempted": candidate[
                    "risk_zero_target_unattempted_sessions"
                ],
            },
        ]
    )
    lines = [
        "# 每日只减仓风险响应 Challenger",
        "",
        "状态：研究/模拟盘审计，不改变生产配置。",
        "",
        "- 选股、因子、五日正常调仓、成本和开盘成交约束均与冠军一致。",
        "- 非正常调仓日只允许风险降仓，不允许恢复仓位或新增买入。",
        "- 风险目标恢复后等待下一次正常调仓。",
        "",
        "## 对照",
        "",
        comparison.to_markdown(index=False),
        "",
        "## 冻结门槛",
        "",
        checks.to_markdown(index=False),
        "",
        "## 结论",
        "",
        f"- {verdict}。",
        f"- 总收益留存：{_pct(gate['return_retention'])}；Sharpe 留存：{_pct(gate['sharpe_retention'])}。",
        f"- 平均换手增量：{_pct(gate['turnover_increase'])}；最大回撤恶化：{_pct(gate['drawdown_degradation'])}。",
        disposition,
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily risk-response challenger.")
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

    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    validate_trading_sessions(
        close.index, Path(args.benchmark), context="daily risk input panel"
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

    common = {
        "close": close,
        "open_px": open_px,
        "features": features,
        "label": realized_label,
        "ic": ic,
        "train_days": int(strategy["train_days"]),
        "retrain_frequency": int(strategy["retrain_frequency"]),
        "top_n": int(strategy["top_n"]),
        "rebalance_frequency": int(strategy["rebalance_frequency"]),
        "max_position_weight": float(strategy["max_position_weight"]),
        "leverage": float(strategy["leverage"]),
        "commission_bps": float(strategy["commission_bps"]),
        "impact_bps": float(strategy["impact_bps"]),
        "max_buy_open_gap": 0.06,
        "limit_buffer": 0.995,
        "market_exposure": market_exposure,
        "initial_capital": float(registration["baseline"]["initial_capital"]),
        "max_training_horizon": max(horizons),
        "feature_directions": feature_directions,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]] = {}
    for policy in ("scheduled_only", "daily_decrease_only"):
        equity, weights, trades = run_walk_forward(
            **common, risk_rebalance_policy=policy
        )
        metrics = calculate_walk_forward_metrics(
            equity, float(registration["baseline"]["initial_capital"])
        )
        metrics["risk_rebalance_policy"] = policy
        results[policy] = (equity, weights, trades, metrics)
        variant_dir = output_dir / policy
        variant_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(variant_dir / "equity_curve.csv", index=False, encoding="utf-8")
        trades.to_csv(variant_dir / "trade_audit.csv", index=False, encoding="utf-8")
        (variant_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if policy == "scheduled_only":
            validate_control_metrics(metrics, registration["baseline"])

        events = trades[
            trades["risk_exposure_decrease_event"] | trades["risk_rebalance_due"]
        ]
        events.to_csv(
            variant_dir / "risk_response_events.csv", index=False, encoding="utf-8"
        )

    control = results["scheduled_only"][3]
    candidate = results["daily_decrease_only"][3]
    phase_attribution = build_relative_phase_attribution(
        results["scheduled_only"][0], results["daily_decrease_only"][0]
    )
    phase_attribution.to_csv(
        output_dir / "relative_phase_attribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate = evaluate_daily_risk_gates(
        candidate,
        registration["baseline"],
        registration["sequential_acceptance_gates"],
    )
    candidate.update(
        {
            "gate_result": gate,
            "qualified_for_prospective_shadow": bool(gate["passed"]),
            "production_effect": False,
            "trade_instruction": False,
        }
    )
    candidate_dir = output_dir / "daily_decrease_only"
    (candidate_dir / "metrics.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results["scheduled_only"][1].to_csv(
        output_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8"
    )

    summary = {
        "schema_version": 1,
        "experiment_id": registration["experiment_id"],
        "status": "completed_research_only",
        "control_metrics": control,
        "candidate_metrics": candidate,
        "relative_phase_attribution": phase_attribution.to_dict(orient="records"),
        "gate_result": gate,
        "qualified_for_prospective_shadow": bool(gate["passed"]),
        "production_effect": False,
        "trade_instruction": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(control, candidate, gate)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
