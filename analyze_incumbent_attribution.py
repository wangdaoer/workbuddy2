"""Research-only attribution for the strict daily rank-model incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import max_drawdown, sharpe_like
from train_next_open_rank_model import calculate_walk_forward_metrics


REQUIRED_EQUITY_COLUMNS = {
    "date",
    "equity",
    "gross_return",
    "cost",
    "turnover",
    "gross_exposure",
    "market_exposure",
}


def classify_market_regime(exposure: float) -> str:
    value = float(exposure)
    if value <= 1e-12:
        return "drawdown_20d_crash"
    if value >= 0.99:
        return "risk_on"
    return "below_ma"


def load_equity_curve(
    path: Path,
    initial_capital: float,
    rebalance_frequency: int = 5,
) -> pd.DataFrame:
    if rebalance_frequency < 1:
        raise ValueError("rebalance frequency must be positive")
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = sorted(REQUIRED_EQUITY_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"equity curve missing columns: {missing}")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("equity curve is empty")
    if frame["date"].duplicated().any():
        raise ValueError("equity dates must be unique")

    numeric = sorted(REQUIRED_EQUITY_COLUMNS.difference({"date"}))
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("equity curve contains non-finite values")
    frame["net_return"] = frame["gross_return"] - frame["cost"]
    if frame["net_return"].le(-1.0).any():
        raise ValueError("net return must be greater than -100%")

    previous = frame["equity"].shift(1)
    previous.iloc[0] = float(initial_capital)
    implied = previous * (1.0 + frame["net_return"])
    if not np.allclose(implied, frame["equity"], rtol=1e-9, atol=1e-5):
        raise ValueError("equity does not reconcile with gross return minus cost")
    frame["year"] = frame["date"].dt.year.astype(str)
    frame["market_regime"] = frame["market_exposure"].map(classify_market_regime)
    frame["rebalance_due"] = np.arange(len(frame)) % rebalance_frequency == 0
    return frame


def _segment_row(group: pd.DataFrame) -> dict[str, object]:
    returns = group["net_return"]
    nav = (1.0 + returns).cumprod()
    return {
        "sessions": int(len(group)),
        "total_return": float(nav.iloc[-1] - 1.0),
        "log_return_contribution": float(np.log1p(returns).sum()),
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
        "win_rate": float(returns.gt(0.0).mean()),
        "avg_net_return": float(returns.mean()),
        "avg_turnover": float(group["turnover"].mean()),
        "total_cost": float(group["cost"].sum()),
        "avg_gross_exposure": float(group["gross_exposure"].mean()),
        "avg_market_exposure_target": float(group["market_exposure"].mean()),
    }


def segment_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(grouper, sort=True, observed=True):
        values = (key,) if len(group_columns) == 1 else tuple(key)
        row = {name: str(value) for name, value in zip(group_columns, values)}
        row.update(_segment_row(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_attribution_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "by_year": segment_metrics(frame, ["year"]),
        "by_market_regime": segment_metrics(frame, ["market_regime"]),
        "by_year_and_regime": segment_metrics(frame, ["year", "market_regime"]),
    }


def summarize_feature_weights(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if frame.empty or "date" not in frame.columns:
        raise ValueError("rolling feature weights must include dated rows")
    feature_columns = [column for column in frame.columns if column != "date"]
    if not feature_columns:
        raise ValueError("rolling feature weights contain no features")

    rows = []
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"feature weight contains non-finite values: {feature}")
        signs = np.sign(values.to_numpy(dtype=float))
        nonzero_signs = signs[signs != 0]
        sign_flips = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
        rows.append(
            {
                "feature": feature,
                "retrain_observations": int(len(values)),
                "mean_weight": float(values.mean()),
                "mean_abs_weight": float(values.abs().mean()),
                "positive_weight_ratio": float(values.gt(0.0).mean()),
                "negative_weight_ratio": float(values.lt(0.0).mean()),
                "sign_flip_count": sign_flips,
                "latest_weight": float(values.iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_abs_weight", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def build_flow_readiness(
    main_flow: dict[str, object],
    institutional: dict[str, object],
) -> dict[str, object]:
    source_sessions = int(main_flow.get("available_source_sessions", 0))
    source_minimum = int(main_flow.get("minimum_history_sessions", 0))
    completed = int(institutional.get("primary_completed_samples", 0))
    required = int(institutional.get("minimum_completed_samples", 0))
    return {
        "main_net_volume": {
            "status": main_flow.get("status"),
            "source_first_date": main_flow.get("source_first_date"),
            "source_latest_date": main_flow.get("source_latest_date"),
            "available_source_sessions": source_sessions,
            "minimum_history_sessions": source_minimum,
            "minimum_history_ready": source_sessions >= source_minimum,
            "source_coverage": main_flow.get("latest_source_coverage"),
            "selection_effect": bool(main_flow.get("selection_effect", False)),
        },
        "institutional_accumulation": {
            "status": institutional.get("status"),
            "validation_start_date": institutional.get("validation_start_date"),
            "primary_horizon": institutional.get("primary_horizon"),
            "primary_completed_samples": completed,
            "minimum_completed_samples": required,
            "remaining_completed_samples": max(required - completed, 0),
            "gate_evaluation_allowed": bool(
                institutional.get("gate_evaluation_allowed", False)
            ),
            "promotion_allowed": bool(institutional.get("promotion_allowed", False)),
        },
    }


def build_risk_flags(
    frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
) -> dict[str, object]:
    years = tables["by_year"]
    regimes = tables["by_market_regime"]
    positive_logs = years.loc[
        years["log_return_contribution"].gt(0.0), "log_return_contribution"
    ]
    dominant_positive_share = (
        float(positive_logs.max() / positive_logs.sum()) if not positive_logs.empty else None
    )
    positive_year_ratio = float(years["total_return"].gt(0.0).mean())
    total_log_return = float(np.log1p(frame["net_return"]).sum())
    risk_on_rows = regimes[regimes["market_regime"].eq("risk_on")]
    risk_on_log_share = None
    if total_log_return > 0.0 and not risk_on_rows.empty:
        risk_on_log_share = float(
            risk_on_rows.iloc[0]["log_return_contribution"] / total_log_return
        )

    liquidity = weights[weights["feature"].eq("liquidity_20")]
    low_liquidity_tilt = False
    if not liquidity.empty:
        row = liquidity.iloc[0]
        low_liquidity_tilt = bool(
            row["mean_weight"] < 0.0 and row["negative_weight_ratio"] >= 0.8
        )

    risk_off_with_positions = frame[
        frame["market_exposure"].le(1e-12) & frame["gross_exposure"].gt(1e-12)
    ]
    deferred = risk_off_with_positions[~risk_off_with_positions["rebalance_due"]]
    residual = risk_off_with_positions[risk_off_with_positions["rebalance_due"]]
    return {
        "positive_year_ratio": positive_year_ratio,
        "dominant_positive_year_share": dominant_positive_share,
        "year_concentration_flag": bool(
            positive_year_ratio < 0.5
            or (dominant_positive_share is not None and dominant_positive_share > 0.7)
        ),
        "risk_on_log_return_share": risk_on_log_share,
        "risk_on_concentration_flag": bool(
            risk_on_log_share is not None and risk_on_log_share > 0.8
        ),
        "persistent_low_liquidity_tilt_flag": low_liquidity_tilt,
        "risk_off_with_positions_sessions": int(len(risk_off_with_positions)),
        "risk_off_non_rebalance_sessions": int(len(deferred)),
        "risk_off_non_rebalance_dates": deferred["date"].dt.strftime("%Y-%m-%d").tolist(),
        "risk_off_rebalance_residual_sessions": int(len(residual)),
        "risk_off_rebalance_residual_dates": residual["date"].dt.strftime("%Y-%m-%d").tolist(),
    }


def _pct(value: object) -> str:
    return f"{float(value):.2%}"


def build_report(
    metrics: dict[str, object],
    tables: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    flow: dict[str, object],
    risk_flags: dict[str, object],
) -> str:
    years = tables["by_year"].copy()
    regimes = tables["by_market_regime"].copy()
    best_year = years.loc[years["total_return"].idxmax()]
    worst_year = years.loc[years["total_return"].idxmin()]
    positive_years = int(years["total_return"].gt(0.0).sum())
    positive_logs = years.loc[years["log_return_contribution"].gt(0.0), "log_return_contribution"]
    dominant_positive_share = (
        float(positive_logs.max() / positive_logs.sum()) if not positive_logs.empty else np.nan
    )
    best_regime = regimes.loc[regimes["log_return_contribution"].idxmax()]
    worst_regime = regimes.loc[regimes["log_return_contribution"].idxmin()]
    top_weights = weights.head(8)[
        ["feature", "mean_weight", "mean_abs_weight", "positive_weight_ratio", "sign_flip_count"]
    ]
    main_flow = flow["main_net_volume"]
    institution = flow["institutional_accumulation"]

    lines = [
        "# 冠军策略收益来源归因",
        "",
        "状态：研究/模拟盘归因，不改变排序、仓位或生产配置。",
        "",
        "## 全周期",
        "",
        f"- 总收益：{_pct(metrics['total_return'])}",
        f"- 年化收益：{_pct(metrics['annualized_return'])}",
        f"- 最大回撤：{_pct(metrics['max_drawdown'])}",
        f"- Sharpe-like：{float(metrics['sharpe_like']):.3f}",
        "",
        "## 核心发现",
        "",
        f"- 正收益年份：{positive_years}/{len(years)}；最好年份为 {best_year['year']}（{_pct(best_year['total_return'])}），最差年份为 {worst_year['year']}（{_pct(worst_year['total_return'])}）。",
        f"- 最大正对数收益年份占全部正对数收益的 {_pct(dominant_positive_share)}；该值只衡量正收益来源集中度。",
        f"- 贡献最强市场状态：{best_regime['market_regime']}；条件复利收益 {_pct(best_regime['total_return'])}。",
        f"- 贡献最弱市场状态：{worst_regime['market_regime']}；条件复利收益 {_pct(worst_regime['total_return'])}。",
        "- 滚动因子权重是打分倾向，不是逐因子利润贡献；没有持仓级反事实重放时不得混称。",
        "",
        "## 风险标记",
        "",
        f"- 年度收益集中：{'是' if risk_flags['year_concentration_flag'] else '否'}；正收益年份比例 {_pct(risk_flags['positive_year_ratio'])}。",
        f"- risk_on 对数收益占全周期净对数收益 {_pct(risk_flags['risk_on_log_return_share'])}，集中标记为 {'是' if risk_flags['risk_on_concentration_flag'] else '否'}。",
        f"- 持续低流动性倾向：{'是' if risk_flags['persistent_low_liquidity_tilt_flag'] else '否'}。",
        f"- 风险信号为零但处于非调仓日而继续持仓：{risk_flags['risk_off_non_rebalance_sessions']} 个交易日（{', '.join(risk_flags['risk_off_non_rebalance_dates']) or '无'}）。",
        f"- 风险信号为零且处于调仓日后仍有仓位：{risk_flags['risk_off_rebalance_residual_sessions']} 个交易日（{', '.join(risk_flags['risk_off_rebalance_residual_dates']) or '无'}）；只有这类记录才可能进一步审查成交阻塞。",
        "",
        "## 年度归因",
        "",
        years.to_markdown(index=False),
        "",
        "## 市场状态归因",
        "",
        regimes.to_markdown(index=False),
        "",
        "## 年度与市场状态交叉",
        "",
        tables["by_year_and_regime"].to_markdown(index=False),
        "",
        "## 滚动权重诊断",
        "",
        top_weights.to_markdown(index=False),
        "",
        "## 资金流前向成熟度",
        "",
        f"- 主力净量源数据：{main_flow['available_source_sessions']} 个交易日，最低滚动要求 {main_flow['minimum_history_sessions']} 个交易日，当前字段级状态为 {main_flow['status']}。",
        f"- 机构吸筹主周期完成样本：{institution['primary_completed_samples']}/{institution['minimum_completed_samples']}，仍需 {institution['remaining_completed_samples']} 个；当前不得评估研究门槛。",
        "- 字段达到最低滚动天数不等于策略验证成熟，资金流信号继续保持零选股、零权重影响。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute incumbent returns by year and market state.")
    parser.add_argument("--equity", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--main-flow-metadata", required=True)
    parser.add_argument("--institutional-tracking-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-frequency", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    equity = load_equity_curve(
        Path(args.equity),
        args.initial_capital,
        rebalance_frequency=args.rebalance_frequency,
    )
    tables = build_attribution_tables(equity)
    weights = summarize_feature_weights(Path(args.weights))
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics.update(calculate_walk_forward_metrics(equity, args.initial_capital))
    flow = build_flow_readiness(
        json.loads(Path(args.main_flow_metadata).read_text(encoding="utf-8")),
        json.loads(Path(args.institutional_tracking_summary).read_text(encoding="utf-8")),
    )
    risk_flags = build_risk_flags(equity, tables, weights)

    total_log_return = float(np.log1p(equity["net_return"]).sum())
    regime_log_return = float(tables["by_market_regime"]["log_return_contribution"].sum())
    if not np.isclose(total_log_return, regime_log_return, rtol=1e-12, atol=1e-12):
        raise RuntimeError("market-regime log contributions do not reconcile")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(output_dir / "feature_weight_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": 1,
        "status": "research_only",
        "trade_instruction": False,
        "production_effect": False,
        "total_log_return": total_log_return,
        "regime_log_return_reconciliation": regime_log_return,
        "performance_metric_policy": "initial_capital_and_complete_net_return_series",
        "flow_readiness": flow,
        "risk_flags": risk_flags,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(metrics, tables, weights, flow, risk_flags)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
