"""Diagnose why the incumbent loses money outside its single positive year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_prices, pivot_prices


def compound_return(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return 0.0
    if values.le(-1.0).any():
        raise ValueError("returns must be greater than -100%")
    return float(np.expm1(np.log1p(values).sum()))


def max_drawdown_from_returns(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    nav = pd.concat(
        [pd.Series([1.0]), (1.0 + values).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    return float((nav / nav.cummax() - 1.0).min())


def classify_failure_mode(row: pd.Series) -> str:
    if float(row["strategy_return"]) >= 0.0:
        return "positive_year"
    if float(row["gross_log_return"]) > 0.0 and float(row["net_log_return"]) < 0.0:
        return "cost_erased_weak_gross_edge"
    if (
        float(row["selection_spread_mean"]) < 0.0
        and float(row["breadth_above_ma60_mean"]) < 0.5
    ):
        return "weak_breadth_selection_underperformance"
    return "unresolved_negative_year"


def validate_and_merge_equity(
    equity_path: Path,
    trades_path: Path,
    benchmark_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    equity = pd.read_csv(equity_path, parse_dates=["date"])
    trades = pd.read_csv(
        trades_path, parse_dates=["signal_date", "realize_date"]
    )
    if equity["date"].duplicated().any() or trades["realize_date"].duplicated().any():
        raise ValueError("equity and trade dates must be unique")
    if equity["date"].tolist() != trades["realize_date"].tolist():
        raise ValueError("equity dates do not align one-to-one with trade realize dates")

    benchmark = pd.read_csv(benchmark_path, parse_dates=["date"]).set_index("date")
    if benchmark.index.duplicated().any():
        raise ValueError("benchmark dates must be unique")
    benchmark_open_return = pd.to_numeric(
        benchmark["open"], errors="coerce"
    ).pct_change(fill_method=None)
    aligned_benchmark = benchmark_open_return.reindex(equity["date"])
    if aligned_benchmark.isna().any():
        raise ValueError("benchmark open returns do not cover every realized session")

    frame = equity.copy()
    frame["signal_date"] = trades["signal_date"].to_numpy()
    frame["net_return"] = frame["gross_return"] - frame["cost"]
    frame["benchmark_open_return"] = aligned_benchmark.to_numpy()
    frame["unit_stock_return"] = frame["gross_return"] / frame[
        "gross_exposure"
    ].replace(0.0, np.nan)
    frame["selection_spread"] = (
        frame["unit_stock_return"] - frame["benchmark_open_return"]
    )
    frame["year"] = frame["date"].dt.year
    quality = {
        "equity_rows": int(len(equity)),
        "trade_rows": int(len(trades)),
        "first_realize_date": equity["date"].min().strftime("%Y-%m-%d"),
        "last_realize_date": equity["date"].max().strftime("%Y-%m-%d"),
        "first_signal_date": trades["signal_date"].min().strftime("%Y-%m-%d"),
        "last_signal_date": trades["signal_date"].max().strftime("%Y-%m-%d"),
        "equity_trade_date_alignment": True,
        "benchmark_missing_sessions": 0,
    }
    return frame, quality


def build_breadth_panel(panel_path: Path) -> pd.DataFrame:
    raw = load_prices(panel_path, None, None)
    close = pivot_prices(raw, "close").sort_index()
    valid = close.notna()
    ma60 = close.rolling(60, min_periods=60).mean()
    return20 = close.pct_change(20, fill_method=None)
    daily_return = close.pct_change(fill_method=None)
    breadth = close.gt(ma60).where(valid & ma60.notna()).mean(axis=1)
    return pd.DataFrame(
        {
            "breadth_above_ma60": breadth,
            "cross_sectional_median_return20": return20.median(axis=1),
            "cross_sectional_return_dispersion": daily_return.std(axis=1),
            "available_symbols": valid.sum(axis=1),
        }
    )


def align_factor_diagnostics(
    frame: pd.DataFrame,
    weights_path: Path,
    daily_ic_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    weights = pd.read_csv(weights_path, parse_dates=["date"]).set_index("date")
    daily_ic = pd.read_csv(daily_ic_path, parse_dates=["date"]).set_index("date")
    if weights.index.duplicated().any() or daily_ic.index.duplicated().any():
        raise ValueError("weight and IC dates must be unique")
    features = [column for column in weights if column in daily_ic]
    if not features:
        raise ValueError("weights and daily IC have no common features")

    signal_dates = pd.DatetimeIndex(frame["signal_date"])
    aligned_weights = weights[features].reindex(signal_dates, method="ffill")
    aligned_ic = daily_ic[features].reindex(signal_dates)
    if aligned_weights.isna().all(axis=1).any() or aligned_ic.isna().all(axis=1).any():
        raise ValueError("weights and IC do not cover every strategy signal date")

    weighted_proxy = aligned_weights * aligned_ic
    daily = pd.DataFrame(
        {
            "signal_date": signal_dates,
            "realize_date": frame["date"].to_numpy(),
            "year": frame["year"].to_numpy(),
            "weighted_realized_ic": weighted_proxy.sum(axis=1, min_count=1).to_numpy(),
        }
    )
    factor_rows = []
    for year in sorted(frame["year"].unique()):
        mask = frame["year"].eq(year).to_numpy()
        for feature in features:
            factor_rows.append(
                {
                    "year": int(year),
                    "feature": feature,
                    "mean_weight": float(aligned_weights.loc[mask, feature].mean()),
                    "mean_realized_ic": float(aligned_ic.loc[mask, feature].mean()),
                    "mean_weighted_ic_proxy": float(
                        weighted_proxy.loc[mask, feature].mean()
                    ),
                    "positive_ic_ratio": float(aligned_ic.loc[mask, feature].gt(0).mean()),
                }
            )
    quality = {
        "rolling_weight_rows": int(len(weights)),
        "daily_ic_rows": int(len(daily_ic)),
        "common_feature_count": int(len(features)),
        "aligned_signal_rows": int(len(daily)),
        "missing_weight_signal_rows": 0,
        "missing_ic_signal_rows": 0,
    }
    return daily, pd.DataFrame(factor_rows), quality


def add_cycle_position(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cycle = -1
    positions = []
    for due in out["rebalance_due"].astype(bool):
        cycle = 0 if due else cycle + 1
        positions.append(cycle)
    out["days_since_rebalance"] = positions
    return out


def build_cycle_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, cycle), group in frame.groupby(["year", "days_since_rebalance"]):
        rows.append(
            {
                "year": int(year),
                "days_since_rebalance": int(cycle),
                "sessions": int(len(group)),
                "net_log_return": float(np.log1p(group["net_return"]).sum()),
                "avg_net_return": float(group["net_return"].mean()),
                "avg_unit_stock_return": float(group["unit_stock_return"].mean()),
                "avg_benchmark_open_return": float(
                    group["benchmark_open_return"].mean()
                ),
                "avg_selection_spread": float(group["selection_spread"].mean()),
                "win_rate": float(group["net_return"].gt(0.0).mean()),
                "avg_cost": float(group["cost"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_year_diagnostics(
    frame: pd.DataFrame,
    factor_daily: pd.DataFrame,
) -> pd.DataFrame:
    factor_by_year = factor_daily.groupby("year")["weighted_realized_ic"]
    rows = []
    for year, group in frame.groupby("year"):
        spread = group["selection_spread"].dropna()
        spread_t = (
            float(spread.mean() / spread.std(ddof=1) * np.sqrt(len(spread)))
            if len(spread) > 1 and spread.std(ddof=1) > 0.0
            else np.nan
        )
        row = {
            "year": int(year),
            "sessions": int(len(group)),
            "strategy_return": compound_return(group["net_return"]),
            "benchmark_open_return": compound_return(group["benchmark_open_return"]),
            "excess_log_return": float(
                np.log1p(group["net_return"]).sum()
                - np.log1p(group["benchmark_open_return"]).sum()
            ),
            "max_drawdown": max_drawdown_from_returns(group["net_return"]),
            "gross_log_return": float(np.log1p(group["gross_return"]).sum()),
            "net_log_return": float(np.log1p(group["net_return"]).sum()),
            "total_cost": float(group["cost"].sum()),
            "avg_gross_exposure": float(group["gross_exposure"].mean()),
            "avg_market_exposure_target": float(group["market_exposure"].mean()),
            "unit_stock_return_mean": float(group["unit_stock_return"].mean()),
            "selection_spread_mean": float(spread.mean()),
            "selection_spread_tstat": spread_t,
            "selection_spread_positive_ratio": float(spread.gt(0.0).mean()),
            "weighted_realized_ic_mean": float(factor_by_year.mean().loc[year]),
            "weighted_realized_ic_positive_ratio": float(
                factor_by_year.apply(lambda values: values.gt(0.0).mean()).loc[year]
            ),
            "breadth_above_ma60_mean": float(group["breadth_above_ma60"].mean()),
            "breadth_above_ma60_median": float(
                group["breadth_above_ma60"].median()
            ),
            "weak_breadth_session_ratio": float(
                group["breadth_above_ma60"].lt(0.5).mean()
            ),
            "cross_sectional_median_return20_mean": float(
                group["cross_sectional_median_return20"].mean()
            ),
            "cross_sectional_return_dispersion_mean": float(
                group["cross_sectional_return_dispersion"].mean()
            ),
            "weak_breadth_high_exposure_ratio": float(
                (
                    group["breadth_above_ma60"].lt(0.5)
                    & group["market_exposure"].ge(0.6)
                ).mean()
            ),
        }
        row["ic_portfolio_translation_gap"] = bool(
            row["weighted_realized_ic_mean"] > 0.0
            and row["strategy_return"] < 0.0
        )
        row["failure_mode"] = classify_failure_mode(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(years: pd.DataFrame, cycles: pd.DataFrame) -> dict[str, object]:
    negative = years[years["strategy_return"].lt(0.0)]
    positive = years[years["strategy_return"].ge(0.0)]
    cycle_all = cycles.groupby("days_since_rebalance").agg(
        sessions=("sessions", "sum"),
        net_log_return=("net_log_return", "sum"),
        avg_selection_spread=("avg_selection_spread", "mean"),
    )
    return {
        "schema_version": 1,
        "status": "research_only_diagnostic",
        "negative_years": negative["year"].astype(int).tolist(),
        "positive_years": positive["year"].astype(int).tolist(),
        "all_negative_years_breadth_below_half": bool(
            negative["breadth_above_ma60_mean"].lt(0.5).all()
        ),
        "all_negative_years_nonpositive_cross_sectional_momentum": bool(
            negative["cross_sectional_median_return20_mean"].le(0.0).all()
        ),
        "all_negative_years_positive_weighted_ic": bool(
            negative["weighted_realized_ic_mean"].gt(0.0).all()
        ),
        "ic_translation_gap_years": negative.loc[
            negative["ic_portfolio_translation_gap"], "year"
        ].astype(int).tolist(),
        "cost_erased_edge_years": negative.loc[
            negative["failure_mode"].eq("cost_erased_weak_gross_edge"), "year"
        ].astype(int).tolist(),
        "weak_breadth_selection_underperformance_years": negative.loc[
            negative["failure_mode"].eq(
                "weak_breadth_selection_underperformance"
            ),
            "year",
        ].astype(int).tolist(),
        "cycle_diagnostics": cycle_all.reset_index().to_dict(orient="records"),
        "main_conclusion": (
            "The incumbent is breadth-dependent: positive broad-market rank IC "
            "does not reliably monetize in the concentrated top-40 portfolio when "
            "the investable universe has weak breadth. The 510300 risk proxy can "
            "remain permissive during narrow large-cap rallies."
        ),
        "production_effect": False,
        "trade_instruction": False,
    }


def build_markdown_report(
    years: pd.DataFrame,
    cycles: pd.DataFrame,
    summary: dict[str, object],
) -> str:
    display = years[
        [
            "year",
            "strategy_return",
            "benchmark_open_return",
            "breadth_above_ma60_mean",
            "cross_sectional_median_return20_mean",
            "weighted_realized_ic_mean",
            "selection_spread_mean",
            "avg_gross_exposure",
            "total_cost",
            "failure_mode",
        ]
    ]
    cycle_display = cycles.groupby("days_since_rebalance", as_index=False).agg(
        sessions=("sessions", "sum"),
        net_log_return=("net_log_return", "sum"),
        avg_selection_spread=("avg_selection_spread", "mean"),
    )
    lines = [
        "# 冠军策略亏损年份诊断",
        "",
        "状态：研究诊断，不改变生产排序、仓位或风险配置。",
        "",
        "## 核心结论",
        "",
        "- 2023、2024、2026 的共同点不是同一种指数行情，而是可投资股票池宽度偏弱：三年的60日均线宽度均值都低于50%，截面20日中位收益均不为正。",
        "- 滚动权重对应的全截面加权IC在三个亏损年份仍为正，但没有稳定转化为Top-40组合收益，说明RankIC不能代替组合尾部兑现验证。",
        "- 2024和2026属于弱宽度下的选股收益不足；2023有微弱毛收益，但被成本完全吞噬。",
        "- 510300风险代理与实际股票池存在口径错配：大盘或目标暴露仍偏高时，广度较弱的股票池可能继续承压。",
        "",
        "## 年度证据",
        "",
        display.to_markdown(index=False),
        "",
        "## 调仓周期证据",
        "",
        cycle_display.to_markdown(index=False),
        "",
        "## 决策",
        "",
        "- 保持当前冠军不变，不在已观察样本上追加宽度阈值或因子权重调参。",
        "- 将现有动态宽度overlay继续作为前向影子观察，重点验证弱宽度且510300风险信号偏乐观的会话。",
        "- 日报必须同时展示全市场宽度、截面20日中位收益、实际总仓位和510300目标暴露，避免把大盘状态当作股票池状态。",
        "- 新候选的验收必须加入Top-N组合兑现指标，不能只看全截面RankIC。",
        "",
        "## 限制",
        "",
        "- 因子乘积是权重与事后IC的诊断代理，不是逐因子利润归因。",
        "- 2026仅统计至2026-07-22，不能与完整自然年等量比较。",
        "- 本报告使用已观察历史，只能解释失败模式，不能证明新风控规则有效。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose incumbent failure years.")
    parser.add_argument("--equity", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--daily-ic", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, source_quality = validate_and_merge_equity(
        Path(args.equity), Path(args.trades), Path(args.benchmark)
    )
    breadth = build_breadth_panel(Path(args.data))
    signal_breadth = breadth.reindex(pd.DatetimeIndex(frame["signal_date"]))
    if signal_breadth.isna().any(axis=None):
        raise ValueError("breadth diagnostics do not cover every strategy signal date")
    for column in signal_breadth:
        frame[column] = signal_breadth[column].to_numpy()

    factor_daily, factor_year, factor_quality = align_factor_diagnostics(
        frame, Path(args.weights), Path(args.daily_ic)
    )
    frame = add_cycle_position(frame)
    cycles = build_cycle_diagnostics(frame)
    years = build_year_diagnostics(frame, factor_daily)
    summary = build_summary(years, cycles)
    source_quality.update(factor_quality)
    source_quality.update(
        {
            "breadth_signal_rows": int(len(signal_breadth)),
            "breadth_missing_signal_rows": 0,
            "panel_first_date": breadth.index.min().strftime("%Y-%m-%d"),
            "panel_last_date": breadth.index.max().strftime("%Y-%m-%d"),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    years.to_csv(output_dir / "year_diagnostics.csv", index=False, encoding="utf-8-sig")
    cycles.to_csv(output_dir / "cycle_diagnostics.csv", index=False, encoding="utf-8-sig")
    factor_year.to_csv(
        output_dir / "factor_year_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    factor_daily.to_csv(
        output_dir / "weighted_ic_daily.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "source_quality.json").write_text(
        json.dumps(source_quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_markdown_report(years, cycles, summary)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
