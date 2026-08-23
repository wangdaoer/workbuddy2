"""Stability report for core risk filter finalist equity curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import max_drawdown, sharpe_like


DEFAULT_SCHEMES = [
    (
        "原始组合",
        "outputs/high_return_v2/core_satellite_overlay_core93_sat15_total93_max029_rerun_20220101_20260707",
    ),
    (
        "进攻版_1000万_最多3次大跌",
        "outputs/high_return_v2/core_satellite_corefilter_liq10_down3_sat_exit0875_strong_20220101_20260707",
    ),
    (
        "平衡版_2000万_最多3次大跌",
        "outputs/high_return_v2/core_satellite_corefilter_liq20_down3_sat_exit0875_strong_20220101_20260707",
    ),
    (
        "稳健版_2000万_最多1次大跌",
        "outputs/high_return_v2/core_satellite_corefilter_liq20_down1_sat_exit0875_strong_20220101_20260707",
    ),
]


def compute_returns(equity: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    out = equity[["date", "equity"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["date", "equity"]).sort_values("date").reset_index(drop=True)
    prev = out["equity"].shift(1)
    if not prev.empty:
        prev.iloc[0] = float(initial_capital)
    out["daily_return"] = out["equity"] / (prev + 1e-12) - 1.0
    return out


def _compound_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def _window_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float(max_drawdown(nav))


def rolling_window_stats(returns: pd.DataFrame, window: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for idx in range(window - 1, len(returns)):
        window_frame = returns.iloc[idx - window + 1 : idx + 1]
        rows.append(
            {
                "end_date": window_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
                "return": _compound_return(window_frame["daily_return"]),
                "max_drawdown": _window_drawdown(window_frame["daily_return"]),
            }
        )
    key = f"rolling_{window}"
    if not rows:
        return {
            f"{key}_positive_rate": np.nan,
            f"{key}_worst_return": np.nan,
            f"{key}_worst_end_date": "",
            f"{key}_worst_drawdown": np.nan,
        }
    frame = pd.DataFrame(rows)
    worst_return_row = frame.loc[frame["return"].idxmin()]
    return {
        f"{key}_positive_rate": float(frame["return"].gt(0).mean()),
        f"{key}_worst_return": float(worst_return_row["return"]),
        f"{key}_worst_end_date": str(worst_return_row["end_date"]),
        f"{key}_worst_drawdown": float(frame["max_drawdown"].min()),
    }


def _period_labels(dates: pd.Series, period_type: str) -> pd.Series:
    if period_type == "year":
        return dates.dt.year.astype(str)
    if period_type == "half_year":
        half = np.where(dates.dt.month.le(6), "H1", "H2")
        return dates.dt.year.astype(str) + half
    raise ValueError(f"unsupported period_type: {period_type}")


def build_segment_metrics(returns: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_type in ("year", "half_year"):
        enriched = returns.copy()
        enriched["period"] = _period_labels(enriched["date"], period_type)
        for period, group in enriched.groupby("period", sort=True):
            period_returns = group["daily_return"]
            rows.append(
                {
                    "scheme": scheme,
                    "period_type": period_type,
                    "period": str(period),
                    "days": int(len(group)),
                    "return": _compound_return(period_returns),
                    "max_drawdown": _window_drawdown(period_returns),
                    "sharpe_like": float(sharpe_like(period_returns.fillna(0.0))),
                    "win_rate": float(period_returns.gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def compare_segments_to_baseline(segments: pd.DataFrame, baseline_scheme: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_type, type_frame in segments.groupby("period_type", sort=True):
        baseline = type_frame[type_frame["scheme"] == baseline_scheme][["period", "return", "max_drawdown"]]
        for scheme, group in type_frame.groupby("scheme", sort=True):
            if scheme == baseline_scheme:
                continue
            merged = group.merge(baseline, on="period", suffixes=("", "_baseline"))
            if merged.empty:
                continue
            excess = merged["return"] - merged["return_baseline"]
            drawdown_diff = merged["max_drawdown"] - merged["max_drawdown_baseline"]
            rows.append(
                {
                    "scheme": scheme,
                    "period_type": str(period_type),
                    "period_count": int(len(merged)),
                    "return_win_count": int(excess.gt(0).sum()),
                    "return_win_rate": float(excess.gt(0).mean()),
                    "avg_excess_return": float(excess.mean()),
                    "worst_excess_return": float(excess.min()),
                    "drawdown_improved_count": int(drawdown_diff.gt(0).sum()),
                    "drawdown_improved_rate": float(drawdown_diff.gt(0).mean()),
                    "avg_drawdown_diff": float(drawdown_diff.mean()),
                    "worst_drawdown_diff": float(drawdown_diff.min()),
                }
            )
    return pd.DataFrame(rows)


def load_scheme(path: Path, scheme: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metrics_path = path / "metrics.json"
    equity_path = path / "equity_curve.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    equity = pd.read_csv(equity_path)
    returns = compute_returns(equity, float(metrics.get("initial_capital", 1_000_000.0)))
    full = {
        "scheme": scheme,
        "total_return": float(metrics["total_return"]),
        "annualized_return": float(metrics["annualized_return"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "sharpe_like": float(metrics["sharpe_like"]),
        "avg_gross_exposure": float(metrics.get("avg_gross_exposure", np.nan)),
        "avg_positions_count": float(metrics.get("avg_positions_count", np.nan)),
        "avg_core_eligible_count": float(metrics.get("avg_core_eligible_count", np.nan)),
        "avg_core_excluded_count": float(metrics.get("avg_core_excluded_count", np.nan)),
        "output_dir": str(path),
    }
    full.update(rolling_window_stats(returns, 126))
    full.update(rolling_window_stats(returns, 252))
    return full, build_segment_metrics(returns, scheme), returns


def _format_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    return f"{number:.2%}"


def _format_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def _display_full(full: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scheme",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "rolling_126_positive_rate",
        "rolling_252_positive_rate",
        "rolling_252_worst_return",
        "rolling_252_worst_drawdown",
    ]
    out = full[cols].copy()
    out.columns = ["方案", "总收益", "年化", "最大回撤", "Sharpe近似", "126日窗盈利率", "252日窗盈利率", "最差252日收益", "最差252日回撤"]
    for col in ["总收益", "年化", "最大回撤", "126日窗盈利率", "252日窗盈利率", "最差252日收益", "最差252日回撤"]:
        out[col] = out[col].map(_format_pct)
    out["Sharpe近似"] = out["Sharpe近似"].map(lambda x: _format_number(x, 3))
    return out


def _display_compare(compare: pd.DataFrame) -> pd.DataFrame:
    out = compare.copy()
    out["period_type"] = out["period_type"].replace({"year": "年度", "half_year": "半年度"})
    out.columns = [
        "方案",
        "周期",
        "周期数",
        "收益胜出数",
        "收益胜率",
        "平均超额收益",
        "最差超额收益",
        "回撤改善数",
        "回撤改善率",
        "平均回撤改善",
        "最差回撤改善",
    ]
    for col in ["收益胜率", "平均超额收益", "最差超额收益", "回撤改善率", "平均回撤改善", "最差回撤改善"]:
        out[col] = out[col].map(_format_pct)
    return out


def write_report(
    output_prefix: Path,
    full: pd.DataFrame,
    segments: pd.DataFrame,
    compare: pd.DataFrame,
    baseline_scheme: str,
) -> Path:
    best_return = full.sort_values(["total_return", "sharpe_like"], ascending=False).iloc[0]
    best_drawdown = full.sort_values(["max_drawdown", "total_return"], ascending=False).iloc[0]
    balanced = full.sort_values(["rolling_252_positive_rate", "max_drawdown", "total_return"], ascending=False).iloc[0]
    lines = [
        "# 核心风险过滤三候选稳定性验证",
        "",
        "## 结论",
        f"- 收益最高：{best_return['scheme']}，总收益 {_format_pct(best_return['total_return'])}，最大回撤 {_format_pct(best_return['max_drawdown'])}，Sharpe {_format_number(best_return['sharpe_like'], 3)}。",
        f"- 回撤最低：{best_drawdown['scheme']}，最大回撤 {_format_pct(best_drawdown['max_drawdown'])}，总收益 {_format_pct(best_drawdown['total_return'])}。",
        f"- 滚动稳定性优先：{balanced['scheme']}，252日窗盈利率 {_format_pct(balanced['rolling_252_positive_rate'])}，最差252日收益 {_format_pct(balanced['rolling_252_worst_return'])}。",
        "- 当前判断：进攻版可以作为高收益候选，平衡版更适合进入每日默认观察，稳健版保留为回撤约束方案。",
        "",
        "## 全样本与滚动窗口",
        _display_full(full).to_markdown(index=False),
        "",
        f"## 相对 {baseline_scheme} 的分段胜率",
        _display_compare(compare).to_markdown(index=False),
        "",
        "## 年度明细",
        _display_segments(segments, "year").to_markdown(index=False),
        "",
        "## 半年度明细",
        _display_segments(segments, "half_year").to_markdown(index=False),
        "",
        "## 文件",
        f"- full: {output_prefix.name}_full.csv",
        f"- segments: {output_prefix.name}_segments.csv",
        f"- compare: {output_prefix.name}_compare.csv",
    ]
    report = output_prefix.with_suffix(".md")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _display_segments(segments: pd.DataFrame, period_type: str) -> pd.DataFrame:
    out = segments[segments["period_type"] == period_type].copy()
    out = out[["scheme", "period", "return", "max_drawdown", "sharpe_like", "win_rate"]]
    out.columns = ["方案", "周期", "收益", "最大回撤", "Sharpe近似", "日胜率"]
    for col in ["收益", "最大回撤", "日胜率"]:
        out[col] = out[col].map(_format_pct)
    out["Sharpe近似"] = out["Sharpe近似"].map(lambda x: _format_number(x, 3))
    return out


def parse_scheme(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("scheme must use label=path")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize core risk filter finalist stability.")
    parser.add_argument("--scheme", action="append", type=parse_scheme, default=None, help="label=output_dir")
    parser.add_argument("--baseline", default="原始组合")
    parser.add_argument(
        "--output-prefix",
        default="outputs/high_return_v2/core_risk_filter_finalist_stability_20260707",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schemes = args.scheme or [(label, Path(path)) for label, path in DEFAULT_SCHEMES]
    full_rows: list[dict[str, object]] = []
    segment_frames: list[pd.DataFrame] = []
    for label, path in schemes:
        full, segments, _returns = load_scheme(Path(path), label)
        full_rows.append(full)
        segment_frames.append(segments)
    full_table = pd.DataFrame(full_rows).sort_values(["total_return", "sharpe_like"], ascending=False)
    segment_table = pd.concat(segment_frames, ignore_index=True)
    compare_table = compare_segments_to_baseline(segment_table, args.baseline)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    full_table.to_csv(output_prefix.with_name(output_prefix.name + "_full.csv"), index=False, encoding="utf-8-sig")
    segment_table.to_csv(output_prefix.with_name(output_prefix.name + "_segments.csv"), index=False, encoding="utf-8-sig")
    compare_table.to_csv(output_prefix.with_name(output_prefix.name + "_compare.csv"), index=False, encoding="utf-8-sig")
    report = write_report(output_prefix, full_table, segment_table, compare_table, args.baseline)
    print(f"Report: {report}")
    print(_display_full(full_table).to_string(index=False))


if __name__ == "__main__":
    main()
