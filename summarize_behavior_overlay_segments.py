"""Segment diagnostics for base vs personal behavior overlay equity curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import max_drawdown, sharpe_like


def add_return_columns(frame: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    for prefix in ("base", "overlay"):
        equity_col = f"{prefix}_equity"
        prev = pd.to_numeric(out[equity_col], errors="coerce").shift(1)
        if not prev.empty:
            prev.iloc[0] = float(initial_capital)
        out[f"{prefix}_daily_return"] = pd.to_numeric(out[equity_col], errors="coerce") / (prev + 1e-12) - 1.0
    return out


def classify_market_regime(exposure: float) -> str:
    value = float(exposure)
    if value >= 0.99:
        return "risk_on"
    if value >= 0.50:
        return "reduced"
    return "defensive"


def _compound_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def _segment_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float(max_drawdown(nav))


def _win_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float(clean.gt(0).mean())


def segment_metrics(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment, group in frame.groupby(group_col, sort=True):
        base_ret = group["base_daily_return"]
        overlay_ret = group["overlay_daily_return"]
        rows.append(
            {
                "segment": str(segment),
                "days": int(len(group)),
                "base_return": _compound_return(base_ret),
                "overlay_return": _compound_return(overlay_ret),
                "return_diff": _compound_return(overlay_ret) - _compound_return(base_ret),
                "base_max_drawdown": _segment_drawdown(base_ret),
                "overlay_max_drawdown": _segment_drawdown(overlay_ret),
                "drawdown_diff": _segment_drawdown(overlay_ret) - _segment_drawdown(base_ret),
                "base_sharpe_like": float(sharpe_like(base_ret.fillna(0.0))),
                "overlay_sharpe_like": float(sharpe_like(overlay_ret.fillna(0.0))),
                "sharpe_diff": float(sharpe_like(overlay_ret.fillna(0.0)) - sharpe_like(base_ret.fillna(0.0))),
                "base_win_rate": _win_rate(base_ret),
                "overlay_win_rate": _win_rate(overlay_ret),
                "avg_base_turnover": float(group.get("base_turnover", pd.Series(dtype=float)).mean()),
                "avg_overlay_turnover": float(group.get("overlay_turnover", pd.Series(dtype=float)).mean()),
                "avg_base_exposure": float(group.get("base_gross_exposure", pd.Series(dtype=float)).mean()),
                "avg_overlay_exposure": float(group.get("overlay_gross_exposure", pd.Series(dtype=float)).mean()),
                "avg_market_exposure": float(group.get("market_exposure", pd.Series(dtype=float)).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_segment_tables(equity: pd.DataFrame, initial_capital: float) -> dict[str, pd.DataFrame]:
    enriched = add_return_columns(equity, initial_capital)
    enriched["year"] = enriched["date"].dt.year.astype(str)
    enriched["market_regime"] = enriched["market_exposure"].map(classify_market_regime)
    return {
        "enriched": enriched,
        "by_year": segment_metrics(enriched, "year"),
        "by_market_regime": segment_metrics(enriched, "market_regime"),
    }


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2%}"


def write_report(
    output_dir: Path,
    by_year: pd.DataFrame,
    by_market_regime: pd.DataFrame,
    metrics: dict[str, object],
) -> Path:
    best_years = int(by_year["return_diff"].gt(0).sum())
    total_years = int(len(by_year))
    drawdown_better_years = int(by_year["drawdown_diff"].gt(0).sum())
    base = metrics.get("base", {})
    overlay = metrics.get("personal_overlay", {})
    lines = [
        "# 个人习惯层分段验证",
        "",
        "目标：检查收益优先参数是否只依赖某一年或某一种市场环境。",
        "",
        "## 全周期",
        f"- 原模型收益率: {_format_pct(base.get('total_return', np.nan))}",
        f"- 个人习惯层收益率: {_format_pct(overlay.get('total_return', np.nan))}",
        f"- 原模型最大回撤: {_format_pct(base.get('max_drawdown', np.nan))}",
        f"- 个人习惯层最大回撤: {_format_pct(overlay.get('max_drawdown', np.nan))}",
        "",
        "## 年度稳定性",
        f"- 收益跑赢年份: {best_years}/{total_years}",
        f"- 回撤改善年份: {drawdown_better_years}/{total_years}",
        "",
        by_year.to_markdown(index=False),
        "",
        "## 市场状态",
        by_market_regime.to_markdown(index=False),
        "",
        "## 解读",
        "- 如果收益只在单一年份跑赢，后续应降低参数权重。",
        "- 如果 defensive/reduced 状态改善明显，说明这层更偏防守；如果 risk_on 也不拖累，才适合进入实盘前置默认链路。",
    ]
    report = output_dir / "segment_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize base vs personal overlay performance by year and market regime.")
    parser.add_argument("--equity", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    equity = pd.read_csv(args.equity)
    tables = build_segment_tables(equity, args.initial_capital)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["by_year"].to_csv(output_dir / "segment_by_year.csv", index=False, encoding="utf-8-sig")
    tables["by_market_regime"].to_csv(output_dir / "segment_by_market_regime.csv", index=False, encoding="utf-8-sig")
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8")) if args.metrics else {}
    report = write_report(output_dir, tables["by_year"], tables["by_market_regime"], metrics)
    print(f"Year segments: {output_dir / 'segment_by_year.csv'}")
    print(f"Market regime segments: {output_dir / 'segment_by_market_regime.csv'}")
    print(f"Report: {report}")
    print(tables["by_year"].to_string(index=False))
    print(tables["by_market_regime"].to_string(index=False))


if __name__ == "__main__":
    main()
