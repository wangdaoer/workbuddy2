"""Summarize rank model equity by calendar-year and full-period segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sharpe_like(returns: pd.Series) -> float:
    sigma = returns.std(ddof=1)
    if sigma <= 1e-12 or np.isnan(sigma):
        return 0.0
    return float(returns.mean() / sigma * np.sqrt(252))


def summarize_segment(name: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "segment": name,
            "days": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "win_rate": 0.0,
            "avg_turnover": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_market_exposure": 0.0,
            "avg_positions_count": 0.0,
            "total_cost": 0.0,
            "start_equity": 0.0,
            "end_equity": 0.0,
        }

    equity = frame["equity"].astype(float)
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    return {
        "segment": name,
        "start_date": frame["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "days": int(len(frame)),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "max_drawdown": max_drawdown(equity),
        "sharpe_like": sharpe_like(returns),
        "win_rate": float(returns.gt(0).mean()),
        "avg_turnover": float(frame["turnover"].mean()) if "turnover" in frame else 0.0,
        "avg_gross_exposure": float(frame["gross_exposure"].mean()) if "gross_exposure" in frame else 0.0,
        "avg_market_exposure": float(frame["market_exposure"].mean()) if "market_exposure" in frame else 0.0,
        "avg_positions_count": float(frame["positions_count"].mean()) if "positions_count" in frame else 0.0,
        "total_cost": float(frame["cost"].sum()) if "cost" in frame else 0.0,
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
    }


def write_markdown(summary: pd.DataFrame, output: Path) -> None:
    lines = ["# Segment Summary", ""]
    display = summary.copy()
    for col in ["total_return", "max_drawdown", "win_rate", "avg_turnover", "avg_gross_exposure", "avg_market_exposure", "total_cost"]:
        if col in display:
            display[col] = display[col].map(lambda x: f"{x:.2%}")
    if "sharpe_like" in display:
        display["sharpe_like"] = display["sharpe_like"].map(lambda x: f"{x:.3f}")
    for col in ["start_equity", "end_equity", "avg_positions_count"]:
        if col in display:
            display[col] = display[col].map(lambda x: f"{x:,.2f}")
    lines.append(display.to_markdown(index=False))
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize rank model OOS segments.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    equity_path = run_dir / "equity_curve.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Missing equity_curve.csv: {equity_path}")

    equity = pd.read_csv(equity_path, parse_dates=["date"])
    equity = equity.sort_values("date").reset_index(drop=True)
    rows = [summarize_segment("full", equity)]
    for year, group in equity.groupby(equity["date"].dt.year):
        rows.append(summarize_segment(str(year), group.reset_index(drop=True)))

    summary = pd.DataFrame(rows)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "segment_summary.csv", index=False, encoding="utf-8")
    (output_dir / "segment_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(summary, output_dir / "segment_summary.md")
    print(summary.to_string(index=False))
    print(f"Segment summary saved to: {output_dir}")


if __name__ == "__main__":
    main()
