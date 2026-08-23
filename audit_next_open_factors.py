"""Audit whether close-time signals predict next-open tradable returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from execution_rules import next_open_return_label
from run_backtest import BacktestEngine, StrategyConfig, load_prices, pivot_prices


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def audit_config(raw_df: pd.DataFrame, config_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = StrategyConfig.from_dict(load_yaml(config_path))
    close = pivot_prices(raw_df, "close")
    open_px = pivot_prices(raw_df, "open").reindex_like(close)
    amount = pivot_prices(raw_df, "amount") if "amount" in raw_df.columns else close * 0.0

    engine = BacktestEngine(cfg)
    close = engine._clean_price_matrix(close)
    open_px = engine._clean_price_matrix(open_px)
    signal = engine._precompute_signal_panel(close)
    universe = engine._precompute_universe_panel(close, amount)
    if universe is not None:
        signal = signal.where(universe)

    # Feature at close T, buy at open T+1, mark at open T+2.
    label = next_open_return_label(open_px, max_abs_daily_return=cfg.max_abs_daily_return)

    rows = []
    for date in signal.index:
        s = signal.loc[date].dropna()
        y = label.loc[date].reindex(s.index).dropna()
        if len(y) < max(cfg.top_n_long * 2, 20):
            continue
        s = s.reindex(y.index).dropna()
        y = y.reindex(s.index).dropna()
        if len(y) < max(cfg.top_n_long * 2, 20):
            continue
        top = s.nlargest(cfg.top_n_long).index
        bottom = s.nsmallest(cfg.top_n_long).index
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "candidate_count": int(len(s)),
                "top_mean_next_open_return": float(y.reindex(top).mean()),
                "bottom_mean_next_open_return": float(y.reindex(bottom).mean()),
                "all_mean_next_open_return": float(y.mean()),
                "top_win_rate": float(y.reindex(top).gt(0).mean()),
                "spearman_ic": float(s.corr(y, method="spearman")),
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        summary = {
            "config": str(config_path),
            "days": 0,
            "top_mean_next_open_return": 0.0,
            "top_win_rate": 0.0,
            "mean_spearman_ic": 0.0,
        }
        return summary, daily

    summary = {
        "config": str(config_path),
        "signal_type": cfg.signal_type,
        "days": int(len(daily)),
        "avg_candidate_count": float(daily["candidate_count"].mean()),
        "top_mean_next_open_return": float(daily["top_mean_next_open_return"].mean()),
        "bottom_mean_next_open_return": float(daily["bottom_mean_next_open_return"].mean()),
        "all_mean_next_open_return": float(daily["all_mean_next_open_return"].mean()),
        "top_excess_next_open_return": float(
            (daily["top_mean_next_open_return"] - daily["all_mean_next_open_return"]).mean()
        ),
        "top_win_rate": float(daily["top_win_rate"].mean()),
        "mean_spearman_ic": float(daily["spearman_ic"].mean()),
        "positive_ic_rate": float(daily["spearman_ic"].gt(0).mean()),
    }
    return summary, daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit next-open predictive power of strategy signals.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/next_open_factor_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for config in args.config:
        config_path = Path(config)
        summary, daily = audit_config(raw, config_path)
        summaries.append(summary)
        daily_name = config_path.stem + "_daily.csv"
        daily.to_csv(output_dir / daily_name, index=False, encoding="utf-8")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary_df.to_string(index=False))
    print(f"Audit saved to: {output_dir}")


if __name__ == "__main__":
    main()
