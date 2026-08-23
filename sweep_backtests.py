"""Parameter sweep utility for project 019ecbc7-7b5b-7c13-a80a-3f328010bab8."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from run_backtest import BacktestEngine, StrategyConfig, load_prices


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small parameter sweep for the high-risk backtest."
    )
    parser.add_argument("--data", required=True, help="Path to data CSV.")
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument(
        "--sweep",
        default=None,
        help="Optional YAML file with parameter arrays, e.g. {'signal.short_window':[3,5],'risk.max_drawdown':[0.12,0.18]}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many top runs to print by Sharpe-like ratio.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["sharpe_like", "total_return", "return_per_drawdown"],
        default="sharpe_like",
        help="Sort key used to rank sweep candidates.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on sweep runs for iterative exploration.",
    )
    parser.add_argument(
        "--sample-runs",
        type=int,
        default=None,
        help="Randomly sample this many runs from the expanded grid.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used with --sample-runs.",
    )
    return parser.parse_args()


def iter_variations(base: Dict[str, Any], grid: Dict[str, List[Any]]) -> list[Dict[str, Any]]:
    if not grid:
        return [copy.deepcopy(base)]
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            cfg2 = cfg
            parts = k.split(".")
            for p in parts[:-1]:
                cfg2 = cfg2.setdefault(p, {})
            cfg2[parts[-1]] = v
        out.append(cfg)
    return out


def read_sweep_spec(path: Path | None) -> Dict[str, List[Any]]:
    if not path:
        return {}
    raw = load_yaml(path)
    if raw is None:
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, list)}


def score_variant(metrics: Dict[str, float], mode: str) -> float:
    if mode == "total_return":
        return float(metrics.get("total_return", 0.0))
    if mode == "return_per_drawdown":
        dd = abs(float(metrics.get("max_drawdown", 0.0)))
        return float(metrics.get("total_return", 0.0) / (dd + 0.002))
    return float(metrics.get("sharpe_like", 0.0))


def build_default_grid(base: Dict[str, Any]) -> Dict[str, List[Any]]:
    return {
        "signal.short_window": [3, 5, 8],
        "signal.long_window": [15, 20, 25],
        "risk.max_drawdown": [0.12, 0.16, 0.20],
        "portfolio.leverage": [1.1, 1.3, 1.5],
        "cost.commission_bps": [0.9, 1.2, 1.5],
    }


def run_all(data: Path, base_cfg: Dict[str, Any], variants: list[Dict[str, Any]]) -> pd.DataFrame:
    raw_df = load_prices(data, base_cfg.get("start_date"), base_cfg.get("end_date"))
    base_output = Path(base_cfg.get("output", {}).get("output_dir", "outputs"))
    records = []
    score_mode = base_cfg.get("_sweep_score_mode", "sharpe_like")
    failed = 0

    for i, cfg in enumerate(variants, 1):
        cfg.setdefault("output", {})
        run_id = f"run_{i:03d}"
        cfg["output"]["output_dir"] = str(base_output / "sweep" / run_id)
        try:
            conf = StrategyConfig.from_dict(cfg)
            engine = BacktestEngine(conf)
            result = engine.run(raw_df)
        except ValueError:
            failed += 1
            continue

        m = result["metrics"]
        records.append(
            {
                "run_id": run_id,
                "short_window": conf.short_window,
                "long_window": conf.long_window,
                "breakout_window": conf.breakout_window,
                "breakout_threshold": conf.breakout_threshold,
                "acceleration_window": conf.acceleration_window,
                "rebalance_frequency": conf.rebalance_frequency,
                "max_drawdown": conf.max_drawdown,
                "leverage": conf.leverage,
                "max_position_weight": conf.max_position_weight,
                "target_annualized_vol": conf.target_annualized_vol,
                "commission_bps": conf.commission_bps,
                "annualized_return": m["annualized_return"],
                "sharpe_like": m["sharpe_like"],
                "max_drawdown_realized": m["max_drawdown"],
                "total_return": m["total_return"],
                "avg_gross_exposure": m.get("avg_gross_exposure", 0.0),
                "avg_turnover": m.get("avg_turnover", 0.0),
                "score": score_variant(m, score_mode),
                "final_equity": m["final_equity"],
                "output_dir": conf.output_dir,
            }
        )

    if failed:
        print(f"[sweep] Skipped {failed} infeasible runs (insufficient lookback).")

    return pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)


def write_summary(df: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sweep_results.csv", index=False)
    for i, row in df.head(top_k).iterrows():
        print(
            f"[{i+1}] {row['run_id']} | sharpe_like={row['sharpe_like']:.3f}, "
            f"annualized_return={row['annualized_return']:.2%}, maxdd={row['max_drawdown_realized']:.2%}, "
            f"avg_exposure={row['avg_gross_exposure']:.2f}, score={row['score']:.3f}, "
            f"output={row['output_dir']}"
        )


def main() -> None:
    global args
    args = parse_args()
    base_cfg = load_yaml(Path(args.config))
    data_path = Path(args.data)
    if args.sweep:
        grid = read_sweep_spec(Path(args.sweep))
    else:
        grid = build_default_grid(base_cfg)

    print(f"Running sweep with {len(grid)} dimensions; total combinations will expand accordingly.")
    variants = iter_variations(base_cfg, grid)
    print(f"Total variant count: {len(variants)}")
    if args.sample_runs is not None and args.sample_runs < len(variants):
        rng = random.Random(args.random_seed)
        variants = rng.sample(variants, args.sample_runs)
        print(f"Randomly sampled {len(variants)} runs via --sample-runs.")
    if args.max_runs is not None and args.max_runs < len(variants):
        variants = variants[:args.max_runs]
        print(f"Limiting sweep to first {len(variants)} runs via --max-runs.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(base_cfg.get("output", {}).get("output_dir", "outputs")) / f"sweep_{ts}"
    base_output.mkdir(parents=True, exist_ok=True)

    # keep a copy of the base configuration
    (base_output / "base_config_snapshot.yaml").write_text(
        yaml.safe_dump(base_cfg, allow_unicode=True), encoding="utf-8"
    )
    grid_path = base_output / "sweep_grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid, allow_unicode=True), encoding="utf-8")

    # inject to use this sweep folder; each variant writes under sweep/run_x
    base_cfg.setdefault("output", {})
    base_cfg["output"]["output_dir"] = str(base_output)
    base_cfg["_sweep_score_mode"] = args.score_mode

    result = run_all(data_path, base_cfg, variants)
    write_summary(result, base_output, args.top_k)

    if args.top_k and args.top_k <= len(result):
        result.head(args.top_k).to_markdown(base_output / "top_k.md", index=False)

    summary = (
        "# Sweep Summary\n"
        f"- total_runs: {len(result)}\n"
        f"- top_k: {args.top_k}\n"
        f"- score_mode: {args.score_mode}\n"
        f"- best_run: {result.iloc[0]['run_id']}\n"
        f"- best_sharpe_like: {result.iloc[0]['sharpe_like']:.4f}\n"
        f"- best_total_return: {result.iloc[0]['total_return']:.2%}\n"
        f"- best_max_drawdown: {result.iloc[0]['max_drawdown_realized']:.2%}\n"
    )
    (base_output / "sweep_summary.md").write_text(summary, encoding="utf-8")
    print(f"Sweep done. Results saved to: {base_output}")


if __name__ == "__main__":
    main()
