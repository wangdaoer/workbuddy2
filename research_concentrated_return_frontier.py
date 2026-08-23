"""Research concentrated long-only portfolios under strict A-share execution.

This module is research-only. It never changes the production strategy or emits
orders. The 900% return threshold is an exploration target, not a promise.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from run_backtest import (
    BacktestEngine,
    StrategyConfig,
    annualized_return,
    load_prices,
    max_drawdown,
    pivot_prices,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "high_return_v2" / "concentrated_return_frontier"


@dataclass(frozen=True)
class ConcentrationScenario:
    scenario_id: str
    top_n_long: int
    max_position_weight: float
    leverage: float = 1.2


SCENARIOS = (
    ConcentrationScenario("concentrated_3", 3, 0.42),
    ConcentrationScenario("concentrated_5", 5, 0.26),
    ConcentrationScenario("concentrated_8", 8, 0.17),
    ConcentrationScenario("reference_10", 10, 0.14),
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate concentrated portfolios against a 900% research target."
    )
    parser.add_argument("--data", required=True, help="Historical OHLCV panel (CSV or Parquet).")
    parser.add_argument("--config", required=True, help="Strict next-open base YAML config.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--legacy-metrics",
        default="",
        help="Optional legacy metrics.json used only as a labelled reference row.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_variants(base: dict[str, Any]) -> list[tuple[str, dict[str, Any], float, float | None]]:
    """Build paired concentration cases with execution and cost stress tests."""
    variants: list[tuple[str, dict[str, Any], float, float | None]] = []
    base_commission = float(base["cost"]["commission_bps"])
    base_impact = float(base["cost"]["impact_bps"])
    for scenario in SCENARIOS:
        for cost_multiplier, gap_limit in ((1.0, None), (2.0, None), (1.0, 0.03)):
            cfg = copy.deepcopy(base)
            cfg["signal"]["top_n_long"] = scenario.top_n_long
            cfg["signal"]["top_n_short"] = 0
            cfg["portfolio"]["max_position_weight"] = scenario.max_position_weight
            cfg["portfolio"]["leverage"] = scenario.leverage
            cfg["portfolio"]["allow_short"] = False
            cfg["portfolio"]["short_exposure"] = 0.0
            cfg["risk"]["max_drawdown"] = 0.65
            cfg["execution"]["model"] = "next_open"
            cfg["execution"]["block_limit_up_buys"] = True
            cfg["execution"]["block_limit_down_sells"] = True
            cfg["execution"]["max_buy_open_gap"] = gap_limit
            cfg["cost"]["commission_bps"] = base_commission * cost_multiplier
            cfg["cost"]["impact_bps"] = base_impact * cost_multiplier
            suffix = "base_cost" if cost_multiplier == 1.0 else "double_cost"
            if gap_limit is not None:
                suffix = "gap3"
            variants.append((f"{scenario.scenario_id}_{suffix}", cfg, cost_multiplier, gap_limit))
    return variants


def scenario_id_from_variant(variant_id: str) -> str:
    for suffix in ("_base_cost", "_double_cost", "_gap3"):
        if variant_id.endswith(suffix):
            return variant_id[: -len(suffix)]
    raise ValueError(f"Unknown concentration variant suffix: {variant_id}")


def segment_metrics(equity: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, segment in equity.groupby(equity.index.year):
        if len(segment) < 2:
            continue
        normalized = segment / float(segment.iloc[0])
        rows.append(
            {
                "year": int(year),
                "trade_days": int(len(segment)),
                "total_return": float(normalized.iloc[-1] - 1.0),
                "annualized_return": float(annualized_return(normalized)),
                "max_drawdown": float(max_drawdown(normalized)),
            }
        )
    return rows


def target_required_cagr(equity: pd.Series, target_total_return: float = 9.0) -> float:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    return float((1.0 + target_total_return) ** (1.0 / years) - 1.0)


def inactive_tail_days(equity: pd.Series, tolerance: float = 1e-12) -> int:
    changed = equity.pct_change(fill_method=None).abs().fillna(0.0).gt(tolerance)
    if not changed.any():
        return max(len(equity) - 1, 0)
    last_active_position = int(changed.to_numpy().nonzero()[0][-1])
    return int(len(equity) - last_active_position - 1)


def evaluate_research_gates(
    *,
    total_return: float,
    max_drawdown_value: float,
    double_cost_return: float,
    gap3_return: float,
    positive_year_ratio: float,
    inactive_tail: int,
) -> list[str]:
    failures: list[str] = []
    if total_return <= 0.0:
        failures.append("non_positive_full_period_return")
    if max_drawdown_value <= -0.65:
        failures.append("drawdown_fail_line_breached")
    if double_cost_return <= 0.0:
        failures.append("non_positive_double_cost_return")
    if gap3_return <= 0.0:
        failures.append("non_positive_gap3_return")
    if positive_year_ratio < 2.0 / 3.0:
        failures.append("insufficient_positive_year_ratio")
    if inactive_tail >= 20:
        failures.append("inactive_tail_after_strategy_failure")
    return failures


def run_research(
    raw: pd.DataFrame,
    base: dict[str, Any],
    output_dir: Path,
    legacy_metrics_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    normal_curves: dict[str, pd.Series] = {}

    variants = build_variants(base)
    base_variants = [item for item in variants if item[2] == 1.0 and item[3] is None]
    reference_engine = BacktestEngine(StrategyConfig.from_dict(base_variants[0][1]))
    close = reference_engine._clean_price_matrix(pivot_prices(raw, "close"))
    amount = (
        pivot_prices(raw, "amount")
        if "amount" in raw.columns
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    signal_cache = reference_engine._precompute_signal_panel(close)
    universe_cache = reference_engine._precompute_universe_panel(close, amount)

    def execute_variant(variant: tuple[str, dict[str, Any], float, float | None]) -> None:
        variant_id, raw_cfg, cost_multiplier, gap_limit = variant
        cfg = StrategyConfig.from_dict(raw_cfg)
        result = BacktestEngine(cfg).run(
            raw,
            signal_cache=signal_cache,
            universe_cache=universe_cache,
        )
        metrics = result["metrics"]
        scenario_id = scenario_id_from_variant(variant_id)
        row = {
            "variant_id": variant_id,
            "scenario_id": scenario_id,
            "top_n_long": cfg.top_n_long,
            "max_position_weight": cfg.max_position_weight,
            "leverage": cfg.leverage,
            "cost_multiplier": cost_multiplier,
            "max_buy_open_gap": gap_limit,
            **metrics,
        }
        records.append(row)
        for segment in segment_metrics(result["equity"]):
            all_segments.append({"variant_id": variant_id, **segment})
        if cost_multiplier == 1.0 and gap_limit is None:
            normal_curves[scenario_id] = result["equity"]

    for variant in base_variants:
        execute_variant(variant)
    preliminary_best = max(records, key=lambda row: float(row["total_return"]))["scenario_id"]
    stress_variants = [
        item
        for item in variants
        if scenario_id_from_variant(item[0]) == preliminary_best and item not in base_variants
    ]
    for variant in stress_variants:
        execute_variant(variant)

    results = pd.DataFrame(records).sort_values(
        ["total_return", "max_drawdown"], ascending=[False, False]
    )
    segments = pd.DataFrame(all_segments)
    results.to_csv(output_dir / "concentration_results.csv", index=False)
    segments.to_csv(output_dir / "concentration_yearly_stability.csv", index=False)

    normal = results[(results["cost_multiplier"] == 1.0) & results["max_buy_open_gap"].isna()].copy()
    best = normal.sort_values("total_return", ascending=False).iloc[0]
    best_scenario = str(best["scenario_id"])
    best_curve = normal_curves[best_scenario]
    best_double = results[
        (results["scenario_id"] == best_scenario) & (results["cost_multiplier"] == 2.0)
    ].iloc[0]
    best_gap = results[
        (results["scenario_id"] == best_scenario) & results["max_buy_open_gap"].notna()
    ].iloc[0]
    best_years = segments[segments["variant_id"] == f"{best_scenario}_base_cost"]
    positive_year_ratio = float((best_years["total_return"] > 0).mean()) if len(best_years) else 0.0
    target_return = 9.0
    required_cagr = target_required_cagr(best_curve, target_return)
    retention_double_cost = float((1.0 + best_double["total_return"]) / (1.0 + best["total_return"]))
    retention_gap3 = float((1.0 + best_gap["total_return"]) / (1.0 + best["total_return"]))
    inactive_tail = inactive_tail_days(best_curve)
    failed_gates = evaluate_research_gates(
        total_return=float(best["total_return"]),
        max_drawdown_value=float(best["max_drawdown"]),
        double_cost_return=float(best_double["total_return"]),
        gap3_return=float(best_gap["total_return"]),
        positive_year_ratio=positive_year_ratio,
        inactive_tail=inactive_tail,
    )

    legacy_reference: dict[str, Any] = {}
    if legacy_metrics_path and legacy_metrics_path.exists():
        legacy_reference = json.loads(legacy_metrics_path.read_text(encoding="utf-8"))

    decision = {
        "research_only": True,
        "target_total_return": target_return,
        "target_met_under_strict_execution": bool(best["total_return"] >= target_return),
        "required_cagr_for_test_span": required_cagr,
        "best_variant": str(best["variant_id"]),
        "best_total_return": float(best["total_return"]),
        "best_annualized_return": float(best["annualized_return"]),
        "best_max_drawdown": float(best["max_drawdown"]),
        "double_cost_equity_retention": retention_double_cost,
        "gap3_equity_retention": retention_gap3,
        "positive_year_ratio": positive_year_ratio,
        "drawdown_fail_line_breached": bool(best["max_drawdown"] <= -0.65),
        "inactive_tail_days": inactive_tail,
        "failed_gates": failed_gates,
        "research_decision": "rejected" if failed_gates else "eligible_for_new_unseen_shadow_only",
        "promotion_allowed": False,
        "legacy_reference": legacy_reference,
    }
    (output_dir / "concentration_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = (
        "# 集中持仓900%收益研究\n\n"
        "> 研究/模拟盘；900%为累计收益探索线，不构成收益承诺。\n\n"
        f"- 严格口径最佳方案：`{decision['best_variant']}`\n"
        f"- 累计收益：{decision['best_total_return']:.2%}\n"
        f"- 年化收益：{decision['best_annualized_return']:.2%}\n"
        f"- 最大回撤：{decision['best_max_drawdown']:.2%}\n"
        f"- 当前样本跨度达到900%所需年化：{required_cagr:.2%}\n"
        f"- 双倍成本后权益保留率：{retention_double_cost:.2%}\n"
        f"- 限制开盘追高3%后权益保留率：{retention_gap3:.2%}\n"
        f"- 正收益年份占比：{positive_year_ratio:.2%}\n"
        f"- 失效后非活跃尾段：{inactive_tail} 个交易日\n"
        f"- 失败门槛：{', '.join(failed_gates) if failed_gates else '无'}\n"
        f"- 是否达到900%严格探索线：{'是' if decision['target_met_under_strict_execution'] else '否'}\n"
        "- 自动晋级：禁止；须完成滚动样本外和至少3个月模拟盘。\n"
    )
    (output_dir / "concentration_report.md").write_text(report, encoding="utf-8")
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    base = load_yaml(Path(args.config))
    raw = load_prices(Path(args.data), base.get("start_date"), base.get("end_date"))
    decision = run_research(
        raw,
        base,
        Path(args.output_dir),
        Path(args.legacy_metrics) if args.legacy_metrics else None,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
