"""Audit legacy 10000%+ backtest curves against stricter execution variants.

This is a research-only audit helper.  It does not place orders and it does
not promote any strategy into the daily simulation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "high_return_v2" / "legacy_10000pct_reproduction_audit"


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    group: str
    role: str
    label: str
    data_path: Optional[Path]
    config_path: Optional[Path]
    existing_output_dir: Path
    assumption: str
    note: str


@dataclass
class AuditResult:
    case_id: str
    group: str
    role: str
    label: str
    source: str
    metrics_path: str
    data_path: str
    config_path: str
    total_return_pct: Optional[float]
    annualized_return_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    final_equity: Optional[float]
    trade_days: Optional[int]
    avg_gross_exposure: Optional[float]
    avg_turnover: Optional[float]
    existing_total_return_pct: Optional[float]
    delta_vs_existing_total_return_pct: Optional[float]
    assumption: str
    note: str
    status: str


def default_cases() -> List[AuditCase]:
    base = ROOT / "outputs" / "high_return_v2"
    configs = ROOT / "configs"
    return [
        AuditCase(
            case_id="legacy_selected_top300",
            group="legacy_10000pct",
            role="legacy",
            label="Selected Top300 old close-to-close curve",
            data_path=ROOT / "data_panel_history_high_return_top300_20220101_20260626.csv",
            config_path=configs / "high_risk_strategy_high_return_v2_selected.yaml",
            existing_output_dir=base / "selected_top300_20220101_20260626",
            assumption="close_to_close, high leverage, no explicit limit-up/down execution gate",
            note="Old spectacular curve. Treat as reproducibility evidence only.",
        ),
        AuditCase(
            case_id="legacy_history_baseline_fixed",
            group="legacy_10000pct",
            role="legacy_artifact_only",
            label="History baseline fixed old artifact",
            data_path=None,
            config_path=None,
            existing_output_dir=base / "history_baseline_fixed_20220101_20260626",
            assumption="legacy artifact; exact config not pinned in this audit",
            note="Included because it is a concrete 10000%+ historical output.",
        ),
        AuditCase(
            case_id="legacy_dynamic_universe_clean",
            group="legacy_extreme",
            role="legacy",
            label="Dynamic universe old close-to-close curve",
            data_path=ROOT / "data_panel_history_main_chinext_20220101_20260626.csv",
            config_path=configs / "high_risk_strategy_high_return_v2_dynamic_universe.yaml",
            existing_output_dir=base / "dynamic_universe_clean_20220101_20260626",
            assumption="close_to_close, dynamic high-return universe, aggressive leverage",
            note="Extremely large result; primary bias/lookthrough suspect.",
        ),
        AuditCase(
            case_id="balanced_close_to_close",
            group="execution_sensitivity",
            role="legacy_control",
            label="Balanced v3 close-to-close control",
            data_path=ROOT / "data_panel_history_main_chinext_20220101_20260626.csv",
            config_path=configs / "high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30_v3.yaml",
            existing_output_dir=base / "dynamic_universe_balanced_dd30_v3_20220101_20260626",
            assumption="close_to_close with reduced leverage and drawdown controls",
            note="Control case before next-open execution gates.",
        ),
        AuditCase(
            case_id="balanced_next_open",
            group="execution_sensitivity",
            role="strict",
            label="Balanced v3 next-open strict execution",
            data_path=ROOT / "data_panel_history_main_chinext_20220101_20260626.csv",
            config_path=configs / "high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30_v3_next_open.yaml",
            existing_output_dir=base / "dynamic_universe_balanced_dd30_v3_next_open_20220101_20260626",
            assumption="next_open, block limit-up buys, block limit-down sells",
            note="Strict execution version of the balanced family.",
        ),
        AuditCase(
            case_id="balanced_next_open_gap3",
            group="execution_sensitivity",
            role="strict",
            label="Balanced v3 next-open + gap<=3%",
            data_path=ROOT / "data_panel_history_main_chinext_20220101_20260626.csv",
            config_path=configs / "high_risk_strategy_high_return_v2_dynamic_universe_balanced_dd30_v3_next_open_gap3.yaml",
            existing_output_dir=base / "dynamic_universe_balanced_dd30_v3_next_open_gap3_20220101_20260626",
            assumption="next_open, limit gates, max buy open gap 3%",
            note="Stricter anti-chase execution variant.",
        ),
        AuditCase(
            case_id="current_next_open_stock_focus",
            group="current_strict_family",
            role="current_reference",
            label="Current stock-focus next-open family as of 2026-07-13",
            data_path=ROOT / "data_panel_history_main_chinext_20220101_20260713.csv",
            config_path=None,
            existing_output_dir=base
            / "next_open_rank_model_stock_focus_lev093_marketfilter_bench20260713_20220101_20260713",
            assumption="trained next-open rank model with market filter",
            note="Current stricter family reference, not a 10000% reproduction target.",
        ),
    ]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a research-only audit for legacy 10000%+ backtest curves."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for audit CSV/JSON/Markdown outputs.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun cases that have both data and config paths. Default reads existing outputs only.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value) * 100.0, 4)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_case(case: AuditCase, rerun_root: Path) -> Path:
    if not case.data_path or not case.config_path:
        raise ValueError(f"Case {case.case_id} cannot be rerun without data and config paths.")
    out_dir = rerun_root / case.case_id
    cmd = [
        sys.executable,
        str(ROOT / "run_backtest.py"),
        "--data",
        str(case.data_path),
        "--config",
        str(case.config_path),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_dir


def result_from_case(case: AuditCase, source_dir: Path, source: str) -> AuditResult:
    metrics_path = source_dir / "metrics.json"
    metrics = load_metrics(metrics_path)
    existing_metrics = load_metrics(case.existing_output_dir / "metrics.json")
    total_return_pct = pct(metrics.get("total_return"))
    existing_total_return_pct = pct(existing_metrics.get("total_return"))
    delta = None
    if total_return_pct is not None and existing_total_return_pct is not None:
        delta = round(total_return_pct - existing_total_return_pct, 4)
    status = "ok" if metrics else "missing_metrics"
    return AuditResult(
        case_id=case.case_id,
        group=case.group,
        role=case.role,
        label=case.label,
        source=source,
        metrics_path=str(metrics_path),
        data_path=str(case.data_path) if case.data_path else "",
        config_path=str(case.config_path) if case.config_path else "",
        total_return_pct=total_return_pct,
        annualized_return_pct=pct(metrics.get("annualized_return")),
        max_drawdown_pct=pct(metrics.get("max_drawdown")),
        final_equity=float_or_none(metrics.get("final_equity")),
        trade_days=int_or_none(metrics.get("trade_days")),
        avg_gross_exposure=float_or_none(metrics.get("avg_gross_exposure")),
        avg_turnover=float_or_none(metrics.get("avg_turnover")),
        existing_total_return_pct=existing_total_return_pct,
        delta_vs_existing_total_return_pct=delta,
        assumption=case.assumption,
        note=case.note,
        status=status,
    )


def collect_results(cases: Iterable[AuditCase], output_dir: Path, rerun: bool) -> List[AuditResult]:
    results: List[AuditResult] = []
    rerun_root = output_dir / "rerun"
    for case in cases:
        source_dir = case.existing_output_dir
        source = "existing"
        if rerun and case.data_path and case.config_path:
            source_dir = run_case(case, rerun_root)
            source = "rerun"
        results.append(result_from_case(case, source_dir, source))
    return results


def evaluate_audit(results: Iterable[AuditResult]) -> Dict[str, Any]:
    rows = [row for row in results if row.status == "ok"]
    legacy = [row for row in rows if row.role.startswith("legacy")]
    strict = [row for row in rows if row.role == "strict"]
    max_legacy = max((row.total_return_pct or float("-inf") for row in legacy), default=None)
    max_strict = max((row.total_return_pct or float("-inf") for row in strict), default=None)
    legacy_10000_observed = any((row.total_return_pct or 0.0) >= 10_000.0 for row in legacy)
    strict_10000_observed = any((row.total_return_pct or 0.0) >= 10_000.0 for row in strict)
    strict_positive = any((row.total_return_pct or 0.0) > 0.0 for row in strict)

    if legacy_10000_observed and not strict_10000_observed:
        status = "legacy_curve_reproducible_under_old_assumptions_only"
    elif legacy_10000_observed and strict_10000_observed:
        status = "requires_deeper_review_before_any_promotion"
    else:
        status = "no_legacy_10000pct_curve_observed"

    return {
        "audit_status": status,
        "legacy_10000pct_observed": legacy_10000_observed,
        "strict_10000pct_observed": strict_10000_observed,
        "strict_positive_observed": strict_positive,
        "max_legacy_total_return_pct": None if max_legacy == float("-inf") else max_legacy,
        "max_strict_total_return_pct": None if max_strict == float("-inf") else max_strict,
        "promote_to_current_simulation": False,
        "reason": (
            "旧曲线可作为历史复现对象；严格成交版本没有复现 10000%+，禁止直接接入当前模拟盘。"
            if legacy_10000_observed and not strict_10000_observed
            else "仍需完整审计未来函数、股票池穿越、成交假设和容量。"
        ),
    }


def fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def fmt_num(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:,.2f}"


def write_csv(results: List[AuditResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_json(results: List[AuditResult], summary: Dict[str, Any], path: Path) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "research_only": True,
        "summary": summary,
        "results": [asdict(row) for row in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(results: List[AuditResult], summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 旧 10000%+ 策略复现审计",
        "",
        "- 结论: 旧高收益曲线可以作为历史产物复现，但不能作为当前可用策略直接接入模拟盘。",
        f"- 审计状态: `{summary['audit_status']}`",
        f"- 旧假设下是否观察到 10000%+: `{summary['legacy_10000pct_observed']}`",
        f"- 严格成交下是否观察到 10000%+: `{summary['strict_10000pct_observed']}`",
        f"- 是否允许接入当前模拟盘: `{summary['promote_to_current_simulation']}`",
        f"- 审核意见: {summary['reason']}",
        "",
        "## 对照表",
        "",
        "| case | role | source | 总收益 | 旧产物总收益 | 复跑差值 | 年化 | 最大回撤 | 期末权益 | 假设 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.case_id,
                    row.role,
                    row.source,
                    fmt_pct(row.total_return_pct),
                    fmt_pct(row.existing_total_return_pct),
                    fmt_pct(row.delta_vs_existing_total_return_pct),
                    fmt_pct(row.annualized_return_pct),
                    fmt_pct(row.max_drawdown_pct),
                    fmt_num(row.final_equity),
                    row.assumption,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 自检要点",
            "",
            "- 10000%+ 只说明旧回测曲线存在，不等于实盘可复制。",
            "- 优先检查未来函数、股票池穿越、动态高收益股票池、涨停无法买入、跌停无法卖出、开盘跳空和容量。",
            "- 严格成交版本未复现 10000%+ 前，不得把旧策略直接并入当前模拟盘。",
            "- 可提取旧策略里的候选特征做因子研究，但必须重新进入样本外、滚动和模拟盘观察。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(default_cases(), output_dir, bool(args.rerun))
    if not results:
        raise SystemExit("No audit cases produced results.")
    summary = evaluate_audit(results)

    write_csv(results, output_dir / "legacy_10000pct_reproduction_audit.csv")
    write_json(results, summary, output_dir / "legacy_10000pct_reproduction_audit.json")
    write_markdown(results, summary, output_dir / "legacy_10000pct_reproduction_audit.md")

    print(f"audit_status: {summary['audit_status']}")
    print(f"legacy_10000pct_observed: {summary['legacy_10000pct_observed']}")
    print(f"strict_10000pct_observed: {summary['strict_10000pct_observed']}")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
