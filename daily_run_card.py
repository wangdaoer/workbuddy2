"""Write reproducible daily workflow run cards."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


VOLATILE_RECORD_KEYS = {"recorded_at"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_daily_run_card(record: dict[str, Any], generated_at: str | None = None) -> dict[str, Any]:
    stable_record = {key: value for key, value in record.items() if key not in VOLATILE_RECORD_KEYS}
    artifact_map = stable_record.get("artifacts") if isinstance(stable_record.get("artifacts"), dict) else {}
    artifact_map = {
        key: path
        for key, path in artifact_map.items()
        if not str(key).startswith("daily_run_card_")
    }
    stable_record["artifacts"] = artifact_map
    artifacts = [
        _artifact_entry(key, Path(str(path)))
        for key, path in sorted(artifact_map.items())
        if path not in (None, "")
    ]
    commands = list(stable_record.get("commands") or [])
    return {
        "schema_version": 2,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "asof_date": stable_record.get("asof_date"),
        "run_status": stable_record.get("run_status", "unknown"),
        "failure": stable_record.get("failure"),
        "run_type": stable_record.get("run_type"),
        "data_source": stable_record.get("data_source"),
        "data_source_exists": stable_record.get("data_source_exists"),
        "config_hash": stable_json_hash(
            {
                "asof_date": stable_record.get("asof_date"),
                "run_type": stable_record.get("run_type"),
                "data_source": stable_record.get("data_source"),
                "steps": stable_record.get("steps"),
                "commands": commands,
                "argv": stable_record.get("argv"),
            }
        ),
        "record_hash": stable_json_hash(stable_record),
        "command_count": len(commands),
        "commands": commands,
        "execution": stable_record.get("execution") or {},
        "verification": stable_record.get("verification") or {},
        "fallback_fetch_status": stable_record.get("fallback_fetch_status"),
        "top10": stable_record.get("top10") or [],
        "artifacts": artifacts,
        "monitoring_alerts": stable_record.get("monitoring_alerts") or [],
        "block_on_factor_decay_alert": stable_record.get("block_on_factor_decay_alert") or False,
        "block_on_regime_alert": stable_record.get("block_on_regime_alert") or False,
        "production_book_build": stable_record.get("production_book_build") or {},
        "p10g_production_monitor_chart": stable_record.get("p10g_production_monitor_chart") or {},
        "p10g_production_integration": stable_record.get("p10g_production_integration") or {},
        "trend_ignition_shadow": {
            "status": stable_record.get("trend_ignition_shadow_status"),
            "training_end_date": stable_record.get("trend_ignition_shadow_training_end_date"),
            "selection_status": stable_record.get("trend_ignition_shadow_selection_status"),
            "research_gate_passed": stable_record.get("trend_ignition_shadow_research_gate_passed"),
            "source_rows": stable_record.get("trend_ignition_shadow_source_rows"),
            "eligible_rows": stable_record.get("trend_ignition_shadow_eligible_rows"),
            "eligibility_ratio": stable_record.get("trend_ignition_shadow_eligibility_ratio"),
            "bucket_counts": stable_record.get("trend_ignition_shadow_bucket_counts"),
        },
        "trend_ignition_score_forward": {
            "status": stable_record.get("trend_ignition_score_forward_status"),
            "report": stable_record.get("trend_ignition_score_forward_report"),
        },
        "early_pattern_forward": {
            "status": stable_record.get("early_pattern_forward_status"),
            "report": stable_record.get("early_pattern_forward_report"),
        },
        "incumbent_capacity_stress": {
            "status": stable_record.get("incumbent_capacity_stress_status"),
            "report": stable_record.get("incumbent_capacity_stress_report"),
        },
        "capacity_evidence": stable_record.get("capacity_evidence") or {},
        "warnings": _warnings(stable_record, artifacts),
    }


def write_daily_run_card(output_dir: Path, record: dict[str, Any], generated_at: str | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    asof_date = str(record.get("asof_date") or "unknown")
    token = asof_date.replace("-", "")
    card = build_daily_run_card(record, generated_at=generated_at)
    json_path = output_dir / f"daily_run_card_{token}.json"
    markdown_path = output_dir / f"daily_run_card_{token}.md"
    json_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(card), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _artifact_entry(key: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file() if exists else False
    return {
        "key": key,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if is_file else None,
        "sha256": file_sha256(path) if is_file else None,
    }


def _warnings(record: dict[str, Any], artifacts: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not record.get("data_source_exists", True):
        warnings.append("data_source_missing")
    if record.get("run_status") == "failed":
        warnings.append("run_failed")
        warnings.append("artifacts_may_include_same_day_pre_failure_outputs")
    for item in artifacts:
        if not item["exists"]:
            warnings.append(f"artifact_missing:{item['key']}")
    verification = record.get("verification") or {}
    if verification.get("missing_stock_names"):
        warnings.append("stock_names_missing")
    benchmark_status = str(verification.get("benchmark_refresh_status") or "")
    if "degraded" in benchmark_status.lower():
        warnings.append(f"benchmark_refresh:{benchmark_status}")
    tests = verification.get("tests")
    if tests and _test_status_has_warning(str(tests)):
        warnings.append(f"tests:{tests}")
    monitoring_alerts = record.get("monitoring_alerts") or []
    # 既有行为保留：regime / factor_decay 等活跃告警动作仍归到 production_regime_alert: 告警码
    non_mode_actions = [
        str(a.get("recommended_action"))
        for a in monitoring_alerts
        if a.get("recommended_action") and a.get("source") != "alert_mode"
    ]
    if non_mode_actions:
        warnings.append("production_regime_alert:" + ",".join(non_mode_actions))
    # P1-3：硬阻断姿态接入告警通知渠道（## Warnings），与 ## Monitoring Alerts 中的 alert_mode 条目同源
    if any(a.get("source") == "alert_mode" for a in monitoring_alerts):
        warnings.append("alert_mode:blocking")
    production_book_build = record.get("production_book_build") or {}
    if production_book_build:
        pb_status = production_book_build.get("status")
        if pb_status in {"stale", "missing", "invalid"}:
            warnings.append(f"production_book_build:{pb_status}")
    p10g_integration = record.get("p10g_production_integration") or {}
    if p10g_integration:
        pi_status = p10g_integration.get("status")
        if pi_status in {"missing", "invalid"}:
            warnings.append(f"p10g_production_integration:{pi_status}")
    p10g_chart = record.get("p10g_production_monitor_chart") or {}
    if p10g_chart:
        pc_status = p10g_chart.get("status")
        if pc_status in {"missing", "invalid"}:
            warnings.append(f"p10g_production_monitor_chart:{pc_status}")
    # 容量证据日更循环（MOS §109-110）：当日无任何容量证据时告警，
    # 提醒高收益结果只能标记为研究上限。
    if record.get("capacity_evidence_available") is False:
        warnings.append("capacity_evidence_missing")
    return warnings


def _test_status_has_warning(status: str) -> bool:
    normalized = status.strip().lower()
    if normalized in {"passed", "not_run_by_pipeline"}:
        return False
    failed = re.search(r"\b(\d+)\s+failed\b", normalized)
    if failed and int(failed.group(1)) > 0:
        return True
    return re.search(r"\b\d+\s+passed\b", normalized) is None


def _render_forward_section(lines: list[str], section: dict[str, Any], _label: str, enable_flag: str) -> None:
    status = section.get("status") or "unknown"
    if status == "skipped":
        lines.append(
            f"- 状态: **skipped** —— 研究性前向收益检验未启用。"
            f"人工决策：使用 `{enable_flag}` 开启。"
        )
    elif status == "no_data":
        lines.append(
            "- 状态: **no_data** —— 已运行但无样本（评分未集成或当日无符合条件标的），研究性跳过。"
        )
    elif status == "complete":
        lines.append("- 状态: **complete** —— 前向收益报告已生成。")
        if section.get("report"):
            lines.append(f"- 报告: {section.get('report')}")
    else:
        lines.append(f"- 状态: **{status}**")


def _markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# Daily Run Card {card.get('asof_date')}",
        "",
        "Research workflow reproducibility record. This file records inputs, commands, checks, and artifact hashes.",
        "",
        "## Summary",
        f"- Generated at: {card.get('generated_at')}",
        f"- Run status: {card.get('run_status')}",
        f"- Run type: {card.get('run_type')}",
        f"- Data source: {card.get('data_source')}",
        f"- Config hash: {card.get('config_hash')}",
        f"- Record hash: {card.get('record_hash')}",
        f"- Commands: {card.get('command_count')}",
        "",
        "## Failure",
        f"- {card.get('failure') or 'none'}",
        "",
        "## Verification",
    ]
    verification = card.get("verification") or {}
    if verification:
        for key, value in verification.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    execution = card.get("execution") or {}
    lines.extend(
        [
            "",
            "## Execution",
            f"- Max parallel steps: {execution.get('max_parallel_steps')}",
            f"- Wall duration seconds: {execution.get('wall_duration_seconds')}",
            f"- Summed step duration seconds: {execution.get('summed_step_duration_seconds')}",
            f"- Cache hits: {execution.get('cache_hits')}",
            "",
            "| step | status | duration_seconds | cache_hit |",
            "|---|---|---:|---:|",
        ]
    )
    step_executions = execution.get("steps") or []
    if step_executions:
        for item in step_executions:
            lines.append(
                f"| {item.get('name')} | {item.get('status')} | "
                f"{item.get('duration_seconds')} | {item.get('cache_hit')} |"
            )
    else:
        lines.append("| none | not_run | 0 | False |")
    lines.extend(["", "## Artifacts", "| key | exists | size_bytes | sha256 | path |", "|---|---:|---:|---|---|"])
    for item in card.get("artifacts") or []:
        lines.append(
            f"| {item['key']} | {item['exists']} | {item['size_bytes']} | {item['sha256'] or ''} | {item['path']} |"
        )
    warnings = card.get("warnings") or []
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(["", "## Monitoring Alerts"])
    monitoring_alerts = card.get("monitoring_alerts") or []
    if monitoring_alerts:
        for item in monitoring_alerts:
            gross_scale = item.get("recommended_gross_scale")
            scale_bit = f" (gross_scale={gross_scale})" if gross_scale is not None else ""
            lines.append(
                f"- [{item.get('source')}] {item.get('status')} "
                f"asof={item.get('asof_date')} -> {item.get('recommended_action')}{scale_bit}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Alert Mode (P1-3)"])
    block_fd = card.get("block_on_factor_decay_alert")
    block_rg = card.get("block_on_regime_alert")
    lines.append(
        f"- factor_decay: {'BLOCKING (--block-on-factor-decay-alert)' if block_fd else 'ALERT-ONLY (default)'}"
    )
    lines.append(
        f"- regime_monitor: {'BLOCKING (--block-on-regime-alert)' if block_rg else 'ALERT-ONLY (default)'}"
    )
    lines.extend(["", "## Production Book"])
    production_book_build = card.get("production_book_build") or {}
    if production_book_build:
        pb_status = production_book_build.get("status")
        pb_bits = [f"status={pb_status}"]
        if production_book_build.get("asof_date"):
            pb_bits.append(f"asof={production_book_build.get('asof_date')}")
        if production_book_build.get("rows") is not None:
            pb_bits.append(f"rows={production_book_build.get('rows')}")
        if production_book_build.get("equity_latest") is not None:
            pb_bits.append(f"equity_latest={production_book_build.get('equity_latest')}")
        if production_book_build.get("total_return") is not None:
            pb_bits.append(f"total_return={production_book_build.get('total_return')}")
        if production_book_build.get("message"):
            pb_bits.append(f"message={production_book_build.get('message')}")
        lines.append(f"- {' '.join(pb_bits)}")
    else:
        lines.append("- none")
    lines.extend(["", "## P10g Production"])
    p10g_chart = card.get("p10g_production_monitor_chart") or {}
    p10g_integration = card.get("p10g_production_integration") or {}
    if p10g_chart or p10g_integration:
        if p10g_chart:
            pc_bits = [f"monitor_chart_status={p10g_chart.get('status')}"]
            if p10g_chart.get("asof_date"):
                pc_bits.append(f"asof={p10g_chart.get('asof_date')}")
            lines.append(f"- {' '.join(pc_bits)}")
        if p10g_integration:
            pi_bits = [f"integration_status={p10g_integration.get('status')}"]
            if p10g_integration.get("asof_date"):
                pi_bits.append(f"asof={p10g_integration.get('asof_date')}")
            if p10g_integration.get("book_capacity") is not None:
                pi_bits.append(f"book_capacity={p10g_integration.get('book_capacity')}")
            if p10g_integration.get("combined_capacity") is not None:
                pi_bits.append(f"combined_capacity={p10g_integration.get('combined_capacity')}")
            if p10g_integration.get("capacity_skipped"):
                pi_bits.append("capacity_skipped=True")
            if p10g_integration.get("portfolio_skipped"):
                pi_bits.append("portfolio_skipped=True")
            lines.append(f"- {' '.join(pi_bits)}")
    else:
        lines.append("- none")
    lines.extend(["", "## Trend Ignition Shadow"])
    ti = card.get("trend_ignition_shadow") or {}
    if ti:
        ti_status = ti.get("status") or "unknown"
        if ti_status == "skipped":
            lines.append(
                f"- 状态: **skipped** — 点火影子评分未启用。"
                f"人工决策：使用 `--enable-trend-ignition-shadow` 开启日评分。"
            )
            if ti.get("training_end_date"):
                lines.append(f"- scorer training_end_date: {ti.get('training_end_date')}")
            if ti.get("selection_status"):
                lines.append(f"- scorer selection_status: {ti.get('selection_status')}")
            if ti.get("research_gate_passed") is not None:
                lines.append(f"- scorer research_gate_passed: {ti.get('research_gate_passed')}")
        elif ti_status == "complete":
            lines.append(f"- 状态: **complete**")
            if ti.get("training_end_date"):
                lines.append(f"- scorer training_end_date: {ti.get('training_end_date')}")
            if ti.get("selection_status"):
                lines.append(f"- scorer selection_status: {ti.get('selection_status')}")
            if ti.get("research_gate_passed") is not None:
                lines.append(f"- scorer research_gate_passed: {ti.get('research_gate_passed')}")
            if ti.get("source_rows") is not None:
                lines.append(f"- source_rows: {ti.get('source_rows')}")
            if ti.get("eligible_rows") is not None:
                lines.append(f"- eligible_rows: {ti.get('eligible_rows')}")
            if ti.get("eligibility_ratio") is not None:
                lines.append(f"- eligibility_ratio: {float(ti.get('eligibility_ratio')):.2%}")
            if ti.get("bucket_counts"):
                lines.append(f"- score_buckets: {ti.get('bucket_counts')}")
        else:
            lines.append(f"- 状态: **{ti_status}**")
    else:
        lines.append("- none")
    lines.extend(["", "## Trend Ignition Score Forward"])
    tisf = card.get("trend_ignition_score_forward") or {}
    if tisf:
        _render_forward_section(
            lines,
            tisf,
            "trend_ignition_score_forward",
            "--enable-trend-ignition-shadow",
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Early Pattern Forward"])
    epf = card.get("early_pattern_forward") or {}
    if epf:
        _render_forward_section(
            lines,
            epf,
            "early_pattern_forward",
            "--enable-early-pattern-forward",
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Incumbent Capacity Stress"])
    ics = card.get("incumbent_capacity_stress") or {}
    if ics:
        _render_forward_section(
            lines,
            ics,
            "incumbent_capacity_stress",
            "--enable-incumbent-capacity-stress",
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Capacity Evidence"])
    ce = card.get("capacity_evidence") or {}
    if ce:
        if ce.get("capacity_evidence_available"):
            lines.append(
                "- 状态: **available** —— 当日存在容量证据，高收益结果可作为可执行收益上限的依据。"
            )
        else:
            lines.append(
                "- 状态: **missing** —— 当日无任何容量证据。"
                "按 MOS §109-110，高收益结果只能标记为研究上限，不能作为可执行收益。"
            )
        inc_cap = ce.get("incumbent_capacity_stress_capital")
        if inc_cap is not None:
            lines.append(
                f"- incumbent 点时容量压力最大可接受资金：{float(inc_cap):,.0f} 元"
                f"（status={ce.get('incumbent_capacity_stress_status')}）"
            )
        pb = ce.get("production_book_capacity")
        if pb is not None:
            lines.append(f"- P10g 生产簿容量 book_capacity：{float(pb):,.0f} 元")
        pc = ce.get("production_combined_capacity")
        if pc is not None:
            lines.append(f"- P10g 组合容量 combined_capacity：{float(pc):,.0f} 元")
        mos_note = ce.get("mos_note")
        if mos_note:
            lines.append(f"- 备注: {mos_note}")
    else:
        lines.append("- none")
    lines.extend(["", "## Commands"])
    commands = card.get("commands") or []
    lines.extend([f"{idx}. `{command}`" for idx, command in enumerate(commands, start=1)] if commands else ["- none"])
    return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
