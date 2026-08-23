"""Research-only external event and macro-risk observation layer.

World Monitor is an external context provider, not an A-share execution feed.
This module intentionally emits no orders and cannot change model selection or
weights. Missing credentials, unavailable endpoints and stale caches are
reported as degraded data rather than disguised as a neutral risk reading.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WorldEventConfig:
    api_base_url: str = "https://api.worldmonitor.app"
    api_key_env: str = "WORLDMONITOR_API_KEY"
    timeout_seconds: int = 12
    max_cache_age_hours: int = 72
    stale_after_hours: int = 36
    minimum_known_components: int = 2
    endpoint_paths: dict[str, str] | None = None
    component_weights: dict[str, float] | None = None

    def endpoints(self) -> dict[str, str]:
        return self.endpoint_paths or {
            "macro_signals": "/api/economic/v1/get-macro-signals",
            "fear_greed": "/api/market/v1/get-fear-greed-index",
            "economic_stress": "/api/economic/v1/get-economic-stress",
        }

    def weights(self) -> dict[str, float]:
        return self.component_weights or {
            "macro_risk": 0.35,
            "financial_stress": 0.40,
            "fear_risk": 0.25,
        }


DIRECT_RISK_FIELDS = (
    "china_external_risk",
    "energy_shock",
    "shipping_disruption",
    "trade_policy_pressure",
)

CSV_FIELDS = (
    "asof_date",
    "status",
    "global_risk_score",
    "risk_level",
    "macro_risk",
    "financial_stress",
    "fear_risk",
    *DIRECT_RISK_FIELDS,
    "event_source_count",
    "event_confidence",
    "data_freshness",
    "research_only",
    "trade_instruction",
    "selection_effect",
)


def load_config(path: Path | None) -> WorldEventConfig:
    if path is None:
        return WorldEventConfig()
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    payload = payload.get("world_event_shadow", payload)
    if not isinstance(payload, dict):
        raise ValueError("world_event_shadow config must be a mapping")
    allowed = {item.name for item in fields(WorldEventConfig)}
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown world event settings: {', '.join(unknown)}")
    return WorldEventConfig(**payload)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _clamp(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return round(min(max(number, 0.0), 100.0), 6)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _macro_risk(payload: dict[str, Any]) -> float | None:
    if payload.get("unavailable") is True:
        return None
    verdict = str(payload.get("verdict") or "").upper()
    verdict_score = {"BUY": 20.0, "CASH": 80.0}.get(verdict)
    signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
    regime = signals.get("macroRegime") or signals.get("macro_regime") or {}
    regime_status = str(regime.get("status") or "").upper() if isinstance(regime, dict) else ""
    regime_score = {"RISK-ON": 20.0, "DEFENSIVE": 75.0}.get(regime_status)
    known = [value for value in (verdict_score, regime_score) if value is not None]
    return round(sum(known) / len(known), 6) if known else None


def _financial_stress(payload: dict[str, Any]) -> float | None:
    if payload.get("unavailable") is True:
        return None
    return _clamp(payload.get("compositeScore", payload.get("composite_score")))


def _fear_risk(payload: dict[str, Any]) -> float | None:
    if payload.get("unavailable") is True:
        return None
    greed = _clamp(payload.get("compositeScore", payload.get("composite_score")))
    return round(100.0 - greed, 6) if greed is not None else None


def _weighted_score(components: dict[str, float | None], weights: dict[str, float]) -> float | None:
    known = {
        key: value
        for key, value in components.items()
        if value is not None and _number(weights.get(key)) is not None and float(weights[key]) > 0
    }
    if not known:
        return None
    denominator = sum(float(weights[key]) for key in known)
    return round(sum(float(known[key]) * float(weights[key]) for key in known) / denominator, 6)


def _risk_level(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 70:
        return "high"
    if score >= 45:
        return "elevated"
    if score >= 30:
        return "moderate"
    return "low"


def fetch_worldmonitor(config: WorldEventConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    api_key = os.environ.get(config.api_key_env, "").strip()
    if not api_key:
        errors = [
            {"source": name, "status": "missing_api_key", "message": config.api_key_env}
            for name in config.endpoints()
        ]
        return {}, errors

    payloads: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    base = config.api_base_url.rstrip("/")
    for name, path in config.endpoints().items():
        request = urllib.request.Request(
            f"{base}/{path.lstrip('/')}",
            headers={
                "Accept": "application/json",
                "User-Agent": "model3-world-event-shadow/1.0",
                "X-WorldMonitor-Key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("response must be a JSON object")
            payloads[name] = payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"source": name, "status": "fetch_failed", "message": str(exc)})
    return payloads, errors


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("World event snapshot must be a JSON object")
    return payload


def write_payload_cache(path: Path, payloads: dict[str, Any], *, now: datetime | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "cached_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "payloads": payloads,
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_payload_cache(
    path: Path,
    config: WorldEventConfig,
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], float | None]:
    if not path.exists():
        return {}, None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payloads"), dict):
        return {}, None
    cached_at = _parse_timestamp(envelope.get("cached_at"))
    if cached_at is None:
        return {}, None
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_hours = max((current - cached_at).total_seconds() / 3600, 0.0)
    if age_hours > config.max_cache_age_hours:
        return {}, age_hours
    return envelope["payloads"], age_hours


def build_observation(
    payloads: dict[str, Any],
    *,
    asof_date: str,
    config: WorldEventConfig | None = None,
    fetch_errors: list[dict[str, Any]] | None = None,
    source_mode: str = "live_api",
    now: datetime | None = None,
) -> dict[str, Any]:
    config = config or WorldEventConfig()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    macro = payloads.get("macro_signals") or {}
    fear = payloads.get("fear_greed") or {}
    stress = payloads.get("economic_stress") or {}
    normalized = payloads.get("normalized") or {}
    if not isinstance(normalized, dict):
        normalized = {}

    components: dict[str, float | None] = {
        "macro_risk": _macro_risk(macro) if isinstance(macro, dict) else None,
        "financial_stress": _financial_stress(stress) if isinstance(stress, dict) else None,
        "fear_risk": _fear_risk(fear) if isinstance(fear, dict) else None,
    }
    direct = {field: _clamp(normalized.get(field, payloads.get(field))) for field in DIRECT_RISK_FIELDS}
    global_score = _weighted_score(components, config.weights())

    timestamps: list[datetime] = []
    for payload in (macro, fear, stress, normalized):
        if not isinstance(payload, dict):
            continue
        for key in ("timestamp", "seededAt", "seeded_at", "observed_at"):
            parsed = _parse_timestamp(payload.get(key))
            if parsed is not None:
                timestamps.append(parsed)
    latest_timestamp = max(timestamps) if timestamps else None
    age_hours = (now - latest_timestamp).total_seconds() / 3600 if latest_timestamp else None
    if age_hours is None:
        freshness = "unknown"
    elif age_hours <= config.stale_after_hours:
        freshness = "current"
    elif age_hours <= config.max_cache_age_hours:
        freshness = "stale"
    else:
        freshness = "expired"

    known_context = sum(value is not None for value in (*components.values(), *direct.values()))
    source_count = sum(
        isinstance(payloads.get(name), dict) and bool(payloads.get(name))
        for name in config.endpoints()
    )
    confidence = known_context / (len(components) + len(DIRECT_RISK_FIELDS))
    if freshness == "stale":
        confidence *= 0.7
    elif freshness in {"unknown", "expired"}:
        confidence *= 0.4
    confidence = round(confidence, 6)

    if global_score is None:
        status = "degraded"
    elif sum(value is not None for value in components.values()) < config.minimum_known_components:
        status = "partial"
    elif freshness in {"stale", "expired", "unknown"}:
        status = "partial"
    else:
        status = "research_ready"

    reasons: list[str] = []
    if global_score is None:
        reasons.append("缺少足够的外部风险数据，不能将缺失值解释为低风险")
    else:
        reasons.append(f"外部风险综合评分 {global_score:.1f}/100，分层为 {_risk_level(global_score)}")
    if fetch_errors:
        reasons.append(f"{len(fetch_errors)} 个外部端点不可用，已显式降级")
    if freshness != "current":
        reasons.append(f"数据时效状态为 {freshness}")

    return {
        "schema_version": 1,
        "asof_date": asof_date,
        "generated_at": now.isoformat(),
        "status": status,
        "source_mode": source_mode,
        "global_risk_score": global_score,
        "risk_level": _risk_level(global_score),
        **components,
        **direct,
        "event_source_count": int(source_count),
        "known_component_count": int(known_context),
        "event_confidence": confidence,
        "data_freshness": freshness,
        "latest_source_timestamp": latest_timestamp.isoformat() if latest_timestamp else None,
        "source_age_hours": round(age_hours, 3) if age_hours is not None else None,
        "reasons_cn": reasons,
        "fetch_errors": list(fetch_errors or []),
        "research_only": True,
        "trade_instruction": False,
        "selection_effect": False,
        "portfolio_weight_effect": 0.0,
        "automatic_promotion": False,
        "config": asdict(config),
    }


def _write_csv(path: Path, observation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow({field: observation.get(field) for field in CSV_FIELDS})


def _display(value: Any) -> str:
    if value is None:
        return "不可用"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_markdown(path: Path, observation: dict[str, Any]) -> None:
    rows = [
        ("综合风险评分", observation.get("global_risk_score")),
        ("风险分层", observation.get("risk_level")),
        ("宏观风险", observation.get("macro_risk")),
        ("金融压力", observation.get("financial_stress")),
        ("恐惧风险", observation.get("fear_risk")),
        ("中国外部风险", observation.get("china_external_risk")),
        ("能源冲击", observation.get("energy_shock")),
        ("航运中断", observation.get("shipping_disruption")),
        ("贸易政策压力", observation.get("trade_policy_pressure")),
    ]
    lines = [
        f"# 全球事件影子观察 {observation['asof_date']}",
        "",
        f"- 状态：`{observation['status']}`",
        f"- 数据时效：`{observation['data_freshness']}`",
        f"- 事件置信度：`{observation['event_confidence']:.2%}`",
        "- 影响正式选股：`False`",
        "- 交易指令：`False`",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        *[f"| {label} | {_display(value)} |" for label, value in rows],
        "",
        "## 审计说明",
        "",
        *[f"- {reason}" for reason in observation.get("reasons_cn", [])],
        "- 该模块仅提供外部环境背景，不改变 Model 3 的候选、仓位或订单。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_json: Path, observation: dict[str, Any]) -> tuple[Path, Path]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(observation, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    csv_path = output_json.with_suffix(".csv")
    report_path = output_json.with_suffix(".md")
    _write_csv(csv_path, observation)
    _write_markdown(report_path, observation)
    return csv_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only World Monitor context layer.")
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.snapshot is not None:
        payloads = load_snapshot(args.snapshot)
        errors: list[dict[str, Any]] = []
        source_mode = "local_snapshot"
    else:
        payloads, errors = fetch_worldmonitor(config)
        source_mode = "live_api"
        if args.cache is not None:
            cached, cache_age = load_payload_cache(args.cache, config)
            missing_sources = [name for name in config.endpoints() if name not in payloads]
            recovered = [name for name in missing_sources if name in cached]
            if recovered:
                payloads.update({name: cached[name] for name in recovered})
                errors.append(
                    {
                        "source": "payload_cache",
                        "status": "cache_fallback",
                        "message": f"recovered={','.join(recovered)}; age_hours={cache_age:.3f}",
                    }
                )
                source_mode = "live_api_with_cache" if len(recovered) < len(payloads) else "cache_fallback"
            live_payloads = {
                name: payloads[name]
                for name in config.endpoints()
                if name in payloads and name not in recovered
            }
            if live_payloads:
                merged_cache = dict(cached)
                merged_cache.update(live_payloads)
                write_payload_cache(args.cache, merged_cache)
    observation = build_observation(
        payloads,
        asof_date=args.asof_date,
        config=config,
        fetch_errors=errors,
        source_mode=source_mode,
    )
    csv_path, report_path = write_outputs(args.output, observation)
    print(args.output)
    print(csv_path)
    print(report_path)
    print(json.dumps({key: observation[key] for key in ("status", "global_risk_score", "data_freshness", "selection_effect")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
