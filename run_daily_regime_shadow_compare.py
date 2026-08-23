"""Run baseline and regime-gated satellite backtests for daily shadow comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd

from panel_io import iter_panel

from strong_pullback_evolution import load_evolution_config


METRIC_KEYS = (
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_like",
    "avg_turnover",
    "avg_gross_exposure",
)


def load_variant_params(
    config_path: Path,
    candidate_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    config = load_evolution_config(config_path)
    baseline = dict(config.baseline)
    for group in config.search_groups:
        for candidate in group.candidates:
            if candidate.candidate_id == candidate_id:
                return baseline, {**baseline, **dict(candidate.overrides)}
    raise ValueError(f"Candidate not found in evolution config: {candidate_id}")


def build_variant_command(
    *,
    python_exe: str,
    script: Path,
    data: Path,
    benchmark: Path,
    output_dir: Path,
    params: Mapping[str, object],
) -> tuple[str, ...]:
    command = [
        python_exe,
        str(script),
        "--data",
        str(data),
        "--benchmark",
        str(benchmark),
        "--output-dir",
        str(output_dir),
    ]
    basket_guard_enabled = any(
        params.get(key) is not None
        for key in ("basket_guard_return_20d_min", "basket_guard_distance_ma60_min")
    )
    if basket_guard_enabled:
        command.append("--basket-risk-guard")
    for key, value in params.items():
        if value is None:
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(option)
            continue
        command.extend((option, str(value)))
    return tuple(command)


def _validated_metrics(metrics: Mapping[str, object], label: str) -> dict[str, object]:
    result = dict(metrics)
    for key in METRIC_KEYS:
        try:
            value = float(result[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} metrics missing finite {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} metrics missing finite {key}")
        result[key] = value
    return result


def build_comparison_payload(
    *,
    asof_date: str,
    candidate_id: str,
    baseline: Mapping[str, object],
    dynamic: Mapping[str, object],
    latest_dynamic_state: Mapping[str, object],
    benchmark_last_date: str,
) -> dict[str, object]:
    baseline_metrics = _validated_metrics(baseline, "baseline")
    dynamic_metrics = _validated_metrics(dynamic, "dynamic")
    benchmark_fresh = benchmark_last_date == asof_date
    published_latest_state = dict(latest_dynamic_state)
    if not benchmark_fresh:
        published_latest_state = {
            "date": asof_date,
            "risk_regime": "unknown_stale_benchmark",
            "target_leverage": None,
            "gross_exposure": None,
            "market_exposure": None,
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asof": asof_date,
        "asof_date": asof_date,
        "baseline_id": "baseline",
        "dynamic_candidate_id": candidate_id,
        "decision": "experimental_only",
        "decision_reason": "严格滚动验证尚未晋级，正式默认策略保持 baseline。",
        "benchmark_last_date": benchmark_last_date,
        "benchmark_fresh": benchmark_fresh,
        "baseline": baseline_metrics,
        "dynamic": dynamic_metrics,
        "delta": {
            key: float(dynamic_metrics[key]) - float(baseline_metrics[key])
            for key in METRIC_KEYS
        },
        "latest_dynamic_state": published_latest_state,
    }


def _percent(value: object) -> str:
    return f"{float(value):.2%}"


def render_chinese_report(payload: Mapping[str, object]) -> str:
    baseline = payload["baseline"]
    dynamic = payload["dynamic"]
    delta = payload["delta"]
    latest = payload.get("latest_dynamic_state") or {}
    assert isinstance(baseline, Mapping)
    assert isinstance(dynamic, Mapping)
    assert isinstance(delta, Mapping)
    assert isinstance(latest, Mapping)
    target_leverage = latest.get("target_leverage")
    target_text = (
        f"{float(target_leverage):.2f}"
        if isinstance(target_leverage, (int, float)) and math.isfinite(float(target_leverage))
        else "不可用"
    )
    lines = [
        f"# 每日风险敞口双轨对照（{payload['asof_date']}）",
        "",
        "## 当前结论",
        "",
        "- 正式策略：`baseline`。",
        f"- 实验策略：`{payload['dynamic_candidate_id']}`。",
        f"- 状态：`{payload['decision']}`，{payload['decision_reason']}",
        f"- 实验策略最新市场状态：`{latest.get('risk_regime', 'unknown')}`。",
        f"- 实验策略最新目标风险敞口：{target_text}。",
        (
            f"- 基准数据：已更新至 `{payload['benchmark_last_date']}`。"
            if payload.get("benchmark_fresh")
            else f"- 风险提示：基准滞后，末日为 `{payload['benchmark_last_date']}`，"
            f"晚于该日的风险档位不得作为完整执行信号。"
        ),
        "",
        "## 全历史技术对照",
        "",
        "| 指标 | baseline | 实验策略 | 差异（实验-baseline） |",
        "| --- | ---: | ---: | ---: |",
        f"| 总收益 | {_percent(baseline['total_return'])} | {_percent(dynamic['total_return'])} | {_percent(delta['total_return'])} |",
        f"| 年化收益 | {_percent(baseline['annualized_return'])} | {_percent(dynamic['annualized_return'])} | {_percent(delta['annualized_return'])} |",
        f"| 最大回撤 | {_percent(baseline['max_drawdown'])} | {_percent(dynamic['max_drawdown'])} | {_percent(delta['max_drawdown'])} |",
        f"| Sharpe | {float(baseline['sharpe_like']):.3f} | {float(dynamic['sharpe_like']):.3f} | {float(delta['sharpe_like']):.3f} |",
        f"| 平均换手 | {_percent(baseline['avg_turnover'])} | {_percent(dynamic['avg_turnover'])} | {_percent(delta['avg_turnover'])} |",
        f"| 平均仓位 | {_percent(baseline['avg_gross_exposure'])} | {_percent(dynamic['avg_gross_exposure'])} | {_percent(delta['avg_gross_exposure'])} |",
        "",
        "该表只用于每日影子观察，不触发策略晋级、券商连接或自动下单。",
        "",
    ]
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _latest_dynamic_state(equity_path: Path) -> dict[str, object]:
    frame = pd.read_csv(equity_path)
    if frame.empty:
        raise ValueError("Dynamic equity curve is empty")
    row = frame.iloc[-1]
    return {
        "date": str(row.get("date", "")),
        "risk_regime": str(row.get("risk_regime", "unknown")),
        "target_leverage": float(row.get("target_leverage", 0.0)),
        "gross_exposure": float(row.get("gross_exposure", 0.0)),
        "market_exposure": float(row.get("market_exposure", 0.0)),
    }


def _benchmark_last_date(path: Path, asof_date: str) -> str:
    frame = pd.read_csv(path, usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    dates = dates[dates.le(pd.Timestamp(asof_date))]
    if dates.empty:
        raise ValueError(f"Benchmark has no observations on or before {asof_date}")
    return dates.max().strftime("%Y-%m-%d")


def ensure_panel_asof_date(path: Path, asof_date: str) -> None:
    latest: pd.Timestamp | None = None
    for chunk in iter_panel(path, columns=["date"], chunksize=250_000):
        dates = pd.to_datetime(chunk["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"Panel contains invalid dates: {path}")
        chunk_latest = dates.max()
        if latest is None or chunk_latest > latest:
            latest = chunk_latest
    if latest is None:
        raise ValueError(f"Panel contains no rows: {path}")
    expected = pd.Timestamp(asof_date)
    if latest.normalize() != expected.normalize():
        raise ValueError(
            f"Panel last date {latest.strftime('%Y-%m-%d')} does not match asof-date {asof_date}"
        )


def _write_comparison_csv(path: Path, payload: Mapping[str, object]) -> None:
    baseline = payload["baseline"]
    dynamic = payload["dynamic"]
    delta = payload["delta"]
    assert isinstance(baseline, Mapping)
    assert isinstance(dynamic, Mapping)
    assert isinstance(delta, Mapping)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "baseline", "dynamic", "delta"),
        )
        writer.writeheader()
        for key in METRIC_KEYS:
            writer.writerow(
                {
                    "metric": key,
                    "baseline": baseline[key],
                    "dynamic": dynamic[key],
                    "delta": delta[key],
                }
            )


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).resolve()
    data_path = Path(args.data).resolve()
    benchmark_path = Path(args.benchmark).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_panel_asof_date(data_path, args.asof_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_params, dynamic_params = load_variant_params(config_path, args.candidate_id)
    script = Path(__file__).resolve().parent / "run_strong_pullback_satellite.py"
    commands = {
        "baseline": build_variant_command(
            python_exe=args.python_exe,
            script=script,
            data=data_path,
            benchmark=benchmark_path,
            output_dir=output_dir / "baseline",
            params=baseline_params,
        ),
        "dynamic": build_variant_command(
            python_exe=args.python_exe,
            script=script,
            data=data_path,
            benchmark=benchmark_path,
            output_dir=output_dir / "dynamic",
            params=dynamic_params,
        ),
    }
    for label, command in commands.items():
        print(f"[{label}] {' '.join(command)}")
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
    payload = build_comparison_payload(
        asof_date=args.asof_date,
        candidate_id=args.candidate_id,
        baseline=_load_json(output_dir / "baseline" / "metrics.json"),
        dynamic=_load_json(output_dir / "dynamic" / "metrics.json"),
        latest_dynamic_state=_latest_dynamic_state(output_dir / "dynamic" / "equity_curve.csv"),
        benchmark_last_date=_benchmark_last_date(benchmark_path, args.asof_date),
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_comparison_csv(output_dir / "comparison.csv", payload)
    (output_dir / "report.md").write_text(render_chinese_report(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily baseline/dynamic regime shadow comparison.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--candidate-id", default="regime_090_balanced")
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    payload = run_comparison(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
