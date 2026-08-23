"""Forward-track preregistered institutional-accumulation shadow observations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from institutional_accumulation_shadow import (
    AccumulationConfig,
    config_hash,
    implementation_hash,
    load_config,
)
from run_backtest import load_prices, pivot_prices
from track_hidden_accumulation_watch import build_tracking_table
from train_next_open_rank_model import clean_matrix


WATCHLIST_RE = re.compile(r"institutional_accumulation_shadow_(20\d{6})\.csv$")
CN_COLUMNS = {
    "watch_date": "观察日期",
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "institutional_accumulation_level": "疑似建仓分层",
    "institutional_accumulation_score": "疑似建仓评分",
    "signal_active": "完整观察信号",
    "tracking_eligible": "纳入影子跟踪",
    "entry_date": "次日开盘观察日",
    "entry_open": "次日开盘观察价",
    "forward_return_1d": "1日跟踪收益",
    "forward_return_3d": "3日跟踪收益",
    "forward_return_5d": "5日跟踪收益",
    "forward_return_10d": "10日跟踪收益",
    "completed_horizons": "已完成周期",
    "pending_horizons": "待完成周期",
    "tracking_status": "跟踪状态",
}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_shadow_watchlists(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        match = WATCHLIST_RE.fullmatch(path.name)
        if match is None or not path.exists():
            continue
        table = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
        if "tracking_eligible" not in table:
            continue
        selected = table[table["tracking_eligible"].map(_truthy)].copy()
        if selected.empty:
            continue
        token = match.group(1)
        selected["watch_date"] = f"{token[:4]}-{token[4:6]}-{token[6:]}"
        selected["symbol"] = selected["symbol"].astype(str).str.zfill(6)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["watch_date", "symbol"], keep="last").sort_values(
        ["watch_date", "institutional_accumulation_score", "symbol"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def summarize_tracking(
    table: pd.DataFrame,
    *,
    config: AccumulationConfig,
    analysis_end_date: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    gate_frame = table.copy()
    if "signal_active" in gate_frame:
        gate_frame = gate_frame[gate_frame["signal_active"].map(_truthy)]
    else:
        gate_frame = gate_frame.iloc[0:0]
    if "watch_date" in gate_frame:
        registered_dates = pd.to_datetime(gate_frame["watch_date"], errors="coerce")
        gate_frame = gate_frame[
            registered_dates.ge(pd.Timestamp(config.validation_start_date))
        ]
    for horizon in (1, 3, 5, 10):
        column = f"forward_return_{horizon}d"
        values = pd.to_numeric(gate_frame.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                "horizon": horizon,
                "completed_samples": int(len(values)),
                "mean_return": float(values.mean()) if not values.empty else np.nan,
                "median_return": float(values.median()) if not values.empty else np.nan,
                "hit_rate": float(values.gt(0.0).mean()) if not values.empty else np.nan,
            }
        )
    summary_table = pd.DataFrame(rows)
    primary = summary_table[summary_table["horizon"].eq(config.primary_horizon)].iloc[0]
    level_counts = (
        gate_frame.loc[
            pd.to_numeric(
                gate_frame.get(f"forward_return_{config.primary_horizon}d", pd.Series(index=gate_frame.index, dtype=float)),
                errors="coerce",
            ).notna(),
            "institutional_accumulation_level",
        ].value_counts().to_dict()
        if not gate_frame.empty and "institutional_accumulation_level" in gate_frame
        else {}
    )
    enough_total = int(primary["completed_samples"]) >= config.minimum_completed_samples
    enough_buckets = bool(level_counts) and all(
        count >= config.minimum_bucket_samples for count in level_counts.values()
    )
    gate_allowed = enough_total and enough_buckets
    gate_passed = bool(
        gate_allowed
        and float(primary["hit_rate"]) >= config.minimum_primary_hit_rate
        and float(primary["mean_return"]) > config.minimum_primary_mean_return
    )
    payload = {
        "schema_version": 2,
        "status": "manual_review_ready" if gate_allowed else "provisional",
        "analysis_end_date": analysis_end_date,
        "registration_id": config.registration_id,
        "validation_start_date": config.validation_start_date,
        "config_sha256": config_hash(config),
        "scorer_implementation_sha256": implementation_hash(),
        "tracked_rows": int(len(table)),
        "active_signal_rows": int(len(gate_frame)),
        "primary_horizon": config.primary_horizon,
        "minimum_completed_samples": config.minimum_completed_samples,
        "minimum_bucket_samples": config.minimum_bucket_samples,
        "primary_completed_samples": int(primary["completed_samples"]),
        "primary_level_completed_counts": level_counts,
        "gate_evaluation_allowed": gate_allowed,
        "research_gate_passed": gate_passed if gate_allowed else None,
        "provisional_only": not gate_allowed,
        "promotion_allowed": False,
        "automatic_promotion": False,
        "research_only": True,
        "trade_instruction": False,
    }
    return summary_table, payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    table.to_csv(pending, index=False, encoding="utf-8-sig")
    pending.replace(path)


def write_outputs(
    table: pd.DataFrame,
    summary_table: pd.DataFrame,
    summary: dict[str, object],
    output: Path,
) -> dict[str, Path]:
    output = Path(output)
    chinese = output.with_name(f"{output.stem}_cn{output.suffix}")
    summary_csv = output.with_name(f"{output.stem}_summary.csv")
    summary_json = output.with_suffix(".json")
    report = output.with_suffix(".md")
    _atomic_csv(output, table)
    _atomic_csv(chinese, table.rename(columns=CN_COLUMNS))
    _atomic_csv(summary_csv, summary_table)
    _atomic_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 疑似机构提前建仓影子跟踪",
        "",
        "观察日收盘后形成信号，下一交易日开盘作为跟踪入口。结果只用于前视验证，不代表组合收益或交易指令。",
        "",
        f"- 状态: {summary['status']}",
        f"- 实际价格截止日: {summary['analysis_end_date']}",
        f"- 跟踪记录: {summary['tracked_rows']}",
        f"- 完整资金确认记录: {summary['active_signal_rows']}",
        f"- 5日已完成样本: {summary['primary_completed_samples']} / {summary['minimum_completed_samples']}",
        f"- 允许评估门槛: {summary['gate_evaluation_allowed']}",
        "- 自动晋级: 关闭",
        "",
        summary_table.to_markdown(index=False),
    ]
    _atomic_text(report, "\n".join(lines) + "\n")
    return {
        "tracking": output,
        "tracking_cn": chinese,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "report": report,
    }


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("Horizons must be positive integers")
    return horizons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-track institutional accumulation shadow signals.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--watchlist-dir", default="outputs/high_return_v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/institutional_accumulation_shadow.yaml")
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 3, 5, 10))
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config) if args.config else None)
    raw = load_prices(Path(args.data), None, None)
    open_px = clean_matrix(pivot_prices(raw, "open"), args.max_abs_daily_return)
    paths = sorted(Path(args.watchlist_dir).glob("institutional_accumulation_shadow_20*.csv"))
    watchlists = load_shadow_watchlists(paths)
    tracked = build_tracking_table(watchlists, open_px, horizons=args.horizons)
    analysis_end = open_px.index.max().strftime("%Y-%m-%d") if not open_px.empty else None
    summary_table, summary = summarize_tracking(tracked, config=config, analysis_end_date=analysis_end)
    outputs = write_outputs(tracked, summary_table, summary, Path(args.output))
    print(f"Institutional accumulation tracked rows: {len(tracked)}")
    print(f"Validation status: {summary['status']}")
    print(f"Tracking report saved to: {outputs['report']}")


if __name__ == "__main__":
    main()
