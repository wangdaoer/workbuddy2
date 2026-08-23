"""Evaluate forward returns by strategy family from merged daily watchlists."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from merged_daily_outputs import PRIORITY_WATCHLIST_CN_COLUMNS, add_strategy_family_columns
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix


WATCHLIST_RE = re.compile(r"merged_priority_watchlist_(20\d{6})\.csv$")


def clean_symbol(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else text


def date_from_token(token: str) -> str:
    return f"{token[:4]}-{token[4:6]}-{token[6:8]}"


def token_from_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons:
        raise argparse.ArgumentTypeError("At least one horizon is required.")
    if any(horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("Horizons must be positive integers.")
    return horizons


def discover_watchlists(watchlist_dir: Path, *, start: str | None = None, end: str | None = None) -> list[Path]:
    start_token = token_from_date(start) if start else None
    end_token = token_from_date(end) if end else None
    paths: list[tuple[str, Path]] = []
    for path in watchlist_dir.glob("merged_priority_watchlist_*.csv"):
        if path.name.endswith("_cn.csv"):
            continue
        match = WATCHLIST_RE.fullmatch(path.name)
        if not match:
            continue
        token = match.group(1)
        if start_token and token < start_token:
            continue
        if end_token and token > end_token:
            continue
        paths.append((token, path))
    return [path for _, path in sorted(paths)]


def load_strategy_watchlist(path: Path) -> pd.DataFrame:
    match = WATCHLIST_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Cannot infer date token from watchlist filename: {path}")
    token = match.group(1)
    table = pd.read_csv(path, dtype={"symbol": "string"}, encoding="utf-8-sig")
    table["symbol"] = table["symbol"].map(clean_symbol)
    if "strategy_family" not in table.columns or table["strategy_family"].astype("string").fillna("").str.len().eq(0).all():
        table = add_strategy_family_columns(table)
    if "strategy_family_cn" not in table.columns:
        table["strategy_family_cn"] = table["strategy_family"]
    if "strategy_family_reason" not in table.columns:
        table["strategy_family_reason"] = ""
    table["asof_date"] = date_from_token(token)
    table["source_file"] = path.name
    return table


def load_strategy_watchlists(
    watchlist_dir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    frames = [load_strategy_watchlist(path) for path in discover_watchlists(watchlist_dir, start=start, end=end)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _date_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_return(exit_open: object, entry_open: object) -> float:
    try:
        exit_value = float(exit_open)
        entry_value = float(entry_open)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(exit_value) or not np.isfinite(entry_value) or entry_value <= 0:
        return np.nan
    return exit_value / entry_value - 1.0


def _safe_adverse_return(low_values: pd.Series, entry_open: object) -> float:
    try:
        entry_value = float(entry_open)
    except (TypeError, ValueError):
        return np.nan
    lows = pd.to_numeric(low_values, errors="coerce").dropna()
    if lows.empty or not np.isfinite(entry_value) or entry_value <= 0:
        return np.nan
    return float(lows.min() / entry_value - 1.0)


def attach_strategy_family_forward_returns(
    candidates: pd.DataFrame,
    open_px: pd.DataFrame,
    low_px: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    horizons = tuple(int(horizon) for horizon in horizons)
    if candidates.empty:
        out = candidates.copy()
        for horizon in horizons:
            out[f"exit_date_{horizon}d"] = pd.Series(dtype="object")
            out[f"exit_open_{horizon}d"] = pd.Series(dtype="float")
            out[f"forward_return_{horizon}d"] = pd.Series(dtype="float")
            out[f"max_adverse_return_{horizon}d"] = pd.Series(dtype="float")
        return out

    open_px = open_px.apply(pd.to_numeric, errors="coerce").sort_index()
    low_px = low_px.apply(pd.to_numeric, errors="coerce").reindex_like(open_px)
    dates = pd.Index(pd.to_datetime(open_px.index)).sort_values()

    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        symbol = clean_symbol(row.get("symbol", ""))
        asof = pd.Timestamp(row.get("asof_date"))
        asof_pos = int(dates.searchsorted(asof, side="right")) - 1
        entry_pos = asof_pos + 1
        record = row.to_dict()
        record["symbol"] = symbol

        entry_date = dates[entry_pos] if symbol in open_px.columns and 0 <= entry_pos < len(dates) else None
        entry_open = open_px.at[entry_date, symbol] if entry_date is not None else np.nan
        record["entry_date"] = _date_text(entry_date)
        record["entry_open"] = float(entry_open) if pd.notna(entry_open) else np.nan

        for horizon in horizons:
            exit_pos = entry_pos + horizon
            exit_date = dates[exit_pos] if symbol in open_px.columns and 0 <= exit_pos < len(dates) else None
            exit_open = open_px.at[exit_date, symbol] if exit_date is not None else np.nan
            adverse_window = (
                low_px.iloc[entry_pos:exit_pos][symbol]
                if symbol in low_px.columns and 0 <= entry_pos < len(dates) and exit_pos > entry_pos
                else pd.Series(dtype="float")
            )
            record[f"exit_date_{horizon}d"] = _date_text(exit_date)
            record[f"exit_open_{horizon}d"] = float(exit_open) if pd.notna(exit_open) else np.nan
            record[f"forward_return_{horizon}d"] = _safe_return(exit_open, entry_open)
            record[f"max_adverse_return_{horizon}d"] = _safe_adverse_return(adverse_window, entry_open)
        rows.append(record)

    return pd.DataFrame(rows)


def summarize_strategy_family_forward_returns(
    samples: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 3, 5, 10),
    group_cols: tuple[str, ...] = ("strategy_family", "strategy_family_cn"),
) -> pd.DataFrame:
    columns = [
        *group_cols,
        "horizon_days",
        "signal_count",
        "completed_count",
        "pending_count",
        "avg_return",
        "median_return",
        "win_rate",
        "p25_return",
        "p75_return",
        "worst_return",
        "avg_adverse_return",
        "worst_adverse_return",
    ]
    if samples.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    group_source = samples.copy()
    for column in group_cols:
        if column not in group_source.columns:
            group_source[column] = "missing"
        group_source[column] = group_source[column].astype("string").fillna("missing")

    for keys, group in group_source.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for horizon in horizons:
            return_col = f"forward_return_{int(horizon)}d"
            adverse_col = f"max_adverse_return_{int(horizon)}d"
            returns = pd.to_numeric(group.get(return_col), errors="coerce")
            completed = returns.dropna()
            adverse = pd.to_numeric(group.get(adverse_col), errors="coerce").dropna()
            rows.append(
                {
                    **base,
                    "horizon_days": int(horizon),
                    "signal_count": int(len(group)),
                    "completed_count": int(len(completed)),
                    "pending_count": int(len(group) - len(completed)),
                    "avg_return": float(completed.mean()) if len(completed) else np.nan,
                    "median_return": float(completed.median()) if len(completed) else np.nan,
                    "win_rate": float(completed.gt(0).mean()) if len(completed) else np.nan,
                    "p25_return": float(completed.quantile(0.25)) if len(completed) else np.nan,
                    "p75_return": float(completed.quantile(0.75)) if len(completed) else np.nan,
                    "worst_return": float(completed.min()) if len(completed) else np.nan,
                    "avg_adverse_return": float(adverse.mean()) if len(adverse) else np.nan,
                    "worst_adverse_return": float(adverse.min()) if len(adverse) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values([*group_cols, "horizon_days"]).reset_index(drop=True)


def build_strategy_family_health(
    family_summary: pd.DataFrame,
    *,
    preferred_horizons: tuple[int, ...] = (3, 1),
    min_completed: int = 20,
) -> pd.DataFrame:
    columns = [
        "strategy_family",
        "strategy_family_cn",
        "selected_horizon_days",
        "signal_count",
        "completed_count",
        "pending_count",
        "avg_return",
        "win_rate",
        "avg_adverse_return",
        "worst_adverse_return",
        "family_health_status",
        "family_health_status_cn",
        "family_health_reason",
    ]
    if family_summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    grouped = family_summary.groupby(["strategy_family", "strategy_family_cn"], dropna=False)
    for (family, family_cn), group in grouped:
        selected = None
        for horizon in preferred_horizons:
            candidate = group[group["horizon_days"].eq(int(horizon))]
            if not candidate.empty and int(candidate.iloc[0].get("completed_count", 0)) >= min_completed:
                selected = candidate.iloc[0]
                break
        if selected is None:
            completed_group = group[pd.to_numeric(group["completed_count"], errors="coerce").fillna(0).gt(0)]
            selected = completed_group.iloc[0] if not completed_group.empty else group.iloc[0]

        completed_count = int(selected.get("completed_count", 0))
        avg_return = pd.to_numeric(pd.Series([selected.get("avg_return")]), errors="coerce").iloc[0]
        win_rate = pd.to_numeric(pd.Series([selected.get("win_rate")]), errors="coerce").iloc[0]
        avg_adverse = pd.to_numeric(pd.Series([selected.get("avg_adverse_return")]), errors="coerce").iloc[0]
        worst_adverse = pd.to_numeric(pd.Series([selected.get("worst_adverse_return")]), errors="coerce").iloc[0]

        if completed_count < min_completed:
            status = "insufficient"
            status_cn = "样本不足"
            reason = f"已完成样本 {completed_count} 个，低于阈值 {min_completed}，暂不评价收益强弱。"
        elif (pd.notna(avg_return) and avg_return <= -0.03) or (pd.notna(win_rate) and win_rate < 0.35) or (
            pd.notna(worst_adverse) and worst_adverse <= -0.15
        ):
            status = "cool_down"
            status_cn = "降温观察"
            reason = "近期完成样本的平均收益、胜率或最大不利波动偏弱，先降低主观追高冲动。"
        elif (pd.notna(avg_return) and avg_return < 0.0) or (pd.notna(win_rate) and win_rate < 0.45) or (
            pd.notna(avg_adverse) and avg_adverse <= -0.05
        ):
            status = "caution"
            status_cn = "谨慎观察"
            reason = "短期表现未形成正优势，继续观察但不提升权重。"
        else:
            status = "normal"
            status_cn = "正常观察"
            reason = "短期表现未触发降温阈值，可继续按原观察层级跟踪。"

        rows.append(
            {
                "strategy_family": family,
                "strategy_family_cn": family_cn,
                "selected_horizon_days": int(selected.get("horizon_days", 0)),
                "signal_count": int(selected.get("signal_count", 0)),
                "completed_count": completed_count,
                "pending_count": int(selected.get("pending_count", 0)),
                "avg_return": avg_return,
                "win_rate": win_rate,
                "avg_adverse_return": avg_adverse,
                "worst_adverse_return": worst_adverse,
                "family_health_status": status,
                "family_health_status_cn": status_cn,
                "family_health_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("strategy_family").reset_index(drop=True)


HEALTH_CN_COLUMNS = {
    "strategy_family": "策略族",
    "strategy_family_cn": "策略族中文",
    "selected_horizon_days": "评价周期",
    "signal_count": "信号数",
    "completed_count": "已完成样本数",
    "pending_count": "待完成样本数",
    "avg_return": "平均收益",
    "win_rate": "胜率",
    "avg_adverse_return": "平均最大不利波动",
    "worst_adverse_return": "最差最大不利波动",
    "family_health_status": "健康状态",
    "family_health_status_cn": "健康状态中文",
    "family_health_reason": "健康状态原因",
}


CANDIDATE_HEALTH_CN_COLUMNS = {
    "family_health_status": "策略族健康状态",
    "family_health_status_cn": "策略族健康状态中文",
    "family_health_reason": "策略族健康原因",
    "family_health_horizon_days": "策略族评价周期",
    "family_health_completed_count": "策略族已完成样本数",
    "family_health_avg_return": "策略族平均收益",
    "family_health_win_rate": "策略族胜率",
    "strategy_family_warning": "策略族风险提示",
}


SELECTED_HEALTH_CN_COLUMNS = {
    "strategy_family": "策略族",
    "strategy_family_cn": "策略族中文",
    "strategy_family_reason": "策略族原因",
    "family_health_status": "策略族健康状态",
    "family_health_status_cn": "策略族健康状态中文",
    "family_health_reason": "策略族健康原因",
    "family_health_horizon_days": "策略族评价周期",
    "family_health_completed_count": "策略族已完成样本数",
    "family_health_avg_return": "策略族平均收益",
    "family_health_win_rate": "策略族胜率",
    "strategy_family_warning": "策略族风险提示",
}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def _append_text(base: object, addition: object, *, prefix: str = "") -> str:
    base_text = "" if pd.isna(base) else str(base).strip()
    add_text = "" if pd.isna(addition) else str(addition).strip()
    if not add_text:
        return base_text
    text = f"{prefix}{add_text}" if prefix else add_text
    if not base_text:
        return text
    if text in base_text:
        return base_text
    return f"{base_text}；{text}"


def _health_warning(row: pd.Series) -> str:
    status = "" if pd.isna(row.get("family_health_status")) else str(row.get("family_health_status"))
    status_cn = "" if pd.isna(row.get("family_health_status_cn")) else str(row.get("family_health_status_cn"))
    reason = "" if pd.isna(row.get("family_health_reason")) else str(row.get("family_health_reason"))
    if status == "cool_down":
        return f"{status_cn}：{reason} 不追高，优先等待回撤修复或下一轮确认。"
    if status == "caution":
        return f"{status_cn}：{reason} 维持观察，不主动提升权重。"
    if status == "insufficient":
        return f"{status_cn}：{reason} 先观察，不作为加仓依据。"
    return ""


def _health_flag(row: pd.Series) -> str:
    status = "" if pd.isna(row.get("family_health_status")) else str(row.get("family_health_status"))
    if status in {"cool_down", "caution", "insufficient"}:
        return f"strategy_family_{status}"
    return ""


def _health_lookup(family_health: pd.DataFrame) -> pd.DataFrame:
    if family_health.empty:
        return pd.DataFrame(
            columns=[
                "strategy_family",
                "family_health_status",
                "family_health_status_cn",
                "family_health_reason",
                "family_health_horizon_days",
                "family_health_completed_count",
                "family_health_avg_return",
                "family_health_win_rate",
            ]
        )
    source = family_health.copy()
    return pd.DataFrame(
        {
            "strategy_family": source["strategy_family"],
            "family_health_status": source["family_health_status"],
            "family_health_status_cn": source["family_health_status_cn"],
            "family_health_reason": source["family_health_reason"],
            "family_health_horizon_days": source["selected_horizon_days"],
            "family_health_completed_count": source["completed_count"],
            "family_health_avg_return": source["avg_return"],
            "family_health_win_rate": source["win_rate"],
        }
    ).drop_duplicates("strategy_family", keep="last")


def annotate_priority_watchlist_with_health(priority: pd.DataFrame, family_health: pd.DataFrame) -> pd.DataFrame:
    if priority.empty:
        return priority.copy()
    result = priority.copy()
    if "strategy_family" not in result.columns:
        result = add_strategy_family_columns(result)
    health_columns = list(CANDIDATE_HEALTH_CN_COLUMNS)
    result = result.drop(columns=[column for column in health_columns if column in result.columns], errors="ignore")
    result = result.merge(_health_lookup(family_health), on="strategy_family", how="left")
    result["strategy_family_warning"] = result.apply(_health_warning, axis=1)
    health_flags = result.apply(_health_flag, axis=1)
    if "risk_flags" not in result.columns:
        result["risk_flags"] = ""
    result["risk_flags"] = [
        _append_text(existing, flag)
        for existing, flag in zip(result["risk_flags"], health_flags, strict=False)
    ]
    ordered_health = [
        "family_health_status",
        "family_health_status_cn",
        "family_health_reason",
        "family_health_horizon_days",
        "family_health_completed_count",
        "family_health_avg_return",
        "family_health_win_rate",
        "strategy_family_warning",
    ]
    base_columns = [column for column in result.columns if column not in ordered_health]
    insert_at = base_columns.index("strategy_family_reason") + 1 if "strategy_family_reason" in base_columns else len(base_columns)
    ordered = base_columns[:insert_at] + ordered_health + base_columns[insert_at:]
    return result[ordered]


def _infer_selected_strategy_family(selected: pd.DataFrame) -> pd.DataFrame:
    temp = pd.DataFrame(
        {
            "symbol": selected.get("股票代码", pd.Series(dtype=str)),
            "stock_name": selected.get("股票名称", pd.Series(dtype=str)),
            "trend_state": selected.get("趋势状态", pd.Series(dtype=str)),
        }
    )
    return add_strategy_family_columns(temp)[["symbol", "strategy_family", "strategy_family_cn", "strategy_family_reason"]]


def annotate_selected_table_with_health(
    selected: pd.DataFrame,
    annotated_priority: pd.DataFrame,
    family_health: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty or "股票代码" not in selected.columns:
        return selected.copy()
    result = selected.copy()
    generated_columns = list(SELECTED_HEALTH_CN_COLUMNS) + list(SELECTED_HEALTH_CN_COLUMNS.values())
    result = result.drop(columns=generated_columns, errors="ignore")
    result["_symbol"] = result["股票代码"].map(clean_symbol)
    priority_lookup = annotated_priority.copy()
    if not priority_lookup.empty and "symbol" in priority_lookup.columns:
        priority_lookup["symbol"] = priority_lookup["symbol"].map(clean_symbol)
        lookup_columns = [
            "symbol",
            "strategy_family",
            "strategy_family_cn",
            "strategy_family_reason",
            "family_health_status",
            "family_health_status_cn",
            "family_health_reason",
            "family_health_horizon_days",
            "family_health_completed_count",
            "family_health_avg_return",
            "family_health_win_rate",
            "strategy_family_warning",
        ]
        priority_lookup = priority_lookup[[column for column in lookup_columns if column in priority_lookup.columns]].drop_duplicates(
            "symbol", keep="first"
        )
        result = result.merge(priority_lookup, left_on="_symbol", right_on="symbol", how="left")
    missing_family = result.get("strategy_family", pd.Series([pd.NA] * len(result))).astype("string").fillna("").str.len().eq(0)
    if missing_family.any():
        inferred = _infer_selected_strategy_family(result.loc[missing_family])
        result.loc[missing_family, "strategy_family"] = inferred["strategy_family"].to_numpy()
        result.loc[missing_family, "strategy_family_cn"] = inferred["strategy_family_cn"].to_numpy()
        result.loc[missing_family, "strategy_family_reason"] = inferred["strategy_family_reason"].to_numpy()
        enriched_missing = result.loc[missing_family].drop(
            columns=[column for column in CANDIDATE_HEALTH_CN_COLUMNS if column in result.columns],
            errors="ignore",
        )
        enriched_missing = enriched_missing.merge(_health_lookup(family_health), on="strategy_family", how="left")
        for column in CANDIDATE_HEALTH_CN_COLUMNS:
            if column in enriched_missing:
                result.loc[missing_family, column] = enriched_missing[column].to_numpy()
        result.loc[missing_family, "strategy_family_warning"] = enriched_missing.apply(_health_warning, axis=1).to_numpy()

    result = result.rename(columns=SELECTED_HEALTH_CN_COLUMNS)
    if "调整原因" in result.columns and "策略族风险提示" in result.columns:
        result["调整原因"] = [
            _append_text(reason, warning, prefix="策略族提示：")
            for reason, warning in zip(result["调整原因"], result["策略族风险提示"], strict=False)
        ]
    result = result.drop(columns=["_symbol", "symbol"], errors="ignore")
    ordered_extra = [
        "策略族",
        "策略族中文",
        "策略族原因",
        "策略族健康状态中文",
        "策略族健康原因",
        "策略族风险提示",
        "策略族评价周期",
        "策略族已完成样本数",
        "策略族平均收益",
        "策略族胜率",
    ]
    base_columns = [column for column in result.columns if column not in ordered_extra]
    insert_at = base_columns.index("调整原因") + 1 if "调整原因" in base_columns else len(base_columns)
    ordered = base_columns[:insert_at] + [column for column in ordered_extra if column in result.columns] + base_columns[insert_at:]
    return result[ordered]


def annotate_candidate_tables_with_health(output_dir: Path, token: str, family_health: pd.DataFrame) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    priority_path = output_dir / f"merged_priority_watchlist_{token}.csv"
    priority = _read_csv_if_exists(priority_path)
    annotated_priority = annotate_priority_watchlist_with_health(priority, family_health)
    if not annotated_priority.empty:
        annotated_priority.to_csv(priority_path, index=False, encoding="utf-8-sig")
        priority_cn_path = output_dir / f"merged_priority_watchlist_{token}_cn.csv"
        annotated_priority.rename(columns={**PRIORITY_WATCHLIST_CN_COLUMNS, **CANDIDATE_HEALTH_CN_COLUMNS}).to_csv(
            priority_cn_path, index=False, encoding="utf-8-sig"
        )
        paths["priority"] = priority_path
        paths["priority_cn"] = priority_cn_path

    selected_path = output_dir / f"daily_personal_overlay_selected_{token}.csv"
    selected = _read_csv_if_exists(selected_path)
    annotated_selected = annotate_selected_table_with_health(selected, annotated_priority, family_health)
    if not annotated_selected.empty:
        annotated_selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
        paths["selected"] = selected_path
    return paths


def _format_report_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    percent_columns = [
        "avg_return",
        "median_return",
        "win_rate",
        "p25_return",
        "p75_return",
        "worst_return",
        "avg_adverse_return",
        "worst_adverse_return",
    ]
    for column in percent_columns:
        if column in out.columns:
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2%}")
    return out


def write_strategy_family_forward_report(
    samples: pd.DataFrame,
    family_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    family_health: pd.DataFrame,
    output_dir: Path,
    token: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "samples": output_dir / f"strategy_family_forward_returns_{token}.csv",
        "family_summary": output_dir / f"strategy_family_forward_summary_{token}.csv",
        "bucket_summary": output_dir / f"strategy_family_forward_bucket_summary_{token}.csv",
        "family_health": output_dir / f"strategy_family_health_{token}.csv",
        "family_health_cn": output_dir / f"strategy_family_health_{token}_cn.csv",
        "report": output_dir / f"strategy_family_forward_report_{token}.md",
    }
    samples.to_csv(paths["samples"], index=False, encoding="utf-8-sig")
    family_summary.to_csv(paths["family_summary"], index=False, encoding="utf-8-sig")
    bucket_summary.to_csv(paths["bucket_summary"], index=False, encoding="utf-8-sig")
    family_health.to_csv(paths["family_health"], index=False, encoding="utf-8-sig")
    family_health.rename(columns=HEALTH_CN_COLUMNS).to_csv(paths["family_health_cn"], index=False, encoding="utf-8-sig")

    lines = [
        f"# 策略族前瞻表现报告 {date_from_token(token)}",
        "",
        "口径：观察表生成后，下一交易日开盘作为进入价；持有指定交易日后按开盘价退出。最大不利波动使用持有期内最低价相对进入价计算。该报告只用于研究和复盘，不构成自动交易指令。",
        "",
        f"- 信号数：{len(samples)}",
    ]
    if not samples.empty:
        lines.append(f"- 覆盖观察日：{samples['asof_date'].min()} 至 {samples['asof_date'].max()}")
        family_counts = samples["strategy_family_cn"].fillna("未分型").value_counts().to_dict()
        lines.append(f"- 策略族分布：{family_counts}")
    lines.extend(["", "## 按策略族汇总", ""])
    lines.append(_format_report_table(family_summary).to_markdown(index=False) if not family_summary.empty else "_暂无可统计样本。_")
    lines.extend(["", "## 策略族健康状态", ""])
    lines.append(
        _format_report_table(family_health.rename(columns=HEALTH_CN_COLUMNS)).to_markdown(index=False)
        if not family_health.empty
        else "_暂无可评价样本。_"
    )
    lines.extend(["", "## 按策略族和优先层级汇总", ""])
    lines.append(_format_report_table(bucket_summary).to_markdown(index=False) if not bucket_summary.empty else "_暂无可统计样本。_")
    if not samples.empty:
        preview_cols = [
            "asof_date",
            "symbol",
            "stock_name",
            "strategy_family_cn",
            "priority_bucket",
            "entry_date",
            "forward_return_1d",
            "forward_return_3d",
            "forward_return_5d",
            "max_adverse_return_5d",
        ]
        preview = samples[[column for column in preview_cols if column in samples.columns]].head(80).copy()
        for column in [col for col in preview.columns if col.startswith("forward_return_") or col.startswith("max_adverse_return_")]:
            preview[column] = preview[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2%}")
        lines.extend(["", "## 样本预览", "", preview.to_markdown(index=False)])
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def run_strategy_family_forward_evaluation(
    data: Path,
    watchlist_dir: Path,
    output_dir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    horizons: Iterable[int] = (1, 3, 5, 10),
    token: str | None = None,
    max_abs_daily_return: float = 0.22,
) -> dict[str, Path]:
    candidates = load_strategy_watchlists(watchlist_dir, start=start, end=end)
    raw = load_prices(data, None, None)
    open_px = clean_matrix(pivot_prices(raw, "open"), max_abs_daily_return)
    low_px = clean_matrix(pivot_prices(raw, "low").reindex_like(open_px), max_abs_daily_return)
    samples = attach_strategy_family_forward_returns(candidates, open_px, low_px, horizons=horizons)
    family_summary = summarize_strategy_family_forward_returns(samples, horizons=horizons)
    family_health = build_strategy_family_health(family_summary)
    bucket_summary = summarize_strategy_family_forward_returns(
        samples,
        horizons=horizons,
        group_cols=("strategy_family", "strategy_family_cn", "priority_bucket"),
    )
    output_token = token or (token_from_date(end) if end else pd.Timestamp(open_px.index.max()).strftime("%Y%m%d"))
    paths = write_strategy_family_forward_report(samples, family_summary, bucket_summary, family_health, output_dir, output_token)
    paths.update(annotate_candidate_tables_with_health(output_dir, output_token, family_health))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate strategy-family forward returns from merged watchlists.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--watchlist-dir", default="outputs/high_return_v2")
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizons", type=parse_horizons, default=(1, 3, 5, 10))
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_strategy_family_forward_evaluation(
        Path(args.data),
        Path(args.watchlist_dir),
        Path(args.output_dir),
        start=args.start,
        end=args.end,
        horizons=args.horizons,
        token=args.token,
        max_abs_daily_return=args.max_abs_daily_return,
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
