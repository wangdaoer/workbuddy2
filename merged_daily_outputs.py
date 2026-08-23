"""Build merged daily views for the model workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from build_daily_personal_overlay_report import load_name_map
from shadow_account_signals import (
    apply_shadow_account_signals,
    load_shadow_account_review,
    shadow_account_summary_lines,
)


DEFAULT_CONSTRUCTIVE_TREND_STATES = {
    "趋势确认",
    "生命线健康",
    "回调可观察",
    "起爆观察",
    "trend_confirmed",
    "life_line_ok",
    "pullback_watch",
    "ignition_watch",
}

STATE_PATTERN_CN_COLUMNS = {
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "strategy_family": "策略族",
    "strategy_family_cn": "策略族中文",
    "strategy_family_reason": "策略族原因",
    "trend_state": "趋势状态",
    "trend_score": "趋势分",
    "pattern_type": "观察形态",
    "pattern_score": "形态分",
    "state_pattern_bucket": "合并判断",
    "close": "收盘价",
    "return_5d": "5日涨跌幅",
    "return_20d": "20日涨跌幅",
    "return_60d": "60日涨跌幅",
    "return_120d": "120日涨跌幅",
    "close_position": "区间位置",
    "amount_ratio": "成交额放大倍数",
    "pattern_reason": "观察理由",
}

MODEL_DECISION_CN_COLUMNS = {
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "strategy_family": "策略族",
    "strategy_family_cn": "策略族中文",
    "strategy_family_reason": "策略族原因",
    "base_rank": "原模型排名",
    "personal_rank": "个人规则后排名",
    "base_selected": "原模型入选",
    "personal_selected": "个人规则后入选",
    "base_target_weight": "原目标权重",
    "personal_target_weight": "个人规则后权重",
    "target_weight_delta": "权重变化",
    "base_score": "原模型分数",
    "personal_adjusted_score": "个人规则后分数",
    "score_delta": "分数变化",
    "personal_action": "个人规则动作",
    "personal_action_cn": "个人规则动作中文",
    "personal_reasons": "个人规则原因",
    "personal_reasons_cn": "个人规则原因中文",
    "decision_layer": "合并判断",
    "trend_state": "趋势状态",
    "close": "收盘价",
    "return_5d": "5日涨跌幅",
    "return_20d": "20日涨跌幅",
    "return_60d": "60日涨跌幅",
    "close_position": "区间位置",
}

PRIORITY_WATCHLIST_CN_COLUMNS = {
    "symbol": "\u80a1\u7968\u4ee3\u7801",
    "stock_name": "\u80a1\u7968\u540d\u79f0",
    "strategy_family": "\u7b56\u7565\u65cf",
    "strategy_family_cn": "\u7b56\u7565\u65cf\u4e2d\u6587",
    "strategy_family_reason": "\u7b56\u7565\u65cf\u539f\u56e0",
    "priority_bucket": "\u4f18\u5148\u5c42\u7ea7",
    "priority_score": "\u4f18\u5148\u5206",
    "personal_rank": "\u4e2a\u4eba\u89c4\u5219\u540e\u6392\u540d",
    "personal_selected": "\u4e2a\u4eba\u89c4\u5219\u540e\u5165\u9009",
    "personal_target_weight": "\u4e2a\u4eba\u89c4\u5219\u540e\u6743\u91cd",
    "decision_layer": "\u6a21\u578b+\u4e2a\u4eba\u89c4\u5219\u5224\u65ad",
    "state_pattern_bucket": "\u8d8b\u52bf+\u5f62\u6001\u5224\u65ad",
    "trend_state": "\u8d8b\u52bf\u72b6\u6001",
    "pattern_type": "\u89c2\u5bdf\u5f62\u6001",
    "pattern_score": "\u5f62\u6001\u5206",
    "target_weight_delta": "\u6743\u91cd\u53d8\u5316",
    "personal_action": "\u4e2a\u4eba\u89c4\u5219\u52a8\u4f5c",
    "personal_action_cn": "\u4e2a\u4eba\u89c4\u5219\u52a8\u4f5c\u4e2d\u6587",
    "personal_reasons_cn": "\u4e2a\u4eba\u89c4\u5219\u539f\u56e0\u4e2d\u6587",
    "risk_flags": "\u98ce\u9669\u6807\u8bb0",
    "pattern_reason": "\u89c2\u5bdf\u7406\u7531",
    "close": "\u6536\u76d8\u4ef7",
    "return_5d": "5\u65e5\u6da8\u8dcc\u5e45",
    "return_20d": "20\u65e5\u6da8\u8dcc\u5e45",
    "return_60d": "60\u65e5\u6da8\u8dcc\u5e45",
    "close_position": "\u533a\u95f4\u4f4d\u7f6e",
    "shadow_account_signal_score": "\u5f71\u5b50\u8d26\u6237\u5206",
    "shadow_account_signal": "\u5f71\u5b50\u8d26\u6237\u63d0\u793a",
    "shadow_account_signal_cn": "\u5f71\u5b50\u8d26\u6237\u63d0\u793a\u4e2d\u6587",
    "shadow_account_notes": "\u5f71\u5b50\u8d26\u6237\u8bf4\u660e",
    "trend_ignition_score": "\u8d8b\u52bf\u8d77\u7206\u8bc4\u5206",
    "trend_ignition_score_bucket": "\u8d8b\u52bf\u8d77\u7206\u5206\u5c42",
    "trend_ignition_deployment_status": "\u8d8b\u52bf\u8d77\u7206\u7814\u7a76\u72b6\u6001",
    "trend_ignition_research_gate_passed": "\u8d8b\u52bf\u8d77\u7206\u7814\u7a76\u95f8\u95e8",
    "trend_ignition_selection_status": "\u8d8b\u52bf\u8d77\u7206\u9009\u62e9\u72b6\u6001",
}


def _date_token(asof_date: str) -> str:
    return asof_date.replace("-", "")


def _normalize_symbol(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    if digits:
        return digits.zfill(6)
    return text


def _normalize_symbol_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "symbol" not in result.columns:
        result["symbol"] = pd.Series(dtype="object")
    result["symbol"] = result["symbol"].map(_normalize_symbol)
    return result


def _series(frame: pd.DataFrame, column: str, default: object = pd.NA) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _first_series(frame: pd.DataFrame, columns: Iterable[str], default: object = pd.NA) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _bool(series: pd.Series, fallback: pd.Series | None = None) -> pd.Series:
    if series.isna().all() and fallback is not None:
        return _numeric(fallback, 0.0).gt(0)
    mapped = series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y", "selected"}
        if not isinstance(value, bool)
        else value
    )
    return mapped.fillna(False).astype(bool)


def _bool_without_na(series: pd.Series) -> pd.Series:
    return series.map(lambda value: False if pd.isna(value) else bool(value)).astype(bool)


def _choose_non_empty(left: pd.Series, right: pd.Series) -> pd.Series:
    left_text = left.astype("string")
    right_text = right.astype("string")
    return left_text.where(left_text.notna() & (left_text.str.len() > 0), right_text)


def _choose_present(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.where(left.notna(), right)


def _ordered_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def _st_name_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.upper()
    return text.str.contains("ST", regex=False)


def _infer_strategy_family(row: pd.Series) -> tuple[str, str, str]:
    pattern = "" if pd.isna(row.get("pattern_type", pd.NA)) else str(row.get("pattern_type", "")).lower()
    trend = "" if pd.isna(row.get("trend_state", pd.NA)) else str(row.get("trend_state", "")).lower()
    bucket = "" if pd.isna(row.get("state_pattern_bucket", pd.NA)) else str(row.get("state_pattern_bucket", "")).lower()
    decision = "" if pd.isna(row.get("decision_layer", pd.NA)) else str(row.get("decision_layer", "")).lower()
    combined = "|".join([pattern, trend, bucket, decision])

    if any(token in combined for token in ["隐性吸筹", "hidden_accumulation"]):
        return (
            "hidden_accumulation",
            "隐性吸筹",
            "温和放量、上涨日成交更活跃，先作为观察和跟踪。",
        )
    if any(token in combined for token in ["趋势二波", "second_wave", "pullback", "回调"]):
        return (
            "strong_pullback",
            "强势回调二波",
            "强势趋势后的回调或二波启动，适合作为卫星候选。",
        )
    if any(token in combined for token in ["突破", "breakout", "platform", "平台", "起爆"]):
        return (
            "breakout_acceleration",
            "突破加速",
            "平台突破或起爆状态，适合提前观察加速段。",
        )
    if any(token in combined for token in ["趋势确认", "生命线健康", "trend_confirmed", "life_line_ok", "model_focus"]):
        return (
            "trend_momentum",
            "趋势动量",
            "趋势结构较健康，主要由模型排序和趋势状态驱动。",
        )
    return (
        "unclassified",
        "未分型",
        "暂未匹配到明确策略族，仅保留为普通观察。",
    )


def add_strategy_family_columns(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if result.empty:
        result["strategy_family"] = pd.Series(dtype="object")
        result["strategy_family_cn"] = pd.Series(dtype="object")
        result["strategy_family_reason"] = pd.Series(dtype="object")
        return result
    inferred = result.apply(_infer_strategy_family, axis=1, result_type="expand")
    result["strategy_family"] = inferred[0]
    result["strategy_family_cn"] = inferred[1]
    result["strategy_family_reason"] = inferred[2]
    return result


def fill_missing_stock_names(table: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    if not name_map:
        return table
    result = _normalize_symbol_column(table)
    if "stock_name" not in result.columns:
        result["stock_name"] = pd.NA
    mapped_names = result["symbol"].map(name_map)
    current_names = result["stock_name"].astype("string")
    missing_name = current_names.isna() | current_names.str.strip().eq("")
    result.loc[missing_name, "stock_name"] = mapped_names[missing_name]
    return result


def build_state_pattern_scan(
    trend_state: pd.DataFrame,
    watchlist: pd.DataFrame,
    constructive_trend_states: set[str] | None = None,
) -> pd.DataFrame:
    constructive = constructive_trend_states or DEFAULT_CONSTRUCTIVE_TREND_STATES
    trend = _normalize_symbol_column(trend_state)
    patterns = _normalize_symbol_column(watchlist)

    trend_part = pd.DataFrame(
        {
            "symbol": _series(trend, "symbol"),
            "trend_state": _series(trend, "trend_state"),
            "trend_score": _series(trend, "trend_score"),
        }
    ).drop_duplicates("symbol", keep="last")

    merged = patterns.merge(trend_part, on="symbol", how="left", suffixes=("", "_trend"))
    pattern_text = _series(merged, "pattern_type").astype("string")
    trend_text = _series(merged, "trend_state").astype("string")
    has_pattern = pattern_text.notna() & (pattern_text.str.len() > 0)
    has_trend = trend_text.notna() & (trend_text.str.len() > 0)
    constructive_trend = trend_text.isin(constructive)

    merged["state_pattern_bucket"] = "pattern_needs_review"
    merged.loc[has_pattern & constructive_trend, "state_pattern_bucket"] = "pattern_confirmed_by_trend"
    merged.loc[has_pattern & ~has_trend, "state_pattern_bucket"] = "pattern_without_trend_state"
    merged.loc[~has_pattern & constructive_trend, "state_pattern_bucket"] = "trend_only"

    if "pattern_score" in merged.columns:
        merged["_sort_score"] = _numeric(merged["pattern_score"], 0.0)
        merged = merged.sort_values(["_sort_score", "symbol"], ascending=[False, True])
    else:
        merged = merged.sort_values("symbol")

    columns = [
        "symbol",
        "stock_name",
        "strategy_family",
        "strategy_family_cn",
        "strategy_family_reason",
        "trend_state",
        "trend_score",
        "pattern_type",
        "pattern_score",
        "state_pattern_bucket",
        "close",
        "return_5d",
        "return_20d",
        "return_60d",
        "return_120d",
        "close_position",
        "amount_ratio",
        "pattern_reason",
    ]
    merged = add_strategy_family_columns(merged)
    return _ordered_columns(merged.drop(columns=["_sort_score"], errors="ignore"), columns).reset_index(drop=True)


def build_model_decision_table(base: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    base_norm = _normalize_symbol_column(base)
    overlay_norm = _normalize_symbol_column(overlay)

    base_part = pd.DataFrame(
        {
            "symbol": _series(base_norm, "symbol"),
            "stock_name_base": _first_series(base_norm, ("stock_name", "security_name", "name")),
            "base_rank": _first_series(base_norm, ("rank", "model_rank")),
            "base_selected": _bool(_series(base_norm, "selected"), _series(base_norm, "target_weight")),
            "base_target_weight": _numeric(_series(base_norm, "target_weight"), 0.0),
            "base_score": _first_series(base_norm, ("score", "factor_score", "final_score")),
            "trend_state_base": _series(base_norm, "trend_state"),
            "close_base": _series(base_norm, "close"),
            "return_5d_base": _series(base_norm, "return_5d"),
            "return_20d_base": _series(base_norm, "return_20d"),
            "return_60d_base": _series(base_norm, "return_60d"),
            "close_position_base": _series(base_norm, "close_position"),
        }
    ).drop_duplicates("symbol", keep="last")
    personal_target = _numeric(
        _first_series(
            overlay_norm,
            ("personal_adjusted_target_weight", "target_weight_after_behavior", "target_weight"),
        ),
        0.0,
    )
    overlay_part = pd.DataFrame(
        {
            "symbol": _series(overlay_norm, "symbol"),
            "stock_name_personal": _first_series(overlay_norm, ("stock_name", "security_name", "name")),
            "personal_rank": _series(overlay_norm, "personal_rank"),
            "personal_selected": _bool(_series(overlay_norm, "personal_selected"), personal_target),
            "personal_target_weight": personal_target,
            "personal_adjusted_score": _first_series(overlay_norm, ("personal_adjusted_score", "final_score")),
            "personal_action": _series(overlay_norm, "personal_action"),
            "personal_action_cn": _series(overlay_norm, "personal_action_cn"),
            "personal_reasons": _series(overlay_norm, "personal_reasons"),
            "personal_reasons_cn": _series(overlay_norm, "personal_reasons_cn"),
            "trend_state_personal": _series(overlay_norm, "trend_state"),
            "close_personal": _series(overlay_norm, "close"),
            "return_5d_personal": _series(overlay_norm, "return_5d"),
            "return_20d_personal": _series(overlay_norm, "return_20d"),
            "return_60d_personal": _series(overlay_norm, "return_60d"),
            "close_position_personal": _series(overlay_norm, "close_position"),
        }
    ).drop_duplicates("symbol", keep="last")

    merged = base_part.merge(overlay_part, on="symbol", how="outer")
    merged["stock_name"] = _choose_non_empty(merged["stock_name_personal"], merged["stock_name_base"])
    merged["trend_state"] = _choose_non_empty(merged["trend_state_personal"], merged["trend_state_base"])
    merged["close"] = _choose_present(merged["close_personal"], merged["close_base"])
    merged["return_5d"] = _choose_present(merged["return_5d_personal"], merged["return_5d_base"])
    merged["return_20d"] = _choose_present(merged["return_20d_personal"], merged["return_20d_base"])
    merged["return_60d"] = _choose_present(merged["return_60d_personal"], merged["return_60d_base"])
    merged["close_position"] = _choose_present(merged["close_position_personal"], merged["close_position_base"])

    merged["base_target_weight"] = _numeric(merged["base_target_weight"], 0.0)
    merged["personal_target_weight"] = _numeric(merged["personal_target_weight"], 0.0)
    merged["target_weight_delta"] = merged["personal_target_weight"] - merged["base_target_weight"]
    merged["base_score"] = pd.to_numeric(merged["base_score"], errors="coerce")
    merged["personal_adjusted_score"] = pd.to_numeric(merged["personal_adjusted_score"], errors="coerce")
    merged["score_delta"] = merged["personal_adjusted_score"] - merged["base_score"]
    merged["base_selected"] = _bool_without_na(merged["base_selected"])
    merged["personal_selected"] = _bool_without_na(merged["personal_selected"])

    merged["decision_layer"] = "unchanged_candidate"
    base_selected = merged["base_selected"]
    personal_selected = merged["personal_selected"]
    reduced = personal_selected & base_selected & (merged["personal_target_weight"] < merged["base_target_weight"] - 1e-12)
    merged.loc[base_selected & personal_selected, "decision_layer"] = "kept_by_overlay"
    merged.loc[reduced, "decision_layer"] = "reduced_by_overlay"
    merged.loc[base_selected & ~personal_selected, "decision_layer"] = "removed_by_overlay"
    merged.loc[~base_selected & personal_selected, "decision_layer"] = "added_by_overlay"

    merged["_sort_rank"] = pd.to_numeric(merged["personal_rank"], errors="coerce").combine_first(
        pd.to_numeric(merged["base_rank"], errors="coerce")
    )
    merged = merged.sort_values(["_sort_rank", "symbol"], na_position="last")

    columns = [
        "symbol",
        "stock_name",
        "strategy_family",
        "strategy_family_cn",
        "strategy_family_reason",
        "base_rank",
        "personal_rank",
        "base_selected",
        "personal_selected",
        "base_target_weight",
        "personal_target_weight",
        "target_weight_delta",
        "base_score",
        "personal_adjusted_score",
        "score_delta",
        "personal_action",
        "personal_action_cn",
        "personal_reasons",
        "personal_reasons_cn",
        "decision_layer",
        "trend_state",
        "close",
        "return_5d",
        "return_20d",
        "return_60d",
        "close_position",
    ]
    merged = add_strategy_family_columns(merged)
    return _ordered_columns(merged.drop(columns=["_sort_rank"], errors="ignore"), columns).reset_index(drop=True)


def build_priority_watchlist(
    state_pattern_scan: pd.DataFrame,
    model_decision_table: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    state_norm = _normalize_symbol_column(state_pattern_scan)
    model_norm = _normalize_symbol_column(model_decision_table)

    state_part = pd.DataFrame(
        {
            "symbol": _series(state_norm, "symbol"),
            "stock_name_pattern": _series(state_norm, "stock_name"),
            "trend_state_pattern": _series(state_norm, "trend_state"),
            "pattern_type": _series(state_norm, "pattern_type"),
            "pattern_score": _numeric(_series(state_norm, "pattern_score"), 0.0),
            "state_pattern_bucket": _series(state_norm, "state_pattern_bucket"),
            "strategy_family_pattern": _series(state_norm, "strategy_family"),
            "strategy_family_cn_pattern": _series(state_norm, "strategy_family_cn"),
            "strategy_family_reason_pattern": _series(state_norm, "strategy_family_reason"),
            "pattern_reason": _series(state_norm, "pattern_reason"),
            "close_pattern": _series(state_norm, "close"),
            "return_5d_pattern": _series(state_norm, "return_5d"),
            "return_20d_pattern": _series(state_norm, "return_20d"),
            "return_60d_pattern": _series(state_norm, "return_60d"),
            "close_position_pattern": _series(state_norm, "close_position"),
        }
    ).drop_duplicates("symbol", keep="last")
    model_part = pd.DataFrame(
        {
            "symbol": _series(model_norm, "symbol"),
            "stock_name_model": _series(model_norm, "stock_name"),
            "personal_rank": _series(model_norm, "personal_rank"),
            "personal_selected": _bool(_series(model_norm, "personal_selected"), _series(model_norm, "personal_target_weight")),
            "personal_target_weight": _numeric(_series(model_norm, "personal_target_weight"), 0.0),
            "decision_layer": _series(model_norm, "decision_layer"),
            "strategy_family_model": _series(model_norm, "strategy_family"),
            "strategy_family_cn_model": _series(model_norm, "strategy_family_cn"),
            "strategy_family_reason_model": _series(model_norm, "strategy_family_reason"),
            "target_weight_delta": _numeric(_series(model_norm, "target_weight_delta"), 0.0),
            "personal_action": _series(model_norm, "personal_action"),
            "personal_action_cn": _series(model_norm, "personal_action_cn"),
            "personal_reasons_cn": _series(model_norm, "personal_reasons_cn"),
            "trend_state_model": _series(model_norm, "trend_state"),
            "close_model": _series(model_norm, "close"),
            "return_5d_model": _series(model_norm, "return_5d"),
            "return_20d_model": _series(model_norm, "return_20d"),
            "return_60d_model": _series(model_norm, "return_60d"),
            "close_position_model": _series(model_norm, "close_position"),
        }
    ).drop_duplicates("symbol", keep="last")

    merged = state_part.merge(model_part, on="symbol", how="outer")
    merged["stock_name"] = _choose_non_empty(merged["stock_name_model"], merged["stock_name_pattern"])
    merged["trend_state"] = _choose_non_empty(merged["trend_state_model"], merged["trend_state_pattern"])
    merged["strategy_family"] = _choose_non_empty(merged["strategy_family_pattern"], merged["strategy_family_model"])
    merged["strategy_family_cn"] = _choose_non_empty(merged["strategy_family_cn_pattern"], merged["strategy_family_cn_model"])
    merged["strategy_family_reason"] = _choose_non_empty(
        merged["strategy_family_reason_pattern"], merged["strategy_family_reason_model"]
    )
    merged["close"] = _choose_present(merged["close_model"], merged["close_pattern"])
    merged["return_5d"] = _choose_present(merged["return_5d_model"], merged["return_5d_pattern"])
    merged["return_20d"] = _choose_present(merged["return_20d_model"], merged["return_20d_pattern"])
    merged["return_60d"] = _choose_present(merged["return_60d_model"], merged["return_60d_pattern"])
    merged["close_position"] = _choose_present(merged["close_position_model"], merged["close_position_pattern"])

    merged["personal_selected"] = _bool_without_na(merged["personal_selected"])
    confirmed_pattern = merged["state_pattern_bucket"].astype("string").eq("pattern_confirmed_by_trend")
    model_selected = merged["personal_selected"]
    decision = merged["decision_layer"].astype("string").fillna("")

    merged["priority_bucket"] = "review_later"
    merged.loc[confirmed_pattern, "priority_bucket"] = "pattern_watch"
    merged.loc[model_selected, "priority_bucket"] = "model_focus"
    merged.loc[model_selected & confirmed_pattern, "priority_bucket"] = "action_focus"
    st_risk = _st_name_mask(merged["stock_name"])
    merged["risk_flags"] = ""
    merged.loc[st_risk, "priority_bucket"] = "risk_watch"
    merged.loc[st_risk, "risk_flags"] = "st_or_special_treatment"
    merged.loc[st_risk, "personal_selected"] = False
    merged.loc[st_risk, "personal_target_weight"] = 0.0

    bucket_rank = merged["priority_bucket"].map(
        {
            "action_focus": 3,
            "model_focus": 2,
            "risk_watch": 1.5,
            "pattern_watch": 1,
            "review_later": 0,
        }
    ).fillna(0)
    score = bucket_rank.astype(float) * 100.0
    score += _numeric(merged["pattern_score"], 0.0) * 10.0
    score += _numeric(merged["personal_target_weight"], 0.0) * 100.0
    score += decision.eq("added_by_overlay").astype(float) * 10.0
    score += decision.eq("kept_by_overlay").astype(float) * 8.0
    score += decision.eq("reduced_by_overlay").astype(float) * 4.0
    score -= decision.eq("removed_by_overlay").astype(float) * 10.0
    merged["priority_score"] = score.round(4)
    merged["_bucket_rank"] = bucket_rank
    needs_family = merged["strategy_family"].astype("string").fillna("").str.len().eq(0)
    if needs_family.any():
        inferred = add_strategy_family_columns(merged.loc[needs_family])
        for column in ["strategy_family", "strategy_family_cn", "strategy_family_reason"]:
            merged.loc[needs_family, column] = inferred[column]

    merged["_sort_rank"] = pd.to_numeric(merged["personal_rank"], errors="coerce")
    merged = merged.sort_values(
        ["_bucket_rank", "priority_score", "_sort_rank", "pattern_score", "symbol"],
        ascending=[False, False, True, False, True],
        na_position="last",
    )
    if top_n > 0:
        executable = merged[merged["priority_bucket"].isin(["action_focus", "model_focus"])]
        risk_watch = merged[merged["priority_bucket"].eq("risk_watch")].head(5)
        supplemental = merged[~merged["priority_bucket"].isin(["action_focus", "model_focus", "risk_watch"])]
        merged = pd.concat([executable, risk_watch, supplemental], ignore_index=False).head(top_n)

    columns = [
        "symbol",
        "stock_name",
        "strategy_family",
        "strategy_family_cn",
        "strategy_family_reason",
        "priority_bucket",
        "priority_score",
        "personal_rank",
        "personal_selected",
        "personal_target_weight",
        "decision_layer",
        "state_pattern_bucket",
        "trend_state",
        "pattern_type",
        "pattern_score",
        "target_weight_delta",
        "personal_action",
        "personal_action_cn",
        "personal_reasons_cn",
        "risk_flags",
        "pattern_reason",
        "close",
        "return_5d",
        "return_20d",
        "return_60d",
        "close_position",
    ]
    return _ordered_columns(merged.drop(columns=["_sort_rank", "_bucket_rank"], errors="ignore"), columns).reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"symbol": "string"}, encoding="utf-8-sig")


def _pending_path(path: Path) -> Path:
    for suffix in ["_pending", "_pending2", "_pending3"]:
        candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}_pending_latest{path.suffix}")


def _write_table(path: Path, table: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except (PermissionError, IsADirectoryError):
        fallback = _pending_path(path)
        table.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def _markdown_table(table: pd.DataFrame, max_rows: int = 20) -> str:
    if table.empty:
        return "_No rows._"
    view = table.head(max_rows).copy()
    values = [[str(value) if not pd.isna(value) else "" for value in row] for row in view.to_numpy()]
    columns = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in values:
        lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    if len(table) > max_rows:
        lines.append(f"\n_Only first {max_rows} rows shown; full table is in the CSV._")
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    asof_date: str,
    state_pattern_scan: pd.DataFrame,
    model_decision_table: pd.DataFrame,
    shadow_account_review: dict[str, object] | None = None,
) -> dict[str, Path]:
    token = _date_token(asof_date)
    priority_watchlist = build_priority_watchlist(state_pattern_scan, model_decision_table)
    priority_watchlist = apply_shadow_account_signals(
        priority_watchlist, shadow_account_review
    )
    paths = {
        "state_pattern_scan": output_dir / f"merged_state_pattern_scan_{token}.csv",
        "state_pattern_scan_cn": output_dir / f"merged_state_pattern_scan_{token}_cn.csv",
        "model_decision_table": output_dir / f"merged_model_decision_table_{token}.csv",
        "model_decision_table_cn": output_dir / f"merged_model_decision_table_{token}_cn.csv",
        "priority_watchlist": output_dir / f"merged_priority_watchlist_{token}.csv",
        "priority_watchlist_cn": output_dir / f"merged_priority_watchlist_{token}_cn.csv",
        "summary": output_dir / f"merged_daily_outputs_{token}.md",
    }
    paths["state_pattern_scan"] = _write_table(paths["state_pattern_scan"], state_pattern_scan)
    paths["state_pattern_scan_cn"] = _write_table(
        paths["state_pattern_scan_cn"], state_pattern_scan.rename(columns=STATE_PATTERN_CN_COLUMNS)
    )
    paths["model_decision_table"] = _write_table(paths["model_decision_table"], model_decision_table)
    paths["model_decision_table_cn"] = _write_table(
        paths["model_decision_table_cn"], model_decision_table.rename(columns=MODEL_DECISION_CN_COLUMNS)
    )
    paths["priority_watchlist"] = _write_table(paths["priority_watchlist"], priority_watchlist)
    paths["priority_watchlist_cn"] = _write_table(
        paths["priority_watchlist_cn"], priority_watchlist.rename(columns=PRIORITY_WATCHLIST_CN_COLUMNS)
    )

    pattern_counts = state_pattern_scan["state_pattern_bucket"].value_counts().to_dict()
    decision_counts = model_decision_table["decision_layer"].value_counts().to_dict()
    priority_counts = priority_watchlist["priority_bucket"].value_counts().to_dict()
    summary = [
        f"# Merged Daily Outputs {asof_date}",
        "",
        "## Priority Watchlist",
        "",
        f"- rows: {len(priority_watchlist)}",
        f"- buckets: {priority_counts}",
        *[f"- {line}" for line in shadow_account_summary_lines(shadow_account_review)],
        "",
        _markdown_table(priority_watchlist.rename(columns=PRIORITY_WATCHLIST_CN_COLUMNS), max_rows=15),
        "",
        "## State + Pattern Scan",
        "",
        f"- rows: {len(state_pattern_scan)}",
        f"- buckets: {pattern_counts}",
        "",
        _markdown_table(state_pattern_scan.rename(columns=STATE_PATTERN_CN_COLUMNS), max_rows=15),
        "",
        "## Model + Personal Decision Table",
        "",
        f"- rows: {len(model_decision_table)}",
        f"- buckets: {decision_counts}",
        "",
        _markdown_table(model_decision_table.rename(columns=MODEL_DECISION_CN_COLUMNS), max_rows=15),
        "",
    ]
    paths["summary"].write_text("\n".join(summary), encoding="utf-8")
    return paths


def enrich_priority_watchlist_with_trend_ignition(
    priority_csv: Path,
    scores_csv: Path,
    asof_date: str,
    *,
    score_columns: Iterable[str] | None = None,
) -> dict[str, Path]:
    """把趋势起爆影子评分回流进当日合并优先观察清单。

    这是纯研究回流：不改变任何排序/权重/信号列，仅在清单末尾追加趋势起爆评分相关列。
    由于影子评分在 merged_daily_outputs 之后才生成（它读取本步骤写出的清单作为评分源），
    该富集作为后置步骤执行。返回被改写的 raw/cn 路径；若无匹配评分则原样返回。
    """
    priority_csv = Path(priority_csv)
    cn_csv = priority_csv.with_name(f"{priority_csv.stem}_cn{priority_csv.suffix}")
    if not priority_csv.exists() or not scores_csv.exists():
        return {"priority_watchlist": priority_csv, "priority_watchlist_cn": cn_csv}

    keep = list(
        score_columns
        or (
            "trend_ignition_score",
            "trend_ignition_score_bucket",
            "trend_ignition_deployment_status",
            "trend_ignition_research_gate_passed",
            "trend_ignition_selection_status",
        )
    )
    try:
        scores = _read_csv(scores_csv)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return {"priority_watchlist": priority_csv, "priority_watchlist_cn": cn_csv}
    if "asof_date" not in scores.columns or "symbol" not in scores.columns:
        return {"priority_watchlist": priority_csv, "priority_watchlist_cn": cn_csv}
    scores = _normalize_symbol_column(scores)
    scores = scores[scores["asof_date"].astype("string") == str(asof_date)]
    if scores.empty:
        # 当日无匹配评分：不加任何列，原样返回。
        return {"priority_watchlist": priority_csv, "priority_watchlist_cn": cn_csv}
    keep = [column for column in keep if column in scores.columns]
    if not keep:
        return {"priority_watchlist": priority_csv, "priority_watchlist_cn": cn_csv}
    scores = scores[["symbol", *keep]].drop_duplicates("symbol", keep="last")

    table = _normalize_symbol_column(_read_csv(priority_csv))
    merged = table.merge(scores, on="symbol", how="left")
    _write_table(priority_csv, merged)
    if cn_csv.exists():
        cn_table = merged.rename(columns=PRIORITY_WATCHLIST_CN_COLUMNS)
        _write_table(cn_csv, cn_table)
    return {"priority_watchlist": priority_csv, "priority_watchlist_cn": cn_csv}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged daily model outputs.")
    parser.add_argument("--trend-state", required=True)
    parser.add_argument("--early-watchlist", required=True)
    parser.add_argument("--base-candidates", required=True)
    parser.add_argument("--overlay-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--names-source", default=None)
    parser.add_argument("--shadow-account-review", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trend_state = _read_csv(Path(args.trend_state))
    watchlist = _read_csv(Path(args.early_watchlist))
    base = _read_csv(Path(args.base_candidates))
    overlay = _read_csv(Path(args.overlay_candidates))

    state_pattern_scan = build_state_pattern_scan(trend_state, watchlist)
    model_decision_table = build_model_decision_table(base, overlay)
    if args.names_source:
        name_map = load_name_map(Path(args.names_source))
        state_pattern_scan = fill_missing_stock_names(state_pattern_scan, name_map)
        model_decision_table = fill_missing_stock_names(model_decision_table, name_map)
    shadow_account_review = load_shadow_account_review(args.shadow_account_review)
    paths = write_outputs(
        Path(args.output_dir),
        args.asof_date,
        state_pattern_scan,
        model_decision_table,
        shadow_account_review=shadow_account_review,
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
