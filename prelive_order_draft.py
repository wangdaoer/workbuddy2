"""Build manual-only pre-live order drafts from daily model outputs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from market_risk import latest_benchmark_risk_state
from ths_daily_data import normalize_daily_market_file
from production_soft_score import effective_book_gross_scale, latest_regime_alert


DEFAULT_CONFIG = {
    "schema_version": 2,
    "mode": "manual_review_only",
    "broker": {
        "name": "gf_securities",
        "name_cn": "广发证券",
        "connect_broker": False,
        "place_orders": False,
    },
    "account": {
        "account_equity": 10000.0,
        "max_deploy_ratio": 0.50,
        "cash_buffer_ratio": 0.02,
    },
    "risk": {
        "max_single_position_ratio": 0.08,
        "lot_size": 100,
        "min_order_value": 500.0,
        "small_account_min_lot_fill": False,
        "require_manual_confirmation": True,
        "allow_auto_submit": False,
    },
    "production_market_risk": {
        "enabled": False,
        "base_strategy_gross_limit": 0.93,
        "ma_window": 120,
        "risk_off_drawdown_20d": -0.08,
        "below_ma_exposure": 0.60,
        "crash_exposure": 0.0,
    },
    "allocation": {
        "method": "lot_knapsack",
        "quality_template_bonus": 3.0,
        "preferred_board_bonus": 1.5,
        "rank_bonus_scale": 1.0,
        "rank_bonus_ceiling": 120,
        "raw_weight_scale": 10.0,
    },
    "universe_focus": {
        "enabled": False,
        "preferred_board_prefixes": ["300", "301"],
        "exceptional_main_board_prefixes": ["000", "001", "002", "600", "601", "603", "605"],
        "exceptional_main_board_max_rank": 10,
        "exceptional_main_board_trend_states": ["趋势确认", "生命线健康", "回调可观察", "起爆观察"],
        "quality_templates": {},
        "exclude_other_boards": True,
    },
}

SYMBOL_COLUMNS = ("symbol", "股票代码", "证券代码", "code")
NAME_COLUMNS = ("stock_name", "股票名称", "证券名称", "name")
WEIGHT_COLUMNS = (
    "personal_target_weight",
    "个人规则后权重",
    "个人习惯层权重",
    "personal_adjusted_target_weight",
    "target_weight",
)
PRICE_COLUMNS = ("close", "收盘价", "latest_close")
RANK_COLUMNS = ("personal_rank", "个人层排名", "个人规则后排名", "rank")
ACTION_COLUMNS = ("personal_action_cn", "动作", "个人规则动作中文")
REASON_COLUMNS = ("个人规则原因中文", "调整原因", "personal_reasons_cn")
RISK_COLUMNS = ("策略族风险提示", "risk_flags", "风险标记")
TREND_COLUMNS = ("trend_state", "趋势状态")
AMOUNT_COLUMNS = ("avg_amount_20d", "20日平均成交额", "avg_amount", "成交额")
RETURN_5D_COLUMNS = ("return_5d", "5日涨跌幅", "5日收益")
RETURN_20D_COLUMNS = ("return_20d", "20日涨跌幅", "20日收益")
RETURN_60D_COLUMNS = ("return_60d", "60日涨跌幅", "60日收益")
CLOSE_POSITION_COLUMNS = ("close_position", "区间位置", "收盘位置")


def load_config(path: Path | None) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Prelive config must be a mapping: {path}")
    _deep_update(cfg, loaded)
    return cfg


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _series_from(frame: pd.DataFrame, candidates: tuple[str, ...], default: object = "") -> pd.Series:
    col = _column(frame, candidates)
    if col is None:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[col]


def _normalize_symbol(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def _number(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        if not text:
            return default
        return float(text)
    return float(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def classify_board(symbol: str) -> str:
    if symbol.startswith(("300", "301")):
        return "创业板"
    if symbol.startswith(("000", "001", "002", "600", "601", "603", "605")):
        return "主板"
    if symbol.startswith("688"):
        return "科创板"
    return "其他"


def quality_template_labels(row: pd.Series, config: dict[str, Any]) -> list[str]:
    focus = config.get("universe_focus", {})
    templates = focus.get("quality_templates", {})
    labels: list[str] = []
    board = str(row.get("board_cn") or classify_board(str(row.get("symbol") or "")))
    allowed_trends = {str(item) for item in focus.get("exceptional_main_board_trend_states", [])}
    trend = str(row.get("trend_state") or "")
    if allowed_trends and trend not in allowed_trends:
        return labels
    for spec in templates.values():
        if not isinstance(spec, dict) or not spec.get("enabled", False):
            continue
        boards = {str(item) for item in spec.get("boards", [])}
        if boards and board not in boards:
            continue
        if _number(row.get("avg_amount_20d")) < float(spec.get("min_avg_amount_20d", 0.0)):
            continue
        if _number(row.get("return_20d")) < float(spec.get("min_return_20d", -1.0)):
            continue
        if _number(row.get("return_60d")) < float(spec.get("min_return_60d", -1.0)):
            continue
        close_position = _number(row.get("close_position"), 0.0)
        if close_position > 0.0 and close_position > float(spec.get("max_close_position", 1.0)):
            continue
        labels.append(str(spec.get("label_cn") or "质量趋势"))
    return labels


def apply_universe_focus(selected: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    focus = config.get("universe_focus", {})
    if not focus.get("enabled", False) or selected.empty:
        out = selected.copy()
        out["board_cn"] = out["symbol"].map(classify_board)
        out["universe_focus_reason_cn"] = "未启用板块偏好"
        return out

    preferred = tuple(str(item) for item in focus.get("preferred_board_prefixes", []))
    main_prefixes = tuple(str(item) for item in focus.get("exceptional_main_board_prefixes", []))
    max_rank = float(focus.get("exceptional_main_board_max_rank", 10))
    trend_states = {str(item) for item in focus.get("exceptional_main_board_trend_states", [])}
    exclude_other = bool(focus.get("exclude_other_boards", True))

    out = selected.copy()
    out["board_cn"] = out["symbol"].map(classify_board)
    is_preferred = out["symbol"].str.startswith(preferred) if preferred else pd.Series(False, index=out.index)
    is_main = out["symbol"].str.startswith(main_prefixes) if main_prefixes else pd.Series(False, index=out.index)
    rank_ok = pd.to_numeric(out["rank"], errors="coerce").le(max_rank)
    trend_ok = out["trend_state"].astype(str).isin(trend_states) if trend_states else pd.Series(True, index=out.index)
    template_labels = out.apply(lambda row: quality_template_labels(row, config), axis=1)
    template_match = template_labels.map(bool)
    exceptional_main = is_main & ((rank_ok & trend_ok) | template_match)
    if exclude_other:
        out = out[is_preferred | exceptional_main].copy()
        is_preferred = is_preferred.loc[out.index]
        exceptional_main = exceptional_main.loc[out.index]
        template_labels = template_labels.loc[out.index]
        template_match = template_match.loc[out.index]
    out["universe_focus_reason_cn"] = "其他板块保留观察"
    out.loc[is_preferred, "universe_focus_reason_cn"] = "创业板优先"
    out.loc[exceptional_main & ~is_preferred, "universe_focus_reason_cn"] = "特别优秀主板保留"
    label_text = template_labels.map(lambda labels: "、".join(labels))
    out.loc[template_match, "universe_focus_reason_cn"] = label_text.loc[template_match]
    out["quality_template_cn"] = label_text.reindex(out.index).fillna("")
    out["_board_order"] = 2
    out.loc[is_preferred, "_board_order"] = 0
    out.loc[exceptional_main & ~is_preferred, "_board_order"] = 1
    return out.sort_values(["_board_order", "rank", "symbol"], kind="mergesort").drop(columns=["_board_order"])


def load_selected(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str)
    symbol_col = _column(frame, SYMBOL_COLUMNS)
    weight_col = _column(frame, WEIGHT_COLUMNS)
    price_col = _column(frame, PRICE_COLUMNS)
    if symbol_col is None or weight_col is None or price_col is None:
        raise ValueError("Selected input must include symbol, target weight, and close price columns")
    selected_col = _column(frame, ("personal_selected", "个人规则后入选", "selected", "入选"))
    out = pd.DataFrame()
    out["symbol"] = frame[symbol_col].map(_normalize_symbol)
    out["stock_name"] = frame[_column(frame, NAME_COLUMNS)].fillna("") if _column(frame, NAME_COLUMNS) else ""
    out["target_weight_raw"] = frame[weight_col].map(_number)
    out["close"] = frame[price_col].map(_number)
    out["rank"] = frame[_column(frame, RANK_COLUMNS)].map(_number) if _column(frame, RANK_COLUMNS) else range(1, len(frame) + 1)
    out["model_action_cn"] = frame[_column(frame, ACTION_COLUMNS)].fillna("") if _column(frame, ACTION_COLUMNS) else ""
    out["model_reason_cn"] = frame[_column(frame, REASON_COLUMNS)].fillna("") if _column(frame, REASON_COLUMNS) else ""
    out["risk_note_cn"] = frame[_column(frame, RISK_COLUMNS)].fillna("") if _column(frame, RISK_COLUMNS) else ""
    out["trend_state"] = frame[_column(frame, TREND_COLUMNS)].fillna("") if _column(frame, TREND_COLUMNS) else ""
    out["avg_amount_20d"] = _series_from(frame, AMOUNT_COLUMNS, 0.0).map(_number)
    out["return_5d"] = _series_from(frame, RETURN_5D_COLUMNS, 0.0).map(_number)
    out["return_20d"] = _series_from(frame, RETURN_20D_COLUMNS, 0.0).map(_number)
    out["return_60d"] = _series_from(frame, RETURN_60D_COLUMNS, 0.0).map(_number)
    out["close_position"] = _series_from(frame, CLOSE_POSITION_COLUMNS, 0.0).map(_number)
    if selected_col:
        selected = frame[selected_col].map(_bool)
    else:
        selected = out["target_weight_raw"].gt(0.0)
    out = out[selected & out["symbol"].ne("") & out["target_weight_raw"].gt(0.0) & out["close"].gt(0.0)]
    return out.sort_values(["rank", "symbol"], kind="mergesort").reset_index(drop=True)


def load_feature_source(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str)
    symbol_col = _column(frame, SYMBOL_COLUMNS)
    if symbol_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["symbol"] = frame[symbol_col].map(_normalize_symbol)
    out["avg_amount_20d_feature"] = _series_from(frame, AMOUNT_COLUMNS, 0.0).map(_number)
    out["return_5d_feature"] = _series_from(frame, RETURN_5D_COLUMNS, 0.0).map(_number)
    out["return_20d_feature"] = _series_from(frame, RETURN_20D_COLUMNS, 0.0).map(_number)
    out["return_60d_feature"] = _series_from(frame, RETURN_60D_COLUMNS, 0.0).map(_number)
    out["close_position_feature"] = _series_from(frame, CLOSE_POSITION_COLUMNS, 0.0).map(_number)
    trend_col = _column(frame, TREND_COLUMNS)
    if trend_col:
        out["trend_state_feature"] = frame[trend_col].fillna("")
    out = out[out["symbol"].ne("")]
    return out.drop_duplicates("symbol", keep="last")


def load_market_source(path: Path | None, asof_date: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame, _ = normalize_daily_market_file(path, asof_date)
    if frame.empty:
        return pd.DataFrame()
    out = frame[["symbol", "amount", "close"]].copy()
    out["symbol"] = out["symbol"].map(_normalize_symbol)
    out["amount_feature"] = pd.to_numeric(out["amount"], errors="coerce")
    out["close_feature"] = pd.to_numeric(out["close"], errors="coerce")
    return out[out["symbol"].ne("")][["symbol", "amount_feature", "close_feature"]].drop_duplicates("symbol", keep="last")


def enrich_features(selected: pd.DataFrame, feature_source: pd.DataFrame) -> pd.DataFrame:
    if feature_source.empty:
        return selected
    out = selected.merge(feature_source, on="symbol", how="left")
    for base, feature in (
        ("avg_amount_20d", "avg_amount_20d_feature"),
        ("return_5d", "return_5d_feature"),
        ("return_20d", "return_20d_feature"),
        ("return_60d", "return_60d_feature"),
        ("close_position", "close_position_feature"),
    ):
        if feature in out:
            current = pd.to_numeric(out[base], errors="coerce")
            incoming = pd.to_numeric(out[feature], errors="coerce")
            out[base] = current.mask(current.isna() | current.eq(0.0), incoming).fillna(0.0)
    if "trend_state_feature" in out:
        out["trend_state"] = out["trend_state"].astype(str).where(
            out["trend_state"].astype(str).str.len().gt(0),
            out["trend_state_feature"].fillna(""),
        )
    return out[[col for col in out.columns if not col.endswith("_feature")]]


def enrich_market(selected: pd.DataFrame, market_source: pd.DataFrame) -> pd.DataFrame:
    if market_source.empty:
        return selected
    out = selected.merge(market_source, on="symbol", how="left")
    if "amount_feature" in out:
        current = pd.to_numeric(out["avg_amount_20d"], errors="coerce")
        incoming = pd.to_numeric(out["amount_feature"], errors="coerce")
        out["avg_amount_20d"] = current.mask(current.isna() | current.eq(0.0), incoming).fillna(0.0)
    if "close_feature" in out:
        current_close = pd.to_numeric(out["close"], errors="coerce")
        incoming_close = pd.to_numeric(out["close_feature"], errors="coerce")
        out["close"] = current_close.mask(current_close.isna() | current_close.le(0.0), incoming_close).fillna(0.0)
    return out[[col for col in out.columns if not col.endswith("_feature")]]


def load_supplement_candidates(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str)
    symbol_col = _column(frame, SYMBOL_COLUMNS)
    price_col = _column(frame, PRICE_COLUMNS)
    if symbol_col is None or price_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["symbol"] = frame[symbol_col].map(_normalize_symbol)
    out["stock_name"] = _series_from(frame, NAME_COLUMNS, "").fillna("")
    out["target_weight_raw"] = _series_from(frame, WEIGHT_COLUMNS, 0.02).map(_number)
    out["target_weight_raw"] = out["target_weight_raw"].where(out["target_weight_raw"].gt(0.0), 0.02)
    out["close"] = frame[price_col].map(_number)
    out["rank"] = _series_from(frame, RANK_COLUMNS, 9999).map(_number)
    out["model_action_cn"] = _series_from(frame, ACTION_COLUMNS, "质量模板补充观察").fillna("")
    out["model_reason_cn"] = _series_from(frame, REASON_COLUMNS, "由实盘前置质量模板补入").fillna("")
    out["risk_note_cn"] = _series_from(frame, RISK_COLUMNS, "").fillna("")
    out["trend_state"] = _series_from(frame, TREND_COLUMNS, "").fillna("")
    out["avg_amount_20d"] = _series_from(frame, AMOUNT_COLUMNS, 0.0).map(_number)
    out["return_5d"] = _series_from(frame, RETURN_5D_COLUMNS, 0.0).map(_number)
    out["return_20d"] = _series_from(frame, RETURN_20D_COLUMNS, 0.0).map(_number)
    out["return_60d"] = _series_from(frame, RETURN_60D_COLUMNS, 0.0).map(_number)
    out["close_position"] = _series_from(frame, CLOSE_POSITION_COLUMNS, 0.0).map(_number)
    return out[out["symbol"].ne("") & out["close"].gt(0.0)].drop_duplicates("symbol", keep="last")


def append_quality_supplements(selected: pd.DataFrame, supplements: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if supplements.empty:
        return selected
    existing = set(selected["symbol"])
    candidates = supplements[~supplements["symbol"].isin(existing)].copy()
    if candidates.empty:
        return selected
    candidates["board_cn"] = candidates["symbol"].map(classify_board)
    candidates["quality_template_cn"] = candidates.apply(
        lambda row: "、".join(quality_template_labels(row, config)), axis=1
    )
    candidates = candidates[candidates["quality_template_cn"].astype(str).str.len().gt(0)].copy()
    if candidates.empty:
        return selected
    candidates["universe_focus_reason_cn"] = candidates["quality_template_cn"]
    candidates["model_action_cn"] = "质量模板补充观察"
    candidates["model_reason_cn"] = candidates["model_reason_cn"].astype(str).where(
        candidates["model_reason_cn"].astype(str).str.len().gt(0),
        "未进入原入选表，但满足实盘前置质量模板",
    )
    if selected.empty:
        return candidates.reset_index(drop=True)
    return pd.concat([selected, candidates], ignore_index=True, sort=False)


def load_positions(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["symbol", "current_shares"])
    frame = pd.read_csv(path, dtype=str)
    symbol_col = _column(frame, SYMBOL_COLUMNS)
    shares_col = _column(frame, ("current_shares", "shares", "持仓数量", "可用股份", "股份余额"))
    if symbol_col is None or shares_col is None:
        raise ValueError("Position input must include symbol and current shares columns")
    out = pd.DataFrame()
    out["symbol"] = frame[symbol_col].map(_normalize_symbol)
    out["current_shares"] = frame[shares_col].map(_number)
    return out[out["symbol"].ne("")].groupby("symbol", as_index=False)["current_shares"].sum()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    return data if isinstance(data, dict) else {}


def prelive_gate_status(
    arena: dict[str, Any],
    regime_tracking: dict[str, Any],
    breadth_tracking: dict[str, Any],
) -> tuple[str, list[str]]:
    notes = [
        "本文件仅用于人工复核，不是券商订单。",
        "自动下单关闭；下单前必须人工二次确认。"]
    status = "manual_review_only"
    if arena:
        if arena.get("research_only") is not True:
            status = "blocked"
            notes.append("策略竞技场未明确 research_only=true。")
        if arena.get("promotion_decision") not in {None, "hold_production_champion"}:
            notes.append(f"策略竞技场晋级状态: {arena.get('promotion_decision')}")
        observation = arena.get("independent_observation_status")
        if observation and observation != "complete":
            notes.append(f"策略竞技场独立观察仍为 {observation}。")
    else:
        notes.append("缺少策略竞技场状态，保持人工复核。")
    if regime_tracking:
        valid = regime_tracking.get("valid_observation_count")
        target = regime_tracking.get("target_days")
        if valid is not None and target is not None and int(valid) < int(target):
            notes.append(f"强势回调动态策略观察 {valid}/{target}，未成熟。")
    if breadth_tracking:
        if breadth_tracking.get("promotion_allowed") is False:
            notes.append("动态广度风控未允许晋级。")
        valid = breadth_tracking.get("valid_observation_count")
        target = breadth_tracking.get("target_valid_trade_days")
        if valid is not None and target is not None and int(valid) < int(target):
            notes.append(f"动态广度风控观察 {valid}/{target}，未成熟。")
    return status, notes


def load_production_market_risk(
    config: dict[str, Any],
    benchmark_path: Path | None,
    asof_date: str,
) -> dict[str, object]:
    risk = config.get("production_market_risk", {})
    if not risk.get("enabled", False):
        return {
            "enabled": False,
            "risk_exposure": 1.0,
            "risk_reason": "disabled",
            "base_strategy_gross_limit": 1.0,
        }
    if benchmark_path is None:
        raise ValueError("production market risk requires --benchmark")
    state = latest_benchmark_risk_state(
        benchmark_path,
        asof_date,
        ma_window=int(risk.get("ma_window", 120)),
        risk_off_drawdown_20d=float(risk.get("risk_off_drawdown_20d", -0.08)),
        below_ma_exposure=float(risk.get("below_ma_exposure", 0.60)),
        crash_exposure=float(risk.get("crash_exposure", 0.0)),
    )
    state["enabled"] = True
    state["base_strategy_gross_limit"] = max(
        0.0,
        min(float(risk.get("base_strategy_gross_limit", 0.93)), 1.0),
    )
    return state


def production_market_risk_note(state: dict[str, object]) -> str:
    exposure = float(state.get("risk_exposure", 1.0))
    reason = str(state.get("risk_reason") or "risk_on")
    reason_cn = {
        "risk_on": "基准风险开启",
        "below_ma": "基准低于长期均线",
        "drawdown_20d_crash": "基准20日跌幅触发风险关闭",
        "disabled": "正式市场过滤未启用",
    }.get(reason, reason)
    return f"正式市场过滤：{reason_cn}，风险暴露 {exposure:.0%}。"


def allocation_priority_score(row: pd.Series, config: dict[str, Any]) -> float:
    allocation = config.get("allocation", {})
    focus = config.get("universe_focus", {})
    score = max(_number(row.get("target_weight_raw")), 0.0) * float(
        allocation.get("raw_weight_scale", 10.0)
    )
    if str(row.get("quality_template_cn") or "").strip():
        score += float(allocation.get("quality_template_bonus", 3.0))
    preferred = tuple(str(value) for value in focus.get("preferred_board_prefixes", []))
    if preferred and str(row.get("symbol") or "").startswith(preferred):
        score += float(allocation.get("preferred_board_bonus", 1.5))
    rank_ceiling = max(int(allocation.get("rank_bonus_ceiling", 120)), 1)
    rank = max(_number(row.get("rank"), float(rank_ceiling)), 0.0)
    rank_fraction = max(rank_ceiling - min(rank, rank_ceiling), 0.0) / rank_ceiling
    score += rank_fraction * float(allocation.get("rank_bonus_scale", 1.0))
    return float(score)


def _prune_knapsack_states(
    states: dict[int, tuple[float, tuple[int, ...]]],
) -> dict[int, tuple[float, tuple[int, ...]]]:
    pruned: dict[int, tuple[float, tuple[int, ...]]] = {}
    best_utility = float("-inf")
    for cost in sorted(states):
        utility, selected = states[cost]
        if utility > best_utility + 1e-12:
            pruned[cost] = (utility, selected)
            best_utility = utility
    return pruned


def allocate_target_lots(
    selected: pd.DataFrame,
    *,
    deploy_value: float,
    equity: float,
    max_single: float,
    lot_size: int,
    small_account_fill: bool,
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Choose whole-share lots without depending on candidate row order."""

    target_shares = pd.Series(0, index=selected.index, dtype=int)
    priority = selected.apply(lambda row: allocation_priority_score(row, config), axis=1)
    if selected.empty or deploy_value <= 0.0:
        return target_shares, priority

    allocation = config.get("allocation", {})
    if not small_account_fill or allocation.get("method") != "lot_knapsack":
        for idx, row in selected.iterrows():
            lot_value = lot_size * float(row["close"])
            if lot_value <= 0.0:
                continue
            desired = float(row["target_weight"]) * equity
            lots = math.floor(desired / lot_value)
            target_shares.loc[idx] = int(lots * lot_size)
        return target_shares, priority

    items: list[tuple[int, int, float]] = []
    ordered = selected.assign(_priority=priority).sort_values("symbol", kind="mergesort")
    for idx, row in ordered.iterrows():
        lot_value = lot_size * float(row["close"])
        if lot_value <= 0.0 or lot_value > max_single * equity + 1e-9:
            continue
        desired = float(row["target_weight"]) * equity
        requested_lots = math.floor(desired / lot_value)
        if requested_lots <= 0:
            requested_lots = 1
        max_lots = math.floor(max_single * equity / lot_value + 1e-12)
        requested_lots = min(requested_lots, max_lots)
        for lot_number in range(1, requested_lots + 1):
            cost_units = max(int(math.ceil(lot_value)), 1)
            utility = float(row["_priority"]) / lot_number
            items.append((idx, cost_units, utility))

    budget_units = max(int(math.floor(deploy_value)), 0)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for item_number, (_, cost, utility) in enumerate(items):
        updated = dict(states)
        for used, (current_utility, chosen) in states.items():
            next_cost = used + cost
            if next_cost > budget_units:
                continue
            candidate = (current_utility + utility, chosen + (item_number,))
            existing = updated.get(next_cost)
            if existing is None or candidate[0] > existing[0] + 1e-12:
                updated[next_cost] = candidate
        states = _prune_knapsack_states(updated)

    _, chosen = max(
        states.values(),
        key=lambda value: (value[0], -sum(items[item][1] for item in value[1])),
    )
    for item_number in chosen:
        idx = items[item_number][0]
        target_shares.loc[idx] += lot_size
    return target_shares, priority


def build_draft(
    selected: pd.DataFrame,
    positions: pd.DataFrame,
    config: dict[str, Any],
    asof_date: str,
    gate_status: str,
    gate_notes: list[str],
    market_risk_state: dict[str, object] | None = None,
    book_regime_alert: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    account = config.get("account", {})
    risk = config.get("risk", {})
    broker = config.get("broker", {})
    universe_focus = config.get("universe_focus", {})
    equity = float(account.get("account_equity", 10000.0))
    deploy_ratio = max(0.0, min(float(account.get("max_deploy_ratio", 0.50)), 1.0))
    cash_buffer = max(0.0, min(float(account.get("cash_buffer_ratio", 0.02)), 1.0))
    max_single = max(0.0, min(float(risk.get("max_single_position_ratio", 0.08)), 1.0))
    lot_size = max(int(risk.get("lot_size", 100)), 1)
    min_order_value = max(float(risk.get("min_order_value", 0.0)), 0.0)
    small_account_fill = bool(risk.get("small_account_min_lot_fill", False))
    risk_state = market_risk_state or {
        "enabled": False,
        "risk_exposure": 1.0,
        "risk_reason": "disabled",
        "base_strategy_gross_limit": 1.0,
    }
    market_risk_exposure = max(
        0.0,
        min(float(risk_state.get("risk_exposure", 1.0)), 1.0),
    )
    base_strategy_gross_limit = max(
        0.0,
        min(float(risk_state.get("base_strategy_gross_limit", 1.0)), 1.0),
    )
    # 监控→闸门闭环落点：regime 死区告警把簿 gross 暴露压到 recommended_gross_scale，
    # 否则返回 1.0（不干预）。与 market_risk / base_strategy 限制取三者最小作为最终生效上限。
    book_regime_gross_scale = effective_book_gross_scale(book_regime_alert, base_scale=1.0)
    effective_deploy_ratio = min(
        deploy_ratio,
        base_strategy_gross_limit * market_risk_exposure,
        book_regime_gross_scale,
    )
    if book_regime_gross_scale < 1.0:
        gate_notes.append(
            f"regime 死区告警：簿 gross 暴露被压减至 {book_regime_gross_scale:.2f}（监控→闸门闭环已生效）"
        )
    deploy_value = equity * max(0.0, effective_deploy_ratio - cash_buffer)

    weights = selected["target_weight_raw"].clip(lower=0.0, upper=max_single)
    if weights.sum() > 0:
        scaled = weights / weights.sum() * effective_deploy_ratio
        selected = selected.copy()
        selected["target_weight"] = scaled.clip(upper=max_single)
    else:
        selected = selected.copy()
        selected["target_weight"] = 0.0
    if selected["target_weight"].sum() > effective_deploy_ratio:
        selected["target_weight"] *= (
            effective_deploy_ratio / selected["target_weight"].sum()
        )

    merged = selected.merge(positions, on="symbol", how="left")
    merged["current_shares"] = pd.to_numeric(
        merged["current_shares"], errors="coerce"
    ).fillna(0.0)
    allocated_shares, allocation_priority = allocate_target_lots(
        merged,
        deploy_value=deploy_value,
        equity=equity,
        max_single=max_single,
        lot_size=lot_size,
        small_account_fill=small_account_fill,
        config=config,
    )
    merged["target_shares_allocated"] = allocated_shares
    merged["allocation_priority_score"] = allocation_priority
    rows: list[dict[str, Any]] = []
    for item in merged.itertuples(index=False):
        target_value = float(item.target_weight) * equity
        target_shares = int(item.target_shares_allocated)
        lot_value = lot_size * float(item.close)
        target_value_lot = target_shares * float(item.close)
        current_value = float(item.current_shares) * float(item.close)
        delta_shares = target_shares - float(item.current_shares)
        delta_value = target_value_lot - current_value
        if abs(delta_value) < min_order_value:
            action = "人工复核-保持观察"
            review_reason = "低于最小人工调整金额"
        elif delta_shares > 0 and small_account_fill and target_value_lot == lot_value:
            action = "人工复核-小账户一手草稿"
            review_reason = "小资金整手可行候选"
        elif delta_shares > 0:
            action = "人工复核-买入草稿"
            review_reason = "目标仓位高于当前持仓"
        elif delta_shares < 0:
            action = "人工复核-卖出草稿"
            review_reason = "目标仓位低于当前持仓"
        else:
            action = "人工复核-保持观察"
            review_reason = "目标股数与当前持仓一致"
        rows.append(
            {
                "asof_date": asof_date,
                "broker": broker.get("name_cn") or broker.get("name") or "",
                "symbol": item.symbol,
                "stock_name": item.stock_name,
                "board_cn": item.board_cn if hasattr(item, "board_cn") else classify_board(item.symbol),
                "universe_focus_reason_cn": (
                    item.universe_focus_reason_cn
                    if hasattr(item, "universe_focus_reason_cn")
                    else ""
                ),
                "quality_template_cn": (
                    item.quality_template_cn if hasattr(item, "quality_template_cn") else ""
                ),
                "rank": int(item.rank) if float(item.rank).is_integer() else float(item.rank),
                "close": float(item.close),
                "avg_amount_20d": round(float(item.avg_amount_20d), 2)
                if hasattr(item, "avg_amount_20d")
                else 0.0,
                "return_20d": round(float(item.return_20d), 6)
                if hasattr(item, "return_20d")
                else 0.0,
                "return_60d": round(float(item.return_60d), 6)
                if hasattr(item, "return_60d")
                else 0.0,
                "close_position": round(float(item.close_position), 6)
                if hasattr(item, "close_position")
                else 0.0,
                "current_shares": int(item.current_shares),
                "current_market_value": round(current_value, 2),
                "target_weight": round(float(item.target_weight), 6),
                "effective_deploy_ratio": round(effective_deploy_ratio, 6),
                "market_risk_exposure": round(market_risk_exposure, 6),
                "allocation_priority_score": round(float(item.allocation_priority_score), 6),
                "target_market_value": round(target_value_lot, 2),
                "target_shares_lot": int(target_shares),
                "delta_shares": int(delta_shares),
                "delta_value": round(delta_value, 2),
                "manual_action_cn": action,
                "manual_review_required": True,
                "prelive_gate_status": gate_status,
                "trade_instruction": False,
                "broker_order_allowed": False,
                "review_reason_cn": review_reason,
                "model_action_cn": item.model_action_cn,
                "model_reason_cn": item.model_reason_cn,
                "risk_note_cn": item.risk_note_cn,
                "trend_state": item.trend_state,
                "gate_notes_cn": "；".join(gate_notes),
            }
        )
    draft = pd.DataFrame(rows)
    metadata = {
        "schema_version": 2,
        "status": "ready" if len(draft) else "empty",
        "asof_date": asof_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "manual_review_only",
        "broker": broker,
        "account_equity": equity,
        "max_deploy_ratio": deploy_ratio,
        "base_strategy_gross_limit": base_strategy_gross_limit,
        "market_risk_exposure": market_risk_exposure,
        "market_risk_reason": risk_state.get("risk_reason"),
        "production_market_risk": risk_state,
        "book_regime_gross_scale": book_regime_gross_scale,
        "book_regime_alert_active": bool(book_regime_alert and book_regime_alert.get("alert")),
        "effective_deploy_ratio": effective_deploy_ratio,
        "cash_buffer_ratio": cash_buffer,
        "deploy_value_budget": round(deploy_value, 2),
        "target_market_value_sum": round(float(draft.get("target_market_value", pd.Series(dtype=float)).sum()), 2),
        "delta_value_sum": round(float(draft.get("delta_value", pd.Series(dtype=float)).sum()), 2),
        "row_count": int(len(draft)),
        "allocation_method": config.get("allocation", {}).get("method", "target_floor"),
        "prelive_gate_status": gate_status,
        "gate_notes": gate_notes,
        "trade_instruction": False,
        "broker_order_allowed": False,
        "connect_broker": False,
        "place_orders": False,
        "manual_confirmation_required": True,
        "universe_focus": universe_focus,
    }
    return draft, metadata


CN_COLUMNS = {
    "asof_date": "日期",
    "broker": "券商",
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "board_cn": "板块",
    "universe_focus_reason_cn": "板块纳入原因",
    "quality_template_cn": "质量模板",
    "rank": "模型排名",
    "close": "收盘价",
    "avg_amount_20d": "20日平均成交额",
    "return_20d": "20日涨跌幅",
    "return_60d": "60日涨跌幅",
    "close_position": "区间位置",
    "current_shares": "当前持仓股数",
    "current_market_value": "当前市值",
    "target_weight": "目标权重",
    "effective_deploy_ratio": "有效总仓位上限",
    "market_risk_exposure": "正式市场风险暴露",
    "allocation_priority_score": "整手分配优先分",
    "target_market_value": "目标市值",
    "target_shares_lot": "目标整手股数",
    "delta_shares": "需人工复核股数变化",
    "delta_value": "需人工复核金额变化",
    "manual_action_cn": "人工动作草稿",
    "manual_review_required": "需要人工确认",
    "prelive_gate_status": "前置闸门状态",
    "trade_instruction": "是否交易指令",
    "broker_order_allowed": "是否允许券商下单",
    "review_reason_cn": "人工复核原因",
    "model_action_cn": "模型动作",
    "model_reason_cn": "模型原因",
    "risk_note_cn": "风险提示",
    "trend_state": "趋势状态",
    "gate_notes_cn": "闸门说明",
}


def write_outputs(draft: pd.DataFrame, metadata: dict[str, Any], output_prefix: Path) -> dict[str, str]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    cn_path = output_prefix.with_name(f"{output_prefix.name}_cn.csv")
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    draft.to_csv(csv_path, index=False, encoding="utf-8-sig")
    draft.rename(columns=CN_COLUMNS).to_csv(cn_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(draft, metadata), encoding="utf-8")
    return {"csv": str(csv_path), "csv_cn": str(cn_path), "json": str(json_path), "report": str(md_path)}


def _markdown(draft: pd.DataFrame, metadata: dict[str, Any]) -> str:
    lines = [
        f"# 实盘前置人工复核草稿 {metadata.get('asof_date')}",
        "",
        "- 状态：仅人工复核，不是交易指令。",
        "- 券商：{}".format((metadata.get("broker") or {}).get("name_cn") or (metadata.get("broker") or {}).get("name")),
        f"- 账户权益假设：{metadata.get('account_equity')}",
        f"- 最大部署比例：{metadata.get('max_deploy_ratio')}",
        f"- 正式市场风险暴露：{metadata.get('market_risk_exposure')}",
        f"- regime 簿闸门 gross 上限：{metadata.get('book_regime_gross_scale')}",
        f"- 最终有效仓位上限：{metadata.get('effective_deploy_ratio')}",
        f"- 草稿目标市值合计：{metadata.get('target_market_value_sum')}",
        f"- 前置闸门：{metadata.get('prelive_gate_status')}",
        "",
        "## 闸门说明",
        "",
        *[f"- {note}" for note in metadata.get("gate_notes", [])],
        "",
        "## 前 20 条草稿",
        "",
    ]
    if draft.empty:
        lines.append("无可生成草稿。")
    else:
        preview = draft.rename(columns=CN_COLUMNS).head(20)
        lines.append(preview.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## 强制边界",
            "",
            "- `trade_instruction=false`。",
            "- `broker_order_allowed=false`。",
            "- 本文件不得直接导入券商交易端。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manual-only pre-live order drafts.")
    parser.add_argument("--selected", required=True)
    parser.add_argument("--feature-source", default=None)
    parser.add_argument("--supplement-source", default=None)
    parser.add_argument("--market-source", default=None)
    parser.add_argument("--positions", default=None)
    parser.add_argument("--config", default="configs/prelive_gf_manual.yaml")
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--arena-metadata", default=None)
    parser.add_argument("--regime-tracking", default=None)
    parser.add_argument("--dynamic-breadth-tracking", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--book-regime-alert",
        default=None,
        help="regime 告警 JSON 路径；缺省自动发现最新 outputs/*/regime_alert*.json",
    )
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    if cfg.get("broker", {}).get("connect_broker") or cfg.get("broker", {}).get("place_orders"):
        raise ValueError("Prelive stage forbids broker connection and order placement")
    if cfg.get("risk", {}).get("allow_auto_submit"):
        raise ValueError("Prelive stage forbids automatic submission")
    selected = load_selected(Path(args.selected))
    selected = enrich_features(
        selected,
        load_feature_source(Path(args.feature_source) if args.feature_source else None),
    )
    market_source = load_market_source(
        Path(args.market_source) if args.market_source else None,
        args.asof_date,
    )
    selected = enrich_market(selected, market_source)
    supplements = load_supplement_candidates(
        Path(args.supplement_source) if args.supplement_source else None
    )
    supplements = enrich_market(supplements, market_source)
    selected = append_quality_supplements(selected, supplements, cfg)
    selected = apply_universe_focus(selected, cfg)
    positions = load_positions(Path(args.positions) if args.positions else None)
    gate_status, gate_notes = prelive_gate_status(
        _read_json(Path(args.arena_metadata) if args.arena_metadata else None),
        _read_json(Path(args.regime_tracking) if args.regime_tracking else None),
        _read_json(Path(args.dynamic_breadth_tracking) if args.dynamic_breadth_tracking else None),
    )
    market_risk_state = load_production_market_risk(
        cfg,
        Path(args.benchmark) if args.benchmark else None,
        args.asof_date,
    )
    gate_notes.append(production_market_risk_note(market_risk_state))
    book_regime_alert = (
        _read_json(Path(args.book_regime_alert))
        if args.book_regime_alert
        else latest_regime_alert(Path(__file__).resolve().parent, args.asof_date)
    )
    draft, metadata = build_draft(
        selected,
        positions,
        cfg,
        args.asof_date,
        gate_status,
        gate_notes,
        market_risk_state,
        book_regime_alert,
    )
    paths = write_outputs(draft, metadata, Path(args.output_prefix))
    print(json.dumps({"metadata": metadata, "artifacts": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
