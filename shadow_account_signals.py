"""Apply research-only shadow account prompts to candidate tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SIGNAL_CN = {"risk": "风险提示", "prefer": "偏好确认", "neutral": "无额外提示"}


def load_shadow_account_review(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    review_path = Path(path)
    if not review_path.exists():
        return None
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shadow account review must be a JSON object")
    if (
        payload.get("research_only") is not True
        or payload.get("allows_broker_orders") is not False
    ):
        raise ValueError("shadow account review must be research-only")
    _validated_rules(payload)
    return payload


def apply_shadow_account_signals(table: pd.DataFrame, review: dict[str, Any] | None) -> pd.DataFrame:
    out = table.copy()
    if out.empty:
        return _ensure_columns(out)
    rules = _validated_rules(review)
    if not rules:
        return _ensure_columns(out)
    symbols = _symbol_series(out)
    notes: list[list[str]] = [[] for _ in range(len(out))]
    scores = [0 for _ in range(len(out))]
    for rule in rules:
        source = str(rule.get("source") or "")
        value = str(rule.get("value") or "")
        action = str(rule.get("action") or "")
        mask = _rule_mask(out, symbols, source, value)
        if not mask.any():
            continue
        delta = _rule_delta(action)
        label = _rule_label(source, value)
        for index, matched in enumerate(mask.tolist()):
            if not matched:
                continue
            scores[index] += delta
            notes[index].append(label)
    out["shadow_account_signal_score"] = scores
    out["shadow_account_signal"] = [
        "risk" if score < 0 else "prefer" if score > 0 else "neutral"
        for score in scores
    ]
    out["shadow_account_signal_cn"] = out["shadow_account_signal"].map(SIGNAL_CN).fillna(out["shadow_account_signal"])
    out["shadow_account_notes"] = ["; ".join(items) if items else "" for items in notes]
    return out


def _validated_rules(review: dict[str, Any] | None) -> list[dict[str, Any]]:
    if review is None:
        return []
    if not isinstance(review, dict):
        raise ValueError("shadow account review must be a JSON object")
    rules = review.get("rules") or []
    if not isinstance(rules, list) or any(not isinstance(rule, dict) for rule in rules):
        raise ValueError("shadow account review rules must be a list of objects")
    return list(rules)


def shadow_account_summary_lines(review: dict[str, Any] | None, limit: int = 8) -> list[str]:
    if not review:
        return ["未接入影子账户复盘规则。"]
    rules = list(review.get("rules") or [])
    lines = [
        "影子账户规则仅做研究提示，不连接券商，不自动下单。",
        f"已读取 {len(rules)} 条个人交割单复盘规则。",
    ]
    for rule in rules[:limit]:
        source = rule.get("source", "")
        value = rule.get("value", "")
        action = _action_cn(str(rule.get("action", "")))
        lines.append(f"{source}={value}: {action}")
    return lines


def _ensure_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "shadow_account_signal_score" not in out:
        out["shadow_account_signal_score"] = 0
    if "shadow_account_signal" not in out:
        out["shadow_account_signal"] = "neutral"
    if "shadow_account_signal_cn" not in out:
        out["shadow_account_signal_cn"] = SIGNAL_CN["neutral"]
    if "shadow_account_notes" not in out:
        out["shadow_account_notes"] = ""
    return out


def _symbol_series(table: pd.DataFrame) -> pd.Series:
    if "symbol" not in table:
        return pd.Series("", index=table.index, dtype=object)
    text = table["symbol"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    digits = text.str.replace(r"\D", "", regex=True)
    return digits.str[-6:].str.zfill(6).mask(digits.eq(""), text)


def _rule_mask(table: pd.DataFrame, symbols: pd.Series, source: str, value: str) -> pd.Series:
    if source == "symbol_history":
        normalized = "".join(ch for ch in value if ch.isdigit())[-6:].zfill(6)
        return symbols.eq(normalized)
    if source == "entry_position":
        return _entry_position_bucket(_numeric_column(table, "close_position")).eq(value)
    if source == "entry_momentum":
        return _entry_momentum_bucket(_numeric_column(table, "return_20d")).eq(value)
    return pd.Series(False, index=table.index)


def _numeric_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(pd.NA, index=table.index)
    return pd.to_numeric(table[column], errors="coerce")


def _entry_position_bucket(values: pd.Series) -> pd.Series:
    result = pd.Series("", index=values.index, dtype=object)
    result.loc[values.le(0.25)] = "low_quarter"
    result.loc[values.gt(0.25) & values.le(0.50)] = "mid_low"
    result.loc[values.gt(0.50) & values.le(0.75)] = "mid_high"
    result.loc[values.gt(0.75)] = "high_quarter"
    return result


def _entry_momentum_bucket(values: pd.Series) -> pd.Series:
    result = pd.Series("", index=values.index, dtype=object)
    result.loc[values.le(-0.20)] = "20d<-20%"
    result.loc[values.gt(-0.20) & values.le(-0.05)] = "-20%~-5%"
    result.loc[values.gt(-0.05) & values.le(0.05)] = "-5%~+5%"
    result.loc[values.gt(0.05) & values.le(0.20)] = "+5%~+20%"
    result.loc[values.gt(0.20)] = ">+20%"
    return result


def _rule_delta(action: str) -> int:
    if action.startswith("avoid"):
        return -1
    if action.startswith("prefer"):
        return 1
    return 0


def _rule_label(source: str, value: str) -> str:
    return f"{source}:{value}"


def _action_cn(action: str) -> str:
    if action.startswith("avoid"):
        return "风险/降权提示"
    if action.startswith("prefer"):
        return "偏好确认"
    return action or "提示"
