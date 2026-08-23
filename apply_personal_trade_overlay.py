"""Apply personal trade-habit rules to latest candidate tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_daily_personal_overlay_report import load_name_map


DEFAULT_RULES: dict[str, Any] = {
    "low_entry_position_max": 0.50,
    "low_entry_score_bonus": 0.020,
    "high_entry_position_min": 0.75,
    "high_entry_score_penalty": -0.030,
    "high_entry_weight_multiplier": 0.50,
    "weak_20d_return_min": -0.20,
    "weak_20d_return_max": -0.05,
    "weak_20d_score_penalty": -0.015,
    "weak_20d_weight_multiplier": 0.70,
    "damaged_20d_return_max": -0.20,
    "damaged_20d_action": "watch_only",
    "symbol_history_min_trades": 3,
    "bad_symbol_pnl_max": -5000.0,
    "bad_symbol_win_rate_max": 0.35,
    "bad_symbol_score_penalty": -0.020,
    "bad_symbol_weight_multiplier": 0.50,
    "severe_bad_symbol_pnl_max": -10000.0,
    "severe_bad_symbol_action": "watch_only",
    "good_symbol_pnl_min": 5000.0,
    "good_symbol_win_rate_min": 0.55,
    "good_symbol_score_bonus": 0.010,
    "min_default_holding_days": 4,
    "avoid_discretionary_sell_before": "10:00",
    "reselect_top_n": None,
    "default_target_weight": None,
    "selection_mode": "conservative_fill",
    "special_treatment_action": "watch_only",
    "watchlist_preference_bonus": 0.005,
}


ACTION_ORDER = {"allow": 0, "reduce": 1, "watch_only": 2}
ACTION_CN = {"allow": "可保留", "reduce": "降仓", "watch_only": "只观察"}
REASON_CN = {
    "entry_in_low_or_mid_low_zone": "信号日收盘位置较低，符合低位/中低位介入习惯",
    "entry_in_high_chase_zone": "信号日收盘位置偏高，降低追高风险",
    "20d_trend_damaged_watch_only": "20日走势受损，先观察不新开仓",
    "20d_momentum_weak_reduce": "20日动量偏弱，降低仓位",
    "personal_history_severe_loss_symbol": "个人历史在该标的亏损较重，先观察",
    "personal_history_bad_symbol_reduce": "个人历史在该标的胜率/收益较差，降权",
    "personal_history_good_symbol_small_bonus": "个人历史在该标的表现较好，小幅加分",
    "special_treatment_execution_unsupported": "ST股票涨跌停执行规则未完整建模，仅保留风险观察",
    "watchlist_preference_bonus": "自选名单长期关注，小幅加分",
    "no_personal_overlay_adjustment": "未触发个人行为层调整",
}


def _load_rules(path: Path | None) -> dict[str, Any]:
    rules = dict(DEFAULT_RULES)
    if path is None:
        return rules
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import yaml
    except ImportError:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules.update(loaded)
    return rules


def _num(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def load_symbol_history(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["symbol"])
    frame = pd.read_csv(path, dtype={"symbol": str})
    if "symbol" not in frame:
        return pd.DataFrame(columns=["symbol"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    keep = [
        "symbol",
        "trades",
        "pnl",
        "win_rate",
        "avg_ret",
        "avg_holding_days",
        "avg_mfe",
        "avg_mae",
    ]
    return frame[[c for c in keep if c in frame.columns]].drop_duplicates("symbol", keep="last")


def decide_row(row: pd.Series, rules: dict[str, Any]) -> dict[str, object]:
    delta = 0.0
    multiplier = 1.0
    action = "allow"
    reasons: list[str] = []

    close_position = _num(row.get("close_position"))
    ret20 = _num(row.get("return_20d"))
    symbol_trades = _num(row.get("personal_trades"), 0.0)
    symbol_pnl = _num(row.get("personal_pnl"), 0.0)
    symbol_win_rate = _num(row.get("personal_win_rate"), np.nan)
    stock_name = str(
        row.get("stock_name", row.get("security_name", row.get("name", ""))) or ""
    ).upper()

    if np.isfinite(close_position) and close_position <= float(rules["low_entry_position_max"]):
        delta += float(rules["low_entry_score_bonus"])
        reasons.append("entry_in_low_or_mid_low_zone")

    if np.isfinite(close_position) and close_position >= float(rules["high_entry_position_min"]):
        delta += float(rules["high_entry_score_penalty"])
        multiplier *= float(rules["high_entry_weight_multiplier"])
        reasons.append("entry_in_high_chase_zone")

    if np.isfinite(ret20) and ret20 <= float(rules["damaged_20d_return_max"]):
        action = str(rules["damaged_20d_action"])
        multiplier = 0.0
        reasons.append("20d_trend_damaged_watch_only")
    elif (
        np.isfinite(ret20)
        and float(rules["weak_20d_return_min"]) < ret20 <= float(rules["weak_20d_return_max"])
    ):
        delta += float(rules["weak_20d_score_penalty"])
        multiplier *= float(rules["weak_20d_weight_multiplier"])
        reasons.append("20d_momentum_weak_reduce")

    if symbol_trades >= float(rules["symbol_history_min_trades"]):
        if symbol_pnl <= float(rules["severe_bad_symbol_pnl_max"]):
            action = str(rules["severe_bad_symbol_action"])
            multiplier = 0.0
            reasons.append("personal_history_severe_loss_symbol")
        elif symbol_pnl <= float(rules["bad_symbol_pnl_max"]) and (
            not np.isfinite(symbol_win_rate) or symbol_win_rate <= float(rules["bad_symbol_win_rate_max"])
        ):
            delta += float(rules["bad_symbol_score_penalty"])
            multiplier *= float(rules["bad_symbol_weight_multiplier"])
            reasons.append("personal_history_bad_symbol_reduce")
        elif symbol_pnl >= float(rules["good_symbol_pnl_min"]) and (
            np.isfinite(symbol_win_rate) and symbol_win_rate >= float(rules["good_symbol_win_rate_min"])
        ):
            delta += float(rules["good_symbol_score_bonus"])
            reasons.append("personal_history_good_symbol_small_bonus")

    if stock_name.startswith(("ST", "*ST", "SST", "S*ST")):
        action = str(rules["special_treatment_action"])
        multiplier = 0.0
        reasons.append("special_treatment_execution_unsupported")

    # 用户自选名单偏好加分（2026-08-06）：ST/风控降权优先；仅当最终 action=allow 才加。
    if action == "allow" and bool(row.get("user_watchlist", False)):
        bonus = float(rules.get("watchlist_preference_bonus", 0.0))
        if bonus:
            delta += bonus
            reasons.append("watchlist_preference_bonus")

    if action == "allow" and multiplier < 0.999:
        action = "reduce"
    return {
        "personal_score_delta": delta,
        "personal_weight_multiplier": multiplier,
        "personal_action": action,
        "personal_reasons": ";".join(reasons) if reasons else "no_personal_overlay_adjustment",
    }


def _reason_to_cn(reason_text: object) -> str:
    reasons = str(reason_text or "").split(";")
    return "；".join(REASON_CN.get(reason, reason) for reason in reasons if reason)


def _resolve_reselect_top_n(rules: dict[str, Any], reselect_top_n: int | None) -> int | None:
    if reselect_top_n is not None:
        return reselect_top_n
    value = rules.get("reselect_top_n")
    if value is None or value == "":
        return None
    return int(value)


def _resolve_base_target_weight(
    original_weight: pd.Series,
    rules: dict[str, Any],
    base_target_weight: float | None,
) -> float:
    if base_target_weight is not None:
        return float(base_target_weight)
    configured = rules.get("default_target_weight")
    if configured is not None and configured != "":
        return float(configured)
    positive = original_weight[original_weight.gt(0)]
    if positive.empty:
        return 0.0
    return float(positive.median())


def apply_overlay(
    candidates: pd.DataFrame,
    symbol_history: pd.DataFrame,
    rules: dict[str, Any],
    reselect_top_n: int | None = None,
    base_target_weight: float | None = None,
    selection_mode: str | None = None,
) -> pd.DataFrame:
    out = candidates.copy()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    if not symbol_history.empty:
        history = symbol_history.rename(
            columns={
                "trades": "personal_trades",
                "pnl": "personal_pnl",
                "win_rate": "personal_win_rate",
                "avg_ret": "personal_avg_ret",
                "avg_holding_days": "personal_avg_holding_days",
                "avg_mfe": "personal_avg_mfe",
                "avg_mae": "personal_avg_mae",
            }
        )
        out = out.merge(history, on="symbol", how="left")
    else:
        out["personal_trades"] = np.nan
        out["personal_pnl"] = np.nan
        out["personal_win_rate"] = np.nan

    decisions = out.apply(lambda row: pd.Series(decide_row(row, rules)), axis=1)
    out = pd.concat([out, decisions], axis=1)
    out["personal_score_delta"] = pd.to_numeric(out["personal_score_delta"], errors="coerce").fillna(0.0)
    out["personal_weight_multiplier"] = pd.to_numeric(
        out["personal_weight_multiplier"],
        errors="coerce",
    ).fillna(1.0)
    out["baseline_score"] = pd.to_numeric(out["score"], errors="coerce")
    out["behavior_bonus"] = out["personal_score_delta"].clip(lower=0.0)
    out["behavior_penalty"] = out["personal_score_delta"].clip(upper=0.0)
    out["behavior_match_score"] = out["personal_score_delta"]
    out["personal_adjusted_score"] = out["baseline_score"] + out["personal_score_delta"]
    out["final_score"] = out["personal_adjusted_score"]
    out["behavior_action"] = out["personal_action"]
    out["personal_action_cn"] = out["personal_action"].map(ACTION_CN).fillna(out["personal_action"])
    out["behavior_reason"] = out["personal_reasons"]
    out["personal_reasons_cn"] = out["personal_reasons"].map(_reason_to_cn)
    if "target_weight" in out:
        original_weight = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0)
    else:
        original_weight = pd.Series(0.0, index=out.index)
    out["target_weight_before_behavior"] = original_weight
    out["target_weight_after_behavior"] = original_weight * out["personal_weight_multiplier"]
    out["_personal_action_order"] = out["personal_action"].map(ACTION_ORDER).fillna(99).astype(int)
    out = out.sort_values(
        ["_personal_action_order", "personal_adjusted_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    top_n = _resolve_reselect_top_n(rules, reselect_top_n)
    mode = selection_mode or str(rules.get("selection_mode", "conservative_fill"))
    if top_n is not None:
        base_weight = _resolve_base_target_weight(
            out["target_weight_before_behavior"],
            rules,
            base_target_weight,
        )
        eligible = out["personal_action"].ne("watch_only")
        out["personal_selected"] = False
        if mode == "original_adjust":
            originally_selected = out["target_weight_before_behavior"].gt(0.0) & eligible
            selected_idx = out[originally_selected].index
        elif mode == "full_rerank":
            selected_idx = out[eligible].head(max(int(top_n), 0)).index
        elif mode == "conservative_fill":
            originally_selected = out["target_weight_before_behavior"].gt(0.0) & eligible
            selected_idx = list(out[originally_selected].index[: max(int(top_n), 0)])
            seats_left = max(int(top_n), 0) - len(selected_idx)
            if seats_left > 0:
                fillers = out[eligible & ~out.index.isin(selected_idx)].head(seats_left).index
                selected_idx = selected_idx + list(fillers)
        else:
            selected_idx = out[eligible].head(max(int(top_n), 0)).index
        # 名单预留名额（2026-08-06）：top_n 中固定 watchlist_quota 席给用户自选名单票（按分补足）。
        # 只补缺、不超配；被挤出的永远是当前选中里分数最低且非名单的票。
        quota = int(rules.get("watchlist_quota", 0) or 0)
        if quota > 0 and "user_watchlist" in out.columns:
            sel_list = list(selected_idx)
            cur_sel = out.loc[sel_list]
            wl_count = int(cur_sel["user_watchlist"].sum())
            deficit = quota - wl_count
            if deficit > 0:
                rest = out[
                    ~out.index.isin(sel_list)
                    & out["user_watchlist"]
                    & out["personal_action"].ne("watch_only")
                ].sort_values("personal_adjusted_score", ascending=False)
                fill = rest.head(deficit).index.tolist()
                removable = cur_sel[~cur_sel["user_watchlist"]].sort_values("personal_adjusted_score")
                drop = removable.head(len(fill)).index.tolist()
                selected_idx = [i for i in sel_list if i not in drop] + fill
        out.loc[selected_idx, "personal_selected"] = True
        if mode == "original_adjust":
            selected_weight = out["target_weight_after_behavior"]
        else:
            selected_weight = base_weight * out["personal_weight_multiplier"]
        out["personal_adjusted_target_weight"] = np.where(out["personal_selected"], selected_weight, 0.0)
    else:
        if "selected" in out:
            out["personal_selected"] = out["selected"].astype(bool)
        else:
            out["personal_selected"] = out["target_weight_before_behavior"].gt(0.0)
        out["personal_adjusted_target_weight"] = out["target_weight_after_behavior"]

    out["target_weight_after_behavior"] = out["personal_adjusted_target_weight"]
    out["behavior_selected"] = out["personal_selected"]
    out.insert(1, "personal_rank", range(1, len(out) + 1))
    out = out.drop(columns=["_personal_action_order"])
    return out


def write_report(output: Path, table: pd.DataFrame, rules: dict[str, Any]) -> Path:
    report = output.with_suffix(".md")
    action_counts = table["personal_action"].value_counts().to_dict()
    top_cols = [
        "symbol",
        "personal_rank",
        "personal_action",
        "personal_action_cn",
        "personal_selected",
        "personal_adjusted_target_weight",
        "personal_adjusted_score",
        "trend_state",
        "close",
        "return_20d",
        "close_position",
        "personal_pnl",
        "personal_win_rate",
        "personal_reasons_cn",
        "personal_reasons",
    ]
    top = table[[c for c in top_cols if c in table.columns]].head(30)
    selected = table[table["personal_selected"].astype(bool)] if "personal_selected" in table else table.head(0)
    selected_top = selected[[c for c in top_cols if c in selected.columns]]
    if "rank" in selected_top.columns:
        selected_top = selected_top.sort_values("rank")
    gross = float(pd.to_numeric(selected.get("personal_adjusted_target_weight", 0.0), errors="coerce").fillna(0.0).sum())
    if "selected" in table:
        old_selected = set(table.loc[table["selected"].astype(bool), "symbol"].astype(str))
    else:
        old_selected = set()
    new_selected = set(selected["symbol"].astype(str)) if "symbol" in selected else set()
    added = sorted(new_selected - old_selected)
    removed = sorted(old_selected - new_selected)
    lines = [
        "# 个人交易习惯叠加层",
        "",
        "基于历史成交记录提炼的研究用行为叠加层；它只做候选股再排序、降仓和观察过滤，不直接改动主模型训练逻辑。",
        "",
        "## 动作统计",
        json.dumps(action_counts, ensure_ascii=False, indent=2),
        "",
        "## 最终组合",
        f"- 入选数量: {len(selected)}",
        f"- 行为层后目标总仓位: {gross:.2%}",
        f"- 选择模式: {rules.get('selection_mode', 'conservative_fill')}",
        f"- 保留原入选: {len(old_selected & new_selected)}；新增候补: {len(added)}；移出观察: {len(removed)}",
        f"- 新增候补: {', '.join(added) if added else '无'}",
        f"- 移出观察: {', '.join(removed) if removed else '无'}",
        "",
        "## 已吸收的执行规则",
        f"- 默认观察持有下限: {rules['min_default_holding_days']} 个交易日，硬风控触发除外。",
        f"- 避免在 {rules['avoid_discretionary_sell_before']} 前做主观卖出决定，硬风控触发除外。",
        "- 优先保留信号日收盘位置不在高位区的标的。",
        "- 20日走势受损、个人历史亏损较重的标的降仓或只观察。",
        "",
        "## 最终入选清单",
        selected_top.to_markdown(index=False),
        "",
        "## 调整后前30名",
        top.to_markdown(index=False),
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply personal trade-habit overlay to model candidates.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--symbol-history", default=None)
    parser.add_argument("--names-source", default=None)
    parser.add_argument("--rules", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reselect-top-n", type=int, default=None)
    parser.add_argument("--base-target-weight", type=float, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["conservative_fill", "full_rerank", "original_adjust"],
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates, dtype={"symbol": str})
    candidates["symbol"] = candidates["symbol"].astype(str).str.zfill(6)
    if args.names_source:
        name_map = load_name_map(Path(args.names_source))
        candidates["stock_name"] = candidates["symbol"].map(name_map).fillna("")
    rules = _load_rules(Path(args.rules) if args.rules else None)
    symbol_history = load_symbol_history(Path(args.symbol_history) if args.symbol_history else None)
    out = apply_overlay(
        candidates,
        symbol_history,
        rules,
        reselect_top_n=args.reselect_top_n,
        base_target_weight=args.base_target_weight,
        selection_mode=args.selection_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    report = write_report(output, out, rules)
    print(f"Overlay rows={len(out)} output={output}")
    print(f"Report={report}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
