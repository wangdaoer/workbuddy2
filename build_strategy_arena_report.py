"""Build a research-only strategy arena from existing independent backtests.

The arena borrows only the fair-comparison idea: every portfolio entrant is
recomputed on the same intersection of trading dates.  Signal-only research
layers are kept in a separate division because they do not yet have an
independent equity ledger.  Historical dominance never promotes a strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


CN_PORTFOLIO_COLUMNS = {
    "entrant_id": "决策器编号",
    "entrant_name": "决策器名称",
    "league_id": "联赛编号",
    "league_name": "联赛名称",
    "contract_id": "执行合同",
    "role": "竞技角色",
    "source_first_date": "源净值开始日",
    "source_latest_date": "源净值结束日",
    "common_start_date": "公共区间开始日",
    "common_end_date": "公共区间结束日",
    "common_trading_days": "公共交易日数",
    "source_coverage_ratio": "公共区间覆盖率",
    "total_return": "公共区间总收益",
    "annualized_return": "公共区间年化收益",
    "max_drawdown": "公共区间最大回撤",
    "sharpe_like": "公共区间Sharpe",
    "calmar_like": "公共区间Calmar",
    "annualized_volatility": "年化波动率",
    "positive_day_rate": "上涨日占比",
    "worst_daily_return": "最差单日收益",
    "latest_daily_return": "最新单日收益",
    "average_turnover": "平均换手率",
    "total_cost": "累计成本",
    "average_gross_exposure": "平均总敞口",
    "latest_gross_exposure": "最新总敞口",
    "historical_pareto_dominates_league_reference": "历史指标支配联赛基准",
    "same_contract_comparable": "同合同可比较",
    "arena_observation_days": "竞技场观察日数",
    "independent_observation_status": "独立观察状态",
    "independent_observation_count": "独立观察有效天数",
    "independent_observation_target": "独立观察目标天数",
    "promotion_eligible": "允许晋级",
    "promotion_reason": "晋级结论",
}

CN_PAIRWISE_COLUMNS = {
    "league_id": "联赛编号",
    "left_id": "决策器A",
    "right_id": "决策器B",
    "common_return_days": "共同收益日数",
    "daily_return_correlation": "日收益相关性",
    "same_direction_rate": "同涨同跌占比",
    "mean_absolute_return_difference": "平均绝对收益差",
    "right_minus_left_total_return": "B减A总收益",
    "right_minus_left_max_drawdown": "B减A最大回撤",
    "right_minus_left_sharpe": "B减ASharpe",
}

CN_SIGNAL_COLUMNS = {
    "observer_id": "观察器编号",
    "observer_name": "观察器名称",
    "division": "观察分区",
    "evidence_type": "证据类型",
    "signal_count": "信号数量",
    "completed_outcome_count": "已完成结果数",
    "evaluation_horizon_days": "评价周期",
    "average_forward_return": "平均前向收益",
    "win_rate": "胜率",
    "worst_adverse_return": "最差不利波动",
    "status": "观察状态",
    "portfolio_comparable": "可与组合净值比较",
    "promotion_eligible": "允许晋级",
    "note": "说明",
}


@dataclass(frozen=True)
class ArenaEntrant:
    entrant_id: str
    entrant_name: str
    league_id: str
    league_name: str
    contract_id: str
    role: str
    equity_path: Path
    metrics_path: Path


def _json_object(path: Path, *, required: bool = True) -> dict[str, object]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int(value: object) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else 0


def load_equity_curve(path: Path, *, asof_date: str | pd.Timestamp) -> pd.DataFrame:
    """Load one auditable curve and reject duplicate or future observations."""

    frame = pd.read_csv(path, low_memory=False)
    required = {"date", "equity"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Equity curve is missing columns {missing}: {path}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError(f"Equity curve contains invalid dates: {path}")
    if frame["date"].duplicated().any():
        raise ValueError(f"Equity curve contains duplicate dates: {path}")
    asof = pd.Timestamp(asof_date).normalize()
    future = frame[frame["date"].gt(asof)]
    if not future.empty:
        raise ValueError(
            f"Equity curve contains dates after {asof.date()}: {path}; "
            f"latest={future['date'].max().date()}"
        )
    if frame["equity"].isna().any() or frame["equity"].le(0.0).any():
        raise ValueError(f"Equity curve contains non-positive or missing equity: {path}")
    for column in ("turnover", "cost", "gross_exposure"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if len(frame) < 2:
        raise ValueError(f"Equity curve requires at least two rows: {path}")
    return frame.sort_values("date").set_index("date")


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1.0).min())


def _sharpe_like(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    volatility = float(clean.std(ddof=1))
    if volatility <= 0.0 or not np.isfinite(volatility):
        return np.nan
    return float(clean.mean() / volatility * math.sqrt(252.0))


def common_trading_index(curves: Mapping[str, pd.DataFrame], *, min_days: int) -> pd.DatetimeIndex:
    if not curves:
        raise ValueError("At least one arena entrant is required")
    common = set.intersection(*(set(frame.index) for frame in curves.values()))
    index = pd.DatetimeIndex(sorted(common))
    if len(index) < int(min_days):
        raise ValueError(f"Arena common history has {len(index)} rows; requires {min_days}")
    return index


def _portfolio_metrics(
    entrant: ArenaEntrant,
    curve: pd.DataFrame,
    common_index: pd.DatetimeIndex,
    source_metrics: Mapping[str, object],
) -> dict[str, object]:
    aligned = curve.loc[common_index].copy()
    nav = aligned["equity"] / float(aligned["equity"].iloc[0])
    returns = nav.pct_change(fill_method=None).dropna()
    periods = max(len(returns), 1)
    total_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (252.0 / periods) - 1.0)
    max_drawdown = _max_drawdown(nav)
    sharpe = _sharpe_like(returns)
    volatility = float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) >= 2 else np.nan
    return {
        "entrant_id": entrant.entrant_id,
        "entrant_name": entrant.entrant_name,
        "league_id": entrant.league_id,
        "league_name": entrant.league_name,
        "contract_id": entrant.contract_id,
        "role": entrant.role,
        "source_first_date": curve.index.min().strftime("%Y-%m-%d"),
        "source_latest_date": curve.index.max().strftime("%Y-%m-%d"),
        "common_start_date": common_index.min().strftime("%Y-%m-%d"),
        "common_end_date": common_index.max().strftime("%Y-%m-%d"),
        "common_trading_days": int(len(common_index)),
        "source_coverage_ratio": float(len(common_index) / len(curve)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_like": sharpe,
        "calmar_like": (
            annualized_return / abs(max_drawdown)
            if pd.notna(max_drawdown) and max_drawdown < 0.0
            else np.nan
        ),
        "annualized_volatility": volatility,
        "positive_day_rate": float(returns.gt(0.0).mean()) if len(returns) else np.nan,
        "worst_daily_return": float(returns.min()) if len(returns) else np.nan,
        "latest_daily_return": float(returns.iloc[-1]) if len(returns) else np.nan,
        "average_turnover": float(aligned["turnover"].mean()) if aligned["turnover"].notna().any() else np.nan,
        "total_cost": float(aligned["cost"].sum()) if aligned["cost"].notna().any() else np.nan,
        "average_gross_exposure": (
            float(aligned["gross_exposure"].mean())
            if aligned["gross_exposure"].notna().any()
            else np.nan
        ),
        "latest_gross_exposure": (
            float(aligned["gross_exposure"].iloc[-1])
            if pd.notna(aligned["gross_exposure"].iloc[-1])
            else np.nan
        ),
        "source_reported_total_return": pd.to_numeric(
            source_metrics.get("total_return"), errors="coerce"
        ),
        "source_reported_max_drawdown": pd.to_numeric(
            source_metrics.get("max_drawdown"), errors="coerce"
        ),
        "source_reported_sharpe_like": pd.to_numeric(
            source_metrics.get("sharpe_like"), errors="coerce"
        ),
        "historical_pareto_dominates_league_reference": False,
        "same_contract_comparable": True,
        "arena_observation_days": 0,
        "independent_observation_status": "not_started",
        "independent_observation_count": 0,
        "independent_observation_target": 0,
        "promotion_eligible": False,
        "promotion_reason": "历史回测结果不能自动晋级。",
    }


def _is_pareto_dominant(challenger: pd.Series, champion: pd.Series) -> bool:
    metrics = ("total_return", "max_drawdown", "sharpe_like")
    values = [
        (pd.to_numeric(challenger.get(metric), errors="coerce"), pd.to_numeric(champion.get(metric), errors="coerce"))
        for metric in metrics
    ]
    if any(pd.isna(left) or pd.isna(right) for left, right in values):
        return False
    no_worse = all(left >= right for left, right in values)
    strictly_better = any(left > right for left, right in values)
    return bool(no_worse and strictly_better)


def apply_observation_evidence(
    leaderboard: pd.DataFrame,
    tracking_summary: Mapping[str, object],
) -> pd.DataFrame:
    result = leaderboard.copy()
    production = result[result["role"].eq("production_champion")]
    if len(production) != 1:
        raise ValueError("Arena requires exactly one production_champion")
    for league_id, league in result.groupby("league_id", sort=False):
        references = league[league["role"].isin({"production_champion", "league_reference"})]
        if len(references) != 1:
            raise ValueError(f"Arena league {league_id} requires exactly one reference")
        reference = references.iloc[0]
        contract_ids = set(league["contract_id"].astype(str))
        same_contract = len(contract_ids) == 1
        result.loc[league.index, "same_contract_comparable"] = same_contract
        for index, row in league.iterrows():
            is_reference = index == reference.name
            result.at[index, "historical_pareto_dominates_league_reference"] = bool(
                False
                if is_reference or not same_contract
                else _is_pareto_dominant(row, reference)
            )

    tracking_status = str(tracking_summary.get("status") or "missing")
    valid_count = int(tracking_summary.get("valid_observation_count") or 0)
    target_count = int(tracking_summary.get("target_days") or 0)
    for index, row in result.iterrows():
        entrant_id = str(row["entrant_id"])
        if row["role"] == "production_champion":
            result.at[index, "independent_observation_status"] = "production_reference"
            result.at[index, "promotion_reason"] = "当前正式冠军；竞技场只记录，不自动改写生产状态。"
        elif entrant_id == "core_breadth_guard":
            result.at[index, "independent_observation_status"] = "collecting"
            result.at[index, "independent_observation_target"] = 60
            result.at[index, "promotion_reason"] = (
                "同族 breadth-guard 挑战者从竞技场启用后的新样本开始观察；"
                "至少 60 个成熟观察日并通过人工复核前禁止晋级。"
            )
        elif entrant_id in {"pullback_baseline", "pullback_dynamic"}:
            result.at[index, "independent_observation_status"] = tracking_status
            result.at[index, "independent_observation_count"] = valid_count
            result.at[index, "independent_observation_target"] = target_count
            if entrant_id == "pullback_dynamic":
                delta = pd.to_numeric(tracking_summary.get("cumulative_return_delta"), errors="coerce")
                delta_text = "未知" if pd.isna(delta) else f"{float(delta):.2%}"
                result.at[index, "promotion_reason"] = (
                    f"动态挑战者独立观察 {valid_count}/{target_count} 日，状态 {tracking_status}，"
                    f"相对卫星基线累计差 {delta_text}；仍需人工复核且禁止自动晋级。"
                )
            else:
                result.at[index, "promotion_reason"] = (
                    "强势回调联赛基准，只与同合同动态敞口挑战者比较；不参与主策略晋级。"
                )
        else:
            result.at[index, "promotion_reason"] = "尚未建立独立前向观察账本。"
    result["promotion_eligible"] = False
    return result


def build_portfolio_division(
    entrants: Iterable[ArenaEntrant],
    *,
    asof_date: str,
    tracking_summary: Mapping[str, object] | None = None,
    min_common_days: int = 252,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DatetimeIndex]]:
    entrants = tuple(entrants)
    curves = {
        entrant.entrant_id: load_equity_curve(entrant.equity_path, asof_date=asof_date)
        for entrant in entrants
    }
    common_indexes: dict[str, pd.DatetimeIndex] = {}
    for league_id in sorted({entrant.league_id for entrant in entrants}):
        league_curves = {
            entrant.entrant_id: curves[entrant.entrant_id]
            for entrant in entrants
            if entrant.league_id == league_id
        }
        common_indexes[league_id] = common_trading_index(
            league_curves, min_days=min_common_days
        )
    rows = []
    for entrant in entrants:
        metrics = _json_object(entrant.metrics_path)
        rows.append(
            _portfolio_metrics(
                entrant,
                curves[entrant.entrant_id],
                common_indexes[entrant.league_id],
                metrics,
            )
        )
    leaderboard = apply_observation_evidence(pd.DataFrame(rows), tracking_summary or {})
    role_order = {
        "production_champion": 0,
        "league_reference": 0,
        "research_challenger": 1,
        "shadow_challenger": 2,
    }
    leaderboard["_role_order"] = leaderboard["role"].map(role_order).fillna(99)
    leaderboard = leaderboard.sort_values(
        ["league_id", "_role_order", "entrant_id"]
    ).drop(columns="_role_order")
    return leaderboard.reset_index(drop=True), curves, common_indexes


def build_pairwise_comparison(
    leaderboard: pd.DataFrame,
    curves: Mapping[str, pd.DataFrame],
    common_indexes: Mapping[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    metrics = leaderboard.set_index("entrant_id")
    rows: list[dict[str, object]] = []
    for league_id, league in leaderboard.groupby("league_id", sort=False):
        common_index = common_indexes[str(league_id)]
        ids = league["entrant_id"].tolist()
        returns = {
            entrant_id: curves[entrant_id]
            .loc[common_index, "equity"]
            .pct_change(fill_method=None)
            .dropna()
            for entrant_id in ids
        }
        for left_pos, left_id in enumerate(ids):
            for right_id in ids[left_pos + 1 :]:
                pair = pd.concat(
                    [returns[left_id].rename("left"), returns[right_id].rename("right")],
                    axis=1,
                    join="inner",
                ).dropna()
                left_metrics = metrics.loc[left_id]
                right_metrics = metrics.loc[right_id]
                rows.append(
                    {
                        "league_id": league_id,
                        "left_id": left_id,
                        "right_id": right_id,
                        "common_return_days": int(len(pair)),
                        "daily_return_correlation": (
                            float(pair["left"].corr(pair["right"]))
                            if len(pair) >= 2
                            else np.nan
                        ),
                        "same_direction_rate": (
                            float((np.sign(pair["left"]) == np.sign(pair["right"])).mean())
                            if len(pair)
                            else np.nan
                        ),
                        "mean_absolute_return_difference": (
                            float((pair["left"] - pair["right"]).abs().mean())
                            if len(pair)
                            else np.nan
                        ),
                        "right_minus_left_total_return": float(
                            right_metrics["total_return"] - left_metrics["total_return"]
                        ),
                        "right_minus_left_max_drawdown": float(
                            right_metrics["max_drawdown"] - left_metrics["max_drawdown"]
                        ),
                        "right_minus_left_sharpe": float(
                            right_metrics["sharpe_like"] - left_metrics["sharpe_like"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_signal_observation_division(
    family_health: pd.DataFrame,
    *,
    czsc_metadata: Mapping[str, object] | None = None,
    flow_metadata: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in family_health.iterrows():
        rows.append(
            {
                "observer_id": f"family:{row.get('strategy_family', '')}",
                "observer_name": row.get("strategy_family_cn", row.get("strategy_family", "")),
                "division": "next_open_signal_family",
                "evidence_type": "realized_forward_return",
                "signal_count": _safe_int(row.get("signal_count")),
                "completed_outcome_count": _safe_int(row.get("completed_count")),
                "evaluation_horizon_days": _safe_int(row.get("selected_horizon_days")),
                "average_forward_return": pd.to_numeric(row.get("avg_return"), errors="coerce"),
                "win_rate": pd.to_numeric(row.get("win_rate"), errors="coerce"),
                "worst_adverse_return": pd.to_numeric(
                    row.get("worst_adverse_return"), errors="coerce"
                ),
                "status": row.get("family_health_status", "missing"),
                "portfolio_comparable": False,
                "promotion_eligible": False,
                "note": row.get("family_health_reason", ""),
            }
        )

    czsc = dict(czsc_metadata or {})
    if czsc:
        rows.append(
            {
                "observer_id": "shadow:czsc_structure",
                "observer_name": "CZSC缠论结构",
                "division": "structure_shadow",
                "evidence_type": "unrealized_snapshot_labels",
                "signal_count": int(czsc.get("candidate_count") or 0),
                "completed_outcome_count": 0,
                "evaluation_horizon_days": 0,
                "average_forward_return": np.nan,
                "win_rate": np.nan,
                "worst_adverse_return": np.nan,
                "status": czsc.get("status", "missing"),
                "portfolio_comparable": False,
                "promotion_eligible": False,
                "note": (
                    f"已分析 {int(czsc.get('analyzed_count') or 0)} 只，结构共振 "
                    f"{int(czsc.get('pattern_confluence_count') or 0)} 只；尚无独立前向收益。"
                ),
            }
        )

    flow = dict(flow_metadata or {})
    if flow:
        rows.append(
            {
                "observer_id": "shadow:main_net_volume",
                "observer_name": "主力净量影子",
                "division": "flow_shadow",
                "evidence_type": "unrealized_vendor_flow",
                "signal_count": int(flow.get("latest_source_rows") or 0),
                "completed_outcome_count": 0,
                "evaluation_horizon_days": 0,
                "average_forward_return": np.nan,
                "win_rate": np.nan,
                "worst_adverse_return": np.nan,
                "status": flow.get("status", "missing"),
                "portfolio_comparable": False,
                "promotion_eligible": False,
                "note": (
                    f"有效源交易日 {int(flow.get('available_source_sessions') or 0)}，"
                    f"5日可用股票 {int(flow.get('eligible_5d_rows') or 0)}；尚无独立前向收益。"
                ),
            }
        )
    return pd.DataFrame(rows)


def update_arena_history(
    history_path: Path,
    leaderboard: pd.DataFrame,
    *,
    asof_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot_columns = [
        "entrant_id",
        "entrant_name",
        "league_id",
        "league_name",
        "contract_id",
        "role",
        "source_latest_date",
        "total_return",
        "max_drawdown",
        "sharpe_like",
        "latest_daily_return",
        "average_turnover",
        "average_gross_exposure",
        "historical_pareto_dominates_league_reference",
        "same_contract_comparable",
        "arena_observation_days",
        "independent_observation_status",
        "independent_observation_count",
        "independent_observation_target",
        "promotion_eligible",
        "promotion_reason",
    ]
    if history_path.exists():
        history = pd.read_csv(history_path, dtype={"entrant_id": str, "asof_date": str})
    else:
        history = pd.DataFrame(columns=["asof_date", *snapshot_columns])
    asof_text = pd.Timestamp(asof_date).strftime("%Y-%m-%d")
    prior = history[history["asof_date"].ne(asof_text)].copy()
    prospective_counts = prior.groupby("entrant_id")["asof_date"].nunique().to_dict()
    updated = leaderboard.copy()
    updated["arena_observation_days"] = [
        int(prospective_counts.get(entrant_id, 0)) + 1
        for entrant_id in updated["entrant_id"]
    ]
    breadth_mask = updated["entrant_id"].eq("core_breadth_guard")
    if breadth_mask.any():
        matured = (updated.loc[breadth_mask, "arena_observation_days"] - 1).clip(lower=0)
        updated.loc[breadth_mask, "independent_observation_count"] = matured.astype(int)
        updated.loc[breadth_mask, "independent_observation_target"] = 60
        updated.loc[breadth_mask, "independent_observation_status"] = np.where(
            matured.ge(60), "sample_target_reached_pending_gates", "collecting"
        )
    snapshot = updated[snapshot_columns].copy()
    snapshot.insert(0, "asof_date", asof_text)
    combined = (
        snapshot.reset_index(drop=True)
        if prior.empty
        else pd.concat([prior, snapshot], ignore_index=True, sort=False)
    )
    combined = combined.sort_values(["asof_date", "entrant_id"]).reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = history_path.with_suffix(f"{history_path.suffix}.tmp")
    combined.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(history_path)
    return updated, combined


def build_arena_metadata(
    leaderboard: pd.DataFrame,
    pairwise: pd.DataFrame,
    signal_division: pd.DataFrame,
    entrants: Iterable[ArenaEntrant],
    *,
    asof_date: str,
    tracking_summary: Mapping[str, object],
    generated_at: str | None = None,
) -> dict[str, object]:
    champion = leaderboard[leaderboard["role"].eq("production_champion")].iloc[0]
    league_references: dict[str, str] = {}
    league_leaders: dict[str, dict[str, str]] = {}
    pareto_by_league: dict[str, list[str]] = {}
    common_windows: dict[str, dict[str, object]] = {}
    for league_id, league in leaderboard.groupby("league_id", sort=False):
        reference = league[
            league["role"].isin({"production_champion", "league_reference"})
        ].iloc[0]
        league_references[str(league_id)] = str(reference["entrant_id"])
        league_leaders[str(league_id)] = {
            "total_return": str(league.loc[league["total_return"].idxmax(), "entrant_id"]),
            "max_drawdown": str(league.loc[league["max_drawdown"].idxmax(), "entrant_id"]),
            "sharpe_like": str(league.loc[league["sharpe_like"].idxmax(), "entrant_id"]),
        }
        pareto_by_league[str(league_id)] = league[
            league["historical_pareto_dominates_league_reference"]
            .fillna(False)
            .astype(bool)
        ]["entrant_id"].tolist()
        common_windows[str(league_id)] = {
            "start_date": str(league["common_start_date"].iloc[0]),
            "end_date": str(league["common_end_date"].iloc[0]),
            "trading_days": int(league["common_trading_days"].iloc[0]),
        }
    source_hashes: dict[str, dict[str, str]] = {}
    for entrant in entrants:
        source_hashes[entrant.entrant_id] = {
            "equity": _file_sha256(entrant.equity_path),
            "metrics": _file_sha256(entrant.metrics_path),
        }
    source_dates_match = leaderboard["source_latest_date"].eq(
        pd.Timestamp(asof_date).strftime("%Y-%m-%d")
    ).all()
    return {
        "schema_version": 1,
        "status": "research_ready" if source_dates_match else "partial",
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "asof_date": pd.Timestamp(asof_date).strftime("%Y-%m-%d"),
        "production_champion": str(champion["entrant_id"]),
        "league_references": league_references,
        "portfolio_entrant_count": int(len(leaderboard)),
        "signal_observer_count": int(len(signal_division)),
        "common_windows": common_windows,
        "all_sources_current_to_asof": bool(source_dates_match),
        "historical_metric_leaders_by_league": league_leaders,
        "historical_pareto_by_league": pareto_by_league,
        "pairwise_comparison_count": int(len(pairwise)),
        "independent_observation_status": tracking_summary.get("status", "missing"),
        "independent_observation_count": int(
            tracking_summary.get("valid_observation_count") or 0
        ),
        "independent_observation_target": int(tracking_summary.get("target_days") or 0),
        "independent_cumulative_return_delta": tracking_summary.get(
            "cumulative_return_delta"
        ),
        "automatic_promotion": False,
        "promotion_decision": "hold_production_champion",
        "promotion_reason": (
            "晋级只允许在同一联赛、同一执行合同内评估；历史结果只用于发现挑战者。"
            "必须完成预注册独立观察和人工复核，本报告无权修改正式策略。"
        ),
        "cross_family_promotion_comparison": False,
        "execution_timing_declaration": "signals_after_close_execute_next_open",
        "llm_trading_enabled": False,
        "external_alpha_arena_code_used": False,
        "research_only": True,
        "selection_effect": False,
        "portfolio_weight_effect": False,
        "source_hashes": source_hashes,
    }


def _format_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    percent_tokens = (
        "return",
        "drawdown",
        "volatility",
        "rate",
        "turnover",
        "cost",
        "exposure",
        "correlation",
        "difference",
    )
    for column in out.columns:
        if column.endswith(("_days", "_count")):
            continue
        if any(token in column for token in percent_tokens):
            numeric = pd.to_numeric(out[column], errors="coerce")
            if numeric.notna().any():
                out[column] = numeric.map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    for column in ("sharpe_like", "calmar_like", "right_minus_left_sharpe"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    return out


def write_arena_outputs(
    leaderboard: pd.DataFrame,
    pairwise: pd.DataFrame,
    signal_division: pd.DataFrame,
    metadata: Mapping[str, object],
    output_dir: Path,
    *,
    token: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "portfolio": output_dir / f"strategy_arena_portfolio_{token}.csv",
        "portfolio_cn": output_dir / f"strategy_arena_portfolio_{token}_cn.csv",
        "pairwise": output_dir / f"strategy_arena_pairwise_{token}.csv",
        "signal_division": output_dir / f"strategy_arena_signal_division_{token}.csv",
        "metadata": output_dir / f"strategy_arena_{token}.json",
        "report": output_dir / f"strategy_arena_{token}.md",
    }
    leaderboard.to_csv(paths["portfolio"], index=False, encoding="utf-8-sig")
    leaderboard.rename(columns=CN_PORTFOLIO_COLUMNS).to_csv(
        paths["portfolio_cn"], index=False, encoding="utf-8-sig"
    )
    pairwise.to_csv(paths["pairwise"], index=False, encoding="utf-8-sig")
    signal_division.to_csv(paths["signal_division"], index=False, encoding="utf-8-sig")
    paths["metadata"].write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    portfolio_columns = [
        "entrant_id",
        "entrant_name",
        "league_name",
        "contract_id",
        "role",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_like",
        "annualized_volatility",
        "average_turnover",
        "average_gross_exposure",
        "historical_pareto_dominates_league_reference",
        "same_contract_comparable",
        "arena_observation_days",
        "independent_observation_status",
        "independent_observation_count",
        "independent_observation_target",
        "promotion_eligible",
        "promotion_reason",
    ]
    lines = [
        f"# 策略决策器竞技场 {metadata['asof_date']}",
        "",
        "组合分区只比较共同交易日上的独立净值曲线；信号观察分区没有独立净值，不能与组合排名混合。",
        "",
        f"- 正式冠军: `{metadata['production_champion']}`",
        f"- 各联赛基准: {metadata['league_references']}",
        f"- 各联赛公共区间: {metadata['common_windows']}",
        f"- 各联赛历史收益 / 回撤 / Sharpe 领先者: {metadata['historical_metric_leaders_by_league']}",
        f"- 各联赛历史 Pareto 挑战者: {metadata['historical_pareto_by_league']}",
        f"- 独立观察: {metadata['independent_observation_count']} / {metadata['independent_observation_target']}，状态 `{metadata['independent_observation_status']}`",
        "- 晋级结论: 保持正式冠军；禁止自动晋级。",
        "",
        "## 组合竞技场",
        "",
        _format_percent_columns(leaderboard[portfolio_columns])
        .rename(columns=CN_PORTFOLIO_COLUMNS)
        .to_markdown(index=False, disable_numparse=True),
        "",
        "## 两两差异",
        "",
        (
            _format_percent_columns(pairwise)
            .rename(columns=CN_PAIRWISE_COLUMNS)
            .to_markdown(index=False, disable_numparse=True)
            if not pairwise.empty
            else "_暂无两两比较。_"
        ),
        "",
        "## 信号观察分区",
        "",
        (
            _format_percent_columns(signal_division)
            .rename(columns=CN_SIGNAL_COLUMNS)
            .to_markdown(index=False, disable_numparse=True)
            if not signal_division.empty
            else "_暂无信号观察器。_"
        ),
        "",
        "## 使用边界",
        "",
        "- 历史全样本占优只用于筛选挑战者，不是独立样本证据。",
        "- 不同策略族只允许同屏展示，不允许跨联赛获得晋级资格。",
        "- 所有策略必须继续遵守收盘后生成信号、下一交易日开盘成交的时间约束。",
        "- CZSC、主力净量等影子层在建立前向结果账本前，不参与组合排名和晋级。",
        "- 本报告不调用大模型生成交易，不改变选股或目标仓位。",
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def run_arena(
    entrants: Iterable[ArenaEntrant],
    *,
    asof_date: str,
    output_dir: Path,
    history_path: Path,
    tracking_summary_path: Path,
    family_health_path: Path | None = None,
    czsc_metadata_path: Path | None = None,
    flow_metadata_path: Path | None = None,
    min_common_days: int = 252,
    generated_at: str | None = None,
) -> tuple[dict[str, Path], dict[str, object]]:
    entrants = tuple(entrants)
    tracking = _json_object(tracking_summary_path, required=False)
    if tracking.get("latest_asof_date") not in {None, pd.Timestamp(asof_date).strftime("%Y-%m-%d")}:
        tracking = {}
    leaderboard, curves, common_indexes = build_portfolio_division(
        entrants,
        asof_date=asof_date,
        tracking_summary=tracking,
        min_common_days=min_common_days,
    )
    leaderboard, _ = update_arena_history(
        history_path, leaderboard, asof_date=asof_date
    )
    pairwise = build_pairwise_comparison(leaderboard, curves, common_indexes)
    family_health = (
        pd.read_csv(family_health_path, low_memory=False)
        if family_health_path is not None and family_health_path.exists()
        else pd.DataFrame()
    )
    signal_division = build_signal_observation_division(
        family_health,
        czsc_metadata=(
            _json_object(czsc_metadata_path, required=False)
            if czsc_metadata_path is not None
            else {}
        ),
        flow_metadata=(
            _json_object(flow_metadata_path, required=False)
            if flow_metadata_path is not None
            else {}
        ),
    )
    metadata = build_arena_metadata(
        leaderboard,
        pairwise,
        signal_division,
        entrants,
        asof_date=asof_date,
        tracking_summary=tracking,
        generated_at=generated_at,
    )
    token = pd.Timestamp(asof_date).strftime("%Y%m%d")
    paths = write_arena_outputs(
        leaderboard, pairwise, signal_division, metadata, output_dir, token=token
    )
    paths["history"] = history_path
    return paths, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the research-only strategy arena.")
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--champion-equity", required=True)
    parser.add_argument("--champion-metrics", required=True)
    parser.add_argument("--breadth-equity", required=True)
    parser.add_argument("--breadth-metrics", required=True)
    parser.add_argument("--baseline-equity", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--dynamic-equity", required=True)
    parser.add_argument("--dynamic-metrics", required=True)
    parser.add_argument("--tracking-summary", required=True)
    parser.add_argument("--family-health", default=None)
    parser.add_argument("--czsc-metadata", default=None)
    parser.add_argument("--flow-metadata", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--min-common-days", type=int, default=252)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entrants = (
        ArenaEntrant(
            "core_rank",
            "多因子主策略",
            "core_next_open",
            "多因子次日开盘联赛",
            "next_open_rank_v1",
            "production_champion",
            Path(args.champion_equity),
            Path(args.champion_metrics),
        ),
        ArenaEntrant(
            "core_breadth_guard",
            "多因子广度风控挑战者",
            "core_next_open",
            "多因子次日开盘联赛",
            "next_open_rank_v1",
            "shadow_challenger",
            Path(args.breadth_equity),
            Path(args.breadth_metrics),
        ),
        ArenaEntrant(
            "pullback_baseline",
            "强势回调卫星基线",
            "pullback_satellite",
            "强势回调卫星联赛",
            "strong_pullback_satellite_v1",
            "league_reference",
            Path(args.baseline_equity),
            Path(args.baseline_metrics),
        ),
        ArenaEntrant(
            "pullback_dynamic",
            "强势回调动态风险敞口",
            "pullback_satellite",
            "强势回调卫星联赛",
            "strong_pullback_satellite_v1",
            "shadow_challenger",
            Path(args.dynamic_equity),
            Path(args.dynamic_metrics),
        ),
    )
    paths, metadata = run_arena(
        entrants,
        asof_date=args.asof_date,
        output_dir=Path(args.output_dir),
        history_path=Path(args.history),
        tracking_summary_path=Path(args.tracking_summary),
        family_health_path=Path(args.family_health) if args.family_health else None,
        czsc_metadata_path=Path(args.czsc_metadata) if args.czsc_metadata else None,
        flow_metadata_path=Path(args.flow_metadata) if args.flow_metadata else None,
        min_common_days=args.min_common_days,
    )
    print(json.dumps(metadata, ensure_ascii=False))
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
