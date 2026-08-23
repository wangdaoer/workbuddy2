"""Research-only proxy for possible institutional accumulation before a price move.

The public A-share tape cannot identify the beneficial owner behind an order. This
module therefore combines point-in-time price structure, persistent turnover and
two independently sourced THS money-flow fields. It emits an observation layer;
it never changes selection, weights or orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from panel_io import iter_panel, panel_columns

from build_daily_personal_overlay_report import load_name_map
from execution_rules import normalize_symbol


@dataclass(frozen=True)
class AccumulationConfig:
    min_price_history: int = 80
    min_flow_history: int = 5
    max_daily_return: float = 0.07
    min_return_20d: float = -0.05
    max_return_20d: float = 0.20
    min_drawdown_20d: float = -0.12
    min_distance_ma20: float = -0.03
    max_distance_ma20: float = 0.10
    max_range_15d: float = 0.25
    min_ma20_ma60_ratio: float = 0.96
    min_amount_ratio_5d: float = 1.10
    max_amount_ratio_5d: float = 2.50
    min_up_amount_persistence_10d: float = 0.50
    max_single_day_amount_spike_20d: float = 3.0
    min_flow_positive_ratio_5d: float = 0.60
    min_flow_rank_5d: float = 0.60
    watch_score: float = 60.0
    strong_score: float = 75.0
    top_n: int = 80
    minimum_completed_samples: int = 80
    minimum_bucket_samples: int = 20
    primary_horizon: int = 5
    minimum_primary_hit_rate: float = 0.52
    minimum_primary_mean_return: float = 0.0
    registration_id: str = "institutional_accumulation_shadow_v1"
    registration_date: str = "2026-07-18"
    validation_start_date: str = "2026-07-20"


CN_COLUMNS = {
    "date": "信号日期",
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "institutional_accumulation_level": "疑似建仓分层",
    "institutional_accumulation_score": "疑似建仓评分",
    "accumulation_reason": "观察理由",
    "close": "收盘价",
    "return_1d": "当日涨跌",
    "return_5d": "5日涨跌",
    "return_20d": "20日涨跌",
    "drawdown_20d": "20日高点回撤",
    "distance_ma20": "相对MA20",
    "ma20_ma60_ratio": "MA20相对MA60",
    "range_15d": "15日振幅",
    "amount_ratio_5d": "近5日成交额放大",
    "up_amount_persistence_10d": "上涨放量连续性",
    "single_day_amount_spike_20d": "20日单日爆量倍数",
    "main_net_volume_ratio_5d": "5日平均主力净量",
    "main_net_volume_positive_ratio_5d": "5日主力净量为正占比",
    "main_net_volume_rank_5d": "5日主力净量分位",
    "main_net_inflow_amount_ratio_5d": "5日大单净额成交额比",
    "main_net_inflow_positive_ratio_5d": "5日大单净流入占比",
    "price_setup": "价格结构通过",
    "volume_setup": "成交持续性通过",
    "flow_evidence_available": "资金流证据可用",
    "flow_confirmed": "资金流确认",
    "tracking_eligible": "纳入影子跟踪",
    "signal_active": "完整观察信号",
    "research_only": "仅供研究",
    "trade_instruction": "交易指令",
    "selection_effect": "影响主模型选股",
}

REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "amount")
OPTIONAL_FLOW_COLUMNS = ("main_net_volume_ratio", "main_net_inflow")


def load_config(path: Path | None) -> AccumulationConfig:
    if path is None:
        return AccumulationConfig()
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    payload = payload.get("institutional_accumulation", payload)
    if not isinstance(payload, dict):
        raise ValueError("institutional_accumulation config must be a mapping")
    allowed = {item.name for item in fields(AccumulationConfig)}
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown institutional accumulation settings: {', '.join(unknown)}")
    return AccumulationConfig(**payload)


def config_hash(config: AccumulationConfig) -> str:
    canonical = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def implementation_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_panel(
    path: Path,
    *,
    asof_date: str | pd.Timestamp | None = None,
    lookback_calendar_days: int = 540,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    header = panel_columns(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"Panel is missing required columns: {', '.join(missing)}")
    usecols = list(REQUIRED_COLUMNS) + [col for col in OPTIONAL_FLOW_COLUMNS if col in header]
    asof = pd.Timestamp(asof_date) if asof_date is not None else None
    start = asof - pd.Timedelta(days=int(lookback_calendar_days)) if asof is not None else None
    chunks: list[pd.DataFrame] = []
    for chunk in iter_panel(
        path,
        columns=usecols,
        dtype={"symbol": str},
        chunksize=250_000,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        if asof is not None:
            chunk = chunk[chunk["date"].between(start, asof, inclusive="both")]
        if not chunk.empty:
            chunks.append(chunk)
    panel = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    target_date = asof if asof is not None else panel["date"].max()
    target_rows = panel["date"].eq(target_date) if not pd.isna(target_date) else pd.Series(False, index=panel.index)
    availability = {
        column: bool(
            column in panel.columns
            and pd.to_numeric(panel.loc[target_rows, column], errors="coerce").notna().any()
        )
        for column in OPTIONAL_FLOW_COLUMNS
    }
    for column in OPTIONAL_FLOW_COLUMNS:
        if column not in panel:
            panel[column] = np.nan
    return panel, availability


def _rolling_transform(grouped: pd.core.groupby.DataFrameGroupBy, column: str, window: int, method: str) -> pd.Series:
    if method == "mean":
        return grouped[column].transform(lambda s: s.rolling(window, min_periods=window).mean())
    if method == "median":
        return grouped[column].transform(lambda s: s.rolling(window, min_periods=window).median())
    if method == "max":
        return grouped[column].transform(lambda s: s.rolling(window, min_periods=window).max())
    if method == "min":
        return grouped[column].transform(lambda s: s.rolling(window, min_periods=window).min())
    if method == "count":
        return grouped[column].transform(lambda s: s.rolling(window, min_periods=window).count())
    raise ValueError(f"Unsupported rolling method: {method}")


def compute_accumulation_features(
    panel: pd.DataFrame,
    config: AccumulationConfig | None = None,
) -> pd.DataFrame:
    config = config or AccumulationConfig()
    required = set(REQUIRED_COLUMNS)
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].map(normalize_symbol)
    numeric_columns = ["open", "high", "low", "close", "amount", *OPTIONAL_FLOW_COLUMNS]
    for column in numeric_columns:
        if column not in out:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out[out["symbol"].ne("") & out["close"].gt(0.0)]
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = out.groupby("symbol", sort=False)

    out["price_history"] = grouped["close"].transform(lambda s: s.notna().cumsum())
    for horizon in (1, 5, 20):
        out[f"return_{horizon}d"] = grouped["close"].pct_change(horizon, fill_method=None)
    out["ma20"] = _rolling_transform(grouped, "close", 20, "mean")
    out["ma60"] = _rolling_transform(grouped, "close", 60, "mean")
    out["ma20_ma60_ratio"] = out["ma20"] / out["ma60"].replace(0.0, np.nan)
    out["distance_ma20"] = out["close"] / out["ma20"].replace(0.0, np.nan) - 1.0
    out["high_20d"] = _rolling_transform(grouped, "high", 20, "max")
    out["drawdown_20d"] = out["close"] / out["high_20d"].replace(0.0, np.nan) - 1.0
    high_15 = _rolling_transform(grouped, "high", 15, "max")
    low_15 = _rolling_transform(grouped, "low", 15, "min")
    out["range_15d"] = high_15 / low_15.replace(0.0, np.nan) - 1.0

    out["amount_recent_5d"] = _rolling_transform(grouped, "amount", 5, "mean")
    out["amount_prior_15d"] = grouped["amount"].transform(
        lambda s: s.shift(5).rolling(15, min_periods=15).median()
    )
    out["amount_ratio_5d"] = out["amount_recent_5d"] / out["amount_prior_15d"].replace(0.0, np.nan)
    prior_amount_median = grouped["amount"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).median()
    )
    up_amount = (
        out["return_1d"].gt(0.0)
        & out["amount"].gt(prior_amount_median)
        & prior_amount_median.notna()
    ).astype(float)
    out["_up_amount"] = up_amount
    out["up_amount_persistence_10d"] = out.groupby("symbol", sort=False)["_up_amount"].transform(
        lambda s: s.rolling(10, min_periods=10).mean()
    )
    amount_max_20 = _rolling_transform(grouped, "amount", 20, "max")
    amount_median_20 = _rolling_transform(grouped, "amount", 20, "median")
    out["single_day_amount_spike_20d"] = amount_max_20 / amount_median_20.replace(0.0, np.nan)

    out["flow_observation_count_5d"] = _rolling_transform(grouped, "main_net_volume_ratio", 5, "count")
    out["main_net_volume_ratio_5d"] = _rolling_transform(grouped, "main_net_volume_ratio", 5, "mean")
    positive_volume = out["main_net_volume_ratio"].gt(0.0).where(out["main_net_volume_ratio"].notna())
    out["_positive_volume"] = positive_volume.astype(float)
    out["main_net_volume_positive_ratio_5d"] = out.groupby("symbol", sort=False)["_positive_volume"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )
    out["main_net_volume_rank_5d"] = out.groupby("date", sort=False)["main_net_volume_ratio_5d"].rank(
        method="average", pct=True
    )

    out["main_net_inflow_amount_ratio"] = out["main_net_inflow"] / out["amount"].replace(0.0, np.nan)
    inflow_grouped = out.groupby("symbol", sort=False)
    out["inflow_observation_count_5d"] = _rolling_transform(
        inflow_grouped, "main_net_inflow_amount_ratio", 5, "count"
    )
    out["main_net_inflow_amount_ratio_5d"] = _rolling_transform(
        inflow_grouped, "main_net_inflow_amount_ratio", 5, "mean"
    )
    positive_inflow = out["main_net_inflow"].gt(0.0).where(out["main_net_inflow"].notna())
    out["_positive_inflow"] = positive_inflow.astype(float)
    out["main_net_inflow_positive_ratio_5d"] = out.groupby("symbol", sort=False)["_positive_inflow"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()
    )

    out["price_setup"] = (
        out["price_history"].ge(config.min_price_history)
        & out["return_1d"].le(config.max_daily_return)
        & out["return_20d"].between(config.min_return_20d, config.max_return_20d, inclusive="both")
        & out["drawdown_20d"].ge(config.min_drawdown_20d)
        & out["distance_ma20"].between(config.min_distance_ma20, config.max_distance_ma20, inclusive="both")
        & out["range_15d"].le(config.max_range_15d)
        & out["ma20_ma60_ratio"].ge(config.min_ma20_ma60_ratio)
    )
    out["volume_setup"] = (
        out["amount_ratio_5d"].between(config.min_amount_ratio_5d, config.max_amount_ratio_5d, inclusive="both")
        & out["up_amount_persistence_10d"].ge(config.min_up_amount_persistence_10d)
        & out["single_day_amount_spike_20d"].le(config.max_single_day_amount_spike_20d)
    )
    out["net_volume_available"] = out["flow_observation_count_5d"].ge(config.min_flow_history)
    out["net_inflow_available"] = out["inflow_observation_count_5d"].ge(config.min_flow_history)
    out["flow_evidence_available"] = out["net_volume_available"] | out["net_inflow_available"]
    out["net_volume_confirmed"] = (
        out["net_volume_available"]
        & out["main_net_volume_ratio_5d"].gt(0.0)
        & out["main_net_volume_positive_ratio_5d"].ge(config.min_flow_positive_ratio_5d)
        & out["main_net_volume_rank_5d"].ge(config.min_flow_rank_5d)
    )
    out["net_inflow_confirmed"] = (
        out["net_inflow_available"]
        & out["main_net_inflow_amount_ratio_5d"].gt(0.0)
        & out["main_net_inflow_positive_ratio_5d"].ge(config.min_flow_positive_ratio_5d)
    )
    out["flow_confirmed"] = out["net_volume_confirmed"] | out["net_inflow_confirmed"]

    def scaled(series: pd.Series, low: float, high: float) -> pd.Series:
        return ((series - low) / (high - low)).clip(0.0, 1.0).fillna(0.0)

    score = pd.Series(0.0, index=out.index)
    close_ma20_ratio = out["close"] / out["ma20"].replace(0.0, np.nan)
    score += scaled(close_ma20_ratio, 0.97, 1.05) * 8.0
    score += scaled(out["ma20_ma60_ratio"], config.min_ma20_ma60_ratio, 1.05) * 8.0
    return_midpoint = (config.min_return_20d + config.max_return_20d) / 2.0
    return_half_range = (config.max_return_20d - config.min_return_20d) / 2.0
    score += (1.0 - (out["return_20d"] - return_midpoint).abs() / return_half_range).clip(0.0, 1.0).fillna(0.0) * 8.0
    score += scaled(out["drawdown_20d"], config.min_drawdown_20d, 0.0) * 6.0
    score += (1.0 - out["range_15d"] / config.max_range_15d).clip(0.0, 1.0).fillna(0.0) * 5.0
    amount_midpoint = (config.min_amount_ratio_5d + config.max_amount_ratio_5d) / 2.0
    amount_half_range = (config.max_amount_ratio_5d - config.min_amount_ratio_5d) / 2.0
    score += (1.0 - (out["amount_ratio_5d"] - amount_midpoint).abs() / amount_half_range).clip(0.0, 1.0).fillna(0.0) * 12.0
    score += scaled(out["up_amount_persistence_10d"], config.min_up_amount_persistence_10d, 1.0) * 10.0
    spike_quality = (
        1.0
        - scaled(
            out["single_day_amount_spike_20d"],
            1.0,
            config.max_single_day_amount_spike_20d,
        )
    ).where(out["single_day_amount_spike_20d"].notna(), 0.0)
    score += spike_quality * 8.0
    score += (out["net_volume_available"] & out["main_net_volume_ratio_5d"].gt(0.0)).astype(float) * 8.0
    score += (out["net_volume_available"] & out["main_net_volume_positive_ratio_5d"].ge(config.min_flow_positive_ratio_5d)).astype(float) * 7.0
    score += (out["net_volume_available"] & out["main_net_volume_rank_5d"].ge(config.min_flow_rank_5d)).astype(float) * 5.0
    score += (out["net_inflow_available"] & out["main_net_inflow_amount_ratio_5d"].gt(0.0)).astype(float) * 8.0
    score += (out["net_inflow_available"] & out["main_net_inflow_positive_ratio_5d"].ge(config.min_flow_positive_ratio_5d)).astype(float) * 7.0
    out["institutional_accumulation_score"] = score.clip(0.0, 100.0)
    out["tracking_eligible"] = out["price_setup"] & out["volume_setup"]
    out["signal_active"] = (
        out["tracking_eligible"]
        & out["flow_evidence_available"]
        & out["flow_confirmed"]
        & out["institutional_accumulation_score"].ge(config.watch_score)
    )
    out["institutional_accumulation_level"] = np.select(
        [
            out["signal_active"]
            & out["net_volume_confirmed"]
            & out["net_inflow_confirmed"]
            & out["institutional_accumulation_score"].ge(config.strong_score),
            out["signal_active"],
            out["tracking_eligible"] & ~out["flow_evidence_available"],
            out["tracking_eligible"],
        ],
        ["强建仓迹象", "建仓观察", "资金流预热观察", "价量观察待确认"],
        default="不纳入",
    )
    return out.drop(columns=["_up_amount", "_positive_volume", "_positive_inflow"])


def _supported_stock(symbol: str) -> bool:
    return str(symbol).startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605"))


def _is_st_name(value: object) -> bool:
    text = str(value or "").strip().upper()
    return text.startswith(("ST", "*ST", "SST", "S*ST"))


def _reason(row: pd.Series) -> str:
    evidence = ["价格结构未过热", "成交额连续放大"]
    if bool(row.get("net_volume_confirmed", False)):
        evidence.append("主力净量连续为正")
    if bool(row.get("net_inflow_confirmed", False)):
        evidence.append("大单净额占成交额为正")
    if not bool(row.get("flow_evidence_available", False)):
        evidence.append("资金流历史不足，暂不确认机构属性")
    elif not bool(row.get("flow_confirmed", False)):
        evidence.append("资金流未确认")
    return "；".join(evidence)


def build_latest_watchlist(
    features: pd.DataFrame,
    *,
    asof_date: str | pd.Timestamp,
    name_map: dict[str, str] | None = None,
    top_n: int = 80,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    asof = pd.Timestamp(asof_date)
    eligible_dates = features.loc[features["date"].le(asof), "date"]
    if eligible_dates.empty:
        return pd.DataFrame()
    source_date = eligible_dates.max()
    latest = features[features["date"].eq(source_date)].copy()
    latest = latest[latest["symbol"].map(_supported_stock)]
    latest["stock_name"] = latest["symbol"].map(name_map or {}).fillna("")
    st_mask = latest["stock_name"].map(_is_st_name)
    latest = latest[latest["tracking_eligible"] & ~st_mask].copy()
    latest["accumulation_reason"] = latest.apply(_reason, axis=1)
    latest["research_only"] = True
    latest["trade_instruction"] = False
    latest["selection_effect"] = False
    latest = latest.sort_values(
        ["signal_active", "institutional_accumulation_score", "symbol"],
        ascending=[False, False, True],
    ).head(int(top_n))
    preferred = [
        "date", "symbol", "stock_name", "institutional_accumulation_level",
        "institutional_accumulation_score", "accumulation_reason", "close",
        "return_1d", "return_5d", "return_20d", "drawdown_20d", "distance_ma20",
        "ma20_ma60_ratio", "range_15d", "amount_ratio_5d",
        "up_amount_persistence_10d", "single_day_amount_spike_20d",
        "main_net_volume_ratio_5d", "main_net_volume_positive_ratio_5d",
        "main_net_volume_rank_5d", "main_net_inflow_amount_ratio_5d",
        "main_net_inflow_positive_ratio_5d", "price_setup", "volume_setup",
        "flow_evidence_available", "net_volume_confirmed", "net_inflow_confirmed",
        "flow_confirmed", "tracking_eligible", "signal_active", "research_only",
        "trade_instruction", "selection_effect",
    ]
    return latest[[column for column in preferred if column in latest]].reset_index(drop=True)


def build_metadata(
    features: pd.DataFrame,
    watchlist: pd.DataFrame,
    *,
    requested_asof_date: str,
    availability: dict[str, bool],
    config: AccumulationConfig,
) -> dict[str, object]:
    source_date = features["date"].max() if not features.empty else pd.NaT
    flow_dates = (
        features.loc[
            features[[column for column in OPTIONAL_FLOW_COLUMNS if column in features]].notna().any(axis=1),
            "date",
        ].dropna().drop_duplicates()
        if not features.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    flow_sessions = int(len(flow_dates))
    source_column_latest_dates: dict[str, str | None] = {}
    source_columns_historical_available: dict[str, bool] = {}
    for column in OPTIONAL_FLOW_COLUMNS:
        usable = features[column].notna() if column in features else pd.Series(False, index=features.index)
        latest_usable_date = features.loc[usable, "date"].max() if not features.empty else pd.NaT
        source_columns_historical_available[column] = bool(usable.any())
        source_column_latest_dates[column] = (
            latest_usable_date.strftime("%Y-%m-%d")
            if not pd.isna(latest_usable_date)
            else None
        )
    requested = pd.Timestamp(requested_asof_date)
    if pd.isna(source_date) or source_date < requested:
        status = "stale"
    elif not any(availability.values()):
        status = "current_flow_unavailable"
    elif flow_sessions < config.min_flow_history:
        status = "warmup"
    else:
        status = "research_ready"
    active_rows = int(watchlist.get("signal_active", pd.Series(dtype=bool)).fillna(False).sum())
    return {
        "schema_version": 3,
        "status": status,
        "requested_asof_date": requested.strftime("%Y-%m-%d"),
        "price_usable_latest_date": source_date.strftime("%Y-%m-%d") if not pd.isna(source_date) else None,
        "flow_source_first_date": flow_dates.min().strftime("%Y-%m-%d") if not flow_dates.empty else None,
        "flow_source_latest_date": flow_dates.max().strftime("%Y-%m-%d") if not flow_dates.empty else None,
        "available_flow_sessions": flow_sessions,
        "minimum_flow_history": config.min_flow_history,
        "source_columns_available": availability,
        "source_columns_historical_available": source_columns_historical_available,
        "source_column_latest_dates": source_column_latest_dates,
        "watchlist_rows": int(len(watchlist)),
        "active_signal_rows": active_rows,
        "registration_id": config.registration_id,
        "registration_date": config.registration_date,
        "validation_start_date": config.validation_start_date,
        "config_sha256": config_hash(config),
        "implementation_sha256": implementation_hash(),
        "provisional_only": True,
        "gate_evaluation_allowed": False,
        "promotion_allowed": False,
        "automatic_promotion": False,
        "research_only": True,
        "trade_instruction": False,
        "selection_effect": False,
        "portfolio_weight_effect": False,
        "limitations": [
            "公开行情不能识别真实账户身份，本信号只表示疑似大资金吸收行为。",
            "同花顺资金流字段属于供应商估算口径，不等同于交易所披露的机构席位。",
            "完整信号要求至少5个连续可用交易日，缺失值不会填零。",
        ],
    }


def _atomic_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(text, encoding=encoding)
    pending.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    frame.to_csv(pending, index=False, encoding="utf-8-sig")
    pending.replace(path)


def write_outputs(
    watchlist: pd.DataFrame,
    metadata: dict[str, object],
    output: Path,
) -> dict[str, Path]:
    output = Path(output)
    chinese = output.with_name(f"{output.stem}_cn{output.suffix}")
    meta_path = output.with_suffix(".json")
    report_path = output.with_suffix(".md")
    _atomic_csv(output, watchlist)
    _atomic_csv(chinese, watchlist.rename(columns=CN_COLUMNS))
    _atomic_text(meta_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    status_text = {
        "warmup": "资金流预热中",
        "research_ready": "影子研究可用",
        "stale": "数据日期落后",
    }.get(str(metadata.get("status")), str(metadata.get("status")))
    lines = [
        "# 疑似机构提前建仓影子观察",
        "",
        "本报告组合价量与同花顺资金流估算，仅表示疑似大资金吸收行为，不能识别真实机构账户，不是买入指令。",
        "",
        f"- 状态: {status_text}",
        f"- 信号日期: {metadata.get('price_usable_latest_date')}",
        f"- 资金流可用交易日: {metadata.get('available_flow_sessions')} / {metadata.get('minimum_flow_history')}",
        f"- 影子跟踪候选: {metadata.get('watchlist_rows')}",
        f"- 完整资金确认信号: {metadata.get('active_signal_rows')}",
        f"- 预注册编号: {metadata.get('registration_id')}",
        "- 主模型影响: 无",
    ]
    if not watchlist.empty:
        show = watchlist.head(40).copy()
        for column in [
            "return_1d", "return_5d", "return_20d", "drawdown_20d", "distance_ma20",
            "main_net_volume_ratio_5d", "main_net_inflow_amount_ratio_5d",
        ]:
            if column in show:
                show[column] = show[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2%}")
        columns = [
            "symbol", "stock_name", "institutional_accumulation_level",
            "institutional_accumulation_score", "return_20d", "amount_ratio_5d",
            "main_net_volume_ratio_5d", "signal_active", "accumulation_reason",
        ]
        lines.extend(["", show[[column for column in columns if column in show]].rename(columns=CN_COLUMNS).to_markdown(index=False)])
    _atomic_text(report_path, "\n".join(lines) + "\n")
    return {"csv": output, "chinese_csv": chinese, "metadata": meta_path, "report": report_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only institutional accumulation proxy.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--config", default="configs/institutional_accumulation_shadow.yaml")
    parser.add_argument("--names-source")
    parser.add_argument("--top-n", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config) if args.config else None)
    panel, availability = load_panel(Path(args.data), asof_date=args.asof_date)
    features = compute_accumulation_features(panel, config)
    names = load_name_map(Path(args.names_source)) if args.names_source else {}
    watchlist = build_latest_watchlist(
        features,
        asof_date=args.asof_date,
        name_map=names,
        top_n=args.top_n or config.top_n,
    )
    metadata = build_metadata(
        features,
        watchlist,
        requested_asof_date=args.asof_date,
        availability=availability,
        config=config,
    )
    if metadata["status"] != "research_ready" and not watchlist.empty:
        watchlist["signal_active"] = False
        watchlist["institutional_accumulation_level"] = np.where(
            watchlist["tracking_eligible"], "资金流预热观察", watchlist["institutional_accumulation_level"]
        )
        metadata["active_signal_rows"] = 0
    paths = write_outputs(watchlist, metadata, Path(args.output))
    print(f"Institutional accumulation status: {metadata['status']}")
    print(f"Watch rows: {len(watchlist)}; active signals: {metadata['active_signal_rows']}")
    print(f"Report saved to: {paths['report']}")


if __name__ == "__main__":
    main()
