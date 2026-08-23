"""Build a research-only daily Chan-structure snapshot with CZSC.

The adapter intentionally validates one CZSC API line.  Structural endpoints
such as a fractal date or a BI end date are descriptive only: every row is
timestamped at the requested as-of close and is actionable no earlier than the
next trading-session open.  The output never changes candidate selection or
portfolio weights.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from panel_io import iter_panel, panel_columns

from build_daily_personal_overlay_report import load_name_map
from execution_rules import normalize_symbol


VALIDATED_CZSC_VERSION = "0.10.12"
REQUIRED_SIGNAL_FUNCTIONS = (
    "cxt_second_bs_V240524",
    "cxt_overlap_V240612",
    "cxt_bi_trend_V230824",
    "cxt_bi_status_V230102",
    "cxt_fx_power_V221107",
    "cxt_third_buy_V230228",
)

CN_COLUMNS = {
    "date": "分析日期",
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "candidate_sources": "候选来源",
    "model_rank": "模型排名",
    "model_selected": "模型已选",
    "trend_state": "原趋势状态",
    "early_pattern_type": "提前观察模式",
    "early_pattern_score": "提前观察评分",
    "analysis_status": "结构分析状态",
    "analysis_error": "结构分析说明",
    "history_bars": "有效历史根数",
    "history_start_date": "有效历史开始日",
    "history_end_date": "有效历史结束日",
    "history_truncated_at": "异常跳变后重置日",
    "completed_bi_count": "已完成笔数量",
    "last_bi_direction": "最后完成笔方向",
    "last_bi_start_date": "最后完成笔开始日",
    "last_bi_end_date": "最后完成笔结束日",
    "chan_last_bi_length": "最后完成笔长度",
    "chan_last_bi_power_price": "最后完成笔价格力度",
    "chan_last_bi_power_volume": "最后完成笔成交量力度",
    "chan_last_bi_snr": "最后完成笔信噪比",
    "chan_last_bi_angle": "最后完成笔斜率",
    "diagnostic_unfinished_direction": "未完成结构方向",
    "diagnostic_unfinished_bar_count": "未完成结构K线数",
    "last_confirmed_fractal_date": "最后确认分型日期",
    "last_confirmed_fractal_mark": "最后确认分型类型",
    "last_confirmed_fractal_value": "最后确认分型价格",
    "zone_valid": "三笔中枢有效",
    "zone_start_date": "三笔中枢开始日",
    "zone_end_date": "三笔中枢结束日",
    "zone_zg": "中枢上沿",
    "zone_zz": "中枢中轴",
    "zone_zd": "中枢下沿",
    "zone_location": "收盘相对中枢位置",
    "zone_position": "中枢位置比例",
    "chan_last_zs_dist_to_zg": "距中枢上沿",
    "chan_last_zs_dist_to_zz": "距中枢中轴",
    "chan_last_zs_dist_to_zd": "距中枢下沿",
    "chan_second_bs_overlap_count": "二买卖重叠次数",
    "chan_trend_channel_pos": "近期笔通道位置",
    "shadow_second_bs": "二买卖影子标签",
    "shadow_overlap_support": "顺畅笔支撑压力标签",
    "shadow_bi_trend": "笔趋势影子标签",
    "shadow_bi_status_confirmed": "确认分型表里标签",
    "shadow_fx_power": "分型强弱标签",
    "shadow_like_third_buy": "类三买影子标签",
    "third_buy_zone_consistent": "类三买与中枢一致",
    "pattern_confluence": "提前形态结构共振",
    "chan_setup": "结构观察结论",
    "risk_flags": "结构风险提示",
    "signal_known_at": "信号已知时点",
    "earliest_action": "最早可执行时点",
    "endpoint_backfill": "是否回填拐点",
    "research_only": "仅供研究",
    "selection_effect": "影响选股",
    "portfolio_weight_effect": "影响仓位",
}


@dataclass(frozen=True)
class CzscRuntime:
    available: bool
    version: str | None
    reason: str | None
    module: ModuleType | None = None
    signals: ModuleType | None = None


def load_czsc_runtime(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
    *,
    validated_version: str = VALIDATED_CZSC_VERSION,
) -> CzscRuntime:
    """Load CZSC lazily and reject unvalidated, API-incompatible versions."""

    try:
        module = import_module("czsc")
        signals = import_module("czsc.signals")
    except (ImportError, ModuleNotFoundError) as exc:
        return CzscRuntime(False, None, f"czsc_not_installed: {exc}")

    version = str(getattr(module, "__version__", "unknown"))
    if version != validated_version:
        return CzscRuntime(
            False,
            version,
            f"unsupported_czsc_version: expected {validated_version}, got {version}",
        )
    missing = [name for name in REQUIRED_SIGNAL_FUNCTIONS if not hasattr(signals, name)]
    required_types = [name for name in ("CZSC", "RawBar", "Freq") if not hasattr(module, name)]
    if missing or required_types:
        details = ", ".join(missing + required_types)
        return CzscRuntime(False, version, f"czsc_api_incomplete: {details}")
    return CzscRuntime(True, version, None, module=module, signals=signals)


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "已选"}


def _first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
            return str(value).strip()
    return ""


def _numeric_min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.min()) if numeric.notna().any() else np.nan


def _numeric_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else np.nan


def load_candidate_union(paths: Iterable[Path]) -> pd.DataFrame:
    """Merge model and early-watch candidates without changing their decisions."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
        if "symbol" not in frame:
            raise ValueError(f"Candidate file is missing symbol: {path}")
        frame = frame.copy()
        frame["symbol"] = frame["symbol"].map(normalize_symbol)
        frame = frame[frame["symbol"].ne("")]
        frame["_candidate_source"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "stock_name",
                "candidate_sources",
                "model_rank",
                "model_selected",
                "trend_state",
                "early_pattern_type",
                "early_pattern_score",
            ]
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for symbol, group in combined.groupby("symbol", sort=True):
        name_values = group["stock_name"] if "stock_name" in group else pd.Series(dtype=object)
        pattern_values = group["pattern_type"] if "pattern_type" in group else pd.Series(dtype=object)
        if "early_pattern_type" in group:
            pattern_values = pd.concat([pattern_values, group["early_pattern_type"]])
        ranks = []
        for column in ("personal_rank", "rank", "model_rank"):
            if column in group:
                ranks.append(group[column])
        rank_values = pd.concat(ranks, ignore_index=True) if ranks else pd.Series(dtype=float)
        selected_values: list[object] = []
        for column in ("personal_selected", "selected", "model_selected"):
            if column in group:
                selected_values.extend(group[column].tolist())
        trend_values = group["trend_state"] if "trend_state" in group else pd.Series(dtype=object)
        score_values = []
        for column in ("pattern_score", "early_pattern_score"):
            if column in group:
                score_values.append(group[column])
        scores = pd.concat(score_values, ignore_index=True) if score_values else pd.Series(dtype=float)
        rows.append(
            {
                "symbol": symbol,
                "stock_name": _first_nonempty(name_values),
                "candidate_sources": ";".join(sorted(set(group["_candidate_source"].astype(str)))),
                "model_rank": _numeric_min(rank_values),
                "model_selected": any(_truthy(value) for value in selected_values),
                "trend_state": _first_nonempty(trend_values),
                "early_pattern_type": _first_nonempty(pattern_values),
                "early_pattern_score": _numeric_max(scores),
            }
        )
    return pd.DataFrame(rows)


def load_panel_prices(
    path: Path,
    symbols: Iterable[str],
    *,
    asof_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Read candidate histories only, excluding all rows after the as-of close."""

    header = panel_columns(path)
    required = {"date", "symbol", "open", "high", "low", "close"}
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"Panel is missing required columns: {', '.join(missing)}")
    columns = [column for column in [*required, "volume", "amount"] if column in header]
    normalized_symbols = {normalize_symbol(symbol) for symbol in symbols}
    normalized_symbols.discard("")
    asof = pd.Timestamp(asof_date).normalize()
    chunks: list[pd.DataFrame] = []
    for chunk in iter_panel(
        path,
        columns=columns,
        dtype={"symbol": str},
        chunksize=250_000,
    ):
        chunk["symbol"] = chunk["symbol"].map(normalize_symbol)
        chunk = chunk.loc[chunk["symbol"].isin(normalized_symbols)].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
        chunk = chunk[chunk["date"].notna() & chunk["date"].le(asof)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=columns)
    out = pd.concat(chunks, ignore_index=True)
    return (
        out.drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return "" if raw is None else str(raw)


def _date_text(value: object) -> str | None:
    date = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(date) else date.strftime("%Y-%m-%d")


def _safe_float(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _truncate_after_price_jump(
    history: pd.DataFrame,
    max_abs_daily_return: float,
) -> tuple[pd.DataFrame, str | None]:
    close = pd.to_numeric(history["close"], errors="coerce")
    jumps = close.pct_change(fill_method=None).abs().gt(float(max_abs_daily_return))
    positions = np.flatnonzero(jumps.to_numpy())
    if not len(positions):
        return history, None
    start = int(positions[-1])
    return history.iloc[start:].copy(), _date_text(history.iloc[start]["date"])


def _signal_value(runtime: CzscRuntime, name: str, czsc_obj: object) -> tuple[str, str]:
    try:
        result = getattr(runtime.signals, name)(czsc_obj)
        if not result:
            return "", ""
        key, value = next(iter(result.items()))
        return str(value), f"{key}_{value}"
    except Exception as exc:  # A single optional signal must not erase the structure snapshot.
        return "", f"signal_error:{name}:{type(exc).__name__}:{exc}"


def _parts(signal_value: str) -> tuple[str, str, str]:
    values = str(signal_value).split("_") if signal_value else []
    values += [""] * (3 - len(values))
    return values[0], values[1], values[2]


def _latest_three_bi_zone(bis: list[object]) -> dict[str, object]:
    """Return the latest valid three-finished-BI overlap known at the snapshot."""

    for start in range(len(bis) - 3, -1, -1):
        window = bis[start : start + 3]
        highs = [_safe_float(getattr(bi, "high", np.nan)) for bi in window]
        lows = [_safe_float(getattr(bi, "low", np.nan)) for bi in window]
        if any(pd.isna(value) for value in highs + lows):
            continue
        zg = min(highs)
        zd = max(lows)
        if zg <= zd:
            continue
        return {
            "zone_valid": True,
            "zone_start_date": _date_text(getattr(window[0], "sdt", None)),
            "zone_end_date": _date_text(getattr(window[-1], "edt", None)),
            "zone_zg": zg,
            "zone_zd": zd,
            "zone_zz": (zg + zd) / 2.0,
            "zone_gg": max(highs),
            "zone_dd": min(lows),
            "zone_bi_start_index": start,
        }
    return {
        "zone_valid": False,
        "zone_start_date": None,
        "zone_end_date": None,
        "zone_zg": np.nan,
        "zone_zd": np.nan,
        "zone_zz": np.nan,
        "zone_gg": np.nan,
        "zone_dd": np.nan,
        "zone_bi_start_index": np.nan,
    }


def _second_bs_overlap_count(bis: list[object], window: int = 9) -> int:
    if len(bis) < window:
        return 0
    selected = bis[-window:]
    last = selected[-1]
    last_fx = getattr(last, "fx_b", None)
    if last_fx is None:
        return 0
    last_low = _safe_float(getattr(last_fx, "low", np.nan))
    last_high = _safe_float(getattr(last_fx, "high", np.nan))
    count = 0
    for bi in selected[:-1]:
        length = _safe_float(getattr(bi, "length", 0))
        if pd.isna(length) or int(length) < 7:
            continue
        fx = getattr(bi, "fx_b", None)
        low = _safe_float(getattr(fx, "low", np.nan))
        high = _safe_float(getattr(fx, "high", np.nan))
        if pd.notna(low) and pd.notna(high) and max(low, last_low) < min(high, last_high):
            count += 1
    return count


def _trend_channel_position(bis: list[object], close: float, window: int = 4) -> float:
    selected = bis[-window:]
    if not selected:
        return np.nan
    high = max(_safe_float(getattr(bi, "high", np.nan)) for bi in selected)
    low = min(_safe_float(getattr(bi, "low", np.nan)) for bi in selected)
    if pd.isna(high) or pd.isna(low) or high <= low:
        return np.nan
    return (close - low) / (high - low)


def _base_result(
    symbol: str,
    asof_date: str,
    *,
    status: str,
    error: str = "",
) -> dict[str, object]:
    return {
        "date": pd.Timestamp(asof_date).strftime("%Y-%m-%d"),
        "symbol": symbol,
        "analysis_status": status,
        "analysis_error": error,
        "signal_known_at": "asof_close",
        "earliest_action": "next_trading_session_open",
        "endpoint_backfill": False,
        "multi_frequency": False,
        "research_only": True,
        "production_feature_eligible": False,
        "selection_effect": False,
        "portfolio_weight_effect": False,
    }


def analyze_symbol_history(
    history: pd.DataFrame,
    runtime: CzscRuntime,
    *,
    symbol: str,
    asof_date: str,
    min_bars: int = 100,
    max_abs_daily_return: float = 0.22,
) -> dict[str, object]:
    if not runtime.available or runtime.module is None or runtime.signals is None:
        return _base_result(
            symbol,
            asof_date,
            status="runtime_unavailable",
            error=runtime.reason or "CZSC runtime unavailable",
        )

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["date"].le(pd.Timestamp(asof_date).normalize())]
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame[
        frame[["open", "high", "low", "close"]].gt(0.0).all(axis=1)
        & frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
        & frame["low"].le(frame[["open", "close", "high"]].min(axis=1))
    ]
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    frame, truncated_at = _truncate_after_price_jump(frame, max_abs_daily_return)
    if len(frame) < int(min_bars):
        result = _base_result(
            symbol,
            asof_date,
            status="insufficient_history",
            error=f"requires {min_bars} valid bars after price-jump reset, got {len(frame)}",
        )
        result.update(
            {
                "history_bars": int(len(frame)),
                "history_start_date": _date_text(frame["date"].min()) if not frame.empty else None,
                "history_end_date": _date_text(frame["date"].max()) if not frame.empty else None,
                "history_truncated_at": truncated_at,
            }
        )
        return result

    volume_coverage = float(frame["volume"].notna().mean())
    amount_coverage = float(frame["amount"].notna().mean())
    bars = []
    try:
        for index, row in enumerate(frame.itertuples(index=False)):
            bars.append(
                runtime.module.RawBar(
                    symbol=symbol,
                    dt=pd.Timestamp(row.date).to_pydatetime(),
                    freq=runtime.module.Freq.D,
                    open=float(row.open),
                    close=float(row.close),
                    high=float(row.high),
                    low=float(row.low),
                    vol=0.0 if pd.isna(row.volume) else float(row.volume),
                    amount=0.0 if pd.isna(row.amount) else float(row.amount),
                    id=index,
                )
            )
        czsc_obj = runtime.module.CZSC(bars, max_bi_num=100)
    except Exception as exc:
        return _base_result(
            symbol,
            asof_date,
            status="analysis_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    bis = list(getattr(czsc_obj, "bi_list", []) or [])
    fxs = list(getattr(czsc_obj, "fx_list", []) or [])
    last_bi = bis[-1] if bis else None
    last_fx = fxs[-1] if fxs else None
    close = float(frame.iloc[-1]["close"])
    zone = _latest_three_bi_zone(bis)
    zg, zz, zd = zone["zone_zg"], zone["zone_zz"], zone["zone_zd"]
    if zone["zone_valid"]:
        zone_location = "中枢上方" if close > zg else "中枢下方" if close < zd else "中枢内部"
        zone_position = (close - zd) / (zg - zd)
    else:
        zone_location = "无有效三笔中枢"
        zone_position = np.nan

    signal_specs = {
        "second": "cxt_second_bs_V240524",
        "overlap": "cxt_overlap_V240612",
        "trend": "cxt_bi_trend_V230824",
        "status": "cxt_bi_status_V230102",
        "fx_power": "cxt_fx_power_V221107",
        "third_buy": "cxt_third_buy_V230228",
    }
    signals = {
        key: _signal_value(runtime, name, czsc_obj) for key, name in signal_specs.items()
    }
    second_v1, second_v2, _ = _parts(signals["second"][0])
    overlap_v1, overlap_v2, _ = _parts(signals["overlap"][0])
    trend_v1, trend_v2, _ = _parts(signals["trend"][0])
    status_v1, status_v2, _ = _parts(signals["status"][0])
    fx_v1, fx_v2, _ = _parts(signals["fx_power"][0])
    third_v1, third_v2, _ = _parts(signals["third_buy"][0])
    third_buy_flag = third_v1 in {"三买", "类三买"}
    third_buy_zone_consistent = bool(third_buy_flag and zone_location == "中枢上方")
    support_flag = overlap_v1 == "支撑"

    setups: list[str] = []
    if second_v1 == "二买":
        setups.append("二买")
    if third_buy_zone_consistent:
        setups.append("类三买且位于中枢上方")
    elif third_buy_flag:
        setups.append("类三买原始标签待中枢复核")
    if support_flag:
        setups.append("顺畅笔支撑")
    if zone_location == "中枢上方" and _enum_value(getattr(last_bi, "direction", "")) == "向下":
        setups.append("中枢上方回调")
    risks: list[str] = []
    if second_v1 == "二卖":
        risks.append("二卖")
    if overlap_v1 == "压力":
        risks.append("顺畅笔压力")
    if zone_location == "中枢下方":
        risks.append("跌破三笔中枢")
    if trend_v1 in {"向下", "下降趋势"}:
        risks.append("笔趋势向下")

    unfinished = getattr(czsc_obj, "ubi", {}) or {}
    result = _base_result(symbol, asof_date, status="analyzed")
    result.update(
        {
            "close": close,
            "history_bars": int(len(frame)),
            "history_start_date": _date_text(frame["date"].min()),
            "history_end_date": _date_text(frame["date"].max()),
            "history_truncated_at": truncated_at,
            "volume_coverage": volume_coverage,
            "amount_coverage": amount_coverage,
            "completed_bi_count": int(len(bis)),
            "last_bi_direction": _enum_value(getattr(last_bi, "direction", "")),
            "last_bi_start_date": _date_text(getattr(last_bi, "sdt", None)),
            "last_bi_end_date": _date_text(getattr(last_bi, "edt", None)),
            "last_bi_high": _safe_float(getattr(last_bi, "high", np.nan)),
            "last_bi_low": _safe_float(getattr(last_bi, "low", np.nan)),
            "chan_last_bi_length": _safe_float(getattr(last_bi, "length", np.nan)),
            "chan_last_bi_power_price": _safe_float(
                getattr(last_bi, "power_price", getattr(last_bi, "power", np.nan))
            ),
            "chan_last_bi_power_volume": _safe_float(getattr(last_bi, "power_volume", np.nan)),
            "chan_last_bi_snr": _safe_float(getattr(last_bi, "power_snr", np.nan)),
            "chan_last_bi_angle": _safe_float(getattr(last_bi, "slope", np.nan)),
            "diagnostic_unfinished_direction": _enum_value(unfinished.get("direction", "")),
            "diagnostic_unfinished_bar_count": int(len(unfinished.get("raw_bars", []) or [])),
            "last_confirmed_fractal_date": _date_text(getattr(last_fx, "dt", None)),
            "last_confirmed_fractal_mark": _enum_value(getattr(last_fx, "mark", "")),
            "last_confirmed_fractal_value": _safe_float(getattr(last_fx, "fx", np.nan)),
            **zone,
            "zone_location": zone_location,
            "zone_position": zone_position,
            "chan_last_zs_dist_to_zg": (close - zg) / close if pd.notna(zg) else np.nan,
            "chan_last_zs_dist_to_zz": (close - zz) / close if pd.notna(zz) else np.nan,
            "chan_last_zs_dist_to_zd": (close - zd) / close if pd.notna(zd) else np.nan,
            "chan_second_bs_overlap_count": _second_bs_overlap_count(bis),
            "chan_trend_channel_pos": _trend_channel_position(bis, close),
            "shadow_second_bs": "_".join(value for value in (second_v1, second_v2) if value),
            "shadow_overlap_support": "_".join(value for value in (overlap_v1, overlap_v2) if value),
            "shadow_bi_trend": "_".join(value for value in (trend_v1, trend_v2) if value),
            "shadow_bi_status_confirmed": "_".join(value for value in (status_v1, status_v2) if value),
            "shadow_fx_power": "_".join(value for value in (fx_v1, fx_v2) if value),
            "shadow_like_third_buy": "_".join(value for value in (third_v1, third_v2) if value),
            "second_buy_flag": second_v1 == "二买",
            "second_sell_flag": second_v1 == "二卖",
            "overlap_support_flag": support_flag,
            "overlap_pressure_flag": overlap_v1 == "压力",
            "third_buy_flag": third_buy_flag,
            "third_buy_zone_consistent": third_buy_zone_consistent,
            "pattern_confluence": False,
            "chan_setup": ";".join(setups) if setups else "暂无结构触发",
            "risk_flags": ";".join(risks),
            "raw_signal_second_bs": signals["second"][1],
            "raw_signal_overlap": signals["overlap"][1],
            "raw_signal_bi_trend": signals["trend"][1],
            "raw_signal_bi_status": signals["status"][1],
            "raw_signal_fx_power": signals["fx_power"][1],
            "raw_signal_third_buy": signals["third_buy"][1],
            "uses_unfinished_context_for_some_shadow_labels": True,
        }
    )
    return result


def build_structure_shadow(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    runtime: CzscRuntime,
    *,
    asof_date: str,
    name_map: dict[str, str] | None = None,
    min_bars: int = 100,
    max_abs_daily_return: float = 0.22,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        symbol = normalize_symbol(candidate.get("symbol"))
        history = prices[prices["symbol"].eq(symbol)] if not prices.empty else prices
        result = analyze_symbol_history(
            history,
            runtime,
            symbol=symbol,
            asof_date=asof_date,
            min_bars=min_bars,
            max_abs_daily_return=max_abs_daily_return,
        )
        result.update(candidate)
        if not str(result.get("stock_name", "")).strip():
            result["stock_name"] = (name_map or {}).get(symbol, "")
        structural_trigger = any(
            bool(result.get(key))
            for key in ("second_buy_flag", "overlap_support_flag", "third_buy_zone_consistent")
        )
        result["pattern_confluence"] = bool(result.get("early_pattern_type") and structural_trigger)
        rows.append(result)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    for column, default in (
        ("pattern_confluence", False),
        ("third_buy_zone_consistent", False),
        ("second_buy_flag", False),
        ("overlap_support_flag", False),
        ("model_selected", False),
        ("model_rank", np.nan),
    ):
        if column not in table:
            table[column] = default
    table["_rank_sort"] = pd.to_numeric(table["model_rank"], errors="coerce").fillna(1e9)
    table = table.sort_values(
        [
            "pattern_confluence",
            "third_buy_zone_consistent",
            "second_buy_flag",
            "overlap_support_flag",
            "model_selected",
            "_rank_sort",
            "symbol",
        ],
        ascending=[False, False, False, False, False, True, True],
    )
    return table.drop(columns="_rank_sort").reset_index(drop=True)


def build_metadata(
    table: pd.DataFrame,
    runtime: CzscRuntime,
    *,
    asof_date: str,
    panel_path: Path,
    candidate_paths: Iterable[Path],
    min_bars: int,
    max_abs_daily_return: float,
) -> dict[str, object]:
    statuses = table.get("analysis_status", pd.Series(dtype=str)).value_counts().to_dict()
    analyzed = int(statuses.get("analyzed", 0))
    if not runtime.available:
        status = "unavailable"
    elif analyzed == 0:
        status = "no_eligible_history"
    elif analyzed < len(table):
        status = "partial"
    else:
        status = "research_ready"
    def count(column: str) -> int:
        if column not in table:
            return 0
        return sum(_truthy(value) for value in table[column])

    return {
        "schema_version": 1,
        "status": status,
        "asof_date": pd.Timestamp(asof_date).strftime("%Y-%m-%d"),
        "panel_path": str(panel_path),
        "candidate_paths": [str(Path(path)) for path in candidate_paths],
        "candidate_count": int(len(table)),
        "analysis_status_counts": {str(key): int(value) for key, value in statuses.items()},
        "analyzed_count": analyzed,
        "pattern_confluence_count": count("pattern_confluence"),
        "second_buy_count": count("second_buy_flag"),
        "second_sell_count": count("second_sell_flag"),
        "third_buy_raw_count": count("third_buy_flag"),
        "third_buy_zone_consistent_count": count("third_buy_zone_consistent"),
        "overlap_support_count": count("overlap_support_flag"),
        "overlap_pressure_count": count("overlap_pressure_flag"),
        "validated_czsc_version": VALIDATED_CZSC_VERSION,
        "runtime_czsc_version": runtime.version,
        "runtime_available": runtime.available,
        "runtime_reason": runtime.reason,
        "min_history_bars": int(min_bars),
        "max_abs_daily_return": float(max_abs_daily_return),
        "frequency": "日线",
        "multi_frequency": False,
        "signal_time_semantics": "仅使用截至asof收盘可见快照；最早下一交易日开盘使用",
        "historical_endpoint_backfill": False,
        "completed_bi_features_only": True,
        "revisable_shadow_labels_present": True,
        "unfinished_context_is_diagnostic_only": True,
        "research_only": True,
        "production_feature_eligible": False,
        "selection_effect": False,
        "portfolio_weight_effect": False,
    }


def _report_table(frame: pd.DataFrame, top_n: int) -> str:
    columns = [
        "symbol",
        "stock_name",
        "model_rank",
        "early_pattern_type",
        "last_bi_direction",
        "zone_location",
        "shadow_second_bs",
        "shadow_overlap_support",
        "shadow_like_third_buy",
        "shadow_bi_trend",
        "chan_setup",
        "risk_flags",
    ]
    selected = frame[[column for column in columns if column in frame]].head(top_n)
    return selected.rename(columns=CN_COLUMNS).to_markdown(index=False, disable_numparse=True)


def write_outputs(
    table: pd.DataFrame,
    metadata: dict[str, object],
    output: Path,
    *,
    top_n: int = 30,
) -> dict[str, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    cn_path = output.with_name(f"{output.stem}_cn{output.suffix}")
    table.rename(columns=CN_COLUMNS).to_csv(cn_path, index=False, encoding="utf-8-sig")
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        return frame[column].map(_truthy).astype(bool)

    def text_series(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series("", index=frame.index, dtype=str)
        return frame[column].fillna("").astype(str)

    analyzed = (
        table[table["analysis_status"].eq("analyzed")]
        if not table.empty and "analysis_status" in table
        else table.iloc[0:0]
    )
    sections = (
        ("提前形态与缠论结构共振", analyzed[bool_mask(analyzed, "pattern_confluence")]),
        ("类三买且位于中枢上方", analyzed[bool_mask(analyzed, "third_buy_zone_consistent")]),
        ("二买结构", analyzed[bool_mask(analyzed, "second_buy_flag")]),
        ("顺畅笔支撑", analyzed[bool_mask(analyzed, "overlap_support_flag")]),
        (
            "中枢上方回调观察",
            analyzed[
                text_series(analyzed, "zone_location").eq("中枢上方")
                & text_series(analyzed, "last_bi_direction").eq("向下")
            ],
        ),
        (
            "结构风险",
            analyzed[text_series(analyzed, "risk_flags").ne("")],
        ),
    )
    lines = [
        "# CZSC 日线缠论结构影子观察",
        "",
        "本报告只把分型、已完成笔、三笔中枢和二/三买等结构作为研究标签，不改变当前模型排名、选股和仓位。",
        "",
        f"- 状态: {metadata['status']}",
        f"- 分析日期: {metadata['asof_date']}",
        f"- CZSC 版本: {metadata.get('runtime_czsc_version') or '未安装'}（验证版本 {metadata['validated_czsc_version']}）",
        f"- 候选 / 完成分析: {metadata['candidate_count']} / {metadata['analyzed_count']}",
        f"- 提前形态共振: {metadata['pattern_confluence_count']}",
        f"- 二买 / 一致类三买 / 顺畅笔支撑: {metadata['second_buy_count']} / {metadata['third_buy_zone_consistent_count']} / {metadata['overlap_support_count']}",
        "- 时间约束: 当日收盘后形成快照，最早下一交易日开盘使用；严禁把后来确认的笔或分型回填到历史端点。",
        "- 风险提示: 未完成结构会继续演化；所有离散标签当前均为影子观察，不具备生产因子资格。",
    ]
    if metadata.get("runtime_reason"):
        lines.extend(["", f"运行环境说明: `{metadata['runtime_reason']}`"])
    for title, frame in sections:
        lines.extend(["", f"## {title}", ""])
        lines.append("_暂无样本。_" if frame.empty else _report_table(frame, top_n))
    report_path = output.with_suffix(".md")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "csv": output,
        "chinese_csv": cn_path,
        "metadata": metadata_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the research-only CZSC structure shadow.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--names-source", default=None)
    parser.add_argument("--min-bars", type=int, default=100)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_paths = [Path(path) for path in args.candidates]
    candidates = load_candidate_union(candidate_paths)
    names = (
        load_name_map(Path(args.names_source))
        if args.names_source and Path(args.names_source).exists()
        else {}
    )
    runtime = load_czsc_runtime()
    if runtime.available:
        prices = load_panel_prices(
            Path(args.data),
            candidates.get("symbol", pd.Series(dtype=str)),
            asof_date=args.asof_date,
        )
    else:
        prices = pd.DataFrame()
    table = build_structure_shadow(
        candidates,
        prices,
        runtime,
        asof_date=args.asof_date,
        name_map=names,
        min_bars=args.min_bars,
        max_abs_daily_return=args.max_abs_daily_return,
    )
    metadata = build_metadata(
        table,
        runtime,
        asof_date=args.asof_date,
        panel_path=Path(args.data),
        candidate_paths=candidate_paths,
        min_bars=args.min_bars,
        max_abs_daily_return=args.max_abs_daily_return,
    )
    paths = write_outputs(table, metadata, Path(args.output), top_n=args.top_n)
    print(json.dumps(metadata, ensure_ascii=False))
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
