"""Build a research-only shadow report from the THS ``主力净量`` field.

The source export expresses ``主力净量`` in percentage points. The ingestion
layer converts it to a decimal ratio before this module receives it. Missing
values remain missing and this shadow never changes candidate selection or
portfolio weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from panel_io import iter_panel, panel_columns

from build_daily_personal_overlay_report import load_name_map
from execution_rules import normalize_symbol


SOURCE_COLUMN = "main_net_volume_ratio"
DEFAULT_WINDOWS = (5, 10)
DEFAULT_LOOKBACK_CALENDAR_DAYS = 120
CN_COLUMNS = {
    "date": "日期",
    "symbol": "股票代码",
    "stock_name": "股票名称",
    "main_net_volume_ratio": "主力净量",
    "main_net_volume_ratio_5d": "5日平均主力净量",
    "main_net_volume_ratio_10d": "10日平均主力净量",
    "positive_ratio_5d": "5日净流入占比",
    "positive_ratio_10d": "10日净流入占比",
    "flow_rank_1d": "当日主力净量分位",
    "flow_rank_5d": "5日主力净量分位",
    "return_5d": "5日价格收益",
    "return_rank_5d": "5日价格收益分位",
    "flow_price_divergence_5d": "5日资金价格分位差",
    "flow_observation_count": "有效资金流天数",
    "flow_direction": "资金方向",
    "flow_price_state": "资金价格状态",
    "shadow_status": "影子状态",
    "shadow_reason": "观察理由",
    "early_pattern_match": "提前形态交集",
    "hidden_accumulation_match": "隐性吸筹交集",
    "early_pattern_type": "提前观察模式",
    "early_pattern_score": "提前观察评分",
    "research_only": "仅供研究",
    "selection_effect": "影响选股",
}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "高质量观察"}


def load_flow_panel(
    path: Path,
    *,
    asof_date: str | pd.Timestamp | None = None,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
) -> tuple[pd.DataFrame, bool]:
    header = panel_columns(path)
    required = {"date", "symbol"}
    missing = sorted(required.difference(header))
    if missing:
        raise ValueError(f"Panel is missing required columns: {', '.join(missing)}")
    columns = [column for column in ["date", "symbol", "close", SOURCE_COLUMN] if column in header]
    asof = pd.Timestamp(asof_date) if asof_date is not None else None
    start = asof - pd.Timedelta(days=int(lookback_calendar_days)) if asof is not None else None
    chunks: list[pd.DataFrame] = []
    for chunk in iter_panel(
        path,
        columns=columns,
        dtype={"symbol": str},
        chunksize=250_000,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        if asof is not None:
            chunk = chunk[chunk["date"].between(start, asof, inclusive="both")]
        if not chunk.empty:
            chunks.append(chunk)
    panel = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
    source_available = SOURCE_COLUMN in panel.columns
    if not source_available:
        panel[SOURCE_COLUMN] = np.nan
    return panel, source_available


def compute_main_net_volume_features(
    panel: pd.DataFrame,
    *,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    required = {"date", "symbol", SOURCE_COLUMN}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out[SOURCE_COLUMN] = pd.to_numeric(out[SOURCE_COLUMN], errors="coerce")
    if "close" in out:
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out[out["symbol"].ne("")]
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = out.groupby("symbol", sort=False)
    out["flow_observation_count"] = grouped[SOURCE_COLUMN].transform(
        lambda values: values.notna().cumsum()
    )
    positive = out[SOURCE_COLUMN].gt(0.0).where(out[SOURCE_COLUMN].notna()).astype(float)
    out["_positive_flow"] = positive

    normalized_windows = tuple(sorted({int(window) for window in windows if int(window) > 0}))
    for window in normalized_windows:
        out[f"main_net_volume_ratio_{window}d"] = grouped[SOURCE_COLUMN].transform(
            lambda values, window=window: values.rolling(window, min_periods=window).mean()
        )
        out[f"positive_ratio_{window}d"] = out.groupby("symbol", sort=False)[
            "_positive_flow"
        ].transform(
            lambda values, window=window: values.rolling(window, min_periods=window).mean()
        )

    if "close" in out:
        out["return_5d"] = grouped["close"].pct_change(5, fill_method=None)
    else:
        out["return_5d"] = np.nan

    out["flow_rank_1d"] = out.groupby("date", sort=False)[SOURCE_COLUMN].rank(
        method="average", pct=True
    )
    ratio_5d = "main_net_volume_ratio_5d"
    if ratio_5d in out:
        out["flow_rank_5d"] = out.groupby("date", sort=False)[ratio_5d].rank(
            method="average", pct=True
        )
    else:
        out["flow_rank_5d"] = np.nan
    out["return_rank_5d"] = out.groupby("date", sort=False)["return_5d"].rank(
        method="average", pct=True
    )
    out["flow_price_divergence_5d"] = out["flow_rank_5d"] - out["return_rank_5d"]
    return out.drop(columns=["_positive_flow"])


def _prepare_early_watchlist(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "symbol" not in frame:
        return pd.DataFrame(columns=["symbol"])
    out = frame.copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    keep = [
        column
        for column in [
            "symbol",
            "pattern_type",
            "pattern_score",
            "hidden_accumulation_trade_watch",
        ]
        if column in out
    ]
    out = out[keep].drop_duplicates("symbol", keep="first")
    return out.rename(
        columns={
            "pattern_type": "early_pattern_type",
            "pattern_score": "early_pattern_score",
            "hidden_accumulation_trade_watch": "hidden_accumulation_match",
        }
    )


def _flow_price_state(row: pd.Series) -> str:
    divergence = pd.to_numeric(row.get("flow_price_divergence_5d"), errors="coerce")
    ratio_5d = pd.to_numeric(row.get("main_net_volume_ratio_5d"), errors="coerce")
    if pd.isna(ratio_5d):
        return "等待5日样本"
    if divergence >= 0.20 and ratio_5d > 0.0:
        return "资金领先价格"
    if divergence <= -0.20 and ratio_5d < 0.0:
        return "价格强于资金"
    return "资金价格同步"


def _shadow_reason(row: pd.Series, minimum_history: int) -> str:
    count_value = pd.to_numeric(row.get("flow_observation_count"), errors="coerce")
    count = int(count_value) if pd.notna(count_value) else 0
    ratio_5d = pd.to_numeric(row.get("main_net_volume_ratio_5d"), errors="coerce")
    state = str(row.get("flow_price_state") or "")
    if pd.isna(ratio_5d):
        return f"仅有{count}个有效交易日，未满{minimum_history}日，暂不参与选股"
    if state == "资金领先价格":
        return "5日主力净量强于同期价格表现，作为疑似吸筹交叉验证"
    if ratio_5d < 0.0:
        return "5日主力净量为负，提示派发或承接不足风险"
    return "5日资金与价格方向基本同步，继续观察持续性"


def build_latest_shadow_table(
    features: pd.DataFrame,
    *,
    asof_date: str | pd.Timestamp | None = None,
    name_map: dict[str, str] | None = None,
    early_watchlist: pd.DataFrame | None = None,
    minimum_history: int = 5,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    asof = pd.Timestamp(asof_date) if asof_date is not None else features["date"].max()
    eligible_dates = features.loc[
        features["date"].le(asof) & features[SOURCE_COLUMN].notna(), "date"
    ]
    if eligible_dates.empty:
        return pd.DataFrame()
    source_date = eligible_dates.max()
    latest = features[
        features["date"].eq(source_date) & features[SOURCE_COLUMN].notna()
    ].copy()
    latest["date"] = source_date.strftime("%Y-%m-%d")
    latest.insert(2, "stock_name", latest["symbol"].map(name_map or {}).fillna(""))

    watch = _prepare_early_watchlist(early_watchlist)
    if not watch.empty:
        latest = latest.merge(watch, on="symbol", how="left", validate="one_to_one")
    for column in ["early_pattern_type", "early_pattern_score", "hidden_accumulation_match"]:
        if column not in latest:
            latest[column] = np.nan
    latest["early_pattern_type"] = latest["early_pattern_type"].fillna("")
    latest["early_pattern_match"] = latest["early_pattern_type"].ne("")
    latest["hidden_accumulation_match"] = latest["hidden_accumulation_match"].map(_truthy)
    latest["flow_direction"] = np.select(
        [latest[SOURCE_COLUMN].gt(0.0), latest[SOURCE_COLUMN].lt(0.0)],
        ["净流入", "净流出"],
        default="中性",
    )
    latest["flow_price_state"] = latest.apply(_flow_price_state, axis=1)
    latest["shadow_status"] = np.where(
        latest.get("main_net_volume_ratio_5d", pd.Series(np.nan, index=latest.index)).notna(),
        "5日影子可观察",
        "预热中",
    )
    latest["shadow_reason"] = latest.apply(
        _shadow_reason, axis=1, minimum_history=minimum_history
    )
    latest["research_only"] = True
    latest["selection_effect"] = False
    latest["_absolute_flow"] = latest[SOURCE_COLUMN].abs()
    latest = latest.sort_values(
        ["hidden_accumulation_match", "early_pattern_match", "_absolute_flow", "symbol"],
        ascending=[False, False, False, True],
    )
    return latest.drop(columns=["_absolute_flow"]).reset_index(drop=True)


def build_metadata(
    features: pd.DataFrame,
    latest: pd.DataFrame,
    *,
    requested_asof_date: str,
    source_column_available: bool,
    minimum_history: int,
) -> dict[str, object]:
    bounded = features[features["date"].le(pd.Timestamp(requested_asof_date))].copy()
    available = bounded[bounded[SOURCE_COLUMN].notna()]
    source_dates = available["date"].drop_duplicates().sort_values()
    latest_source_date = source_dates.max() if not source_dates.empty else pd.NaT
    if pd.isna(latest_source_date):
        latest_universe_rows = 0
        latest_source_rows = 0
    else:
        latest_day = bounded[bounded["date"].eq(latest_source_date)]
        latest_universe_rows = int(len(latest_day))
        latest_source_rows = int(latest_day[SOURCE_COLUMN].notna().sum())
    eligible_5d = int(
        latest.get("main_net_volume_ratio_5d", pd.Series(dtype=float)).notna().sum()
    )
    if not source_column_available or source_dates.empty:
        status = "unavailable"
    elif len(source_dates) < minimum_history or eligible_5d == 0:
        status = "warmup"
    else:
        status = "research_ready"
    return {
        "schema_version": 1,
        "status": status,
        "requested_asof_date": requested_asof_date,
        "source_column": SOURCE_COLUMN,
        "source_column_available": bool(source_column_available),
        "source_semantics": "同花顺主力净量；源值为百分比点，入口除以100后保存为十进制比例",
        "not_equivalent_to": "main_net_inflow（大单净额，金额单位）",
        "source_first_date": (
            source_dates.min().strftime("%Y-%m-%d") if not source_dates.empty else None
        ),
        "source_latest_date": (
            latest_source_date.strftime("%Y-%m-%d") if not pd.isna(latest_source_date) else None
        ),
        "available_source_sessions": int(len(source_dates)),
        "minimum_history_sessions": int(minimum_history),
        "latest_universe_rows": latest_universe_rows,
        "latest_source_rows": latest_source_rows,
        "latest_source_coverage": (
            latest_source_rows / latest_universe_rows if latest_universe_rows else 0.0
        ),
        "eligible_5d_rows": eligible_5d,
        "early_pattern_overlap_count": int(latest.get("early_pattern_match", pd.Series(dtype=bool)).sum()),
        "hidden_accumulation_overlap_count": int(
            latest.get("hidden_accumulation_match", pd.Series(dtype=bool)).sum()
        ),
        "rolling_factor_usable_latest_date": (
            latest_source_date.strftime("%Y-%m-%d") if eligible_5d and not pd.isna(latest_source_date) else None
        ),
        "missing_values_filled_with_zero": False,
        "research_only": True,
        "selection_effect": False,
        "portfolio_weight_effect": False,
    }


def _font_properties():
    from matplotlib.font_manager import FontProperties

    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        path = windir / "Fonts" / name
        if path.exists():
            return FontProperties(fname=str(path))
    return FontProperties()


def write_zero_center_chart(table: pd.DataFrame, output: Path, *, top_n: int = 30) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    table = table.copy()
    for column, default in ((SOURCE_COLUMN, np.nan), ("symbol", ""), ("stock_name", "")):
        if column not in table:
            table[column] = default
    font = _font_properties()
    half = max(1, int(top_n) // 2)
    positive = table[table[SOURCE_COLUMN].gt(0.0)].nlargest(half, SOURCE_COLUMN)
    negative = table[table[SOURCE_COLUMN].lt(0.0)].nsmallest(half, SOURCE_COLUMN)
    chart = pd.concat([negative, positive], ignore_index=True).sort_values(SOURCE_COLUMN)
    height = max(4.5, min(12.0, 0.34 * max(len(chart), 1) + 1.8))
    fig, ax = plt.subplots(figsize=(11, height))
    if chart.empty:
        ax.text(0.5, 0.5, "Main net volume data unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [
            f"{row.stock_name} {row.symbol}" if str(row.stock_name).strip() else str(row.symbol)
            for row in chart.itertuples()
        ]
        colors = np.where(chart[SOURCE_COLUMN].ge(0.0), "#D64545", "#18864B")
        ax.barh(labels, chart[SOURCE_COLUMN], color=colors, height=0.72)
        max_abs = float(chart[SOURCE_COLUMN].abs().max())
        limit = max(max_abs * 1.12, 0.001)
        ax.set_xlim(-limit, limit)
        ax.axvline(0.0, color="#202428", linewidth=1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", alpha=0.20)
        ax.set_title("主力净量零轴观察", fontproperties=font, fontsize=14)
        ax.set_xlabel("源值已转换为十进制比例；红色为净流入，绿色为净流出", fontproperties=font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font)
            label.set_fontsize(8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_report_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    percent_columns = [
        SOURCE_COLUMN,
        "main_net_volume_ratio_5d",
        "main_net_volume_ratio_10d",
        "positive_ratio_5d",
        "positive_ratio_10d",
        "return_5d",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
            )
    return display.rename(columns=CN_COLUMNS)


def write_outputs(
    table: pd.DataFrame,
    metadata: dict[str, object],
    output: Path,
    *,
    top_n: int = 30,
) -> dict[str, Path]:
    table = table.copy()
    required_output_columns = {
        "date": "",
        "symbol": "",
        "stock_name": "",
        SOURCE_COLUMN: np.nan,
        "early_pattern_match": False,
        "hidden_accumulation_match": False,
    }
    for column, default in required_output_columns.items():
        if column not in table:
            table[column] = default
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    cn_output = output.with_name(f"{output.stem}_cn{output.suffix}")
    table.rename(columns=CN_COLUMNS).to_csv(cn_output, index=False, encoding="utf-8-sig")
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    chart_path = output.with_suffix(".png")
    write_zero_center_chart(table, chart_path, top_n=top_n)

    preview_columns = [
        "symbol",
        "stock_name",
        SOURCE_COLUMN,
        "main_net_volume_ratio_5d",
        "flow_price_state",
        "early_pattern_type",
        "shadow_status",
        "shadow_reason",
    ]
    positive = table[table[SOURCE_COLUMN].gt(0.0)].nlargest(top_n // 2, SOURCE_COLUMN)
    negative = table[table[SOURCE_COLUMN].lt(0.0)].nsmallest(top_n // 2, SOURCE_COLUMN)
    overlap = table[table["early_pattern_match"].fillna(False).astype(bool)].head(top_n)
    lines = [
        "# 主力净量影子观察",
        "",
        "该报告吸收零轴正负柱的展示方式，但指标值直接来自每日同花顺 `主力净量` 字段。它不从图表库推导公式，也不改变当前选股和仓位。",
        "",
        f"- 状态: {metadata['status']}",
        f"- 数据日期: {metadata['source_latest_date']}",
        f"- 有效交易日: {metadata['available_source_sessions']} / 最低要求 {metadata['minimum_history_sessions']}",
        f"- 最新覆盖率: {metadata['latest_source_coverage']:.2%}",
        f"- 5日可用股票数: {metadata['eligible_5d_rows']}",
        f"- 与提前形态交集: {metadata['early_pattern_overlap_count']}",
        f"- 与高质量隐性吸筹交集: {metadata['hidden_accumulation_overlap_count']}",
        "- 风险提示: 主力净量是供应商口径的点时字段，不代表未来买盘，也不能单独判定吸筹或出货。",
        "",
        f"![主力净量零轴观察]({chart_path.name})",
    ]
    for title, frame in (("提前形态交集", overlap), ("净流入靠前", positive), ("净流出风险", negative)):
        lines.extend(["", f"## {title}", ""])
        if frame.empty:
            lines.append("_暂无样本。_")
        else:
            display = _format_report_table(frame[[column for column in preview_columns if column in frame]])
            lines.append(display.to_markdown(index=False, disable_numparse=True))
    report_path = output.with_suffix(".md")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "csv": output,
        "chinese_csv": cn_output,
        "metadata": metadata_path,
        "chart": chart_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the THS main-net-volume shadow report.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--names-source", default=None)
    parser.add_argument("--early-watchlist", default=None)
    parser.add_argument("--minimum-history", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel, source_available = load_flow_panel(
        Path(args.data),
        asof_date=args.asof_date,
    )
    features = compute_main_net_volume_features(panel)
    early_watchlist = None
    if args.early_watchlist and Path(args.early_watchlist).exists():
        early_watchlist = pd.read_csv(Path(args.early_watchlist), dtype={"symbol": str})
    names = load_name_map(Path(args.names_source)) if args.names_source else {}
    latest = build_latest_shadow_table(
        features,
        asof_date=args.asof_date,
        name_map=names,
        early_watchlist=early_watchlist,
        minimum_history=args.minimum_history,
    )
    metadata = build_metadata(
        features,
        latest,
        requested_asof_date=args.asof_date,
        source_column_available=source_available,
        minimum_history=args.minimum_history,
    )
    paths = write_outputs(latest, metadata, Path(args.output), top_n=args.top_n)
    print(json.dumps(metadata, ensure_ascii=False))
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
