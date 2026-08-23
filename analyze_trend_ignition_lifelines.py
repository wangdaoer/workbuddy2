"""Extract ignition points and lifelines for long trend runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix
from research_database import ResearchDatabase, normalize_a_share_symbol


PANEL_PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]


def load_tdx_history_prices(
    research_db: str | Path,
    tdx_db: str | Path,
    *,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    asset_type: str = "stock",
) -> pd.DataFrame:
    db = ResearchDatabase(research_db)
    normalized_symbols = None
    if symbols:
        normalized_symbols = []
        for symbol in symbols:
            normalized = normalize_a_share_symbol(symbol)
            if normalized is None:
                raise ValueError(f"Invalid A-share symbol: {symbol!r}")
            normalized_symbols.append(normalized)
    prices = db.query_tdx_history_normalized(
        tdx_db,
        symbols=sorted(set(normalized_symbols)) if normalized_symbols else None,
        asset_type=asset_type,
        start=None if start is None else pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=None if end is None else pd.Timestamp(end).strftime("%Y-%m-%d"),
    )
    if prices.empty:
        return pd.DataFrame(columns=PANEL_PRICE_COLUMNS)
    prices = prices[PANEL_PRICE_COLUMNS].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
    return prices.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def parse_tdx_symbols(
    symbols_text: str | None = None,
    symbols_file: str | Path | None = None,
) -> list[str] | None:
    symbols: list[str] = []
    if symbols_text:
        symbols.extend(item.strip() for item in symbols_text.split(",") if item.strip())
    if symbols_file:
        path = Path(symbols_file)
        symbols.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    cleaned = []
    for symbol in symbols:
        normalized = normalize_a_share_symbol(symbol)
        if normalized is None:
            raise ValueError(f"Invalid A-share symbol: {symbol!r}")
        cleaned.append(normalized)
    cleaned = sorted(set(cleaned))
    return cleaned or None


def find_ignition_candidates(
    close: pd.Series,
    high: pd.Series,
    amount: pd.Series,
    *,
    breakout_window: int = 60,
    amount_window: int = 20,
    amount_multiplier: float = 1.5,
    breakout_buffer: float = 0.01,
) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    amount = pd.to_numeric(amount, errors="coerce").reindex(close.index)

    prev_high = high.rolling(
        breakout_window, min_periods=max(20, breakout_window // 2)
    ).max().shift(1)
    amount_median = amount.replace(0, np.nan).rolling(
        amount_window, min_periods=max(5, amount_window // 2)
    ).median().shift(1)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    amount_ratio = amount / (amount_median + 1e-12)

    return_20d = close / close.shift(20) - 1.0
    return_60d = close / close.shift(60) - 1.0
    volatility_20d = close.pct_change(fill_method=None).rolling(20, min_periods=20).std()
    ma20_over_ma60 = ma20 / (ma60 + 1e-12) - 1.0
    close_over_ma20 = close / (ma20 + 1e-12) - 1.0
    rolling_high_120 = high.rolling(120, min_periods=60).max()
    drawdown_120d = close / (rolling_high_120 + 1e-12) - 1.0
    amount_mean_5 = amount.replace(0, np.nan).rolling(5, min_periods=3).mean()
    amount_trend_5_20 = amount_mean_5 / (amount_median + 1e-12)

    breakout = close.gt(prev_high * (1.0 + breakout_buffer))
    volume_ok = amount_ratio.ge(amount_multiplier)
    trend_ok = close.gt(ma60) & ma20.ge(ma60 * 0.98)
    valid = breakout & volume_ok & trend_ok
    breakout_count_20d = breakout.astype(float).rolling(20, min_periods=5).sum()

    out = pd.DataFrame(
        {
            "close": close,
            "prev_high": prev_high,
            "breakout_pct": close / (prev_high + 1e-12) - 1.0,
            "amount": amount,
            "amount_ratio": amount_ratio,
            "ma20": ma20,
            "ma60": ma60,
            "return_20d": return_20d,
            "return_60d": return_60d,
            "volatility_20d": volatility_20d,
            "ma20_over_ma60": ma20_over_ma60,
            "close_over_ma20": close_over_ma20,
            "drawdown_120d": drawdown_120d,
            "amount_trend_5_20": amount_trend_5_20,
            "breakout_count_20d": breakout_count_20d,
        }
    )
    return out.loc[valid].dropna(subset=["close", "prev_high"])


def choose_lifeline(
    close: pd.Series,
    ignition_date: pd.Timestamp,
    peak_date: pd.Timestamp,
    *,
    windows: tuple[int, ...] = (20, 30, 60, 120),
    breach_buffer: float = 0.03,
    max_breach_days: int = 3,
    end_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    close = pd.to_numeric(close, errors="coerce").sort_index()
    segment = close.loc[ignition_date:peak_date].dropna()
    candidates: list[dict[str, object]] = []

    for window in windows:
        line = close.rolling(window, min_periods=max(5, window // 2)).mean()
        aligned = pd.concat(
            [segment.rename("close"), line.reindex(segment.index).rename("line")],
            axis=1,
        ).dropna()
        if aligned.empty:
            continue
        distance = aligned["close"] / (aligned["line"] + 1e-12) - 1.0
        breach = distance.lt(-breach_buffer)
        candidates.append(
            {
                "lifeline_ma": window,
                "breach_days_to_peak": int(breach.sum()),
                "min_distance_to_lifeline": float(distance.min()),
                "avg_distance_to_lifeline": float(distance.mean()),
                "line": line,
            }
        )

    if not candidates:
        return {
            "lifeline_ma": None,
            "breach_days_to_peak": None,
            "min_distance_to_lifeline": None,
            "avg_distance_to_lifeline": None,
            "lifeline_break_date": None,
            "lifeline_status": "no_lifeline",
        }

    valid = [row for row in candidates if row["breach_days_to_peak"] <= max_breach_days]
    if valid:
        chosen = sorted(valid, key=lambda row: int(row["lifeline_ma"]))[0]
    else:
        chosen = sorted(
            candidates,
            key=lambda row: (
                int(row["breach_days_to_peak"]),
                -float(row["min_distance_to_lifeline"]),
                int(row["lifeline_ma"]),
            ),
        )[0]

    line = chosen.pop("line")
    break_date = None
    if end_date is not None:
        after_peak = close.loc[peak_date:end_date].iloc[1:].dropna()
        if not after_peak.empty:
            after_line = line.reindex(after_peak.index)
            broken = after_peak.lt(after_line * (1.0 - breach_buffer))
            if broken.any():
                break_date = broken[broken].index[0]

    chosen["lifeline_break_date"] = break_date.strftime("%Y-%m-%d") if break_date is not None else None
    chosen["lifeline_status"] = "broken" if break_date is not None else "alive_or_unbroken"
    return chosen


def _max_drawdown(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return 0.0
    return float((values / values.cummax() - 1.0).min())


def _compute_ignition_prior_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series | None,
    open: pd.Series | None,
    amount: pd.Series,
    ignition_date: pd.Timestamp,
) -> dict[str, float]:
    """点火前特征: 全部只用 ignition_date 及之前的数据, 严格防未来泄漏.

    返回 10 个候选特征; 回溯不足时填 NaN (由下游 dropna/缺失桶处理).
    """
    out: dict[str, float] = {}
    keys = [
        "pre_amount_trend_20_60", "pre_amount_accel", "pre_volume_z",
        "pre_consolidation_60", "pre_drift_slope_60", "pre_cnl_amplitude_60",
        "ign_day_amplitude", "ign_day_upper_shadow", "ign_day_close_pos", "ign_day_gap",
    ]
    # 前缀截至点火当日, 点火前特征再取 .iloc[:-1] (不含点火当日)
    pre_close = close.loc[:ignition_date]
    if len(pre_close) < 2:
        return {k: float("nan") for k in keys}
    pc = pre_close.iloc[:-1]
    ph = high.loc[:ignition_date].iloc[:-1]
    pa = amount.loc[:ignition_date].iloc[:-1].replace(0, np.nan)
    # ---- 点火前量能 (建仓痕迹) ----
    if len(pc) >= 20:
        am20 = pa.rolling(20, min_periods=10).median()
        am60 = pa.rolling(60, min_periods=30).median()
        am120 = pa.rolling(120, min_periods=60).median()
        out["pre_amount_trend_20_60"] = float(am20.iloc[-1] / (am60.iloc[-1] + 1e-12))
        out["pre_amount_accel"] = float(am20.iloc[-1] / (am120.iloc[-1] + 1e-12) - 1.0)
        pa_std120 = pa.rolling(120, min_periods=60).std().iloc[-1]
        out["pre_volume_z"] = float((am20.iloc[-1] - am120.iloc[-1]) / (pa_std120 + 1e-12))
        # ---- 横盘蓄势结构 ----
        win_h = ph.tail(60)
        win_c = pc.tail(60)
        win_l = low.loc[:ignition_date].iloc[:-1].tail(60) if low is not None else win_h
        out["pre_consolidation_60"] = float((win_h.max() - win_l.min()) / (win_c.mean() + 1e-12))
        y = win_c.values.astype(float)
        if len(y) > 2 and float(np.std(y)) > 0:
            slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
            out["pre_drift_slope_60"] = slope / (float(y.mean()) + 1e-12)
        else:
            out["pre_drift_slope_60"] = 0.0
        out["pre_cnl_amplitude_60"] = float(((win_h - win_c) / (win_c.shift(1) + 1e-12)).mean())
    else:
        for k in ["pre_amount_trend_20_60", "pre_amount_accel", "pre_volume_z",
                  "pre_consolidation_60", "pre_drift_slope_60", "pre_cnl_amplitude_60"]:
            out[k] = float("nan")
    # ---- 首板质量 (点火当日, point-in-time 合法) ----
    ic_close = float(close.loc[ignition_date])
    ic_high = float(high.loc[ignition_date])
    prev_c = float(pre_close.iloc[-2])
    out["ign_day_amplitude"] = float((ic_high - (float(low.loc[ignition_date]) if low is not None else ic_close)) / (prev_c + 1e-12))
    out["ign_day_upper_shadow"] = float((ic_high - ic_close) / (ic_close + 1e-12))
    ic_low = float(low.loc[ignition_date]) if low is not None else ic_close
    out["ign_day_close_pos"] = float((ic_close - ic_low) / (ic_high - ic_low + 1e-12))
    ic_open = float(open.loc[ignition_date]) if open is not None else ic_close
    out["ign_day_gap"] = float((ic_open - prev_c) / (prev_c + 1e-12))
    return out


def summarize_trends_for_symbol(
    symbol: str,
    close: pd.Series,
    high: pd.Series,
    amount: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    open: pd.Series | None = None,
    low: pd.Series | None = None,
    min_trend_return: float = 0.80,
    ignition_lookback_days: int = 120,
    breakout_window: int = 60,
    amount_window: int = 20,
    amount_multiplier: float = 1.5,
    breakout_buffer: float = 0.01,
    lifeline_breach_buffer: float = 0.03,
    lifeline_max_breach_days: int = 3,
    lifeline_windows: tuple[int, ...] = (20, 30, 60, 120),
    include_below_threshold: bool = False,
    include_prestart_ignitions: bool = True,
    outcome_horizon_days: int = 365,
    ignition_cooldown_bars: int = 20,
) -> list[dict[str, object]]:
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    amount = pd.to_numeric(amount, errors="coerce").reindex(close.index)
    if close.loc[:end].dropna().empty:
        return []

    ignition_start = (
        start - pd.Timedelta(days=ignition_lookback_days)
        if include_prestart_ignitions
        else start
    )
    ignitions = find_ignition_candidates(
        close,
        high,
        amount,
        breakout_window=breakout_window,
        amount_window=amount_window,
        amount_multiplier=amount_multiplier,
        breakout_buffer=breakout_buffer,
    )
    ignitions = ignitions.loc[(ignitions.index >= ignition_start) & (ignitions.index <= end)]

    valid_dates = close.loc[:end].dropna().index
    date_positions = pd.Series(np.arange(len(valid_dates)), index=valid_dates)
    records: list[dict[str, object]] = []
    cooldown_until_position = -1
    for ignition_date, ignition in ignitions.iterrows():
        ignition_position = int(date_positions.loc[ignition_date])
        if ignition_position <= cooldown_until_position:
            continue
        outcome_end = min(end, ignition_date + pd.Timedelta(days=outcome_horizon_days))
        future_window = close.loc[max(ignition_date, start):outcome_end].dropna()
        if future_window.empty:
            continue
        peak_date = future_window.idxmax()
        peak_close = float(future_window.loc[peak_date])
        ignition_close = float(close.loc[ignition_date])
        if ignition_close <= 0:
            continue
        peak_return = peak_close / ignition_close - 1.0
        if peak_return < min_trend_return and not include_below_threshold:
            continue

        period_start_close = close.loc[close.index >= start].dropna()
        period_start_price = float(period_start_close.iloc[0]) if not period_start_close.empty else np.nan
        close_to_end = close.loc[:outcome_end].dropna()
        end_close = float(close_to_end.iloc[-1]) if not close_to_end.empty else np.nan
        trend_path = close.loc[ignition_date:peak_date]
        lifeline = choose_lifeline(
            close,
            ignition_date,
            peak_date,
            windows=lifeline_windows,
            breach_buffer=lifeline_breach_buffer,
            max_breach_days=lifeline_max_breach_days,
            end_date=outcome_end,
        )
        pre_feats = _compute_ignition_prior_features(
            close, high, low, open, amount, ignition_date
        )
        records.append(
            {
                **pre_feats,
                "symbol": str(symbol).zfill(6),
                "ignition_date": ignition_date.strftime("%Y-%m-%d"),
                "ignition_close": ignition_close,
                "peak_date": peak_date.strftime("%Y-%m-%d"),
                "peak_close": peak_close,
                "peak_return": peak_return,
                "analysis_end_date": end.strftime("%Y-%m-%d"),
                "outcome_end_date": outcome_end.strftime("%Y-%m-%d"),
                "period_return_to_peak": peak_close / period_start_price - 1.0 if period_start_price > 0 else np.nan,
                "return_to_end_from_ignition": end_close / ignition_close - 1.0 if end_close > 0 else np.nan,
                "days_to_peak": int((peak_date - ignition_date).days),
                "max_drawdown_to_peak": _max_drawdown(trend_path),
                "breakout_pct": float(ignition["breakout_pct"]),
                "amount_ratio": float(ignition["amount_ratio"]),
                "return_20d": float(ignition["return_20d"]),
                "return_60d": float(ignition["return_60d"]),
                "volatility_20d": float(ignition["volatility_20d"]),
                "ma20_over_ma60": float(ignition["ma20_over_ma60"]),
                "close_over_ma20": float(ignition["close_over_ma20"]),
                "drawdown_120d": float(ignition["drawdown_120d"]),
                "amount_trend_5_20": float(ignition["amount_trend_5_20"]),
                "breakout_count_20d": float(ignition["breakout_count_20d"]),
                **lifeline,
            }
        )
        cooldown_until_position = ignition_position + max(0, ignition_cooldown_bars)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze long trend ignition points and lifelines.")
    parser.add_argument("--data")
    parser.add_argument("--research-db", default="data/research.sqlite3")
    parser.add_argument("--tdx-db", default="data/tdx_history.sqlite3")
    parser.add_argument("--use-tdx-history", action="store_true")
    parser.add_argument("--tdx-symbols", default=None, help="Comma separated symbols for TDX history mode.")
    parser.add_argument(
        "--tdx-symbols-file",
        default=None,
        help="UTF-8 text file with one symbol per line for TDX history mode.",
    )
    parser.add_argument("--output-dir", default="outputs/high_return_v2/trend_ignition_lifelines_202506_202606")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default="2026-06-26")
    parser.add_argument("--min-trend-return", type=float, default=0.80)
    parser.add_argument("--ignition-lookback-days", type=int, default=120)
    parser.add_argument("--breakout-window", type=int, default=60)
    parser.add_argument("--amount-window", type=int, default=20)
    parser.add_argument("--amount-multiplier", type=float, default=1.5)
    parser.add_argument("--breakout-buffer", type=float, default=0.01)
    parser.add_argument("--lifeline-breach-buffer", type=float, default=0.03)
    parser.add_argument("--lifeline-max-breach-days", type=int, default=3)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--period", default=None, help="Chronological period label stored in training samples.")
    parser.add_argument("--outcome-horizon-days", type=int, default=365)
    parser.add_argument("--ignition-cooldown-bars", type=int, default=20)
    parser.add_argument("--include-prestart-ignitions", action="store_true")
    args = parser.parse_args(argv)
    if not args.use_tdx_history and not args.data:
        parser.error("--data is required unless --use-tdx-history is set")
    return args


def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "symbol": "股票代码",
            "ignition_date": "起爆日",
            "ignition_close": "起爆收盘价",
            "peak_date": "峰值日",
            "peak_close": "峰值收盘价",
            "peak_return": "起爆至峰值收益",
            "period_return_to_peak": "区间起点至峰值收益",
            "return_to_end_from_ignition": "起爆至区间末收益",
            "days_to_peak": "起爆到峰值天数",
            "max_drawdown_to_peak": "起爆到峰值最大回撤",
            "breakout_pct": "突破幅度",
            "amount_ratio": "成交额放大倍数",
            "return_20d": "近20日收益",
            "return_60d": "近60日收益",
            "volatility_20d": "近20日波动率",
            "ma20_over_ma60": "MA20相对MA60偏离",
            "close_over_ma20": "收盘价相对MA20偏离",
            "drawdown_120d": "距120日高点回撤",
            "amount_trend_5_20": "近5日成交额趋势比",
            "breakout_count_20d": "近20日突破次数",
            "lifeline_ma": "生命线均线",
            "breach_days_to_peak": "峰值前跌破生命线天数",
            "min_distance_to_lifeline": "相对生命线最小距离",
            "avg_distance_to_lifeline": "相对生命线平均距离",
            "lifeline_break_date": "生命线跌破日",
            "lifeline_status": "生命线状态",
        }
    )


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if args.use_tdx_history:
        symbols = parse_tdx_symbols(args.tdx_symbols, args.tdx_symbols_file)
        raw = load_tdx_history_prices(
            args.research_db,
            args.tdx_db,
            symbols=symbols,
            start=start - pd.Timedelta(days=args.ignition_lookback_days + args.breakout_window * 3),
            end=end,
            asset_type="stock",
        )
    else:
        raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    opens = pivot_prices(raw, "open").reindex_like(close)
    lows = pivot_prices(raw, "low").reindex_like(close)

    records: list[dict[str, object]] = []
    for symbol in close.columns:
        records.extend(
            summarize_trends_for_symbol(
                symbol=str(symbol).zfill(6),
                close=close[symbol],
                high=high[symbol],
                amount=amount[symbol],
                open=opens[symbol],
                low=lows[symbol],
                start=start,
                end=end,
                min_trend_return=args.min_trend_return,
                ignition_lookback_days=args.ignition_lookback_days,
                breakout_window=args.breakout_window,
                amount_window=args.amount_window,
                amount_multiplier=args.amount_multiplier,
                breakout_buffer=args.breakout_buffer,
                lifeline_breach_buffer=args.lifeline_breach_buffer,
                lifeline_max_breach_days=args.lifeline_max_breach_days,
                include_below_threshold=True,
                include_prestart_ignitions=args.include_prestart_ignitions,
                outcome_horizon_days=args.outcome_horizon_days,
                ignition_cooldown_bars=args.ignition_cooldown_bars,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.DataFrame(records)
    if not samples.empty:
        samples["period"] = args.period or f"{start:%Y%m%d}_{end:%Y%m%d}"
        samples = samples.sort_values(["ignition_date", "symbol"]).reset_index(drop=True)
    samples.to_csv(output_dir / "trend_ignition_samples.csv", index=False, encoding="utf-8-sig")
    trends = (
        samples[pd.to_numeric(samples["peak_return"], errors="coerce").ge(args.min_trend_return)].copy()
        if not samples.empty
        else samples.copy()
    )
    if not trends.empty:
        trends = trends.sort_values(["peak_return", "amount_ratio"], ascending=[False, False])
    trends.to_csv(output_dir / "trend_ignition_lifelines.csv", index=False, encoding="utf-8-sig")
    cn = to_chinese_columns(trends)
    cn.to_csv(output_dir / "trend_ignition_lifelines_cn.csv", index=False, encoding="utf-8-sig")
    top = cn.head(args.top_n)
    top.to_csv(output_dir / "top_trends_cn.csv", index=False, encoding="utf-8-sig")

    summary = {
        "start": args.start,
        "end": args.end,
        "min_trend_return": args.min_trend_return,
        "outcome_horizon_days": args.outcome_horizon_days,
        "sample_count": int(len(samples)),
        "trend_count": int(len(trends)),
        "symbol_count": int(trends["symbol"].nunique()) if not trends.empty else 0,
        "median_peak_return": float(trends["peak_return"].median()) if not trends.empty else None,
        "median_days_to_peak": float(trends["days_to_peak"].median()) if not trends.empty else None,
        "lifeline_distribution": trends["lifeline_ma"].value_counts(dropna=False).to_dict()
        if not trends.empty
        else {},
        "broken_lifeline_ratio": float(trends["lifeline_status"].eq("broken").mean()) if not trends.empty else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Trend ignition/lifeline outputs saved to: {output_dir}")
    if not top.empty:
        print(top.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
