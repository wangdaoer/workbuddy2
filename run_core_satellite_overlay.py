"""Walk-forward core rank model with a strict strong-pullback satellite sleeve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import apply_open_constraints, next_open_return_label
from run_backtest import load_prices, max_drawdown, pivot_prices, sharpe_like
from run_strong_pullback_satellite import (
    apply_rebound_exit,
    build_daily_candidates,
    build_raw_metric_frames,
    should_apply_rebound_exit,
)
from train_next_open_rank_model import (
    build_breadth_exposure,
    build_features,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    normalize_weights,
)


def select_rank_targets(
    score: pd.Series,
    top_n: int,
    sleeve_leverage: float,
    max_position_weight: float,
    min_score: float | None = None,
) -> pd.Series:
    pool = score.replace([np.inf, -np.inf], np.nan).dropna()
    if min_score is not None:
        pool = pool[pool.ge(min_score)]
    if pool.empty or top_n <= 0 or sleeve_leverage <= 0:
        return pd.Series(dtype=float)
    selected = pool.nlargest(top_n)
    weight = min(max_position_weight, sleeve_leverage / len(selected))
    return pd.Series(weight, index=selected.index, dtype=float)


def combine_target_weights(
    core: pd.Series,
    satellite: pd.Series,
    all_symbols: pd.Index,
    total_leverage: float,
    max_position_weight: float,
) -> pd.Series:
    combined = (
        core.reindex(all_symbols).fillna(0.0)
        + satellite.reindex(all_symbols).fillna(0.0)
    )
    combined = combined.clip(lower=0.0, upper=max_position_weight)
    gross = float(combined.sum())
    if gross <= total_leverage or gross <= 1e-12:
        return combined

    capped = combined.ge(max_position_weight - 1e-12)
    capped_weight = float(combined[capped].sum())
    if capped_weight >= total_leverage:
        scaled = combined * (total_leverage / gross)
        return scaled.clip(lower=0.0, upper=max_position_weight)

    remaining_budget = total_leverage - capped_weight
    uncapped = combined.where(~capped, 0.0)
    uncapped_sum = float(uncapped.sum())
    if uncapped_sum > 1e-12:
        combined.loc[~capped] = uncapped.loc[~capped] * (remaining_budget / uncapped_sum)
    return combined


def apply_overlay_satellite_rebound_exit(
    current_satellite: pd.Series,
    target_satellite: pd.Series,
    close_row: pd.Series,
    entry_price: pd.Series,
    market_exposure: float,
    rebound_exit_return: float | None = None,
    rebound_exit_scale: float = 0.0,
    rebound_exit_market_exposure_max: float | None = None,
    rebound_exit_market_exposure_min: float | None = None,
) -> tuple[pd.Series, dict[str, str]]:
    enabled = should_apply_rebound_exit(
        rebound_exit_return,
        market_exposure=market_exposure,
        market_exposure_max=rebound_exit_market_exposure_max,
        market_exposure_min=rebound_exit_market_exposure_min,
    )
    if not enabled:
        return target_satellite.copy(), {}
    return apply_rebound_exit(
        current_satellite,
        target_satellite,
        close_row,
        entry_price,
        rebound_exit_return=rebound_exit_return,
        rebound_exit_scale=rebound_exit_scale,
    )


def build_core_eligibility_mask(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    row_index: int,
    min_close: float | None = None,
    min_avg_amount_20d: float | None = None,
    max_large_down_days_20: int | None = None,
    large_down_threshold: float = -0.045,
    exclude_symbols: set[str] | None = None,
    avg_amount_20d: pd.DataFrame | None = None,
    large_down_days_20: pd.DataFrame | None = None,
) -> pd.Series:
    eligible = pd.Series(True, index=close.columns, dtype=bool)
    if row_index < 0 or row_index >= len(close.index):
        return eligible

    close_row = close.iloc[row_index]
    if min_close is not None:
        eligible &= close_row.ge(float(min_close)).fillna(False)

    if min_avg_amount_20d is not None:
        avg_amount_frame = avg_amount_20d
        if avg_amount_frame is None:
            avg_amount_frame = amount.replace(0, np.nan).rolling(20).median()
        avg_amount = avg_amount_frame.iloc[row_index]
        eligible &= avg_amount.ge(float(min_avg_amount_20d)).fillna(False)

    if max_large_down_days_20 is not None:
        large_down_frame = large_down_days_20
        if large_down_frame is None:
            large_down_frame = close.pct_change(fill_method=None).le(float(large_down_threshold)).rolling(20).sum()
        large_down_days = large_down_frame.iloc[row_index]
        eligible &= large_down_days.le(int(max_large_down_days_20)).fillna(False)

    if exclude_symbols:
        excluded = {str(symbol).zfill(6) for symbol in exclude_symbols}
        eligible.loc[eligible.index.astype(str).isin(excluded)] = False

    return eligible


def load_excluded_symbols(path: str | None) -> set[str]:
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Exclude symbols file not found: {file_path}")
    if file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path, dtype=str)
        for column in ["symbol", "security_code", "code", "代码"]:
            if column in frame.columns:
                return {str(value).strip().zfill(6) for value in frame[column].dropna() if str(value).strip()}
        first_column = frame.columns[0]
        return {str(value).strip().zfill(6) for value in frame[first_column].dropna() if str(value).strip()}
    symbols: set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split(",")[0].strip()
        if token:
            symbols.add(token.zfill(6))
    return symbols


def score_from_weights(features: dict[str, pd.DataFrame], weights: pd.Series, row_index: int) -> pd.Series:
    score = None
    for name, weight in weights.items():
        if name not in features:
            continue
        part = features[name].iloc[row_index].fillna(0.0) * weight
        score = part if score is None else score + part
    if score is None:
        return pd.Series(dtype=float)
    return score


def run_overlay_walk_forward(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    train_days: int,
    retrain_frequency: int,
    rebalance_frequency: int,
    core_top_n: int,
    satellite_top_n: int,
    core_leverage: float,
    satellite_leverage: float,
    total_leverage: float,
    max_position_weight: float,
    min_core_score: float | None,
    min_satellite_score: float | None,
    commission_bps: float,
    impact_bps: float,
    max_buy_open_gap: float,
    limit_buffer: float,
    market_exposure: pd.Series,
    initial_capital: float,
    filter_kwargs: dict[str, float],
    core_min_close: float | None = None,
    core_min_avg_amount_20d: float | None = None,
    core_max_large_down_days_20: int | None = None,
    core_large_down_threshold: float = -0.045,
    core_exclude_symbols: set[str] | None = None,
    satellite_rebound_exit_return: float | None = None,
    satellite_rebound_exit_scale: float = 0.0,
    satellite_rebound_exit_market_exposure_max: float | None = None,
    satellite_rebound_exit_market_exposure_min: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_features(close, open_px, high, low, amount)
    raw_metrics = build_raw_metric_frames(close, open_px, high, low, amount)
    label = next_open_return_label(open_px)
    ic = daily_ic(features, label)

    equity = initial_capital
    positions = pd.Series(0.0, index=close.columns)
    core_sleeve = pd.Series(0.0, index=close.columns)
    satellite_sleeve = pd.Series(0.0, index=close.columns)
    satellite_entry_price = pd.Series(np.nan, index=close.columns, dtype=float)
    current_weights = pd.Series(1.0 / len(features), index=list(features))
    nav_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    core_avg_amount_20d = (
        amount.replace(0, np.nan).rolling(20).median()
        if core_min_avg_amount_20d is not None
        else None
    )
    core_large_down_days_20 = (
        close.pct_change(fill_method=None).le(float(core_large_down_threshold)).rolling(20).sum()
        if core_max_large_down_days_20 is not None
        else None
    )

    for i in range(train_days, len(close.index) - 2):
        signal_date = close.index[i]
        previous_satellite_sleeve = satellite_sleeve.copy()
        satellite_rebound_hits: dict[str, str] = {}
        if (i - train_days) % retrain_frequency == 0:
            current_weights = normalize_weights(ic.iloc[i - train_days : i].mean())
            weight_rows.append({"date": signal_date.strftime("%Y-%m-%d"), **current_weights.to_dict()})

        core_count = 0
        core_eligible_count = 0
        core_excluded_count = 0
        satellite_count = 0
        candidate_count = 0
        exposure = float(market_exposure.reindex([signal_date]).iloc[0])
        is_rebalance = (i - train_days) % rebalance_frequency == 0
        score = pd.Series(dtype=float)
        core = pd.Series(dtype=float)
        if is_rebalance:
            score = score_from_weights(features, current_weights, i)
            core_eligibility = build_core_eligibility_mask(
                close,
                amount,
                row_index=i,
                min_close=core_min_close,
                min_avg_amount_20d=core_min_avg_amount_20d,
                max_large_down_days_20=core_max_large_down_days_20,
                large_down_threshold=core_large_down_threshold,
                exclude_symbols=core_exclude_symbols,
                avg_amount_20d=core_avg_amount_20d,
                large_down_days_20=core_large_down_days_20,
            )
            core_eligible_count = int(core_eligibility.sum())
            core_excluded_count = int((~core_eligibility).sum())
            score = score.reindex(close.columns).where(core_eligibility).dropna()
            core = select_rank_targets(
                score,
                top_n=core_top_n,
                sleeve_leverage=core_leverage,
                max_position_weight=max_position_weight,
                min_score=min_core_score,
            )
            core_sleeve = core.reindex(close.columns).fillna(0.0)
            candidates = build_daily_candidates(
                features,
                raw_metrics,
                current_weights,
                i,
                **filter_kwargs,
            )
            if min_satellite_score is not None:
                candidates = candidates[candidates["score"].ge(min_satellite_score)]
            candidate_count = int(len(candidates))
            satellite = select_rank_targets(
                candidates["score"] if not candidates.empty else pd.Series(dtype=float),
                top_n=satellite_top_n,
                sleeve_leverage=satellite_leverage,
                max_position_weight=max_position_weight,
            )
            satellite_sleeve = satellite.reindex(close.columns).fillna(0.0)
            satellite_count = int(satellite.ne(0).sum())
            core_count = int(core.ne(0).sum())
        else:
            target = positions.copy()

        satellite_rebound_enabled = should_apply_rebound_exit(
            satellite_rebound_exit_return,
            market_exposure=exposure,
            market_exposure_max=satellite_rebound_exit_market_exposure_max,
            market_exposure_min=satellite_rebound_exit_market_exposure_min,
        )
        satellite_sleeve, satellite_rebound_hits = apply_overlay_satellite_rebound_exit(
            previous_satellite_sleeve,
            satellite_sleeve,
            close.iloc[i],
            satellite_entry_price,
            market_exposure=exposure,
            rebound_exit_return=satellite_rebound_exit_return,
            rebound_exit_scale=satellite_rebound_exit_scale,
            rebound_exit_market_exposure_max=satellite_rebound_exit_market_exposure_max,
            rebound_exit_market_exposure_min=satellite_rebound_exit_market_exposure_min,
        )
        if is_rebalance or satellite_rebound_hits:
            target = combine_target_weights(
                core_sleeve,
                satellite_sleeve,
                all_symbols=close.columns,
                total_leverage=total_leverage,
                max_position_weight=max_position_weight,
            )
            target = target * exposure

        if is_rebalance:
            selected = target[target.gt(0)]
            for symbol, weight in selected.items():
                in_satellite = bool(satellite_sleeve.reindex([symbol]).fillna(0.0).iloc[0] > 0.0)
                selection_rows.append(
                    {
                        "signal_date": signal_date.strftime("%Y-%m-%d"),
                        "symbol": str(symbol).zfill(6),
                        "target_weight": float(weight),
                        "in_core": bool(symbol in core.index),
                        "in_satellite": bool(in_satellite),
                        "score": float(score.reindex([symbol]).iloc[0]) if symbol in score.index else np.nan,
                        "satellite_rebound_exit_enabled": bool(satellite_rebound_enabled),
                        "satellite_rebound_exit_count": int(len(satellite_rebound_hits)),
                    }
                )

        target = apply_open_constraints(
            positions,
            target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=max_buy_open_gap,
            limit_buffer=limit_buffer,
        )
        active_satellite_rebound_hits = {
            str(symbol): reason
            for symbol, reason in satellite_rebound_hits.items()
            if symbol in satellite_sleeve.index
            and float(satellite_sleeve.loc[symbol]) < float(previous_satellite_sleeve.loc[symbol])
        }

        turnover = float((target - positions).abs().sum())
        cost = turnover * (commission_bps + impact_bps) / 1e4
        realized = label.iloc[i].reindex(close.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross_return = float((target * realized).sum())
        equity *= 1.0 + gross_return - cost

        next_open = open_px.iloc[i + 1].reindex(close.columns)
        new_satellite_entries = satellite_sleeve.gt(0.0) & previous_satellite_sleeve.le(0.0) & target.gt(0.0)
        satellite_exits = satellite_sleeve.le(0.0)
        satellite_entry_price.loc[new_satellite_entries] = next_open.loc[new_satellite_entries]
        satellite_entry_price.loc[satellite_exits] = np.nan
        positions = target

        nav_rows.append(
            {
                "date": close.index[i + 2].strftime("%Y-%m-%d"),
                "equity": equity,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "gross_exposure": float(positions.abs().sum()),
                "market_exposure": exposure,
                "positions_count": int(positions.ne(0).sum()),
                "core_count": core_count,
                "core_eligible_count": core_eligible_count,
                "core_excluded_count": core_excluded_count,
                "satellite_count": satellite_count,
                "satellite_candidate_count": candidate_count,
                "satellite_rebound_exit_enabled": bool(satellite_rebound_enabled),
                "satellite_rebound_exit_count": int(len(active_satellite_rebound_hits)),
                "satellite_rebound_exit_symbols": ",".join(sorted(active_satellite_rebound_hits)),
            }
        )
        trade_rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "realize_date": close.index[i + 2].strftime("%Y-%m-%d"),
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "positions_count": int(positions.ne(0).sum()),
                "core_count": core_count,
                "core_eligible_count": core_eligible_count,
                "core_excluded_count": core_excluded_count,
                "satellite_count": satellite_count,
                "satellite_candidate_count": candidate_count,
                "satellite_rebound_exit_enabled": bool(satellite_rebound_enabled),
                "satellite_rebound_exit_count": int(len(active_satellite_rebound_hits)),
                "satellite_rebound_exit_symbols": ",".join(sorted(active_satellite_rebound_hits)),
            }
        )

    return (
        pd.DataFrame(nav_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(selection_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest core rank model with strong-pullback satellite overlay.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/core_satellite_overlay")
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--retrain-frequency", type=int, default=20)
    parser.add_argument("--rebalance-frequency", type=int, default=5)
    parser.add_argument("--core-top-n", type=int, default=35)
    parser.add_argument("--satellite-top-n", type=int, default=5)
    parser.add_argument("--core-leverage", type=float, default=0.70)
    parser.add_argument("--satellite-leverage", type=float, default=0.20)
    parser.add_argument("--total-leverage", type=float, default=0.90)
    parser.add_argument("--max-position-weight", type=float, default=0.08)
    parser.add_argument("--min-core-score", type=float, default=None)
    parser.add_argument("--min-satellite-score", type=float, default=None)
    parser.add_argument("--core-min-close", type=float, default=None)
    parser.add_argument("--core-min-avg-amount-20d", type=float, default=None)
    parser.add_argument("--core-max-large-down-days-20", type=int, default=None)
    parser.add_argument("--core-large-down-threshold", type=float, default=-0.045)
    parser.add_argument("--core-exclude-symbols-file", default=None)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--impact-bps", type=float, default=0.7)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--max-buy-open-gap", type=float, default=0.05)
    parser.add_argument("--limit-buffer", type=float, default=0.995)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--market-ma-window", type=int, default=120)
    parser.add_argument("--market-risk-off-drawdown-20d", type=float, default=-0.08)
    parser.add_argument("--market-below-ma-exposure", type=float, default=0.65)
    parser.add_argument("--market-crash-exposure", type=float, default=0.0)
    parser.add_argument("--breadth-filter", action="store_true")
    parser.add_argument("--breadth-ma-window", type=int, default=60)
    parser.add_argument("--breadth-threshold", type=float, default=0.45)
    parser.add_argument("--breadth-below-exposure", type=float, default=0.55)
    parser.add_argument("--breadth-crash-threshold", type=float, default=0.32)
    parser.add_argument("--breadth-crash-exposure", type=float, default=0.20)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--min-close", type=float, default=2.0)
    parser.add_argument("--min-avg-amount-20d", type=float, default=30_000_000.0)
    parser.add_argument("--min-pullback-5d", type=float, default=0.03)
    parser.add_argument("--max-pullback-5d", type=float, default=0.18)
    parser.add_argument("--min-prior-return-20", type=float, default=0.08)
    parser.add_argument("--min-prior-return-60", type=float, default=0.18)
    parser.add_argument("--min-return-20d", type=float, default=-0.12)
    parser.add_argument("--min-return-60d", type=float, default=0.0)
    parser.add_argument("--min-distance-ma60", type=float, default=-0.10)
    parser.add_argument("--max-intraday-return", type=float, default=0.05)
    parser.add_argument("--satellite-rebound-exit-return", type=float, default=None)
    parser.add_argument("--satellite-rebound-exit-scale", type=float, default=0.0)
    parser.add_argument("--satellite-rebound-exit-market-exposure-max", type=float, default=None)
    parser.add_argument("--satellite-rebound-exit-market-exposure-min", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core_exclude_symbols = load_excluded_symbols(args.core_exclude_symbols_file)
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    market_exposure = load_market_exposure(
        args.benchmark,
        close.index,
        ma_window=args.market_ma_window,
        risk_off_drawdown_20d=args.market_risk_off_drawdown_20d,
        below_ma_exposure=args.market_below_ma_exposure,
        crash_exposure=args.market_crash_exposure,
    )
    if args.breadth_filter:
        breadth_exposure = build_breadth_exposure(
            close,
            ma_window=args.breadth_ma_window,
            threshold=args.breadth_threshold,
            below_exposure=args.breadth_below_exposure,
            crash_threshold=args.breadth_crash_threshold,
            crash_exposure=args.breadth_crash_exposure,
        )
        market_exposure = pd.concat([market_exposure, breadth_exposure], axis=1).min(axis=1)

    filter_kwargs = {
        "min_close": args.min_close,
        "min_avg_amount_20d": args.min_avg_amount_20d,
        "min_pullback_5d": args.min_pullback_5d,
        "max_pullback_5d": args.max_pullback_5d,
        "min_prior_return_20": args.min_prior_return_20,
        "min_prior_return_60": args.min_prior_return_60,
        "min_return_20d": args.min_return_20d,
        "min_return_60d": args.min_return_60d,
        "min_distance_ma60": args.min_distance_ma60,
        "max_intraday_return": args.max_intraday_return,
    }
    equity, weights, trades, selections = run_overlay_walk_forward(
        close=close,
        open_px=open_px,
        high=high,
        low=low,
        amount=amount,
        train_days=args.train_days,
        retrain_frequency=args.retrain_frequency,
        rebalance_frequency=args.rebalance_frequency,
        core_top_n=args.core_top_n,
        satellite_top_n=args.satellite_top_n,
        core_leverage=args.core_leverage,
        satellite_leverage=args.satellite_leverage,
        total_leverage=args.total_leverage,
        max_position_weight=args.max_position_weight,
        min_core_score=args.min_core_score,
        min_satellite_score=args.min_satellite_score,
        commission_bps=args.commission_bps,
        impact_bps=args.impact_bps,
        max_buy_open_gap=args.max_buy_open_gap,
        limit_buffer=args.limit_buffer,
        market_exposure=market_exposure,
        initial_capital=args.initial_capital,
        filter_kwargs=filter_kwargs,
        core_min_close=args.core_min_close,
        core_min_avg_amount_20d=args.core_min_avg_amount_20d,
        core_max_large_down_days_20=args.core_max_large_down_days_20,
        core_large_down_threshold=args.core_large_down_threshold,
        core_exclude_symbols=core_exclude_symbols,
        satellite_rebound_exit_return=args.satellite_rebound_exit_return,
        satellite_rebound_exit_scale=args.satellite_rebound_exit_scale,
        satellite_rebound_exit_market_exposure_max=args.satellite_rebound_exit_market_exposure_max,
        satellite_rebound_exit_market_exposure_min=args.satellite_rebound_exit_market_exposure_min,
    )
    if equity.empty:
        raise ValueError("No backtest rows generated. Check train-days and input history length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(output_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(output_dir / "selected_positions.csv", index=False, encoding="utf-8-sig")

    nav = pd.Series(equity["equity"].to_numpy(), index=pd.to_datetime(equity["date"]))
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    total_return = float(nav.iloc[-1] / args.initial_capital - 1.0)
    metrics = {
        "strategy": "core_satellite_overlay",
        "initial_capital": args.initial_capital,
        "final_equity": float(nav.iloc[-1]),
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (252 / max(len(nav), 1)) - 1.0),
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
        "trade_days": int(len(nav)),
        "avg_turnover": float(equity["turnover"].mean()),
        "avg_gross_exposure": float(equity["gross_exposure"].mean()),
        "avg_market_exposure": float(equity["market_exposure"].mean()),
        "avg_positions_count": float(equity["positions_count"].mean()),
        "avg_core_count": float(equity["core_count"].mean()),
        "avg_core_eligible_count": float(equity["core_eligible_count"].mean()),
        "avg_core_excluded_count": float(equity["core_excluded_count"].mean()),
        "avg_satellite_count": float(equity["satellite_count"].mean()),
        "avg_satellite_candidate_count": float(equity["satellite_candidate_count"].mean()),
        "core_top_n": args.core_top_n,
        "satellite_top_n": args.satellite_top_n,
        "core_leverage": args.core_leverage,
        "satellite_leverage": args.satellite_leverage,
        "total_leverage": args.total_leverage,
        "max_position_weight": args.max_position_weight,
        "min_core_score": args.min_core_score,
        "min_satellite_score": args.min_satellite_score,
        "core_risk_filter": {
            "min_close": args.core_min_close,
            "min_avg_amount_20d": args.core_min_avg_amount_20d,
            "max_large_down_days_20": args.core_max_large_down_days_20,
            "large_down_threshold": args.core_large_down_threshold,
            "exclude_symbols_file": args.core_exclude_symbols_file,
            "exclude_symbols_count": len(core_exclude_symbols),
        },
        "filter": filter_kwargs,
        "satellite_rebound_exit": {
            "enabled": args.satellite_rebound_exit_return is not None,
            "return": args.satellite_rebound_exit_return,
            "scale": args.satellite_rebound_exit_scale,
            "market_exposure_max": args.satellite_rebound_exit_market_exposure_max,
            "market_exposure_min": args.satellite_rebound_exit_market_exposure_min,
            "enabled_day_count": int(equity["satellite_rebound_exit_enabled"].sum()),
            "trigger_count": int(equity["satellite_rebound_exit_count"].sum()),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Core + satellite overlay outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
