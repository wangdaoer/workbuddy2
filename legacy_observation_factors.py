from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from execution_rules import (
    LIMIT_DOWN_PRICE_COLUMNS,
    LIMIT_RATE_COLUMNS,
    LIMIT_UP_PRICE_COLUMNS,
    drop_terminal_zero_placeholders,
    normalize_symbol,
)


REQUIRED_PANEL_COLUMNS = ("date", "symbol", "close")
MONEY_FLOW_COLUMNS = (
    "net_money_flow",
    "net_mf_amount",
    "main_net_inflow",
    "money_flow",
)

OBSERVATION_SCORE_WEIGHTS: Mapping[str, float] = {
    "momentum_20": 0.20,
    "momentum_60": 0.15,
    "breakout_distance_20": 0.20,
    "trend_acceleration": 0.15,
    "liquidity_20": 0.10,
    "liquidity_stability_20": 0.10,
    "flow_persistence": 0.10,
    "volatility_20": -0.10,
    "capacity_risk": -0.10,
}
OBSERVATION_SCORE_INPUTS = tuple(OBSERVATION_SCORE_WEIGHTS)
RELEASED_FACTOR_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    *MONEY_FLOW_COLUMNS,
    "position_notional",
    *LIMIT_RATE_COLUMNS,
    *LIMIT_UP_PRICE_COLUMNS,
    *LIMIT_DOWN_PRICE_COLUMNS,
    *OBSERVATION_SCORE_INPUTS,
    "history_eligible",
    "liquidity_eligible",
    "score_eligible",
    "observation_score",
    "limit_up_risk",
    "limit_down_risk",
    "opening_gap_risk",
    "signal_lag_sessions",
    "research_only",
    "trade_instruction",
)


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")

    missing = sorted(set(REQUIRED_PANEL_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    out = drop_terminal_zero_placeholders(df)
    if out[list(REQUIRED_PANEL_COLUMNS)].isna().any().any():
        raise ValueError("required panel columns cannot contain missing values")

    try:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        out["close"] = pd.to_numeric(out["close"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid panel values: {exc}") from exc

    out["symbol"] = out["symbol"].map(normalize_symbol)
    close_values = out["close"].to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or (close_values <= 0.0).any():
        raise ValueError("close must contain finite positive values")

    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate date-symbol rows are not allowed")

    return out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def _rolling_by_symbol(
    out: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
) -> pd.Series:
    grouped = out.groupby("symbol", sort=False)[column]
    return grouped.transform(
        lambda values: getattr(values.rolling(window, min_periods=window), operation)()
    )


def _available_numeric_column(out: pd.DataFrame, column: str) -> bool:
    if column not in out.columns:
        return False
    out[column] = pd.to_numeric(out[column], errors="raise")
    values = out[column].to_numpy(dtype=float, na_value=np.nan)
    return bool(np.isfinite(values).any())


def _rank_observation_score(out: pd.DataFrame) -> pd.Series:
    numerator = pd.Series(0.0, index=out.index)
    denominator = pd.Series(0.0, index=out.index)

    for column, weight in OBSERVATION_SCORE_WEIGHTS.items():
        ranks = out.groupby("date", sort=False)[column].rank(method="average", pct=True)
        usable = ranks.notna()
        numerator = numerator.add(ranks.fillna(0.0) * weight)
        denominator = denominator.add(usable.astype(float) * abs(weight))

    score = numerator.div(denominator.where(denominator.gt(0.0)))
    return score.where(out["history_eligible"])


def _first_numeric_series(
    out: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[str | None, pd.Series]:
    column = next((name for name in candidates if name in out.columns), None)
    if column is None:
        return None, pd.Series(np.nan, index=out.index, dtype=float)
    return column, pd.to_numeric(out[column], errors="coerce")


def _add_execution_risk_annotations(
    out: pd.DataFrame,
    *,
    opening_gap_threshold: float,
    limit_buffer: float = 0.995,
) -> dict[str, object]:
    previous_close = out.groupby("symbol", sort=False)["close"].shift(1)
    board_rate = out["symbol"].map(
        lambda symbol: 0.20 if symbol.startswith(("300", "301")) else 0.10
    )

    rate_column, explicit_rate = _first_numeric_series(out, LIMIT_RATE_COLUMNS)
    explicit_rate = explicit_rate.where(explicit_rate.le(1.0), explicit_rate / 100.0)
    valid_rate = explicit_rate.gt(0.0) & explicit_rate.lt(1.0)
    effective_rate = explicit_rate.where(valid_rate, board_rate)

    up_price_column, explicit_up = _first_numeric_series(
        out, LIMIT_UP_PRICE_COLUMNS
    )
    down_price_column, explicit_down = _first_numeric_series(
        out, LIMIT_DOWN_PRICE_COLUMNS
    )
    valid_up = explicit_up.gt(previous_close)
    valid_down = explicit_down.gt(0.0) & explicit_down.lt(previous_close)

    up_trigger = previous_close * (1.0 + effective_rate * float(limit_buffer))
    down_trigger = previous_close * (1.0 - effective_rate * float(limit_buffer))
    up_trigger = (
        previous_close + (explicit_up - previous_close) * float(limit_buffer)
    ).where(valid_up, up_trigger)
    down_trigger = (
        previous_close - (previous_close - explicit_down) * float(limit_buffer)
    ).where(valid_down, down_trigger)

    comparable = previous_close.notna() & previous_close.gt(0.0)
    limit_up_risk = pd.Series(pd.NA, index=out.index, dtype="boolean")
    limit_down_risk = pd.Series(pd.NA, index=out.index, dtype="boolean")
    limit_up_risk.loc[comparable] = out.loc[comparable, "close"].ge(
        up_trigger.loc[comparable]
    )
    limit_down_risk.loc[comparable] = out.loc[comparable, "close"].le(
        down_trigger.loc[comparable]
    )
    out["limit_up_risk"] = limit_up_risk
    out["limit_down_risk"] = limit_down_risk

    if "open" in out.columns:
        open_values = pd.to_numeric(out["open"], errors="coerce")
        next_open = open_values.groupby(out["symbol"], sort=False).shift(-1)
        gap_comparable = next_open.gt(0.0) & out["close"].gt(0.0)
        opening_gap_risk = pd.Series(pd.NA, index=out.index, dtype="boolean")
        opening_gap_risk.loc[gap_comparable] = (
            next_open.loc[gap_comparable]
            .div(out.loc[gap_comparable, "close"])
            .sub(1.0)
            .gt(float(opening_gap_threshold))
        )
    else:
        opening_gap_risk = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["opening_gap_risk"] = opening_gap_risk

    explicit_both = (valid_rate | valid_up) & (valid_rate | valid_down)
    comparable_explicit = explicit_both.loc[comparable]
    if not comparable_explicit.empty and comparable_explicit.all():
        coverage = "point_in_time_explicit"
    elif comparable_explicit.any():
        coverage = "partial_mixed"
    else:
        coverage = "partial_board_default"

    unsupported = (
        "ST 5% and other exceptional limits are unsupported without "
        "point-in-time limit-rate or limit-price source fields."
    )
    return {
        "limit_status_coverage": coverage,
        "limit_rate_column": rate_column,
        "limit_up_price_column": up_price_column,
        "limit_down_price_column": down_price_column,
        "unsupported_limit_status": unsupported,
        "opening_gap_threshold": float(opening_gap_threshold),
        "opening_gap_evaluation_only": True,
        "risk_availability": {
            "limit_up_risk": bool(comparable.any()),
            "limit_down_risk": bool(comparable.any()),
            "opening_gap_risk": bool(
                "open" in out.columns and opening_gap_risk.notna().any()
            ),
        },
    }


def compute_observation_factors(
    df: pd.DataFrame,
    *,
    opening_gap_threshold: float = 0.03,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = validate_panel(df)
    out.drop(
        columns=["signal_lag_days", "earliest_action_date"],
        errors="ignore",
        inplace=True,
    )
    grouped = out.groupby("symbol", sort=False)

    out["momentum_20"] = grouped["close"].pct_change(20, fill_method=None)
    out["momentum_60"] = grouped["close"].pct_change(60, fill_method=None)
    prior_high = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).max().shift(1)
    )
    out["breakout_distance_20"] = out["close"].div(prior_high) - 1.0

    return_3 = grouped["close"].pct_change(3, fill_method=None)
    return_6 = grouped["close"].pct_change(6, fill_method=None)
    out["trend_acceleration"] = return_3 - return_6 / 2.0

    daily_return = grouped["close"].pct_change(fill_method=None)
    out["volatility_20"] = daily_return.groupby(out["symbol"], sort=False).transform(
        lambda values: values.rolling(20, min_periods=20).std()
    )

    amount_available = _available_numeric_column(out, "amount")
    if amount_available:
        positive_amount = out["amount"].where(out["amount"].gt(0.0))
        out["_positive_amount"] = positive_amount
        trailing_amount = _rolling_by_symbol(out, "_positive_amount", 20, "median")
        trailing_mean = _rolling_by_symbol(out, "_positive_amount", 20, "mean")
        trailing_std = _rolling_by_symbol(out, "_positive_amount", 20, "std")
        coefficient_of_variation = trailing_std.div(trailing_mean.abs().replace(0.0, np.nan))
        out["liquidity_20"] = np.log1p(trailing_amount)
        out["liquidity_stability_20"] = 1.0 / (1.0 + coefficient_of_variation)
        out.drop(columns=["_positive_amount"], inplace=True)
    else:
        trailing_amount = pd.Series(np.nan, index=out.index, dtype=float)
        out["liquidity_20"] = np.nan
        out["liquidity_stability_20"] = np.nan

    money_flow_column = next(
        (column for column in MONEY_FLOW_COLUMNS if _available_numeric_column(out, column)),
        None,
    )
    if money_flow_column is None:
        out["flow_persistence"] = np.nan
    else:
        positive_flow = out[money_flow_column].gt(0.0).where(
            out[money_flow_column].notna()
        )
        out["_positive_flow"] = positive_flow.astype(float)
        out["flow_persistence"] = _rolling_by_symbol(
            out, "_positive_flow", 20, "mean"
        )
        out.drop(columns=["_positive_flow"], inplace=True)

    capacity_available = amount_available and _available_numeric_column(
        out, "position_notional"
    )
    if capacity_available:
        out["capacity_risk"] = out["position_notional"].abs().div(trailing_amount)
    else:
        out["capacity_risk"] = np.nan

    out["history_eligible"] = out[
        [
            "momentum_20",
            "momentum_60",
            "breakout_distance_20",
            "trend_acceleration",
            "volatility_20",
        ]
    ].notna().all(axis=1)
    out["liquidity_eligible"] = out["liquidity_20"].notna()
    out["score_eligible"] = out["history_eligible"]
    out["observation_score"] = _rank_observation_score(out)
    execution_risk_metadata = _add_execution_risk_annotations(
        out,
        opening_gap_threshold=opening_gap_threshold,
    )

    out["signal_lag_sessions"] = 1
    out["research_only"] = True
    out["trade_instruction"] = False

    factor_availability = {
        factor: bool(factor in out and out[factor].notna().any())
        for factor in OBSERVATION_SCORE_INPUTS
    }
    latest_date = out["date"].max()
    latest_rows = out["date"].eq(latest_date)
    latest_row_count = int(latest_rows.sum())
    factor_latest_dates: dict[str, str | None] = {}
    factor_current_coverage: dict[str, float] = {}
    for factor in factor_availability:
        usable = out[factor].notna() if factor in out else pd.Series(False, index=out.index)
        latest_usable_date = out.loc[usable, "date"].max()
        factor_latest_dates[factor] = (
            latest_usable_date.strftime("%Y-%m-%d")
            if not pd.isna(latest_usable_date)
            else None
        )
        factor_current_coverage[factor] = (
            float(usable.loc[latest_rows].mean()) if latest_row_count else 0.0
        )
    factor_current_availability = {
        factor: coverage > 0.0
        for factor, coverage in factor_current_coverage.items()
    }
    factor_current_complete = {
        factor: coverage == 1.0
        for factor, coverage in factor_current_coverage.items()
    }
    factor_input_latest_dates: dict[str, str | None] = {}
    for column in ("amount", *MONEY_FLOW_COLUMNS):
        if column not in out:
            continue
        usable = pd.to_numeric(out[column], errors="coerce").notna()
        latest_usable_date = out.loc[usable, "date"].max()
        factor_input_latest_dates[column] = (
            latest_usable_date.strftime("%Y-%m-%d")
            if not pd.isna(latest_usable_date)
            else None
        )
    money_flow_current_coverage = (
        float(out.loc[latest_rows, money_flow_column].notna().mean())
        if money_flow_column is not None and latest_row_count
        else 0.0
    )
    metadata: dict[str, object] = {
        "factor_availability": factor_availability,
        "factor_latest_dates": factor_latest_dates,
        "factor_current_availability": factor_current_availability,
        "factor_current_complete": factor_current_complete,
        "factor_current_coverage": factor_current_coverage,
        "factor_input_latest_dates": factor_input_latest_dates,
        "money_flow_column": money_flow_column,
        "money_flow_current_available": money_flow_current_coverage > 0.0,
        "money_flow_current_coverage": money_flow_current_coverage,
        "score_inputs": list(OBSERVATION_SCORE_INPUTS),
        "signal_lag_sessions": 1,
        "signal_timing": (
            "close_t_actionable_no_earlier_than_next_observed_trading_session"
        ),
        "research_only": True,
        "trade_instruction": False,
        **execution_risk_metadata,
    }
    released_columns = [
        column for column in dict.fromkeys(RELEASED_FACTOR_COLUMNS) if column in out
    ]
    return out.loc[:, released_columns].copy(), metadata


def _full_period_returns(panel: pd.DataFrame) -> pd.Series:
    grouped = panel.groupby("symbol", sort=False)["close"]
    first = grouped.first()
    last = grouped.last()
    returns = last.div(first) - 1.0
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def audit_static_universe(
    selected: pd.DataFrame,
    broad: pd.DataFrame,
) -> dict[str, object]:
    selected_panel = validate_panel(selected)
    broad_panel = validate_panel(broad)
    selected_returns = _full_period_returns(selected_panel)
    broad_returns = _full_period_returns(broad_panel)
    if selected_returns.empty or broad_returns.empty:
        raise ValueError("static-universe audit requires full-period returns")

    percentiles = sorted(
        float(broad_returns.le(value).mean()) for value in selected_returns
    )
    median_percentile = float(np.median(percentiles))
    winner_threshold = float(broad_returns.quantile(0.75))
    broad_winners = set(broad_returns.index[broad_returns.ge(winner_threshold)])
    selected_symbols = set(selected_returns.index)
    overlap = sorted(selected_symbols.intersection(broad_winners))

    percentile_summary = {
        "minimum": float(np.min(percentiles)),
        "median": median_percentile,
        "maximum": float(np.max(percentiles)),
    }
    return {
        "selected_full_period_return_median": float(selected_returns.median()),
        "broad_full_period_return_median": float(broad_returns.median()),
        "selected_return_percentiles": percentiles,
        "selected_return_percentile_distribution": percentile_summary,
        "selected_median_percentile": median_percentile,
        "broad_winner_return_threshold": winner_threshold,
        "selected_winner_overlap_symbols": overlap,
        "selected_winner_overlap_count": len(overlap),
        "selected_winner_overlap_rate": len(overlap) / len(selected_symbols),
        "lookthrough_suspect": median_percentile >= 0.75,
        "research_only": True,
        "trade_instruction": False,
    }
