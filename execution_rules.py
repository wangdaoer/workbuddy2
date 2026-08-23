"""Shared next-open labels and open-execution constraints."""

from __future__ import annotations

import re

import pandas as pd


ZERO_PLACEHOLDER_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
LIMIT_RATE_COLUMNS = ("limit_rate", "price_limit_rate", "daily_limit_rate")
LIMIT_UP_PRICE_COLUMNS = ("limit_up_price", "up_limit_price")
LIMIT_DOWN_PRICE_COLUMNS = ("limit_down_price", "down_limit_price")


def normalize_symbol(value: object) -> str:
    """Return a validated, zero-padded six-digit stock identifier."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"invalid stock identifier: {value!r}")
    try:
        if bool(pd.isna(value)):
            raise ValueError(f"invalid stock identifier: {value!r}")
    except (TypeError, ValueError):
        raise ValueError(f"invalid stock identifier: {value!r}") from None

    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,6})(?:\.0+)?", text)
    if match is None:
        raise ValueError(f"invalid stock identifier: {value!r}")
    return match.group(1).zfill(6)


def drop_terminal_zero_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Drop only trailing rows that are provable all-zero market-data placeholders."""
    required = {"date", "symbol", "close"}
    if not required.issubset(df.columns):
        return df.copy()

    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    zero_close = close.eq(0.0)
    if not zero_close.any():
        return out

    if not set(ZERO_PLACEHOLDER_COLUMNS).issubset(out.columns):
        raise ValueError(
            "zero close rows may be ignored only as terminal all-zero "
            "OHLCV/amount placeholders"
        )

    companions = out.loc[zero_close, ZERO_PLACEHOLDER_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if companions.isna().any().any() or not companions.eq(0.0).all(axis=None):
        raise ValueError(
            "zero close rows may be ignored only as terminal all-zero "
            "OHLCV/amount placeholders"
        )

    try:
        symbols = out["symbol"].map(normalize_symbol)
        dates = pd.to_datetime(out["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "zero close rows may be ignored only as terminal all-zero "
            "OHLCV/amount placeholders"
        ) from exc

    check = pd.DataFrame(
        {"symbol": symbols, "date": dates, "zero_close": zero_close},
        index=out.index,
    ).sort_values(["symbol", "date"], kind="mergesort")
    for _, group in check.groupby("symbol", sort=False):
        zero_positions = group["zero_close"].to_numpy().nonzero()[0]
        if not len(zero_positions):
            continue
        first_zero = int(zero_positions[0])
        if first_zero == 0 or not group["zero_close"].iloc[first_zero:].all():
            raise ValueError(
                "zero close rows may be ignored only as terminal all-zero "
                "OHLCV/amount placeholders"
            )

    return out.loc[~zero_close].copy()


def limit_thresholds(
    columns: pd.Index,
    limit_rate_row: pd.Series | None = None,
) -> pd.Series:
    values = {}
    for code in columns:
        text = normalize_symbol(code)
        values[code] = 0.20 if text.startswith(("300", "301")) else 0.10
    thresholds = pd.Series(values, dtype=float)
    if limit_rate_row is None:
        return thresholds

    explicit = pd.to_numeric(
        limit_rate_row.reindex(columns), errors="coerce"
    ).astype(float)
    explicit = explicit.where(explicit.le(1.0), explicit / 100.0)
    explicit = explicit.where(explicit.gt(0.0) & explicit.lt(1.0))
    return explicit.fillna(thresholds)


def next_open_return_label(
    open_px: pd.DataFrame,
    max_abs_daily_return: float | None = None,
) -> pd.DataFrame:
    return forward_open_return_label(
        open_px,
        horizon_days=1,
        max_abs_daily_return=max_abs_daily_return,
    )


def forward_open_return_label(
    open_px: pd.DataFrame,
    horizon_days: int,
    max_abs_daily_return: float | None = None,
) -> pd.DataFrame:
    """Return from next open to the open after ``horizon_days`` sessions."""

    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    label = open_px.shift(-(horizon_days + 1)) / (open_px.shift(-1) + 1e-12) - 1.0
    if max_abs_daily_return is not None and max_abs_daily_return > 0:
        daily_equivalent = (1.0 + label).pow(1.0 / horizon_days) - 1.0
        label = label.mask(daily_equivalent.abs().gt(float(max_abs_daily_return)))
    return label


def apply_open_constraints(
    current: pd.Series,
    target: pd.Series,
    open_row: pd.Series,
    prev_close_row: pd.Series,
    max_buy_open_gap: float | None,
    limit_buffer: float,
    block_limit_up_buys: bool = True,
    block_limit_down_sells: bool = True,
    limit_rate_row: pd.Series | None = None,
    limit_up_price_row: pd.Series | None = None,
    limit_down_price_row: pd.Series | None = None,
) -> pd.Series:
    adjusted, _ = apply_open_constraints_with_diagnostics(
        current=current,
        target=target,
        open_row=open_row,
        prev_close_row=prev_close_row,
        max_buy_open_gap=max_buy_open_gap,
        limit_buffer=limit_buffer,
        block_limit_up_buys=block_limit_up_buys,
        block_limit_down_sells=block_limit_down_sells,
        limit_rate_row=limit_rate_row,
        limit_up_price_row=limit_up_price_row,
        limit_down_price_row=limit_down_price_row,
    )
    return adjusted


def open_constraint_masks(
    current: pd.Series,
    target: pd.Series,
    open_row: pd.Series,
    prev_close_row: pd.Series,
    max_buy_open_gap: float | None,
    limit_buffer: float,
    block_limit_up_buys: bool = True,
    block_limit_down_sells: bool = True,
    limit_rate_row: pd.Series | None = None,
    limit_up_price_row: pd.Series | None = None,
    limit_down_price_row: pd.Series | None = None,
) -> dict[str, pd.Series]:
    current = current.reindex(target.index).fillna(0.0)
    open_aligned = open_row.reindex(target.index)
    prev_close = prev_close_row.reindex(target.index)
    open_gap = open_aligned / (prev_close + 1e-12) - 1.0
    limit = limit_thresholds(target.index, limit_rate_row=limit_rate_row)
    up_trigger = prev_close * (1.0 + limit * float(limit_buffer))
    down_trigger = prev_close * (1.0 - limit * float(limit_buffer))

    if limit_up_price_row is not None:
        explicit_up = pd.to_numeric(
            limit_up_price_row.reindex(target.index), errors="coerce"
        )
        usable_up = explicit_up.gt(prev_close)
        explicit_up_trigger = prev_close + (
            explicit_up - prev_close
        ) * float(limit_buffer)
        up_trigger = explicit_up_trigger.where(usable_up, up_trigger)
    if limit_down_price_row is not None:
        explicit_down = pd.to_numeric(
            limit_down_price_row.reindex(target.index), errors="coerce"
        )
        usable_down = explicit_down.gt(0.0) & explicit_down.lt(prev_close)
        explicit_down_trigger = prev_close - (
            prev_close - explicit_down
        ) * float(limit_buffer)
        down_trigger = explicit_down_trigger.where(usable_down, down_trigger)

    increasing = target.gt(current)
    decreasing = target.lt(current)

    return {
        "blocked_limit_up_buys": (
            increasing & open_aligned.ge(up_trigger)
            if block_limit_up_buys
            else pd.Series(False, index=target.index)
        ),
        "blocked_limit_down_sells": (
            decreasing & open_aligned.le(down_trigger)
            if block_limit_down_sells
            else pd.Series(False, index=target.index)
        ),
        "blocked_open_gap_buys": (
            increasing & open_gap.gt(float(max_buy_open_gap))
            if max_buy_open_gap is not None
            else pd.Series(False, index=target.index)
        ),
    }


def apply_open_constraints_with_diagnostics(
    current: pd.Series,
    target: pd.Series,
    open_row: pd.Series,
    prev_close_row: pd.Series,
    max_buy_open_gap: float | None,
    limit_buffer: float,
    block_limit_up_buys: bool = True,
    block_limit_down_sells: bool = True,
    limit_rate_row: pd.Series | None = None,
    limit_up_price_row: pd.Series | None = None,
    limit_down_price_row: pd.Series | None = None,
) -> tuple[pd.Series, dict[str, int]]:
    masks = open_constraint_masks(
        current=current,
        target=target,
        open_row=open_row,
        prev_close_row=prev_close_row,
        max_buy_open_gap=max_buy_open_gap,
        limit_buffer=limit_buffer,
        block_limit_up_buys=block_limit_up_buys,
        block_limit_down_sells=block_limit_down_sells,
        limit_rate_row=limit_rate_row,
        limit_up_price_row=limit_up_price_row,
        limit_down_price_row=limit_down_price_row,
    )
    blocked = pd.Series(False, index=target.index)
    for mask in masks.values():
        blocked |= mask

    current_aligned = current.reindex(target.index).fillna(0.0)
    adjusted = target.where(~blocked, current_aligned)

    executable_long_buys = (
        target.gt(current_aligned)
        & target.gt(0.0)
        & current_aligned.ge(0.0)
        & ~blocked
    )
    target_gross_budget = float(target.abs().sum())
    target_long_budget = float(target.clip(lower=0.0).sum())
    exceeds_gross_budget = float(adjusted.abs().sum()) > target_gross_budget
    exceeds_long_budget = float(adjusted.clip(lower=0.0).sum()) > target_long_budget
    if executable_long_buys.any() and (exceeds_gross_budget or exceeds_long_budget):
        base = adjusted.copy()
        base.loc[executable_long_buys] = current_aligned.loc[executable_long_buys]
        gross_capacity = max(target_gross_budget - float(base.abs().sum()), 0.0)
        long_capacity = max(target_long_budget - float(base.clip(lower=0.0).sum()), 0.0)
        requested = target.loc[executable_long_buys] - current_aligned.loc[executable_long_buys]
        requested_total = float(requested.sum())
        scale = (
            min(1.0, gross_capacity / requested_total, long_capacity / requested_total)
            if requested_total > 0.0
            else 0.0
        )
        adjusted.loc[executable_long_buys] = (
            current_aligned.loc[executable_long_buys] + requested * scale
        )

    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    counts["blocked_orders_total"] = int(blocked.sum())
    tolerance = 1e-12
    if float(adjusted.abs().sum()) > target_gross_budget + tolerance:
        counts["gross_budget_overrun"] = 1
    if float(adjusted.clip(lower=0.0).sum()) > target_long_budget + tolerance:
        counts["long_budget_overrun"] = 1
    return adjusted, counts
