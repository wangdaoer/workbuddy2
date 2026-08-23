"""Research-only forward shadow account for A-share pre-live drafts.

The module consumes yesterday's ``prelive_order_draft_YYYYMMDD.csv`` and
simulates whole-lot execution at today's open.  It never connects to a broker
and never emits broker-importable orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from execution_rules import (
    LIMIT_DOWN_PRICE_COLUMNS,
    LIMIT_RATE_COLUMNS,
    LIMIT_UP_PRICE_COLUMNS,
    normalize_symbol,
    open_constraint_masks,
)
from panel_io import panel_columns, read_panel, write_panel_atomic


SCHEMA_VERSION = 1
LOT_SIZE = 100
RESEARCH_METADATA = {
    "research_only": True,
    "trade_instruction": False,
    "broker_order_allowed": False,
}
PREVIOUS_CLOSE_COLUMNS = ("prev_close", "pre_close", "previous_close")
SNAPSHOT_PATTERN = re.compile(r"prelive_account_shadow_snapshot_(\d{8})\.json$")
DRAFT_PATTERN = re.compile(r"prelive_order_draft_(\d{8})\.csv$")


@dataclass(frozen=True)
class ShadowAccountConfig:
    """Execution assumptions for the ten-thousand-yuan shadow account."""

    initial_cash: float = 10_000.0
    lot_size: int = LOT_SIZE
    # 费用表（A 股实盘标准，见 MOS 规则 13）：佣金万三(3bps，买卖双侧) + 法定印花税万分之五(5bps，仅卖出侧)
    commission_bps: float = 3.0
    minimum_commission: float = 5.0
    sell_stamp_tax_bps: float = 5.0
    # 滑点/市场冲击为独立于手续费的执行假设（非费用），保持 5bps
    slippage_bps: float = 5.0
    max_buy_open_gap: float | None = 0.03
    limit_buffer: float = 1.0
    block_limit_up_buys: bool = True
    block_limit_down_sells: bool = True

    def validate(self) -> None:
        if not math.isfinite(float(self.initial_cash)) or self.initial_cash <= 0.0:
            raise ValueError("initial_cash must be a finite positive number")
        if self.lot_size != LOT_SIZE:
            raise ValueError("A-share shadow account lot_size must be exactly 100")
        for name in (
            "commission_bps",
            "minimum_commission",
            "sell_stamp_tax_bps",
            "slippage_bps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.max_buy_open_gap is not None:
            gap = float(self.max_buy_open_gap)
            if not math.isfinite(gap) or gap < 0.0:
                raise ValueError("max_buy_open_gap must be non-negative or None")
        if not math.isfinite(float(self.limit_buffer)) or not 0.0 < self.limit_buffer <= 1.0:
            raise ValueError("limit_buffer must be in (0, 1]")


POSITION_COLUMNS = [
    "date",
    "symbol",
    "shares",
    "avg_cost",
    "close",
    "market_value",
    "unrealized_pnl",
    "weight",
    "research_only",
    "trade_instruction",
    "broker_order_allowed",
]
TRADE_COLUMNS = [
    "trade_id",
    "date",
    "source_draft_date",
    "sequence",
    "symbol",
    "side",
    "opening_shares",
    "target_shares",
    "requested_shares",
    "filled_shares",
    "reference_open",
    "execution_price",
    "gross_amount",
    "commission",
    "stamp_tax",
    "net_cash_change",
    "realized_pnl",
    "research_only",
    "trade_instruction",
    "broker_order_allowed",
]
HISTORY_COLUMNS = [
    "date",
    "source_draft_date",
    "previous_snapshot_date",
    "opening_cash",
    "ending_cash",
    "market_value",
    "equity",
    "daily_pnl",
    "daily_return",
    "total_pnl",
    "total_return",
    "trade_count",
    "buy_count",
    "sell_count",
    "turnover_value",
    "commission",
    "stamp_tax",
    "realized_pnl_today",
    "realized_pnl_cumulative",
    "unrealized_pnl",
    "blocked_order_count",
    "cash_constrained_order_count",
    "research_only",
    "trade_instruction",
    "broker_order_allowed",
]


def _coerce_config(
    config: ShadowAccountConfig | Mapping[str, Any] | None,
) -> ShadowAccountConfig:
    if config is None:
        result = ShadowAccountConfig()
    elif isinstance(config, ShadowAccountConfig):
        result = config
    elif isinstance(config, Mapping):
        values = dict(config)
        if "min_commission" in values and "minimum_commission" not in values:
            values["minimum_commission"] = values.pop("min_commission")
        if "stamp_tax_bps" in values and "sell_stamp_tax_bps" not in values:
            values["sell_stamp_tax_bps"] = values.pop("stamp_tax_bps")
        result = ShadowAccountConfig(**values)
    else:
        raise TypeError("config must be ShadowAccountConfig, a mapping, or None")
    result.validate()
    return result


def load_shadow_config(path: str | Path) -> ShadowAccountConfig:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("shadow-account config must be a mapping")
    section = loaded.get("prelive_account_shadow", loaded)
    if not isinstance(section, dict):
        raise ValueError("prelive_account_shadow config must be a mapping")
    return _coerce_config(section)


def _parse_date(value: str | pd.Timestamp) -> pd.Timestamp:
    text = str(value).strip()
    try:
        parsed = pd.to_datetime(text, format="%Y%m%d" if re.fullmatch(r"\d{8}", text) else None)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid execution date: {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"invalid execution date: {value!r}")
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _optional_float(value: str) -> float | None:
    if str(value).strip().lower() in {"none", "null", "off", "disabled"}:
        return None
    return float(value)


def _date_text(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _date_token(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def _money(value: float) -> float:
    return round(float(value) + 1e-12, 2)


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite: {value!r}")
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False, double_precision=10))


def load_order_draft(
    path: Path,
    execution_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Load and validate a previous-day whole-lot target draft."""

    execution = _parse_date(execution_date)
    if not path.exists():
        raise FileNotFoundError(path)
    draft = pd.read_csv(path, dtype=str, low_memory=False)
    required = {"symbol", "target_shares_lot"}
    missing = sorted(required - set(draft.columns))
    if missing:
        raise ValueError(f"order draft missing required columns: {missing}")

    try:
        draft["symbol"] = draft["symbol"].map(normalize_symbol)
    except ValueError as exc:
        raise ValueError(f"order draft contains an invalid symbol: {exc}") from exc
    if draft["symbol"].duplicated().any():
        duplicates = draft.loc[draft["symbol"].duplicated(False), "symbol"].unique().tolist()
        raise ValueError(f"order draft contains duplicate symbols: {duplicates}")

    targets = pd.to_numeric(draft["target_shares_lot"], errors="coerce")
    if targets.isna().any() or not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise ValueError("target_shares_lot must contain finite numbers")
    if not np.allclose(targets, np.round(targets), atol=1e-9):
        raise ValueError("target_shares_lot must contain integer share counts")
    draft["target_shares_lot"] = np.round(targets).astype(int)
    if draft["target_shares_lot"].lt(0).any():
        raise ValueError("target_shares_lot cannot be negative")
    if draft["target_shares_lot"].mod(LOT_SIZE).ne(0).any():
        raise ValueError("target_shares_lot must be a multiple of 100 shares")

    for column, forbidden in (
        ("trade_instruction", True),
        ("broker_order_allowed", True),
    ):
        if column in draft and draft[column].map(_truthy).eq(forbidden).any():
            raise ValueError(f"order draft violates research-only boundary: {column}=true")

    dates: list[pd.Timestamp] = []
    match = DRAFT_PATTERN.fullmatch(path.name)
    if match:
        dates.append(_parse_date(match.group(1)))
    if "asof_date" in draft and not draft.empty:
        asof_dates = draft["asof_date"].dropna().astype(str).str.strip()
        asof_dates = asof_dates[asof_dates.ne("")].map(_parse_date).unique().tolist()
        dates.extend(pd.Timestamp(value) for value in asof_dates)
    unique_dates = sorted(set(dates))
    if len(unique_dates) != 1:
        raise ValueError(
            "order draft date must be unambiguous in its filename or asof_date column"
        )
    draft_date = unique_dates[0]
    if draft_date >= execution:
        raise ValueError("order draft must be dated before the execution date")
    draft = draft.reset_index(drop=True)
    draft["draft_order"] = np.arange(len(draft), dtype=int)
    return draft, draft_date


def _first_existing(columns: pd.Index | list[str], candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def load_execution_market(
    path: Path,
    execution_date: str | pd.Timestamp,
    relevant_symbols: set[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load today's open/close and point-in-time previous closes via panel_io."""

    execution = _parse_date(execution_date)
    if not path.exists():
        raise FileNotFoundError(path)
    available = panel_columns(path)
    required = {"date", "symbol", "open", "close"}
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(f"price panel missing required columns: {missing}")
    optional = [
        name
        for name in (
            *PREVIOUS_CLOSE_COLUMNS,
            *LIMIT_RATE_COLUMNS,
            *LIMIT_UP_PRICE_COLUMNS,
            *LIMIT_DOWN_PRICE_COLUMNS,
        )
        if name in available
    ]
    selected_columns = ["date", "symbol", "open", "close", *dict.fromkeys(optional)]
    panel = read_panel(
        path,
        columns=selected_columns,
        dtype={"symbol": "string"},
        parse_dates=["date"],
        low_memory=False,
    )
    if panel.empty:
        raise ValueError("price panel is empty")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    try:
        panel["symbol"] = panel["symbol"].map(normalize_symbol)
    except ValueError as exc:
        raise ValueError(f"price panel contains an invalid symbol: {exc}") from exc
    if not panel["date"].eq(execution).any():
        raise ValueError(f"price panel has no rows for execution date {_date_text(execution)}")
    previous_dates = panel.loc[panel["date"].lt(execution), "date"]
    previous_market_date = previous_dates.max() if not previous_dates.empty else None

    relevant = panel.loc[
        panel["symbol"].isin(relevant_symbols) & panel["date"].le(execution)
    ].copy()
    if relevant.duplicated(["date", "symbol"]).any():
        raise ValueError("price panel contains duplicate date + symbol rows")
    current = relevant.loc[relevant["date"].eq(execution)].copy()
    current = current.set_index("symbol", drop=True).sort_index()

    missing_symbols = sorted(relevant_symbols - set(current.index))
    if missing_symbols:
        raise ValueError(f"price panel is missing current-day prices for: {missing_symbols}")
    for column in ("open", "close"):
        current[column] = pd.to_numeric(current[column], errors="coerce")
        values = current[column].to_numpy(dtype=float)
        if len(values) and (not np.isfinite(values).all() or (values <= 0.0).any()):
            raise ValueError(f"current-day {column} must contain finite positive prices")

    previous_close = pd.Series(np.nan, index=current.index, dtype=float)
    explicit_column = _first_existing(current.columns, PREVIOUS_CLOSE_COLUMNS)
    if explicit_column is not None:
        previous_close = pd.to_numeric(current[explicit_column], errors="coerce")
    historical = relevant.loc[relevant["date"].lt(execution)].copy()
    if not historical.empty:
        historical["close"] = pd.to_numeric(historical["close"], errors="coerce")
        historical = historical.sort_values(["symbol", "date"], kind="mergesort")
        inferred = historical.groupby("symbol", sort=False)["close"].last()
        previous_close = previous_close.fillna(inferred.reindex(current.index))

    for column in optional:
        current[column] = pd.to_numeric(current[column], errors="coerce")
    current.attrs["previous_market_date"] = previous_market_date
    return current, previous_close.astype(float)


def _snapshot_date(path: Path) -> pd.Timestamp | None:
    match = SNAPSHOT_PATTERN.fullmatch(path.name)
    return _parse_date(match.group(1)) if match else None


def _load_opening_state(
    output_dir: Path,
    execution_date: pd.Timestamp,
    config: ShadowAccountConfig,
) -> dict[str, Any]:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in output_dir.glob("prelive_account_shadow_snapshot_*.json"):
        date_value = _snapshot_date(path)
        if date_value is not None:
            candidates.append((date_value, path))
    future = [date_value for date_value, _ in candidates if date_value > execution_date]
    if future:
        raise ValueError("cannot rebuild a historical date while later account snapshots exist")
    previous = [(date_value, path) for date_value, path in candidates if date_value < execution_date]
    if not previous:
        return {
            "previous_snapshot_date": None,
            "previous_snapshot_path": None,
            "cash": _money(config.initial_cash),
            "equity": _money(config.initial_cash),
            "initial_cash": _money(config.initial_cash),
            "positions": {},
            "realized_pnl_cumulative": 0.0,
            "commission_cumulative": 0.0,
            "stamp_tax_cumulative": 0.0,
        }

    previous_date, previous_path = max(previous, key=lambda item: item[0])
    try:
        payload = json.loads(previous_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid previous snapshot JSON: {previous_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"previous snapshot must contain a JSON object: {previous_path}")
    if any(payload.get(key) != value for key, value in RESEARCH_METADATA.items()):
        raise ValueError("previous snapshot does not preserve the research-only boundary")
    if _parse_date(payload.get("date")) != previous_date:
        raise ValueError("previous snapshot date does not match its filename")

    initial_cash = _number(payload.get("initial_cash"), "snapshot initial_cash")
    if not math.isclose(initial_cash, config.initial_cash, abs_tol=0.005):
        raise ValueError("initial_cash cannot change after the account has been initialized")
    cash = _number(payload.get("cash"), "snapshot cash")
    equity = _number(payload.get("equity"), "snapshot equity")
    if cash < -1e-9:
        raise ValueError("previous snapshot cash cannot be negative")

    positions: dict[str, dict[str, float | int]] = {}
    raw_positions = payload.get("positions", [])
    if not isinstance(raw_positions, list):
        raise ValueError("previous snapshot positions must be a list")
    for row in raw_positions:
        if not isinstance(row, dict):
            raise ValueError("previous snapshot contains an invalid position row")
        symbol = normalize_symbol(row.get("symbol"))
        shares = int(_number(row.get("shares"), "snapshot shares"))
        avg_cost = _number(row.get("avg_cost", 0.0), "snapshot avg_cost")
        if shares <= 0 or shares % LOT_SIZE != 0 or avg_cost < 0.0:
            raise ValueError("previous snapshot positions must be positive 100-share lots")
        if symbol in positions:
            raise ValueError(f"previous snapshot contains duplicate symbol: {symbol}")
        positions[symbol] = {"shares": shares, "avg_cost": avg_cost}
    return {
        "previous_snapshot_date": _date_text(previous_date),
        "previous_snapshot_path": previous_path,
        "cash": _money(cash),
        "equity": _money(equity),
        "initial_cash": _money(initial_cash),
        "positions": positions,
        "realized_pnl_cumulative": _number(
            payload.get("realized_pnl_cumulative", 0.0),
            "snapshot realized_pnl_cumulative",
        ),
        "commission_cumulative": _number(
            payload.get("commission_cumulative", 0.0),
            "snapshot commission_cumulative",
        ),
        "stamp_tax_cumulative": _number(
            payload.get("stamp_tax_cumulative", 0.0),
            "snapshot stamp_tax_cumulative",
        ),
    }


def _commission(gross_amount: float, config: ShadowAccountConfig) -> float:
    if gross_amount <= 0.0:
        return 0.0
    variable = gross_amount * config.commission_bps / 10_000.0
    return _money(max(variable, config.minimum_commission))


def _buy_cost(quantity: int, price: float, config: ShadowAccountConfig) -> tuple[float, float, float]:
    gross = _money(quantity * price)
    commission = _commission(gross, config)
    return gross, commission, _money(gross + commission)


def _affordable_buy_quantity(
    desired_shares: int,
    cash: float,
    price: float,
    config: ShadowAccountConfig,
) -> int:
    high = desired_shares // LOT_SIZE
    low = 0
    while low < high:
        middle = (low + high + 1) // 2
        _, _, required_cash = _buy_cost(middle * LOT_SIZE, price, config)
        if required_cash <= cash + 1e-9:
            low = middle
        else:
            high = middle - 1
    return low * LOT_SIZE


def _optional_market_row(
    current_market: pd.DataFrame,
    candidates: tuple[str, ...],
) -> pd.Series | None:
    column = _first_existing(current_market.columns, candidates)
    return current_market[column] if column is not None else None


def simulate_shadow_day(
    draft: pd.DataFrame,
    draft_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    current_market: pd.DataFrame,
    previous_close: pd.Series,
    opening_state: Mapping[str, Any],
    config: ShadowAccountConfig,
) -> dict[str, Any]:
    """Simulate one day, always processing executable sells before buys."""

    positions = {
        symbol: {"shares": int(row["shares"]), "avg_cost": float(row["avg_cost"])}
        for symbol, row in opening_state["positions"].items()
    }
    targets = dict(zip(draft["symbol"], draft["target_shares_lot"], strict=True))
    symbols = sorted(set(positions) | {symbol for symbol, shares in targets.items() if shares > 0})
    current_shares = pd.Series(
        {symbol: int(positions.get(symbol, {}).get("shares", 0)) for symbol in symbols},
        dtype=float,
    )
    target_shares = pd.Series(
        {symbol: int(targets.get(symbol, 0)) for symbol in symbols},
        dtype=float,
    )
    open_row = current_market["open"].reindex(symbols)
    previous_row = previous_close.reindex(symbols)

    buy_changes = target_shares.gt(current_shares)
    sell_changes = target_shares.lt(current_shares)
    needs_previous_close = (
        buy_changes & (config.block_limit_up_buys or config.max_buy_open_gap is not None)
    ) | (sell_changes & config.block_limit_down_sells)
    if previous_row.loc[needs_previous_close].isna().any():
        missing = previous_row.loc[needs_previous_close & previous_row.isna()].index.tolist()
        raise ValueError(f"previous close is required for execution constraints: {missing}")
    if (previous_row.loc[needs_previous_close] <= 0.0).any():
        raise ValueError("previous close must be positive when execution constraints are enabled")

    masks = open_constraint_masks(
        current=current_shares,
        target=target_shares,
        open_row=open_row,
        prev_close_row=previous_row,
        max_buy_open_gap=config.max_buy_open_gap,
        limit_buffer=config.limit_buffer,
        block_limit_up_buys=config.block_limit_up_buys,
        block_limit_down_sells=config.block_limit_down_sells,
        limit_rate_row=_optional_market_row(current_market, LIMIT_RATE_COLUMNS),
        limit_up_price_row=_optional_market_row(current_market, LIMIT_UP_PRICE_COLUMNS),
        limit_down_price_row=_optional_market_row(current_market, LIMIT_DOWN_PRICE_COLUMNS),
    )
    blocked = pd.Series(False, index=target_shares.index)
    for mask in masks.values():
        blocked |= mask.reindex(blocked.index).fillna(False)
    executable_target = target_shares.where(~blocked, current_shares).astype(int)

    order_events: list[dict[str, Any]] = []
    for symbol in blocked[blocked].index:
        reasons = [name for name, mask in masks.items() if bool(mask.get(symbol, False))]
        side = "BUY" if target_shares[symbol] > current_shares[symbol] else "SELL"
        order_events.append(
            {
                "symbol": symbol,
                "side": side,
                "status": "blocked_by_execution_constraint",
                "requested_shares": int(abs(target_shares[symbol] - current_shares[symbol])),
                "filled_shares": 0,
                "reasons": reasons,
                **RESEARCH_METADATA,
            }
        )

    cash = _money(opening_state["cash"])
    opening_cash = cash
    trades: list[dict[str, Any]] = []
    realized_today = 0.0
    sequence = 0
    slippage_rate = config.slippage_bps / 10_000.0

    # All sale proceeds are available before any buy affordability check.
    sale_symbols = sorted(executable_target[executable_target.lt(current_shares)].index)
    for symbol in sale_symbols:
        opening_quantity = int(current_shares[symbol])
        target_quantity = int(executable_target[symbol])
        quantity = opening_quantity - target_quantity
        reference_open = float(current_market.at[symbol, "open"])
        execution_price = round(reference_open * (1.0 - slippage_rate), 6)
        gross = _money(quantity * execution_price)
        commission = _commission(gross, config)
        stamp_tax = _money(gross * config.sell_stamp_tax_bps / 10_000.0)
        proceeds = _money(gross - commission - stamp_tax)
        avg_cost = float(positions[symbol]["avg_cost"])
        realized = _money(proceeds - quantity * avg_cost)
        realized_today = _money(realized_today + realized)
        cash = _money(cash + proceeds)
        if cash < -1e-9:
            raise ValueError("sale costs would make account cash negative")
        if target_quantity:
            positions[symbol]["shares"] = target_quantity
        else:
            del positions[symbol]
        sequence += 1
        trades.append(
            {
                "trade_id": f"{_date_token(execution_date)}-{sequence:04d}-{symbol}-SELL",
                "date": _date_text(execution_date),
                "source_draft_date": _date_text(draft_date),
                "sequence": sequence,
                "symbol": symbol,
                "side": "SELL",
                "opening_shares": opening_quantity,
                "target_shares": target_quantity,
                "requested_shares": quantity,
                "filled_shares": quantity,
                "reference_open": reference_open,
                "execution_price": execution_price,
                "gross_amount": gross,
                "commission": commission,
                "stamp_tax": stamp_tax,
                "net_cash_change": proceeds,
                "realized_pnl": realized,
                **RESEARCH_METADATA,
            }
        )

    draft_priority = draft.set_index("symbol")["draft_order"].to_dict()
    buy_symbols = sorted(
        executable_target[executable_target.gt(current_shares)].index,
        key=lambda symbol: (int(draft_priority.get(symbol, 10**9)), symbol),
    )
    for symbol in buy_symbols:
        opening_quantity = int(current_shares[symbol])
        target_quantity = int(executable_target[symbol])
        desired = target_quantity - opening_quantity
        reference_open = float(current_market.at[symbol, "open"])
        execution_price = round(reference_open * (1.0 + slippage_rate), 6)
        quantity = _affordable_buy_quantity(desired, cash, execution_price, config)
        if quantity == 0:
            order_events.append(
                {
                    "symbol": symbol,
                    "side": "BUY",
                    "status": "blocked_by_insufficient_cash",
                    "requested_shares": desired,
                    "filled_shares": 0,
                    "reasons": ["insufficient_cash"],
                    **RESEARCH_METADATA,
                }
            )
            continue
        gross, commission, required_cash = _buy_cost(quantity, execution_price, config)
        cash = _money(cash - required_cash)
        if cash < -1e-9:
            raise AssertionError("buy affordability check allowed a cash overdraft")
        old_position = positions.get(symbol, {"shares": 0, "avg_cost": 0.0})
        old_shares = int(old_position["shares"])
        total_cost = old_shares * float(old_position["avg_cost"]) + required_cash
        new_shares = old_shares + quantity
        positions[symbol] = {
            "shares": new_shares,
            "avg_cost": round(total_cost / new_shares, 6),
        }
        sequence += 1
        trades.append(
            {
                "trade_id": f"{_date_token(execution_date)}-{sequence:04d}-{symbol}-BUY",
                "date": _date_text(execution_date),
                "source_draft_date": _date_text(draft_date),
                "sequence": sequence,
                "symbol": symbol,
                "side": "BUY",
                "opening_shares": opening_quantity,
                "target_shares": target_quantity,
                "requested_shares": desired,
                "filled_shares": quantity,
                "reference_open": reference_open,
                "execution_price": execution_price,
                "gross_amount": gross,
                "commission": commission,
                "stamp_tax": 0.0,
                "net_cash_change": -required_cash,
                "realized_pnl": 0.0,
                **RESEARCH_METADATA,
            }
        )
        if quantity < desired:
            order_events.append(
                {
                    "symbol": symbol,
                    "side": "BUY",
                    "status": "partially_filled_insufficient_cash",
                    "requested_shares": desired,
                    "filled_shares": quantity,
                    "reasons": ["insufficient_cash"],
                    **RESEARCH_METADATA,
                }
            )

    position_rows: list[dict[str, Any]] = []
    market_value = 0.0
    unrealized_pnl = 0.0
    for symbol in sorted(positions):
        shares = int(positions[symbol]["shares"])
        close = float(current_market.at[symbol, "close"])
        value = _money(shares * close)
        unrealized = _money((close - float(positions[symbol]["avg_cost"])) * shares)
        market_value = _money(market_value + value)
        unrealized_pnl = _money(unrealized_pnl + unrealized)
        position_rows.append(
            {
                "date": _date_text(execution_date),
                "symbol": symbol,
                "shares": shares,
                "avg_cost": round(float(positions[symbol]["avg_cost"]), 6),
                "close": close,
                "market_value": value,
                "unrealized_pnl": unrealized,
                "weight": 0.0,
                **RESEARCH_METADATA,
            }
        )
    equity = _money(cash + market_value)
    for row in position_rows:
        row["weight"] = round(row["market_value"] / equity, 8) if equity > 0.0 else 0.0

    positions_frame = pd.DataFrame(position_rows, columns=POSITION_COLUMNS)
    trades_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    commission_today = _money(sum(float(row["commission"]) for row in trades))
    stamp_tax_today = _money(sum(float(row["stamp_tax"]) for row in trades))
    turnover = _money(sum(float(row["gross_amount"]) for row in trades))
    previous_equity = float(opening_state["equity"])
    initial_cash = float(opening_state["initial_cash"])
    realized_cumulative = _money(
        float(opening_state["realized_pnl_cumulative"]) + realized_today
    )
    metrics = {
        "opening_cash": opening_cash,
        "ending_cash": cash,
        "market_value": market_value,
        "equity": equity,
        "daily_pnl": _money(equity - previous_equity),
        "daily_return": round(equity / previous_equity - 1.0, 10) if previous_equity else 0.0,
        "total_pnl": _money(equity - initial_cash),
        "total_return": round(equity / initial_cash - 1.0, 10),
        "trade_count": len(trades),
        "buy_count": sum(row["side"] == "BUY" for row in trades),
        "sell_count": sum(row["side"] == "SELL" for row in trades),
        "turnover_value": turnover,
        "commission": commission_today,
        "stamp_tax": stamp_tax_today,
        "realized_pnl_today": realized_today,
        "realized_pnl_cumulative": realized_cumulative,
        "unrealized_pnl": unrealized_pnl,
        "blocked_order_count": sum(
            row["status"] == "blocked_by_execution_constraint" for row in order_events
        ),
        "cash_constrained_order_count": sum(
            "insufficient_cash" in row["status"] for row in order_events
        ),
        "position_count": len(position_rows),
        "commission_cumulative": _money(
            float(opening_state["commission_cumulative"]) + commission_today
        ),
        "stamp_tax_cumulative": _money(
            float(opening_state["stamp_tax_cumulative"]) + stamp_tax_today
        ),
    }
    return {
        "positions": positions_frame,
        "trades": trades_frame,
        "order_events": order_events,
        "metrics": metrics,
    }


def _history_frame(
    history_path: Path,
    execution_date: pd.Timestamp,
    row: dict[str, Any],
) -> pd.DataFrame:
    if history_path.exists():
        existing = pd.read_csv(history_path, dtype={"date": str}, low_memory=False)
        missing = sorted(set(HISTORY_COLUMNS) - set(existing.columns))
        if missing:
            raise ValueError(f"existing account history has an incompatible schema: {missing}")
        dates = existing["date"].map(_parse_date)
        if dates.gt(execution_date).any():
            raise ValueError("cannot rebuild history while later history rows exist")
        existing = existing.loc[~dates.eq(execution_date), HISTORY_COLUMNS].copy()
    else:
        existing = pd.DataFrame(columns=HISTORY_COLUMNS)
    current_row = pd.DataFrame([row], columns=HISTORY_COLUMNS)
    combined = (
        current_row
        if existing.empty
        else pd.concat([existing, current_row], ignore_index=True)
    )
    combined["date"] = combined["date"].map(lambda value: _date_text(_parse_date(value)))
    combined = combined.sort_values("date", kind="mergesort").reset_index(drop=True)
    if combined["date"].duplicated().any():
        raise AssertionError("account history contains duplicate dates after idempotent upsert")
    return combined


def _summary_markdown(summary: Mapping[str, Any], trades: pd.DataFrame) -> str:
    metrics = summary["metrics"]
    lines = [
        f"# 一万元A股整手前向影子账户 {_date_text(_parse_date(summary['date']))}",
        "",
        "- 账户性质：仅研究用途的前向影子账户，不连接券商、不下单。",
        "- `research_only=true`",
        "- `trade_instruction=false`",
        "- `broker_order_allowed=false`",
        "- 执行顺序：当日开盘先卖出，后买入；仅模拟100股整手。",
        "",
        "## 当日摘要",
        "",
        f"- 期初现金：{metrics['opening_cash']:.2f} 元",
        f"- 期末现金：{metrics['ending_cash']:.2f} 元",
        f"- 持仓市值：{metrics['market_value']:.2f} 元",
        f"- 账户权益：{metrics['equity']:.2f} 元",
        f"- 当日损益：{metrics['daily_pnl']:.2f} 元",
        f"- 累计损益：{metrics['total_pnl']:.2f} 元",
        f"- 成交笔数：{metrics['trade_count']}",
        f"- 佣金：{metrics['commission']:.2f} 元",
        f"- 卖出印花税：{metrics['stamp_tax']:.2f} 元",
        f"- 执行约束阻止：{metrics['blocked_order_count']} 笔",
        f"- 资金约束影响：{metrics['cash_constrained_order_count']} 笔",
        "",
        "## 模拟成交",
        "",
    ]
    if trades.empty:
        lines.append("当日无模拟成交。")
    else:
        preview = trades[
            [
                "sequence",
                "symbol",
                "side",
                "filled_shares",
                "execution_price",
                "gross_amount",
                "commission",
                "stamp_tax",
            ]
        ].rename(
            columns={
                "sequence": "顺序",
                "symbol": "证券代码",
                "side": "方向",
                "filled_shares": "成交股数",
                "execution_price": "模拟成交价",
                "gross_amount": "成交金额",
                "commission": "佣金",
                "stamp_tax": "印花税",
            }
        )
        lines.append(preview.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## 边界声明",
            "",
            "本报告只用于研究、复盘和人工检查，不构成交易指令，也不得导入券商交易端。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_prelive_account_shadow(
    draft_path: str | Path,
    panel_path: str | Path,
    output_dir: str | Path,
    execution_date: str | pd.Timestamp,
    config: ShadowAccountConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one idempotent research-only shadow-account day."""

    cfg = _coerce_config(config)
    execution = _parse_date(execution_date)
    draft_source = Path(draft_path)
    panel_source = Path(panel_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    draft, draft_date = load_order_draft(draft_source, execution)
    opening_state = _load_opening_state(destination, execution, cfg)
    relevant_symbols = set(opening_state["positions"]) | set(
        draft.loc[draft["target_shares_lot"].gt(0), "symbol"]
    )
    current_market, previous_close = load_execution_market(
        panel_source,
        execution,
        relevant_symbols,
    )
    previous_market_date = current_market.attrs.get("previous_market_date")
    if previous_market_date is not None and pd.Timestamp(previous_market_date) != draft_date:
        raise ValueError(
            "order draft date must equal the latest panel trading date before execution: "
            f"expected {_date_text(pd.Timestamp(previous_market_date))}, "
            f"got {_date_text(draft_date)}"
        )
    simulation = simulate_shadow_day(
        draft=draft,
        draft_date=draft_date,
        execution_date=execution,
        current_market=current_market,
        previous_close=previous_close,
        opening_state=opening_state,
        config=cfg,
    )

    token = _date_token(execution)
    snapshot_path = destination / f"prelive_account_shadow_snapshot_{token}.json"
    positions_path = destination / f"prelive_account_shadow_positions_{token}.csv"
    trades_path = destination / f"prelive_account_shadow_trades_{token}.csv"
    history_path = destination / "prelive_account_shadow_history.csv"
    summary_path = destination / "prelive_account_shadow_summary.json"
    report_path = destination / "prelive_account_shadow_summary.md"
    previous_path = opening_state["previous_snapshot_path"]
    provenance = {
        "draft_file": draft_source.name,
        "draft_sha256": _sha256(draft_source),
        "panel_file": panel_source.name,
        "panel_sha256": _sha256(panel_source),
        "previous_snapshot_file": previous_path.name if previous_path else None,
        "previous_snapshot_sha256": _sha256(previous_path) if previous_path else None,
    }
    calculation_payload = {
        "date": _date_text(execution),
        "source_draft_date": _date_text(draft_date),
        "config": asdict(cfg),
        "provenance": provenance,
    }
    calculation_id = hashlib.sha256(
        json.dumps(calculation_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    metrics = simulation["metrics"]
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "account_type": "a_share_whole_lot_forward_shadow",
        "date": _date_text(execution),
        "source_draft_date": _date_text(draft_date),
        "previous_snapshot_date": opening_state["previous_snapshot_date"],
        "state_rebuild_rule": "latest_snapshot_strictly_before_execution_date",
        "execution_order": "sell_then_buy_at_current_open",
        "initial_cash": opening_state["initial_cash"],
        "cash": metrics["ending_cash"],
        "market_value": metrics["market_value"],
        "equity": metrics["equity"],
        "realized_pnl_cumulative": metrics["realized_pnl_cumulative"],
        "commission_cumulative": metrics["commission_cumulative"],
        "stamp_tax_cumulative": metrics["stamp_tax_cumulative"],
        "positions": _json_records(simulation["positions"]),
        "trades": _json_records(simulation["trades"]),
        "order_events": simulation["order_events"],
        "metrics": metrics,
        "config": asdict(cfg),
        "provenance": provenance,
        "calculation_id": calculation_id,
        **RESEARCH_METADATA,
    }
    history_row = {
        "date": _date_text(execution),
        "source_draft_date": _date_text(draft_date),
        "previous_snapshot_date": opening_state["previous_snapshot_date"] or "",
        **{name: metrics[name] for name in HISTORY_COLUMNS if name in metrics},
        **RESEARCH_METADATA,
    }
    history = _history_frame(history_path, execution, history_row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "account_type": "a_share_whole_lot_forward_shadow",
        "date": _date_text(execution),
        "source_draft_date": _date_text(draft_date),
        "previous_snapshot_date": opening_state["previous_snapshot_date"],
        "calculation_id": calculation_id,
        "metrics": metrics,
        "history_rows": int(len(history)),
        "artifacts": {
            "snapshot": snapshot_path.name,
            "positions": positions_path.name,
            "trades": trades_path.name,
            "history": history_path.name,
            "summary": summary_path.name,
            "report_cn": report_path.name,
        },
        "boundary_cn": "仅研究用途，不连接券商，不生成或提交券商订单。",
        **RESEARCH_METADATA,
    }

    write_panel_atomic(simulation["positions"], positions_path)
    write_panel_atomic(simulation["trades"], trades_path)
    write_panel_atomic(history, history_path)
    _write_text_atomic(
        snapshot_path,
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(report_path, _summary_markdown(summary, simulation["trades"]))
    return {
        "snapshot": snapshot,
        "positions": simulation["positions"],
        "trades": simulation["trades"],
        "history": history,
        "summary": summary,
        "artifacts": {key: str(destination / name) for key, name in summary["artifacts"].items()},
    }


def run_shadow_account(
    draft_path: str | Path,
    panel_path: str | Path,
    output_dir: str | Path,
    execution_date: str | pd.Timestamp,
    config: ShadowAccountConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-friendly short alias for :func:`run_prelive_account_shadow`."""

    return run_prelive_account_shadow(
        draft_path=draft_path,
        panel_path=panel_path,
        output_dir=output_dir,
        execution_date=execution_date,
        config=config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a research-only 10,000-yuan A-share whole-lot shadow account."
    )
    parser.add_argument("--draft", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--initial-cash", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--commission-bps", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--minimum-commission",
        "--min-commission",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--slippage-bps", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--max-buy-open-gap",
        type=_optional_float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--limit-buffer", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--allow-limit-up-buys", action="store_true")
    parser.add_argument("--allow-limit-down-sells", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_shadow_config(args.config) if args.config else ShadowAccountConfig()
    values = asdict(base)
    for name in (
        "initial_cash",
        "commission_bps",
        "minimum_commission",
        "sell_stamp_tax_bps",
        "slippage_bps",
        "max_buy_open_gap",
        "limit_buffer",
    ):
        if hasattr(args, name):
            values[name] = getattr(args, name)
    if args.allow_limit_up_buys:
        values["block_limit_up_buys"] = False
    if args.allow_limit_down_sells:
        values["block_limit_down_sells"] = False
    result = run_prelive_account_shadow(
        draft_path=args.draft,
        panel_path=args.panel,
        output_dir=args.output_dir,
        execution_date=args.execution_date,
        config=values,
    )
    print(json.dumps({"summary": result["summary"], "artifacts": result["artifacts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
