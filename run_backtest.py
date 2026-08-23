"""High-risk research backtest scaffold for project 019ecbc7-7b5b-7c13-a80a-3f328010bab8."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from execution_rules import (
    LIMIT_DOWN_PRICE_COLUMNS,
    LIMIT_RATE_COLUMNS,
    LIMIT_UP_PRICE_COLUMNS,
    apply_open_constraints_with_diagnostics,
    drop_terminal_zero_placeholders,
)
from panel_io import read_panel

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`."
    ) from exc


@dataclass
class StrategyConfig:
    initial_capital: float
    start_date: Optional[str]
    end_date: Optional[str]
    universe_top_n_by_mcap: int
    universe_dynamic_top_n: int
    universe_selection_mode: str
    universe_selection_min_history: int
    max_abs_daily_return: float
    execution_model: str
    block_limit_up_buys: bool
    block_limit_down_sells: bool
    limit_buffer: float
    max_buy_open_gap: Optional[float]
    short_window: int
    long_window: int
    breakout_window: int
    acceleration_window: int
    breakout_threshold: float
    vol_window: int
    signal_type: str
    top_n_long: int
    top_n_short: int
    signal_only_long_breakout: bool
    breakout_up_weight: float
    breakout_dn_weight: float
    trend_weight: float
    accel_weight: float
    signal_noise_clip: float
    rebalance_frequency: int
    long_exposure: float
    short_exposure: float
    max_position_weight: float
    use_net_neutral: bool
    allow_short: bool
    leverage: float
    max_drawdown: float
    drawdown_cooldown_days: int
    target_annualized_vol: float
    vol_lookback: int
    min_signal_non_na_ratio: float
    commission_bps: float
    impact_bps: float
    output_dir: str
    universe_selection_lag_days: int = 0
    stamp_tax_bps: float = 0.0

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "StrategyConfig":
        return cls(
            initial_capital=cfg["initial_capital"],
            start_date=cfg.get("start_date"),
            end_date=cfg.get("end_date"),
            universe_top_n_by_mcap=cfg.get("universe", {}).get("top_n_by_mcap", 100),
            universe_dynamic_top_n=int(cfg.get("universe", {}).get("dynamic_top_n", 0)),
            universe_selection_mode=cfg.get("universe", {}).get("selection_mode", "none"),
            universe_selection_min_history=int(cfg.get("universe", {}).get("selection_min_history", 120)),
            universe_selection_lag_days=max(
                0, int(cfg.get("universe", {}).get("selection_lag_days", 0))
            ),
            max_abs_daily_return=float(cfg.get("data", {}).get("max_abs_daily_return", 0.35)),
            execution_model=cfg.get("execution", {}).get("model", "close_to_close"),
            block_limit_up_buys=bool(cfg.get("execution", {}).get("block_limit_up_buys", False)),
            block_limit_down_sells=bool(cfg.get("execution", {}).get("block_limit_down_sells", False)),
            limit_buffer=float(cfg.get("execution", {}).get("limit_buffer", 0.995)),
            max_buy_open_gap=cfg.get("execution", {}).get("max_buy_open_gap"),
            short_window=cfg["signal"]["short_window"],
            long_window=cfg["signal"]["long_window"],
            breakout_window=cfg["signal"].get("breakout_window", cfg["signal"]["long_window"]),
            acceleration_window=cfg["signal"].get(
                "acceleration_window", cfg["signal"]["short_window"]
            ),
            breakout_threshold=cfg["signal"].get("breakout_threshold", 0.01),
            vol_window=cfg["signal"]["vol_window"],
            signal_type=cfg["signal"]["signal_type"],
            top_n_long=cfg["signal"]["top_n_long"],
            top_n_short=cfg["signal"]["top_n_short"],
            signal_only_long_breakout=cfg["signal"].get("only_long", False),
            breakout_up_weight=float(cfg["signal"].get("breakout_up_weight", 0.55)),
            breakout_dn_weight=float(cfg["signal"].get("breakout_dn_weight", -0.35)),
            trend_weight=float(cfg["signal"].get("trend_weight", 0.25)),
            accel_weight=float(cfg["signal"].get("accel_weight", 0.25)),
            signal_noise_clip=float(cfg["signal"].get("signal_noise_clip", 5.0)),
            rebalance_frequency=cfg["portfolio"]["rebalance_frequency"],
            long_exposure=cfg["portfolio"]["long_exposure"],
            short_exposure=cfg["portfolio"]["short_exposure"],
            max_position_weight=cfg["portfolio"]["max_position_weight"],
            use_net_neutral=cfg["portfolio"]["use_net_neutral"],
            allow_short=cfg["portfolio"]["allow_short"],
            leverage=cfg["portfolio"]["leverage"],
            max_drawdown=cfg["risk"]["max_drawdown"],
            drawdown_cooldown_days=cfg["risk"]["drawdown_cooldown_days"],
            target_annualized_vol=cfg["risk"]["target_annualized_vol"],
            vol_lookback=cfg["risk"]["vol_lookback"],
            min_signal_non_na_ratio=cfg["risk"]["min_signal_non_na_ratio"],
            commission_bps=cfg["cost"]["commission_bps"],
            impact_bps=cfg["cost"]["impact_bps"],
            stamp_tax_bps=cfg["cost"].get("stamp_tax_bps", 0.0),
            output_dir=cfg["output"]["output_dir"],
        )


def load_config(path: Path) -> StrategyConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return StrategyConfig.from_dict(raw)


def clean_symbol(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else text


def prepare_prices(
    panel: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    *,
    strict_validation: bool = True,
) -> pd.DataFrame:
    df = panel.copy()
    required = {"date", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if strict_validation:
        df = drop_terminal_zero_placeholders(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = np.nan
    if "amount" not in df.columns:
        df["amount"] = np.nan

    for c in ["close", "open", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["amount"] = df["amount"].fillna(df["volume"] * df["close"])
    df["symbol"] = df["symbol"].map(clean_symbol)

    df = df.dropna(subset=["date", "symbol", "close"]).copy()
    df = df[
        df["close"].gt(0)
        & df["open"].gt(0)
        & df["high"].gt(0)
        & df["low"].gt(0)
        & df["volume"].gt(0)
        & df["amount"].gt(0)
    ].copy()
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError("No rows after date filtering.")

    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_prices(data_path: Path, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    return prepare_prices(
        read_panel(data_path),
        start,
        end,
        strict_validation=False,
    )


def pivot_prices(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="symbol", values=value_col).sort_index()


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0] - 1
    trading_periods = len(equity) - 1
    return (1 + total) ** (252 / trading_periods) - 1


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.expanding().max()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def sharpe_like(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = daily_returns - risk_free / 252
    sigma = excess.std(ddof=1)
    if sigma <= 1e-12:
        return 0.0
    return float(excess.mean() / sigma * np.sqrt(252))


class BacktestEngine:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.equity = float(cfg.initial_capital)
        self.nav = []
        self.date_idx = []
        self.drawdown_series = []
        self.trade_log = []
        self.position_gross_history = []
        self.turnover_history = []
        self.execution_constraint_counts = {
            "blocked_limit_up_buys": 0,
            "blocked_limit_down_sells": 0,
            "blocked_open_gap_buys": 0,
            "blocked_orders_total": 0,
        }
        self.signal_cache: Optional[pd.DataFrame] = None
        self.universe_cache: Optional[pd.DataFrame] = None

    def _record_execution_constraint_counts(self, counts: dict[str, int]) -> None:
        for name, count in counts.items():
            self.execution_constraint_counts[name] = (
                self.execution_constraint_counts.get(name, 0) + count
            )

    def _generate_signal(self, close: pd.DataFrame, idx: int, trade_dates: pd.Index) -> pd.Series:
        if idx < self.cfg.long_window:
            return pd.Series(dtype=float)

        date = trade_dates[idx]
        close_slice = close.iloc[: idx + 1]
        if self.signal_cache is not None:
            score = self.signal_cache.iloc[idx].copy()
        else:
            signal_type = self.cfg.signal_type.lower().strip()
            if signal_type == "trend_momentum_ratio":
                score = self._signal_trend_momentum_ratio(close_slice)
            elif signal_type == "trend_breakout_accel":
                score = self._signal_trend_breakout_accel(close_slice)
            elif signal_type == "mean_reversion_mr":
                score = self._signal_mean_reversion(close_slice)
            else:
                raise ValueError(f"Unsupported signal_type: {self.cfg.signal_type}")

        if self.universe_cache is not None:
            allowed = self.universe_cache.iloc[idx].reindex(score.index).fillna(False)
            score = score.where(allowed)

        # Keep only symbols with enough non-na coverage
        min_ratio = self.cfg.min_signal_non_na_ratio
        min_count = int(len(close_slice) * min_ratio)
        enough = close_slice.count().ge(min_count)
        score = score.where(enough)

        signal_non_na_ratio = score.notna().mean()
        if signal_non_na_ratio < self.cfg.min_signal_non_na_ratio:
            return pd.Series(dtype=float)

        return score

    def _zscore(self, s: pd.Series) -> pd.Series:
        s2 = s.dropna()
        if s2.empty:
            return pd.Series(dtype=float)
        mean = s2.mean()
        std = s2.std(ddof=0)
        if std <= 0 or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        out = (s - mean) / (std + 1e-12)
        return out.fillna(0.0)

    def _zscore_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        mean = frame.mean(axis=1, skipna=True)
        std = frame.std(axis=1, ddof=0, skipna=True)
        return frame.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0).fillna(0.0)

    def _precompute_signal_panel(self, close: pd.DataFrame) -> pd.DataFrame | None:
        signal_type = self.cfg.signal_type.lower().strip()
        if signal_type == "trend_momentum_ratio":
            short_ma = close.rolling(self.cfg.short_window).mean()
            long_ma = close.rolling(self.cfg.long_window).mean()
            vol = close.pct_change(fill_method=None).rolling(self.cfg.vol_window).std()
            momentum = (short_ma - long_ma) / (long_ma + 1e-12)
            return self._zscore_frame(momentum / (vol + 1e-12))

        if signal_type == "trend_breakout_accel":
            roll_high = close.rolling(self.cfg.breakout_window).max().shift(1)
            roll_low = close.rolling(self.cfg.breakout_window).min().shift(1)
            breakout_up = close / (roll_high + 1e-12) - 1.0
            breakout_dn = close / (roll_low + 1e-12) - 1.0

            fast_ma = close.rolling(self.cfg.short_window).mean()
            slow_ma = close.rolling(self.cfg.long_window).mean()
            trend = fast_ma / (slow_ma + 1e-12) - 1.0
            accel = close.pct_change(
                self.cfg.acceleration_window, fill_method=None
            ).rolling(self.cfg.acceleration_window).mean()

            score = (
                self._zscore_frame(breakout_up.fillna(0.0)) * self.cfg.breakout_up_weight
                + self._zscore_frame(breakout_dn.fillna(0.0)) * self.cfg.breakout_dn_weight
                + self._zscore_frame(trend.fillna(0.0)) * self.cfg.trend_weight
                + self._zscore_frame(accel) * self.cfg.accel_weight
            )

            long_breakout = breakout_up >= self.cfg.breakout_threshold
            short_breakout = breakout_dn <= -self.cfg.breakout_threshold
            if self.cfg.signal_only_long_breakout:
                score = score.where(long_breakout)
            else:
                score = score.where(long_breakout | short_breakout)
            clip = abs(self.cfg.signal_noise_clip)
            return score.clip(lower=-clip, upper=clip)

        if signal_type == "mean_reversion_mr":
            ma = close.rolling(self.cfg.long_window).mean()
            vol = close.pct_change(fill_method=None).rolling(self.cfg.vol_window).std()
            zret = self._zscore_frame((close - ma) / (ma + 1e-12))
            return -zret / (vol.fillna(vol.median(axis=1), axis=0) + 1e-12)

        raise ValueError(f"Unsupported signal_type: {self.cfg.signal_type}")

    def _precompute_universe_panel(self, close: pd.DataFrame, amount: pd.DataFrame) -> pd.DataFrame | None:
        top_n = self.cfg.universe_dynamic_top_n
        if top_n <= 0:
            return None
        mode = self.cfg.universe_selection_mode.lower().strip()
        if mode not in {"high_return", "momentum"}:
            return None

        returns = close.pct_change(fill_method=None)
        ret_20 = close.pct_change(20, fill_method=None)
        ret_60 = close.pct_change(60, fill_method=None)
        prev_high_20 = close.rolling(20).max().shift(1)
        breakout_20 = close / (prev_high_20 + 1e-12) - 1.0
        vol_20 = returns.rolling(20).std()
        liquidity = np.log1p(amount.replace(0, np.nan).rolling(20).median())

        score = (
            ret_60.rank(axis=1, pct=True) * 0.25
            + ret_20.rank(axis=1, pct=True) * 0.30
            + breakout_20.rank(axis=1, pct=True) * 0.20
            + vol_20.rank(axis=1, pct=True) * 0.10
            + liquidity.rank(axis=1, pct=True) * 0.15
        )
        enough_history = close.notna().rolling(self.cfg.universe_selection_min_history).sum().ge(
            self.cfg.universe_selection_min_history
        )
        score = score.where(enough_history)
        rank = score.rank(axis=1, ascending=False, method="first")
        membership = rank.le(top_n)
        lag_days = self.cfg.universe_selection_lag_days
        if lag_days:
            membership = membership.shift(lag_days, fill_value=False).astype(bool)
        return membership

    def _clean_price_matrix(self, close: pd.DataFrame) -> pd.DataFrame:
        threshold = self.cfg.max_abs_daily_return
        if threshold <= 0:
            return close
        returns = close.pct_change(fill_method=None)
        bad_jump = returns.abs().gt(threshold)
        return close.mask(bad_jump)

    def _apply_execution_constraints(
        self,
        current: pd.Series,
        target: pd.Series,
        open_row: pd.Series,
        prev_close_row: pd.Series,
        limit_rate_row: pd.Series | None = None,
        limit_up_price_row: pd.Series | None = None,
        limit_down_price_row: pd.Series | None = None,
    ) -> tuple[pd.Series, dict[str, int]]:
        return apply_open_constraints_with_diagnostics(
            current=current,
            target=target,
            open_row=open_row,
            prev_close_row=prev_close_row,
            max_buy_open_gap=self.cfg.max_buy_open_gap,
            limit_buffer=self.cfg.limit_buffer,
            block_limit_up_buys=self.cfg.block_limit_up_buys,
            block_limit_down_sells=self.cfg.block_limit_down_sells,
            limit_rate_row=limit_rate_row,
            limit_up_price_row=limit_up_price_row,
            limit_down_price_row=limit_down_price_row,
        )

    def _signal_trend_momentum_ratio(self, close_slice: pd.DataFrame) -> pd.Series:
        close_now = close_slice.iloc[-1]
        short_ma = close_slice.rolling(self.cfg.short_window).mean().iloc[-1]
        long_ma = close_slice.rolling(self.cfg.long_window).mean().iloc[-1]
        vol = close_slice.pct_change(fill_method=None).rolling(self.cfg.vol_window).std().iloc[-1]
        momentum = (short_ma - long_ma) / (long_ma + 1e-12)
        score = momentum / (vol + 1e-12)
        return self._zscore(score)

    def _signal_trend_breakout_accel(self, close_slice: pd.DataFrame) -> pd.Series:
        close_now = close_slice.iloc[-1]

        req_window = max(self.cfg.breakout_window, self.cfg.acceleration_window * 2, self.cfg.short_window, 2)
        if len(close_slice) < req_window:
            return pd.Series(dtype=float)

        roll_high = close_slice.rolling(self.cfg.breakout_window).max().iloc[-1]
        roll_low = close_slice.rolling(self.cfg.breakout_window).min().iloc[-1]
        prev_high = close_slice.rolling(self.cfg.breakout_window).max().iloc[-2]
        prev_low = close_slice.rolling(self.cfg.breakout_window).min().iloc[-2]

        breakout_up = (close_now - prev_high) / (prev_high + 1e-12)
        breakout_dn = (close_now - prev_low) / (prev_low + 1e-12)

        fast_ma = close_slice.rolling(self.cfg.short_window).mean().iloc[-1]
        slow_ma = close_slice.rolling(self.cfg.long_window).mean().iloc[-1]
        trend = fast_ma / (slow_ma + 1e-12) - 1.0

        accel = close_slice.pct_change(
            self.cfg.acceleration_window, fill_method=None
        ).rolling(
            self.cfg.acceleration_window
        ).mean().iloc[-1]
        accel = self._zscore(accel)

        score = (
            self._zscore(breakout_up.fillna(0.0)) * self.cfg.breakout_up_weight
            + self._zscore(breakout_dn.fillna(0.0)) * self.cfg.breakout_dn_weight
            + self._zscore(trend.fillna(0.0)) * self.cfg.trend_weight
            + accel * self.cfg.accel_weight
        )

        long_breakout = breakout_up >= self.cfg.breakout_threshold
        short_breakout = breakout_dn <= -self.cfg.breakout_threshold
        if self.cfg.signal_only_long_breakout:
            score = score.where(long_breakout)
        else:
            score = score.where(long_breakout | short_breakout)
        # Boost directional preference for strong breakouts and keep symmetry
        clip = abs(self.cfg.signal_noise_clip)
        return score.clip(lower=-clip, upper=clip)

    def _signal_mean_reversion(self, close_slice: pd.DataFrame) -> pd.Series:
        close_now = close_slice.iloc[-1]
        req_window = max(self.cfg.long_window, self.cfg.vol_window, 2)
        if len(close_slice) < req_window:
            return pd.Series(dtype=float)

        ma = close_slice.rolling(self.cfg.long_window).mean().iloc[-1]
        vol = close_slice.pct_change(fill_method=None).rolling(self.cfg.vol_window).std().iloc[-1]
        zret = self._zscore((close_now - ma) / (ma + 1e-12))
        return -zret / (vol.fillna(vol.median()) + 1e-12)

    def _cap_and_redistribute(self, raw_weights: pd.Series, exposure: float) -> pd.Series:
        if raw_weights.empty or exposure <= 0:
            return pd.Series(dtype=float)

        base = raw_weights.abs().replace([np.inf, -np.inf], np.nan).dropna()
        if base.empty or base.sum() <= 0:
            return pd.Series(dtype=float)

        position_cap = max(float(self.cfg.max_position_weight), 0.0)
        target_total = min(float(exposure), position_cap * len(base))
        if target_total <= 0:
            return pd.Series(0.0, index=base.index, dtype=float)

        weights = pd.Series(0.0, index=base.index, dtype=float)
        active = base.copy()
        remaining = target_total
        tolerance = 1e-12
        while not active.empty and remaining > tolerance:
            active_total = float(active.sum())
            proposed = (
                active / active_total * remaining
                if active_total > 0
                else pd.Series(remaining / len(active), index=active.index)
            )
            newly_capped = proposed >= position_cap - tolerance
            if not newly_capped.any():
                weights.loc[active.index] = proposed
                remaining = 0.0
                break

            capped_index = proposed.index[newly_capped]
            weights.loc[capped_index] = position_cap
            remaining = max(remaining - position_cap * len(capped_index), 0.0)
            active = active.drop(index=capped_index)

        return weights.clip(lower=0.0, upper=position_cap)

    def _enforce_final_position_cap(self, weights: pd.Series) -> pd.Series:
        clean = weights.replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean[clean.abs() > 1e-15]
        if clean.empty:
            return pd.Series(dtype=float)

        long_weights = clean[clean > 0]
        short_weights = -clean[clean < 0]
        long_total = float(long_weights.sum())
        short_total = float(short_weights.sum())

        if self.cfg.use_net_neutral:
            if long_weights.empty or short_weights.empty:
                return pd.Series(dtype=float)
            feasible_long = min(
                long_total, self.cfg.max_position_weight * len(long_weights)
            )
            feasible_short = min(
                short_total, self.cfg.max_position_weight * len(short_weights)
            )
            balanced_total = min(feasible_long, feasible_short)
            long_total = balanced_total
            short_total = balanced_total

        capped = pd.Series(0.0, index=clean.index, dtype=float)
        if not long_weights.empty:
            capped.loc[long_weights.index] = self._cap_and_redistribute(
                long_weights, long_total
            )
        if not short_weights.empty:
            capped.loc[short_weights.index] = -self._cap_and_redistribute(
                short_weights, short_total
            )
        capped = capped[capped.abs() > 1e-15]

        if not capped.empty and (
            capped.abs() > self.cfg.max_position_weight + 1e-12
        ).any():
            raise RuntimeError("Final target weight exceeds max_position_weight")
        return capped

    def _target_weights(self, signal: pd.Series) -> pd.Series:
        if signal.empty:
            return pd.Series(dtype=float)

        candidates = signal.dropna()
        if candidates.empty:
            return pd.Series(dtype=float)

        long_side = candidates.nlargest(self.cfg.top_n_long)
        short_side = candidates.nsmallest(self.cfg.top_n_short)

        symbols = []
        weights = []
        if not long_side.empty:
            l = self._cap_and_redistribute(long_side, self.cfg.long_exposure)
            symbols.extend(l.index.tolist())
            weights.extend(l.tolist())

        if self.cfg.allow_short and not short_side.empty:
            s = -self._cap_and_redistribute(short_side, self.cfg.short_exposure)
            symbols.extend(s.index.tolist())
            weights.extend(s.tolist())

        target = pd.Series(weights, index=symbols, dtype=float)
        target = target.groupby(level=0).sum()

        if self.cfg.use_net_neutral:
            net = target.sum()
            target -= net / max(len(target), 1)
        gross = target.abs().sum()
        if gross > (self.cfg.long_exposure + self.cfg.short_exposure):
            target = target / gross * (self.cfg.long_exposure + self.cfg.short_exposure)
        target = target * self.cfg.leverage
        return self._enforce_final_position_cap(target)

    def _scale_for_risk(self, weights: pd.Series, returns: pd.DataFrame, idx: int) -> pd.Series:
        if weights.empty:
            return weights

        if idx <= self.cfg.vol_lookback:
            return weights

        hist = returns.iloc[max(0, idx - self.cfg.vol_lookback) : idx + 1]
        ann_vol = hist.std(ddof=0) * np.sqrt(252)
        target_vol = self.cfg.target_annualized_vol
        port_vol = np.sqrt((weights.abs() * ann_vol.fillna(0.0)).pow(2).sum())
        if port_vol <= 0:
            return weights
        scale = min(1.0, target_vol / port_vol)
        return weights * float(scale)

    def run(
        self,
        raw_df: pd.DataFrame,
        *,
        signal_cache: pd.DataFrame | None = None,
        universe_cache: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        close = pivot_prices(raw_df, "close")
        close = self._clean_price_matrix(close)
        amount = (
            pivot_prices(raw_df, "amount")
            if "amount" in raw_df.columns
            else pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        )
        returns = close.pct_change(fill_method=None).fillna(0.0)
        if close.shape[1] == 0 or len(close) < max(self.cfg.long_window, 2):
            raise ValueError("Not enough data for backtest.")
        self.signal_cache = (
            signal_cache.reindex(index=close.index, columns=close.columns)
            if signal_cache is not None
            else self._precompute_signal_panel(close)
        )
        self.universe_cache = (
            universe_cache.reindex(index=close.index, columns=close.columns, fill_value=False)
            if universe_cache is not None
            else self._precompute_universe_panel(close, amount)
        )

        if self.cfg.execution_model.lower().strip() == "next_open":
            open_px = pivot_prices(raw_df, "open").reindex_like(close)
            open_px = self._clean_price_matrix(open_px)
            optional_panels = {}
            for name, candidates in (
                ("limit_rate", LIMIT_RATE_COLUMNS),
                ("limit_up_price", LIMIT_UP_PRICE_COLUMNS),
                ("limit_down_price", LIMIT_DOWN_PRICE_COLUMNS),
            ):
                column = next((item for item in candidates if item in raw_df.columns), None)
                optional_panels[name] = (
                    pivot_prices(raw_df, column).reindex_like(close)
                    if column is not None
                    else None
                )
            return self._run_next_open(close, open_px, amount, **optional_panels)

        positions = pd.Series(dtype=float)
        high_water_mark = self.equity
        cooldown = 0
        for i, date in enumerate(close.index):
            if i == 0:
                self.nav.append(self.equity)
                self.date_idx.append(date)
                self.drawdown_series.append(0.0)
                self.position_gross_history.append(0.0)
                self.turnover_history.append(0.0)
                continue

            daily_ret = returns.loc[date].fillna(0.0)
            gross_pnl = float((positions.reindex(close.columns).fillna(0.0) * daily_ret).sum())
            self.equity *= 1.0 + gross_pnl

            high_water_mark = max(high_water_mark, self.equity)
            drawdown = 1 - self.equity / high_water_mark if high_water_mark > 0 else 0.0
            self.drawdown_series.append(float(drawdown))
            self.date_idx.append(date)
            self.nav.append(self.equity)

            if drawdown > self.cfg.max_drawdown:
                cooldown = max(cooldown, self.cfg.drawdown_cooldown_days)

            if cooldown > 0:
                target = pd.Series(dtype=float)
                cooldown -= 1
                risk_state = "risk_halt"
            elif i % self.cfg.rebalance_frequency != 0:
                target = positions
                risk_state = "hold"
            else:
                signal = self._generate_signal(close, i, close.index)
                target = self._target_weights(signal)
                target = self._scale_for_risk(target, returns, i)
                risk_state = "rebalance"

                # record top signal if no rebalance candidate
                if target.empty and not signal.empty:
                    target = pd.Series(dtype=float)

            target = target.reindex(close.columns).fillna(0.0)
            if target.empty:
                target = pd.Series(0.0, index=close.columns)

            turnover = float((target - positions.reindex(close.columns).fillna(0.0)).abs().sum())
            self.turnover_history.append(turnover)
            cost = turnover * (self.cfg.commission_bps + self.cfg.impact_bps) / 1e4
            cost += (turnover / 2.0) * self.cfg.stamp_tax_bps / 1e4
            if cost > 0:
                self.equity *= 1.0 - cost

            if cost > 0:
                self.trade_log.append(
                    {
                        "date": str(date),
                        "event": "rebalance_or_flatten",
                        "turnover": turnover,
                        "cost": cost,
                        "risk_state": risk_state,
                    }
                )

            self.position_gross_history.append(float(target.abs().sum()))
            positions = target

        nav = pd.Series(self.nav, index=self.date_idx, name="equity")
        daily_ret = nav.pct_change().fillna(0.0)
        metrics = {
            "initial_capital": self.cfg.initial_capital,
            "final_equity": float(nav.iloc[-1]),
            "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
            "annualized_return": float(annualized_return(nav)),
            "max_drawdown": float(max_drawdown(nav)),
            "sharpe_like": float(sharpe_like(daily_ret)),
            "trade_days": int(len(nav)),
            "non_zero_signals": int(np.count_nonzero(np.array(self.drawdown_series))),
            "final_positions_count": int((positions != 0).sum()),
            "avg_gross_exposure": float(pd.Series(self.position_gross_history).mean()),
            "avg_turnover": float(pd.Series(self.turnover_history).mean()),
            **self.execution_constraint_counts,
        }
        return {
            "equity": nav,
            "positions": positions,
            "returns": daily_ret,
            "metrics": metrics,
            "trade_log": pd.DataFrame(self.trade_log),
            "drawdown_series": pd.Series(self.drawdown_series, index=self.date_idx, name="drawdown"),
        }

    def _build_target(
        self,
        close: pd.DataFrame,
        returns: pd.DataFrame,
        positions: pd.Series,
        idx: int,
        cooldown: int,
    ) -> tuple[pd.Series, int, str]:
        if cooldown > 0:
            return pd.Series(dtype=float), cooldown - 1, "risk_halt"
        if idx % self.cfg.rebalance_frequency != 0:
            return positions, cooldown, "hold"
        signal = self._generate_signal(close, idx, close.index)
        target = self._target_weights(signal)
        target = self._scale_for_risk(target, returns, idx)
        if target.empty and not signal.empty:
            target = pd.Series(dtype=float)
        return target, cooldown, "rebalance"

    def _run_next_open(
        self,
        close: pd.DataFrame,
        open_px: pd.DataFrame,
        amount: pd.DataFrame,
        limit_rate: pd.DataFrame | None = None,
        limit_up_price: pd.DataFrame | None = None,
        limit_down_price: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        open_returns = open_px.pct_change(fill_method=None).fillna(0.0)
        close_returns = close.pct_change(fill_method=None).fillna(0.0)
        positions = pd.Series(0.0, index=close.columns)
        queued_target = pd.Series(0.0, index=close.columns)
        high_water_mark = self.equity
        cooldown = 0

        for i, date in enumerate(close.index):
            if i > 0:
                open_ret = open_returns.loc[date].fillna(0.0)
                gross_pnl = float((positions.reindex(close.columns).fillna(0.0) * open_ret).sum())
                self.equity *= 1.0 + gross_pnl

            target_to_execute = queued_target.reindex(close.columns).fillna(0.0)
            if i > 0:
                target_to_execute, counts = self._apply_execution_constraints(
                    positions.reindex(close.columns).fillna(0.0),
                    target_to_execute,
                    open_px.loc[date],
                    close.iloc[i - 1],
                    limit_rate_row=(
                        limit_rate.loc[date] if limit_rate is not None else None
                    ),
                    limit_up_price_row=(
                        limit_up_price.loc[date]
                        if limit_up_price is not None
                        else None
                    ),
                    limit_down_price_row=(
                        limit_down_price.loc[date]
                        if limit_down_price is not None
                        else None
                    ),
                )
                self._record_execution_constraint_counts(counts)

            turnover = float((target_to_execute - positions.reindex(close.columns).fillna(0.0)).abs().sum())
            self.turnover_history.append(turnover)
            cost = turnover * (self.cfg.commission_bps + self.cfg.impact_bps) / 1e4
            cost += (turnover / 2.0) * self.cfg.stamp_tax_bps / 1e4
            if cost > 0:
                self.equity *= 1.0 - cost
                self.trade_log.append(
                    {
                        "date": str(date),
                        "event": "next_open_execute",
                        "turnover": turnover,
                        "cost": cost,
                        "risk_state": "execute",
                    }
                )
            positions = target_to_execute
            self.position_gross_history.append(float(positions.abs().sum()))

            high_water_mark = max(high_water_mark, self.equity)
            drawdown = 1 - self.equity / high_water_mark if high_water_mark > 0 else 0.0
            self.drawdown_series.append(float(drawdown))
            self.date_idx.append(date)
            self.nav.append(self.equity)

            if drawdown > self.cfg.max_drawdown:
                cooldown = max(cooldown, self.cfg.drawdown_cooldown_days)

            queued_target, cooldown, _ = self._build_target(
                close, close_returns, positions, i, cooldown
            )
            queued_target = queued_target.reindex(close.columns).fillna(0.0)

        nav = pd.Series(self.nav, index=self.date_idx, name="equity")
        daily_ret = nav.pct_change(fill_method=None).fillna(0.0)
        metrics = {
            "initial_capital": self.cfg.initial_capital,
            "final_equity": float(nav.iloc[-1]),
            "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
            "annualized_return": float(annualized_return(nav)),
            "max_drawdown": float(max_drawdown(nav)),
            "sharpe_like": float(sharpe_like(daily_ret)),
            "trade_days": int(len(nav)),
            "non_zero_signals": int(np.count_nonzero(np.array(self.drawdown_series))),
            "final_positions_count": int((positions != 0).sum()),
            "avg_gross_exposure": float(pd.Series(self.position_gross_history).mean()),
            "avg_turnover": float(pd.Series(self.turnover_history).mean()),
            **self.execution_constraint_counts,
        }
        return {
            "equity": nav,
            "positions": positions,
            "returns": daily_ret,
            "metrics": metrics,
            "trade_log": pd.DataFrame(self.trade_log),
            "drawdown_series": pd.Series(self.drawdown_series, index=self.date_idx, name="drawdown"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-risk strategy backtest scaffold.")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV or Parquet panel.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory, overrides YAML output_dir.",
    )
    return parser.parse_args()


def write_outputs(results: Dict[str, Any], cfg: StrategyConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results["equity"].to_csv(output_dir / "equity_curve.csv")
    results["drawdown_series"].to_csv(output_dir / "drawdown_series.csv")
    results["trade_log"].to_csv(output_dir / "trade_log.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, ensure_ascii=False, indent=2)

    report = (
        "# 回测报告\n\n"
        f"- 初始资金: {cfg.initial_capital:,.2f}\n"
        f"- 期末权益: {results['metrics']['final_equity']:,.2f}\n"
        f"- 总收益: {results['metrics']['total_return']:.2%}\n"
        f"- 年化收益: {results['metrics']['annualized_return']:.2%}\n"
        f"- 最大回撤: {results['metrics']['max_drawdown']:.2%}\n"
        f"- 风险调整后收益: {results['metrics']['sharpe_like']:.3f}\n"
        f"- 交易日: {results['metrics']['trade_days']}\n"
    )
    (output_dir / "backtest_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.output_dir:
        cfg.output_dir = args.output_dir

    raw = load_prices(Path(args.data), cfg.start_date, cfg.end_date)
    engine = BacktestEngine(cfg)
    result = engine.run(raw)
    write_outputs(result, cfg, Path(cfg.output_dir))
    print(
        "Backtest completed.\n"
        f"  final_equity: {result['metrics']['final_equity']:.2f}\n"
        f"  total_return: {result['metrics']['total_return']:.2%}\n"
        f"  max_drawdown: {result['metrics']['max_drawdown']:.2%}\n"
        f"  sharpe_like: {result['metrics']['sharpe_like']:.3f}\n"
        f"  output_dir: {cfg.output_dir}"
    )


if __name__ == "__main__":
    main()
