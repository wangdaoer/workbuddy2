"""Walk-forward rank model for next-open tradable returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import (
    apply_open_constraints,
    forward_open_return_label,
    next_open_return_label,
)
from market_risk import load_benchmark_market_exposure
from run_backtest import load_prices, max_drawdown, pivot_prices, sharpe_like
from trading_calendar import validate_trading_sessions


MAIN_CHINEXT_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605")
TREND_GATE_BREAKOUT_FLOOR = -0.10
RISK_REBALANCE_POLICIES = ("scheduled_only", "daily_decrease_only")

# 动态流动性筛选：当期流动性达标标的数低于此阈值时，回退到全 universe（避免极端期无票可交易）
MIN_LIVE_SYMBOLS = 50

SLEEVE_FEATURE_DIRECTIONS: dict[str, dict[str, int]] = {
    "trend": {
        "momentum_20": 1,
        "momentum_60": 1,
        "breakout_20": 1,
        "distance_ma20": 1,
    },
    "mean_reversion": {
        "reversal_5": 1,
        "intraday_return": -1,
        "close_position": -1,
    },
    "hybrid_pullback": {
        "strong_pullback_20_5": 1,
    },
    "trend_gated_pullback": {
        "trend_gated_pullback_absorption": 1,
    },
}

QUARANTINED_HYBRID_FEATURES = (
    "strong_pullback_20_5",
    "strong_pullback_60_5",
    "breakout_pullback_20_5",
    "liquid_pullback",
)
MARKET_DIAGNOSTIC_MA_WINDOW = 60
MARKET_DIAGNOSTIC_RETURN_WINDOW = 20

PRODUCTION_FEATURE_NAMES = (
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "reversal_5",
    "breakout_20",
    "distance_ma20",
    "volatility_20",
    "liquidity_20",
    "intraday_return",
    "close_position",
    "strong_pullback_20_5",
    "strong_pullback_60_5",
    "breakout_pullback_20_5",
    "anti_chase_intraday",
    "liquid_pullback",
)


def clean_matrix(frame: pd.DataFrame, max_abs_return: float) -> pd.DataFrame:
    if max_abs_return <= 0:
        return frame
    ret = frame.pct_change(fill_method=None)
    return frame.mask(ret.abs().gt(max_abs_return))


def rank_pct(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def build_trend_gated_pullback_absorption(
    momentum_20: pd.DataFrame,
    breakout_20: pd.DataFrame,
    reversal_5_rank: pd.DataFrame,
    volatility_20_rank: pd.DataFrame,
) -> pd.DataFrame:
    eligible = momentum_20.gt(0.0) & breakout_20.gt(TREND_GATE_BREAKOUT_FLOOR)
    low_volatility_rank = (1.0 - volatility_20_rank).clip(lower=0.0)
    absorption_score = reversal_5_rank * low_volatility_rank
    return rank_pct(absorption_score.where(eligible))


def build_features(close: pd.DataFrame, open_px: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, amount: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    ma20 = close.rolling(20).mean()
    prev_high_20 = high.rolling(20).max().shift(1)
    close_range = (high - low).replace(0, np.nan)
    raw = {
        "momentum_5": close.pct_change(5, fill_method=None),
        "momentum_20": close.pct_change(20, fill_method=None),
        "momentum_60": close.pct_change(60, fill_method=None),
        "reversal_5": -close.pct_change(5, fill_method=None),
        "breakout_20": close / (prev_high_20 + 1e-12) - 1.0,
        "distance_ma20": close / (ma20 + 1e-12) - 1.0,
        "volatility_20": returns.rolling(20).std(),
        "liquidity_20": np.log1p(amount.replace(0, np.nan).rolling(20).median()),
        "intraday_return": close / (open_px + 1e-12) - 1.0,
        "close_position": (close - low) / (close_range + 1e-12),
    }
    ranked = {name: rank_pct(frame) for name, frame in raw.items()}

    # Distill the close-to-close advantage into next-open tradable patterns:
    # keep intermediate-term strength, but avoid buying immediately after short-term exhaustion.
    ranked["strong_pullback_20_5"] = rank_pct(ranked["momentum_20"] * ranked["reversal_5"])
    ranked["strong_pullback_60_5"] = rank_pct(ranked["momentum_60"] * ranked["reversal_5"])
    ranked["breakout_pullback_20_5"] = rank_pct(ranked["breakout_20"] * ranked["reversal_5"])
    ranked["anti_chase_intraday"] = rank_pct((1.0 - ranked["intraday_return"]) * (1.0 - ranked["close_position"]))
    ranked["liquid_pullback"] = rank_pct(ranked["liquidity_20"] * ranked["reversal_5"])
    ranked["trend_gated_pullback_absorption"] = build_trend_gated_pullback_absorption(
        raw["momentum_20"],
        raw["breakout_20"],
        ranked["reversal_5"],
        ranked["volatility_20"],
    )
    return ranked


def _daily_ic_legacy(features, label):
    """原逐点 spearman 实现（已弃用，仅留作回归对比基准）。"""
    rows = []
    for date in label.index:
        y = label.loc[date]
        row = {"date": date}
        for name, feature in features.items():
            x = feature.loc[date]
            both = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
            row[name] = both["x"].corr(both["y"], method="spearman") if len(both) >= 30 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def daily_ic(features: dict[str, pd.DataFrame], label: pd.DataFrame) -> pd.DataFrame:
    """横截面 spearman IC（逐日），向量化实现（spearman = pearson of ranks）。

    与原逐点实现数值等价（NaN 自动排除、有效样本 <30 置 NaN）：
    对每个因子，仅在 factor 与 label 同时非 NaN 的股票子集上计算 spearman，
    即先按 mask=f.notna()&label.notna() 把交集外置 NaN 再做 rank，避免平局
    平均秩因样本集不同而产生偏差（与原 concat().dropna() 语义一致）。
    将 O(日期 × 因子) 的 Python 循环降为 O(因子) 次矩阵运算，提速 ~100×。
    原双循环版本保留为 _daily_ic_legacy 供回归对比。
    """
    out = {}
    for name, f in features.items():
        mask = f.notna() & label.notna()
        rf = f.where(mask).rank(axis=1, pct=False)
        rl = label.where(mask).rank(axis=1, pct=False)
        valid_n = mask.sum(axis=1)
        rfm = rf.mean(axis=1)
        rlm = rl.mean(axis=1)
        rfstd = rf.std(axis=1, ddof=0)
        rlstd = rl.std(axis=1, ddof=0)
        cov = (rf * rl).mean(axis=1) - rfm * rlm
        with np.errstate(divide="ignore", invalid="ignore"):
            ic = cov / (rfstd * rlstd)
        ic = ic.where(valid_n >= 30)
        out[name] = ic
    return pd.DataFrame(out, index=label.index)


def parse_training_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError("training horizons must be comma-separated positive integers") from exc
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("training horizons must be comma-separated positive integers")
    return horizons


def build_multi_horizon_ic(
    features: dict[str, pd.DataFrame],
    open_px: pd.DataFrame,
    horizons: tuple[int, ...],
    max_abs_daily_return: float,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    horizon_ics = {
        horizon: daily_ic(
            features,
            forward_open_return_label(
                open_px,
                horizon_days=horizon,
                max_abs_daily_return=max_abs_daily_return,
            ),
        )
        for horizon in horizons
    }
    combined = pd.concat(horizon_ics.values()).groupby(level=0).mean().sort_index()
    return combined, horizon_ics


def mature_ic_window(
    ic: pd.DataFrame,
    signal_position: int,
    train_days: int,
    max_training_horizon: int,
) -> pd.DataFrame:
    """Return only labels whose exit open is known by the signal-date close."""

    mature_end = signal_position - max_training_horizon
    mature_start = mature_end - train_days
    if mature_start < 0:
        return ic.iloc[0:0]
    return ic.iloc[mature_start:mature_end]


def normalize_weights(weights: pd.Series) -> pd.Series:
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = weights.abs().sum()
    if denom <= 1e-12:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / denom


def directional_ic_weights(
    mean_ic: pd.Series,
    directions: dict[str, int],
) -> pd.Series:
    if set(mean_ic.index) != set(directions):
        raise ValueError("direction map must exactly match sleeve features")
    if any(direction not in (-1, 1) for direction in directions.values()):
        raise ValueError("feature directions must be -1 or 1")

    cleaned = mean_ic.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    direction_series = pd.Series(directions, dtype=float).reindex(cleaned.index)
    aligned_strength = cleaned * direction_series
    constrained = cleaned.where(aligned_strength.gt(0.0), 0.0)
    denom = constrained.abs().sum()
    if denom <= 1e-12:
        return pd.Series(0.0, index=cleaned.index)
    return constrained / denom


def select_feature_sleeve(
    features: dict[str, pd.DataFrame],
    sleeve: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, int] | None]:
    if sleeve == "unconstrained":
        missing = sorted(set(PRODUCTION_FEATURE_NAMES).difference(features))
        if missing:
            raise ValueError(f"production features missing: {missing}")
        return {name: features[name] for name in PRODUCTION_FEATURE_NAMES}, None
    directions = SLEEVE_FEATURE_DIRECTIONS.get(sleeve)
    if directions is None:
        raise ValueError(f"unknown sleeve: {sleeve}")
    missing = sorted(set(directions).difference(features))
    if missing:
        raise ValueError(f"sleeve features missing: {missing}")
    return {name: features[name] for name in directions}, directions.copy()


def load_market_exposure(
    benchmark_path: str | None,
    trade_dates: pd.Index,
    ma_window: int,
    risk_off_drawdown_20d: float,
    below_ma_exposure: float,
    crash_exposure: float,
) -> pd.Series:
    return load_benchmark_market_exposure(
        benchmark_path,
        trade_dates,
        ma_window=ma_window,
        risk_off_drawdown_20d=risk_off_drawdown_20d,
        below_ma_exposure=below_ma_exposure,
        crash_exposure=crash_exposure,
    )


def build_breadth_exposure(
    close: pd.DataFrame,
    ma_window: int,
    threshold: float,
    below_exposure: float,
    crash_threshold: float,
    crash_exposure: float,
) -> pd.Series:
    ma = close.rolling(ma_window).mean()
    breadth = close.gt(ma).mean(axis=1)
    exposure = pd.Series(1.0, index=close.index)
    exposure = exposure.where(~breadth.lt(threshold), below_exposure)
    exposure = exposure.where(~breadth.lt(crash_threshold), crash_exposure)
    return exposure.fillna(1.0).clip(lower=0.0, upper=1.0)


def build_market_diagnostics(
    close: pd.DataFrame,
    market_exposure: pd.Series,
    *,
    ma_window: int = MARKET_DIAGNOSTIC_MA_WINDOW,
    return_window: int = MARKET_DIAGNOSTIC_RETURN_WINDOW,
) -> pd.DataFrame:
    """Build observation-only market state without affecting model weights."""

    if ma_window < 2 or return_window < 1:
        raise ValueError("market diagnostic windows are invalid")
    valid = close.notna()
    moving_average = close.rolling(ma_window, min_periods=ma_window).mean()
    breadth_eligible = valid & moving_average.notna()
    return_window_values = close.pct_change(return_window, fill_method=None)
    diagnostics = pd.DataFrame(
        {
            f"breadth_above_ma{ma_window}": close.gt(moving_average)
            .where(breadth_eligible)
            .mean(axis=1),
            f"cross_sectional_median_return{return_window}": return_window_values.median(
                axis=1
            ),
            "available_symbols": valid.sum(axis=1),
            "breadth_eligible_symbols": breadth_eligible.sum(axis=1),
            "return_eligible_symbols": return_window_values.notna().sum(axis=1),
            "model_market_exposure_target": market_exposure.reindex(close.index),
        },
        index=close.index,
    )
    diagnostics.index.name = "date"
    return diagnostics.reset_index()


def calculate_walk_forward_metrics(
    equity: pd.DataFrame,
    initial_capital: float,
) -> dict[str, float | int]:
    if initial_capital <= 0.0 or not np.isfinite(initial_capital):
        raise ValueError("initial capital must be finite and positive")
    required = {
        "equity",
        "gross_return",
        "cost",
        "turnover",
        "gross_exposure",
        "market_exposure",
        "positions_count",
    }
    missing = sorted(required.difference(equity.columns))
    if missing:
        raise ValueError(f"equity metrics missing columns: {missing}")
    if equity.empty:
        raise ValueError("equity metrics require at least one trading session")

    nav = pd.to_numeric(equity["equity"], errors="coerce")
    net_returns = (
        pd.to_numeric(equity["gross_return"], errors="coerce")
        - pd.to_numeric(equity["cost"], errors="coerce")
    )
    if not np.isfinite(nav.to_numpy(dtype=float)).all() or not np.isfinite(
        net_returns.to_numpy(dtype=float)
    ).all():
        raise ValueError("equity metrics contain non-finite values")

    previous = nav.shift(1)
    previous.iloc[0] = float(initial_capital)
    if not np.allclose(
        previous * (1.0 + net_returns),
        nav,
        rtol=1e-9,
        atol=1e-5,
    ):
        raise ValueError("equity does not reconcile with complete net returns")

    nav_with_initial = pd.concat(
        [pd.Series([float(initial_capital)]), nav.reset_index(drop=True)],
        ignore_index=True,
    )
    total_return = float(nav.iloc[-1] / initial_capital - 1.0)
    annualized = float((1.0 + total_return) ** (252.0 / len(nav)) - 1.0)
    metrics: dict[str, float | int] = {
        "initial_capital": float(initial_capital),
        "final_equity": float(nav.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": float(max_drawdown(nav_with_initial)),
        "sharpe_like": float(sharpe_like(net_returns)),
        "trade_days": int(len(nav)),
        "avg_turnover": float(equity["turnover"].mean()),
        "avg_gross_exposure": float(equity["gross_exposure"].mean()),
        "avg_market_exposure_target": float(equity["market_exposure"].mean()),
        "avg_positions_count": float(equity["positions_count"].mean()),
    }
    for column in (
        "open_constraint_blocked_buy_weight",
        "open_constraint_blocked_sell_weight",
        "capacity_blocked_buy_weight",
        "capacity_blocked_sell_weight",
    ):
        if column in equity:
            metrics[f"total_{column}"] = float(equity[column].sum())
    for column in ("open_constraint_limited_symbols", "capacity_limited_symbols"):
        if column in equity:
            metrics[f"{column}_sessions"] = int(equity[column].gt(0).sum())
            metrics[f"total_{column}"] = int(equity[column].sum())
    risk_session_metrics = {
        "risk_exposure_decrease_event": "risk_exposure_decrease_sessions",
        "risk_rebalance_due": "risk_rebalance_attempt_sessions",
        "risk_deleverage_residual": "risk_deleverage_residual_sessions",
        "risk_unexplained_residual": "risk_unexplained_residual_sessions",
        "risk_zero_target_non_rebalance_with_positions": (
            "risk_zero_target_non_rebalance_with_positions_sessions"
        ),
        "risk_zero_target_unattempted": "risk_zero_target_unattempted_sessions",
        "risk_zero_target_unexplained_residual": (
            "risk_zero_target_unexplained_residual_sessions"
        ),
    }
    for column, metric_name in risk_session_metrics.items():
        if column in equity:
            metrics[metric_name] = int(equity[column].fillna(False).astype(bool).sum())
    if "risk_deleverage_shortfall_weight" in equity:
        metrics["total_risk_deleverage_shortfall_weight"] = float(
            equity["risk_deleverage_shortfall_weight"].sum()
        )
    return metrics


def trade_constraint_audit(
    current: pd.Series,
    requested: pd.Series,
    constrained: pd.Series,
) -> dict[str, float | int]:
    requested_delta = requested.reindex(current.index).fillna(0.0) - current
    constrained_delta = constrained.reindex(current.index).fillna(0.0) - current
    requested_buy = requested_delta.clip(lower=0.0)
    executed_buy = constrained_delta.clip(lower=0.0)
    requested_sell = (-requested_delta).clip(lower=0.0)
    executed_sell = (-constrained_delta).clip(lower=0.0)
    blocked_buy = (requested_buy - executed_buy).clip(lower=0.0)
    blocked_sell = (requested_sell - executed_sell).clip(lower=0.0)
    limited = requested_delta.sub(constrained_delta).abs().gt(1e-12)
    return {
        "limited_symbols": int(limited.sum()),
        "blocked_buy_weight": float(blocked_buy.sum()),
        "blocked_sell_weight": float(blocked_sell.sum()),
    }


def apply_trade_capacity(
    current: pd.Series,
    requested: pd.Series,
    trailing_median_amount: pd.Series,
    equity: float,
    max_daily_amount_participation: float,
) -> tuple[pd.Series, dict[str, float | int]]:
    if not np.isfinite(equity) or equity <= 0.0:
        raise ValueError("capacity equity must be finite and positive")
    participation = float(max_daily_amount_participation)
    if not np.isfinite(participation) or not 0.0 < participation <= 1.0:
        raise ValueError("max daily amount participation must be in (0, 1]")

    desired = requested.reindex(current.index).fillna(0.0)
    amounts = pd.to_numeric(
        trailing_median_amount.reindex(current.index), errors="coerce"
    )
    capacity_weight = (amounts * participation / equity).where(
        amounts.gt(0.0) & np.isfinite(amounts), 0.0
    )
    desired_delta = desired - current
    executed_delta = pd.Series(
        np.clip(
            desired_delta.to_numpy(dtype=float),
            -capacity_weight.to_numpy(dtype=float),
            capacity_weight.to_numpy(dtype=float),
        ),
        index=current.index,
    )
    constrained = current + executed_delta
    return constrained, trade_constraint_audit(current, desired, constrained)


def daily_decrease_only_target(
    positions: pd.Series,
    scheduled_pre_overlay_gross: float,
    current_exposure_ceiling: float,
    market_exposure_target: float,
) -> tuple[pd.Series, float, float, bool]:
    values = (
        scheduled_pre_overlay_gross,
        current_exposure_ceiling,
        market_exposure_target,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("daily risk target inputs must be finite")
    if scheduled_pre_overlay_gross < 0.0:
        raise ValueError("scheduled pre-overlay gross must be nonnegative")
    if not 0.0 <= current_exposure_ceiling <= 1.0:
        raise ValueError("current exposure ceiling must be in [0, 1]")
    if not 0.0 <= market_exposure_target <= 1.0:
        raise ValueError("market exposure target must be in [0, 1]")

    next_ceiling = min(current_exposure_ceiling, market_exposure_target)
    gross_cap = scheduled_pre_overlay_gross * next_ceiling
    current_gross = float(positions.abs().sum())
    if current_gross <= gross_cap + 1e-12:
        return positions.copy(), next_ceiling, gross_cap, False

    scale = gross_cap / current_gross if current_gross > 0.0 else 0.0
    return positions * scale, next_ceiling, gross_cap, True


def run_walk_forward(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    label: pd.DataFrame,
    ic: pd.DataFrame,
    train_days: int,
    retrain_frequency: int,
    top_n: int,
    rebalance_frequency: int,
    max_position_weight: float,
    leverage: float,
    commission_bps: float,
    impact_bps: float,
    max_buy_open_gap: float,
    limit_buffer: float,
    market_exposure: pd.Series,
    initial_capital: float,
    max_training_horizon: int,
    impact_model: str | None = None,
    impact_ref_participation: float = 0.01,
    feature_directions: dict[str, int] | None = None,
    amount: pd.DataFrame | None = None,
    capacity_lookback: int = 20,
    max_daily_amount_participation: float | None = None,
    risk_rebalance_policy: str = "scheduled_only",
    feature_selection: "Callable[[pd.DataFrame], list[str]] | None" = None,
    regime: "pd.Series | None" = None,
    liquid_mask: "pd.DataFrame | None" = None,
    top_n_schedule: "pd.Series | None" = None,
    stamp_tax_bps: float = 0.0,
    horizon_ics: "dict[int, pd.DataFrame] | None" = None,
    horizon_fusion_scheme: str = "icmag",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if risk_rebalance_policy not in RISK_REBALANCE_POLICIES:
        raise ValueError(
            f"risk rebalance policy must be one of {RISK_REBALANCE_POLICIES}"
        )
    capacity_amount = None
    if max_daily_amount_participation is not None:
        if capacity_lookback < 1:
            raise ValueError("capacity lookback must be positive")
        if amount is None:
            raise ValueError("amount data is required when capacity is enabled")
        if not 0.0 < float(max_daily_amount_participation) <= 1.0:
            raise ValueError("max daily amount participation must be in (0, 1]")
        capacity_amount = (
            amount.reindex(index=close.index, columns=close.columns)
            .apply(pd.to_numeric, errors="coerce")
            .where(lambda values: values.gt(0.0))
            .rolling(capacity_lookback, min_periods=capacity_lookback)
            .median()
        )

    if impact_model is not None:
        if impact_model not in ("sqrt", "linear"):
            raise ValueError("impact_model must be None, 'sqrt', or 'linear'")
        if amount is None:
            raise ValueError("amount data is required when impact_model is enabled")
        if not 0.0 < float(impact_ref_participation) <= 1.0:
            raise ValueError("impact_ref_participation must be in (0, 1]")
    # 冲击成本模型用到的实时成交额（与 close 对齐，不做滚动 median）
    amount_aligned = (
        amount.reindex(index=close.index, columns=close.columns)
        .apply(pd.to_numeric, errors="coerce")
        .where(lambda values: values.gt(0.0))
        if amount is not None
        else None
    )

    if liquid_mask is not None:
        if amount is None:
            raise ValueError("amount data is required when liquid_mask is enabled")
        liquid_mask_aligned = liquid_mask.reindex(
            index=close.index, columns=close.columns
        ).fillna(False)
    else:
        liquid_mask_aligned = None

    equity = initial_capital
    positions = pd.Series(0.0, index=close.columns)
    current_weights = pd.Series(1.0 / len(features), index=list(features))
    # 逐周期融合状态（仅 --horizon-fusion 时启用；默认 None，完全走原路径）
    per_horizon_weights: "dict[int, pd.Series] | None" = None
    fusion_w: dict[int, float] = {}
    nav_rows = []
    weight_rows = []
    trade_rows = []
    risk_exposure_ceiling = 1.0
    scheduled_pre_overlay_gross = 0.0

    first_signal_position = train_days + max_training_horizon
    current_live_cols = list(close.columns)
    for i in range(first_signal_position, len(close.index) - 2):
        date = close.index[i]
        step = i - first_signal_position
        rebalance_due = step % rebalance_frequency == 0
        exposure = float(market_exposure.reindex([date]).iloc[0])
        if not np.isfinite(exposure) or not 0.0 <= exposure <= 1.0:
            raise ValueError("market exposure target must be finite and in [0, 1]")
        if step % retrain_frequency == 0:
            if liquid_mask_aligned is not None:
                current_live_cols = [
                    c for c in close.columns
                    if bool(liquid_mask_aligned.loc[date, c])
                ]
                if len(current_live_cols) < MIN_LIVE_SYMBOLS:
                    current_live_cols = list(close.columns)
                # IC 已通过 ic= 参数一次性估计（全样本，横截面基于全市场股票）；
                # 原实现对截断列集重复 daily_ic，全样本规模下单程 >1h。此处直接复用预计算 IC，
                # 回测持仓候选仍按 current_live_cols 过滤（下方 candidates.reindex）。
                # 偏差：IC 估计样本由「当日流动性子集」变为「全市场」（微小，方向性结论不受影响）。
                ic_r = ic
            else:
                current_live_cols = list(close.columns)
                ic_r = ic
            training_ic = mature_ic_window(
                ic_r,
                signal_position=i,
                train_days=train_days,
                max_training_horizon=max_training_horizon,
            )
            if len(training_ic) != train_days:
                raise RuntimeError("strict training IC window is incomplete")
            mean_ic = training_ic.mean()
            if regime is not None:
                # regime 门控加权：用「当前 regime 在成熟窗口内的 IC 子样本」选因子+加权，
                # 使模型在牛/震荡/熊下使用各自有效的因子符号（P2 regime 表洞察的工程化）。
                cur_reg = regime.reindex([date]).iloc[0] if date in regime.index else regime.iloc[-1]
                reg_mask = (regime.reindex(training_ic.index) == cur_reg).values
                sub = training_ic[reg_mask].mean() if reg_mask.sum() >= 30 else mean_ic
                if feature_selection is not None:
                    selected = [f for f in sub.index if pd.notna(sub[f]) and sub[f] > 0]
                    if len(selected) < 3:
                        selected = list(features.keys())
                    current_weights = normalize_weights(sub[selected])
                else:
                    current_weights = normalize_weights(sub)
            elif feature_selection is not None:
                # 自适应因子选择：仅用成熟窗口 IC 为正的因子（消除固定子集的 look-ahead）
                selected = feature_selection(training_ic)
                if len(selected) < 3:
                    selected = list(features.keys())
                current_weights = normalize_weights(mean_ic[selected])
            elif feature_directions is not None:
                current_weights = directional_ic_weights(mean_ic, feature_directions)
            else:
                current_weights = normalize_weights(mean_ic)
            weight_rows.append({"date": date.strftime("%Y-%m-%d"), **current_weights.to_dict()})

            # ---- 逐周期融合（深化多周期融合）：每个周期独立算权重向量，打分层融合 ----
            per_horizon_weights = None
            fusion_w = {}
            if horizon_ics is not None:
                phw: dict[int, pd.Series] = {}
                ph_metric: dict[int, float] = {}
                for _h, _h_ic in horizon_ics.items():
                    _h_train = mature_ic_window(
                        _h_ic,
                        signal_position=i,
                        train_days=train_days,
                        max_training_horizon=max_training_horizon,
                    )
                    if len(_h_train) != train_days:
                        continue
                    _h_mean = _h_train.mean()
                    # 与 current_weights 同一套加权策略，但逐周期独立应用
                    if feature_directions is not None:
                        _w = directional_ic_weights(_h_mean, feature_directions)
                    elif feature_selection is not None:
                        _sel = feature_selection(_h_train)
                        if len(_sel) < 3:
                            _sel = list(features.keys())
                        _w = normalize_weights(_h_mean[_sel])
                    elif regime is not None:
                        _cur_reg = (
                            regime.reindex([date]).iloc[0]
                            if date in regime.index
                            else regime.iloc[-1]
                        )
                        _reg_mask = (regime.reindex(_h_train.index) == _cur_reg).values
                        _sub = _h_train[_reg_mask].mean() if _reg_mask.sum() >= 30 else _h_mean
                        _w = normalize_weights(_sub)
                    else:
                        _w = normalize_weights(_h_mean)
                    if _w.abs().sum() <= 1e-12:
                        continue
                    phw[_h] = _w
                    ph_metric[_h] = float(_h_mean.abs().mean())
                if phw:
                    per_horizon_weights = phw
                    if horizon_fusion_scheme == "equal":
                        _fw = {_h: 1.0 for _h in phw}
                    elif horizon_fusion_scheme == "decay":
                        _fw = {_h: 1.0 / np.sqrt(float(_h)) for _h in phw}
                    else:  # icmag：按该周期成熟窗口平均|IC|幅度分配融合权重
                        _tot = sum(ph_metric.values()) or 1.0
                        _fw = {_h: ph_metric[_h] / _tot for _h in phw}
                    _fw_sum = sum(_fw.values()) or 1.0
                    fusion_w = {_h: _v / _fw_sum for _h, _v in _fw.items()}

        risk_exposure_decrease_event = False
        risk_rebalance_due = False
        pre_trade_gross = float(positions.abs().sum())
        if rebalance_due:
            if current_weights.abs().sum() <= 1e-12:
                pre_overlay_target = pd.Series(0.0, index=close.columns)
            else:
                if per_horizon_weights is not None:
                    # 逐周期打分：各周期线性分→横截面百分位→按融合权重加总
                    fused = None
                    for _h, _w in per_horizon_weights.items():
                        _s = None
                        for _name, _weight in _w.items():
                            _frame = features[_name]
                            _s = (
                                _frame.iloc[i] * _weight
                                if _s is None
                                else _s + _frame.iloc[i] * _weight
                            )
                        _s_rank = _s.rank(pct=True)
                        _fw = fusion_w.get(_h, 1.0)
                        fused = _s_rank * _fw if fused is None else fused + _s_rank * _fw
                    candidates = fused.reindex(current_live_cols).dropna()
                else:
                    score = None
                    for name, weight in current_weights.items():
                        frame = features[name]
                        score = (
                            frame.iloc[i] * weight
                            if score is None
                            else score + frame.iloc[i] * weight
                        )
                    candidates = score.reindex(current_live_cols).dropna()
                if candidates.empty:
                    pre_overlay_target = pd.Series(0.0, index=close.columns)
                else:
                    eff_top_n = (
                        int(top_n_schedule.loc[date])
                        if top_n_schedule is not None and date in top_n_schedule.index
                        else top_n
                    )
                    selected = candidates.nlargest(eff_top_n).index
                    raw_weight = min(max_position_weight, leverage / max(len(selected), 1))
                    pre_overlay_target = pd.Series(0.0, index=close.columns)
                    pre_overlay_target.loc[selected] = raw_weight
                    gross = pre_overlay_target.abs().sum()
                    if gross > leverage:
                        pre_overlay_target = pre_overlay_target / gross * leverage
            scheduled_pre_overlay_gross = float(pre_overlay_target.abs().sum())
            risk_exposure_ceiling = exposure
            risk_target_gross_cap = scheduled_pre_overlay_gross * risk_exposure_ceiling
            target = pre_overlay_target * exposure
        else:
            previous_ceiling = risk_exposure_ceiling
            risk_exposure_decrease_event = exposure < previous_ceiling - 1e-12
            risk_target, risk_exposure_ceiling, risk_target_gross_cap, reduction_due = (
                daily_decrease_only_target(
                    positions,
                    scheduled_pre_overlay_gross,
                    previous_ceiling,
                    exposure,
                )
            )
            if risk_rebalance_policy == "daily_decrease_only" and reduction_due:
                target = risk_target
                risk_rebalance_due = True
            else:
                target = positions.copy()

        zero_target_non_rebalance_with_positions = bool(
            not rebalance_due and exposure <= 1e-12 and pre_trade_gross > 1e-12
        )
        zero_target_unattempted = bool(
            zero_target_non_rebalance_with_positions and not risk_rebalance_due
        )

        requested_target = target.copy()
        open_constrained_target = apply_open_constraints(
            positions,
            requested_target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=max_buy_open_gap,
            limit_buffer=limit_buffer,
        )
        open_audit = trade_constraint_audit(
            positions, requested_target, open_constrained_target
        )
        if capacity_amount is None:
            target = open_constrained_target
            capacity_audit = {
                "limited_symbols": 0,
                "blocked_buy_weight": 0.0,
                "blocked_sell_weight": 0.0,
            }
        else:
            target, capacity_audit = apply_trade_capacity(
                positions,
                open_constrained_target,
                capacity_amount.iloc[i],
                equity,
                float(max_daily_amount_participation),
            )

        post_constraint_gross = float(target.abs().sum())
        risk_shortfall = (
            max(post_constraint_gross - risk_target_gross_cap, 0.0)
            if risk_rebalance_due
            else 0.0
        )
        risk_residual = bool(risk_rebalance_due and risk_shortfall > 1e-12)
        explained_sell_block = float(open_audit["blocked_sell_weight"]) + float(
            capacity_audit["blocked_sell_weight"]
        )
        risk_unexplained_residual = bool(
            risk_residual and explained_sell_block + 1e-12 < risk_shortfall
        )
        zero_target_unexplained_residual = bool(
            zero_target_non_rebalance_with_positions
            and risk_rebalance_due
            and post_constraint_gross > 1e-12
            and explained_sell_block + 1e-12 < post_constraint_gross
        )

        turnover = float((target - positions).abs().sum())
        if impact_model is not None and amount_aligned is not None:
            # 参与度依赖冲击成本（Almgren/Kyle 风格）：
            # participation[s] = |Δ权重_s| × equity / 当日成交额_s
            #   sqrt  : impact_s = impact_bps × sqrt(participation / ref)
            #   linear: impact_s = impact_bps × (participation / ref)
            delta_w = (target - positions).abs()
            traded_value = delta_w * equity
            daily_amt = amount_aligned.iloc[i].reindex(delta_w.index).fillna(0.0)
            part = (traded_value / daily_amt).clip(lower=0.0, upper=1.0)
            if impact_model == "sqrt":
                impact_s = impact_bps * np.sqrt(part / impact_ref_participation)
            else:  # linear
                impact_s = impact_bps * (part / impact_ref_participation)
            cost = float(
                (delta_w * (commission_bps + impact_s.fillna(0.0))).sum() / 1e4
            )
        else:
            cost = turnover * (commission_bps + impact_bps) / 1e4
        # 法定印花税：仅卖出侧，卖出名义额约为换手额的一半
        cost += (turnover / 2.0) * stamp_tax_bps / 1e4
        realized = label.iloc[i].reindex(close.columns).fillna(0.0)
        gross_return = float((target * realized).sum())
        equity *= 1.0 + gross_return - cost
        positions = target

        nav_rows.append(
            {
                "date": close.index[i + 2].strftime("%Y-%m-%d"),
                "equity": equity,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "gross_exposure": float(positions.abs().sum()),
                "market_exposure": float(market_exposure.reindex([date]).iloc[0]),
                "positions_count": int(positions.ne(0).sum()),
                "rebalance_due": bool(rebalance_due),
                "scheduled_rebalance_due": bool(rebalance_due),
                "risk_exposure_decrease_event": bool(risk_exposure_decrease_event),
                "risk_rebalance_due": bool(risk_rebalance_due),
                "execution_rebalance_due": bool(rebalance_due or risk_rebalance_due),
                "risk_exposure_ceiling_target": float(risk_exposure_ceiling),
                "risk_target_gross_cap": float(risk_target_gross_cap),
                "risk_deleverage_shortfall_weight": float(risk_shortfall),
                "risk_deleverage_residual": risk_residual,
                "risk_unexplained_residual": risk_unexplained_residual,
                "risk_zero_target_non_rebalance_with_positions": zero_target_non_rebalance_with_positions,
                "risk_zero_target_unattempted": zero_target_unattempted,
                "risk_zero_target_unexplained_residual": zero_target_unexplained_residual,
                "requested_target_gross_exposure": float(requested_target.abs().sum()),
                "open_constrained_gross_exposure": float(open_constrained_target.abs().sum()),
                "open_constraint_limited_symbols": int(open_audit["limited_symbols"]),
                "open_constraint_blocked_buy_weight": float(open_audit["blocked_buy_weight"]),
                "open_constraint_blocked_sell_weight": float(open_audit["blocked_sell_weight"]),
                "capacity_limited_symbols": int(capacity_audit["limited_symbols"]),
                "capacity_blocked_buy_weight": float(capacity_audit["blocked_buy_weight"]),
                "capacity_blocked_sell_weight": float(capacity_audit["blocked_sell_weight"]),
            }
        )
        trade_rows.append(
            {
                "signal_date": date.strftime("%Y-%m-%d"),
                "realize_date": close.index[i + 2].strftime("%Y-%m-%d"),
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "rebalance_due": bool(rebalance_due),
                "scheduled_rebalance_due": bool(rebalance_due),
                "risk_exposure_decrease_event": bool(risk_exposure_decrease_event),
                "risk_rebalance_due": bool(risk_rebalance_due),
                "execution_rebalance_due": bool(rebalance_due or risk_rebalance_due),
                "risk_target_gross_cap": float(risk_target_gross_cap),
                "risk_deleverage_shortfall_weight": float(risk_shortfall),
                "risk_unexplained_residual": risk_unexplained_residual,
                "risk_zero_target_unattempted": zero_target_unattempted,
                "risk_zero_target_unexplained_residual": zero_target_unexplained_residual,
                "open_constraint_limited_symbols": int(open_audit["limited_symbols"]),
                "capacity_limited_symbols": int(capacity_audit["limited_symbols"]),
                "capacity_blocked_buy_weight": float(capacity_audit["blocked_buy_weight"]),
                "capacity_blocked_sell_weight": float(capacity_audit["blocked_sell_weight"]),
            }
        )

    return pd.DataFrame(nav_rows), pd.DataFrame(weight_rows), pd.DataFrame(trade_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a walk-forward next-open rank model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/next_open_rank_model")
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--retrain-frequency", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--rebalance-frequency", type=int, default=1)
    parser.add_argument("--max-position-weight", type=float, default=0.04)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--impact-bps", type=float, default=0.7)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--max-buy-open-gap", type=float, default=0.06)
    parser.add_argument("--limit-buffer", type=float, default=0.995)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--market-ma-window", type=int, default=120)
    parser.add_argument("--market-risk-off-drawdown-20d", type=float, default=-0.08)
    parser.add_argument("--market-below-ma-exposure", type=float, default=0.60)
    parser.add_argument("--market-crash-exposure", type=float, default=0.0)
    parser.add_argument("--breadth-filter", action="store_true")
    parser.add_argument("--breadth-ma-window", type=int, default=60)
    parser.add_argument("--breadth-threshold", type=float, default=0.45)
    parser.add_argument("--breadth-below-exposure", type=float, default=0.55)
    parser.add_argument("--breadth-crash-threshold", type=float, default=0.32)
    parser.add_argument("--breadth-crash-exposure", type=float, default=0.20)
    parser.add_argument("--initial-capital", type=float, default=1000000.0)
    parser.add_argument("--capacity-lookback", type=int, default=20)
    parser.add_argument("--max-daily-amount-participation", type=float, default=None)
    parser.add_argument(
        "--risk-rebalance-policy",
        choices=RISK_REBALANCE_POLICIES,
        default="scheduled_only",
    )
    parser.add_argument(
        "--training-horizons",
        default="1",
        help="Comma-separated next-open holding horizons used only for IC training.",
    )
    parser.add_argument(
        "--sleeve",
        choices=("unconstrained", *SLEEVE_FEATURE_DIRECTIONS),
        default="unconstrained",
        help="Research-only feature sleeve with fixed IC sign constraints.",
    )
    parser.add_argument(
        "--horizon-fusion",
        action="store_true",
        help="Deepen multi-horizon fusion: train per-horizon weight vectors and fuse at score level "
        "(default IC-averaging is replaced by score-level fusion). Requires --training-horizons with >=2 horizons.",
    )
    parser.add_argument(
        "--horizon-fusion-scheme",
        choices=("icmag", "equal", "decay"),
        default="icmag",
        help="Fusion weight scheme: icmag=by per-horizon mean|IC| magnitude, equal=uniform, decay=1/sqrt(h).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    calendar_audit = None
    if args.benchmark:
        calendar_audit = validate_trading_sessions(
            close.index,
            Path(args.benchmark),
            context="rank model input panel",
        )
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    all_features = build_features(close, open_px, high, low, amount)
    features, feature_directions = select_feature_sleeve(all_features, args.sleeve)
    training_horizons = parse_training_horizons(args.training_horizons)
    if args.horizon_fusion and len(training_horizons) < 2:
        # 融合无意义（单周期退化为原路径），自动补齐为 隔夜/周/月 三周期
        training_horizons = parse_training_horizons("1,5,20")
    realized_label = next_open_return_label(
        open_px, max_abs_daily_return=args.max_abs_daily_return
    )
    ic, horizon_ics = build_multi_horizon_ic(
        features,
        open_px,
        training_horizons,
        args.max_abs_daily_return,
    )
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
    market_diagnostics = build_market_diagnostics(close, market_exposure)
    equity, weights, trades = run_walk_forward(
        close=close,
        open_px=open_px,
        features=features,
        label=realized_label,
        ic=ic,
        train_days=args.train_days,
        retrain_frequency=args.retrain_frequency,
        top_n=args.top_n,
        rebalance_frequency=args.rebalance_frequency,
        max_position_weight=args.max_position_weight,
        leverage=args.leverage,
        commission_bps=args.commission_bps,
        impact_bps=args.impact_bps,
        max_buy_open_gap=args.max_buy_open_gap,
        limit_buffer=args.limit_buffer,
        market_exposure=market_exposure,
        initial_capital=args.initial_capital,
        max_training_horizon=max(training_horizons),
        feature_directions=feature_directions,
        amount=amount,
        capacity_lookback=args.capacity_lookback,
        max_daily_amount_participation=args.max_daily_amount_participation,
        risk_rebalance_policy=args.risk_rebalance_policy,
        horizon_ics=horizon_ics if args.horizon_fusion else None,
        horizon_fusion_scheme=args.horizon_fusion_scheme,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8")
    weights.to_csv(output_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8")
    trades.to_csv(output_dir / "trade_audit.csv", index=False, encoding="utf-8")
    market_diagnostics.to_csv(
        output_dir / "market_diagnostics.csv", index=False, encoding="utf-8"
    )
    ic.to_csv(output_dir / "daily_feature_ic.csv", encoding="utf-8")
    for horizon, horizon_ic in horizon_ics.items():
        horizon_ic.to_csv(
            output_dir / f"daily_feature_ic_{horizon}d.csv",
            encoding="utf-8",
        )

    metrics = {
        **calculate_walk_forward_metrics(equity, args.initial_capital),
        "training_horizons": list(training_horizons),
        "max_training_horizon": max(training_horizons),
        "training_label_policy": "purged_mature_next_open_labels",
        "realized_return_policy": "one_session_next_open_to_following_open",
        "performance_metric_policy": "initial_capital_and_complete_net_return_series",
        "capacity_lookback": args.capacity_lookback,
        "max_daily_amount_participation": args.max_daily_amount_participation,
        "risk_rebalance_policy": args.risk_rebalance_policy,
        "sleeve": args.sleeve,
        "feature_directions": feature_directions,
        "horizon_fusion": bool(args.horizon_fusion),
        "horizon_fusion_scheme": args.horizon_fusion_scheme if args.horizon_fusion else None,
        "quarantined_hybrid_features": list(QUARANTINED_HYBRID_FEATURES),
        "market_diagnostic_policy": {
            "selection_effect": False,
            "ma_window": MARKET_DIAGNOSTIC_MA_WINDOW,
            "return_window": MARKET_DIAGNOSTIC_RETURN_WINDOW,
            "output": "market_diagnostics.csv",
        },
        "trading_calendar_audit": calendar_audit,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Rank model outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
