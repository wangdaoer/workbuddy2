"""P10 因子库扩张：在 P8b 冠军因子集（16 个）基础上新增维度不同的候选因子。

设计原则：
- 不改动生产因子集（PRODUCTION_FEATURE_NAMES / build_features），新增因子独立追加。
- 新增因子覆盖：多周期动量/反转、长周期均线距离、Amihud 非流动性、流动性骤增、
  归一化振幅、隔夜跳空、波动偏度、开盘位、RSI 型有界振荡器，及其与既有反转逻辑的复合。
- 全部 rank 化，与既有因子同尺度，交给 run_walk_forward 的自适应 IC>0 选择。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from train_next_open_rank_model import build_features, rank_pct


def _safe_ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return num / (den + 1e-12)


def build_features_expanded(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    base = build_features(close, open_px, high, low, amount)
    returns = close.pct_change(fill_method=None)
    rng = (high - low).replace(0, np.nan)

    new: dict[str, pd.DataFrame] = {}

    # ---- 多周期动量 / 反转（补齐 5/20/60 之外的周期）----
    new["momentum_10"] = close.pct_change(10, fill_method=None)
    new["momentum_120"] = close.pct_change(120, fill_method=None)
    new["reversal_10"] = -close.pct_change(10, fill_method=None)
    new["reversal_20"] = -close.pct_change(20, fill_method=None)

    # ---- 长周期均线距离（趋势强度）----
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    new["distance_ma60"] = _safe_ratio(close, ma60) - 1.0
    new["distance_ma120"] = _safe_ratio(close, ma120) - 1.0

    # ---- Amihud 非流动性（|收益| / 成交额）----
    illiq = returns.abs() / amount.replace(0, np.nan)
    new["amihud_20"] = illiq.rolling(20).median()

    # ---- 流动性骤增（当日成交额 / 20 日中位数）----
    new["volume_ratio_20"] = _safe_ratio(amount, amount.rolling(20).median())

    # ---- 归一化振幅（(H-L)/C 的 20 日均值）----
    new["range_20"] = (rng / close).rolling(20).mean()

    # ---- 隔夜跳空（今开 / 昨收 - 1）----
    new["gap_1"] = _safe_ratio(open_px, close.shift(1)) - 1.0

    # ---- 波动偏度（上行波动 - 下行波动）----
    up = returns.clip(lower=0.0)
    dn = returns.clip(upper=0.0).abs()
    new["vol_skew_20"] = up.rolling(20).std() - dn.rolling(20).std()

    # ---- 开盘位（(O-L)/(H-L)，与上影/下影相关，和既有 close_position 互补）----
    new["open_position"] = _safe_ratio(open_px - low, rng)

    # ---- RSI 型有界振荡器（20 日收益的指数加权，压缩到 [-1,1] 再 rank）----
    ema_up = up.ewm(alpha=1 / 14).mean()
    ema_dn = dn.ewm(alpha=1 / 14).mean()
    rsi = _safe_ratio(ema_up, ema_up + ema_dn)          # 0..1
    new["rsi_14"] = rsi - 0.5                            # -0.5..0.5，rank 后方向由选择决定

    # ---- 复合因子（复用既有 rank 因子与新因子，乘积后 rank）----
    new["strong_mom120_reversal5"] = rank_pct(new["momentum_120"] * base["reversal_5"])
    new["liquid_momentum"] = rank_pct(base["liquidity_20"] * base["momentum_20"])
    new["breakout_range"] = rank_pct(base["breakout_20"] * new["range_20"])
    new["high_range_reversal"] = rank_pct(new["range_20"] * base["reversal_5"])
    new["gap_reversal"] = rank_pct(new["gap_1"] * base["reversal_5"])

    # 全部 rank 化（base 已是 rank，仅 new 的基础量需 rank；复合已 rank_pct）
    ranked_new = {name: (rank_pct(frame) if name not in (
        "strong_mom120_reversal5", "liquid_momentum", "breakout_range",
        "high_range_reversal", "gap_reversal") else frame)
        for name, frame in new.items()}

    return {**base, **ranked_new}
