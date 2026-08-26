"""吸收 GitHub 因子库：WorldQuant 101 Formulaic Alphas 精选子集（model3 适配版）。

来源：Zura Kakushadze《101 Formulaic Alphas》(arXiv:1601.00991)。
参考实现：lvlh2/alpha101（GitHub，仅作公式对照，本模块不依赖该包，自实现以保证 PIT 安全）。

设计纪律（与 factor_expansion.py 一致）：
- 不改动生产因子集（PRODUCTION_FEATURE_NAMES / build_features）。
- 仅用面板既有字段 close/open/high/low/amount（无 VWAP、无行业、无额外量价字段）。
- 所有公式只用 t 及之前 bar 的数据；WQ101 公式基于当根 bar 收盘可得量，天然 PIT 安全
  （model3 标签 = next_open，在 t+2 开盘成熟，与当日因子无冲突）。
- 最终统一 rank_pct，与既有因子同尺度，交给 walk-forward 的 IC>0 自适应选择。

显式跳过（面板缺字段）：
- 需 VWAP 的 alpha（#2/3/7/8/11/19/32/37/53/56/57/62/73/83/89…）
- 需 INDUSTRY 行业中性化的 alpha（#10/27/28/30/31/33/34/66/68/69/76/81/85…）
- 个别用 amount（成交额）近似 adv/vwap 的，已在因子名/注释标注偏差。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from train_next_open_rank_model import rank_pct


# ----------------------------------------------------------------------------
# 时序 / 横截面 算子（向量化；index=date, columns=symbol）
# ----------------------------------------------------------------------------
def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 rank 百分位（WQ 的 rank）。"""
    return df.rank(axis=1, pct=True)


def _ts_mean(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).mean()


def _ts_std(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).std()


def _ts_min(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).min()


def _ts_max(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).max()


def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).corr(y)


def _delta(df: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    return df - df.shift(d)


def _delay(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.shift(d)


def _ts_rank(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序 rank：当前值在 trailing d 窗口内的百分位排名 (1/d .. 1)。向量化。"""
    arr = np.asarray(df.values, dtype=np.float32)
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    if T >= d:
        sw = sliding_window_view(arr, d, axis=0)          # (T-d+1, N, d)
        last = sw[:, :, -1:]                               # (T-d+1, N, 1)
        cnt = (sw <= last).sum(axis=2)                    # (T-d+1, N)
        out[d - 1:] = cnt / d
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _decay_linear(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """时序加权移动平均：窗口内线性衰减权重（最新= d，最旧= 1）。向量化。"""
    arr = np.asarray(df.values, dtype=np.float32)
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    w = np.arange(1, d + 1, dtype=np.float32)
    w = w / w.sum()
    if T >= d:
        sw = sliding_window_view(arr, d, axis=0)          # (T-d+1, N, d)
        out[d - 1:] = np.einsum("tnk,k->tn", sw, w)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _guard(div: pd.DataFrame) -> pd.DataFrame:
    """把除零产生的 inf 归为 NaN，避免 rank_pct 失真。"""
    return div.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# 精选 alpha 子集（仅 OHLC + amount）
# ----------------------------------------------------------------------------
def build_wq_alpha_factors(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ret = close.pct_change(fill_method=None)
    rng = (high - low).replace(0, np.nan)                 # 当日振幅
    rng5_sum = rng.rolling(5).sum().replace(0, np.nan)    # alpha#6/20 分母
    amt_rank = _rank_cs(amount)
    typ = (high + low + close) / 3.0                      # 典型价（vwap 近似）

    out: dict[str, pd.DataFrame] = {}

    # Alpha#1: 价量相关（用 amount 近似 adv）—— 量价背离的前瞻结构
    out["wq_a1_pricevol_corr5"] = _rank_cs(_ts_corr(close, amt_rank, 5))

    # Alpha#4: 弱势（low 的横截面 rank 在 9 日窗口内的时序低位）
    out["wq_a4_low_weak9"] = -_ts_rank(_rank_cs(low), 9)

    # Alpha#6: 日内弱势归一（(开-收) 排名 / 5 日振幅和）
    out["wq_a6_intraday_weak"] = _guard(_rank_cs(open_px - close) / rng5_sum)

    # Alpha#9: 近期最低价（5 日窗口内最低收盘的 rank）
    out["wq_a9_recent_min"] = _rank_cs(_ts_min(_rank_cs(close), 5))

    # Alpha#12: 量价背离（量增 * 价跌 为正向信号）
    out["wq_a12_voldiv"] = np.sign(_delta(amount, 1)) * (-_delta(close, 1))

    # Alpha#14: 价格时序低位（5 日窗口内 close 的 rank 弱势）
    out["wq_a14_price_rank5"] = -_rank_cs(_ts_rank(close, 5))

    # Alpha#15: 高低相关形态（high-low 相关在 5 日窗口持续为负向）
    out["wq_a15_hl_corr_shape"] = -_ts_corr(high, low, 5).pipe(_rank_cs).rolling(5).sum()

    # Alpha#17: 典型价 vs 收盘（vwap 代理；标注偏差）
    out["wq_a17_typ_vs_close"] = _rank_cs(typ - close)

    # Alpha#20: 开盘-收盘弱势归一（与 #6 同向，量纲更稳）
    out["wq_a20_openclose_weak"] = -_guard(_rank_cs(open_px - close) / rng5_sum)

    # Alpha#23: 高低比（5 日均高 / 5 日均低 - 1）
    out["wq_a23_hl_ratio5"] = _ts_mean(high, 5) / (_ts_mean(low, 5) + 1e-12) - 1.0

    # Alpha#24: 收益衰减（5 日收益做 decay_linear，近期加权）
    out["wq_a24_ret_decay5"] = _decay_linear(_delta(close, 5), 5)

    # Alpha#26: 高低相关峰值（5 日 high-low 相关最大值，负向）
    out["wq_a26_hl_corr_max5"] = -_ts_corr(high, low, 5).rolling(5).max()

    # Alpha#44: 双时序 rank 复合（价格 + 量）
    out["wq_a44_dual_rank"] = _ts_rank(close, 10) + _ts_rank(amt_rank, 10)

    # Alpha#51: 自高位回落（close / 20 日最高 high - 1）
    out["wq_a51_off_high20"] = close / (_ts_max(high, 20) + 1e-12) - 1.0

    # Alpha#54: 振幅弱势（振幅/收盘 越小越弱）
    out["wq_a54_range_weak"] = -_rank_cs(rng / (close + 1e-12))

    # Alpha#101: 实体占比（(收-开)/(高-低+eps)）
    out["wq_a101_body_ratio"] = (close - open_px) / (rng + 0.001)

    # Alpha#13: sqrt 形态（sqrt(high*close) - sqrt(low*close)）
    out["wq_a13_sqrt_shape"] = np.sqrt(high * close) - np.sqrt(low * close)

    # Alpha#19 简化（无 vwap）：1 日收益反转
    out["wq_a19_ret_reversal"] = -ret

    # Alpha#9b: 量时序 rank（量能相对近期位置）
    out["wq_a9b_vol_rank10"] = _ts_rank(amt_rank, 10)

    # 统一 rank_pct，与既有因子同尺度（spearman IC 不变）
    return {name: rank_pct(frame) for name, frame in out.items()}
