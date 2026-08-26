"""吸收 GitHub 因子库：QLib Alpha158 因子集（微软 MIT，纯 OHLC+amount 子集）。

来源：microsoft/qlib —— qlib/contrib/data/handler.py 的 Alpha158 数据集。
参考公式文档：Alpha158/Alpha360 因子计算公式（dev.to/henry_lin_3ac6363747f45b4）。
QLib 标签定义即 Ref($close,-2)/Ref($close,-1)-1（t+2 收益），与 model3 的 next_open
标签天然同频，故吸收价值高。

设计纪律（与 wq_alpha_factors.py 完全一致）：
- 不改动生产因子集（PRODUCTION_FEATURE_NAMES / build_features）。
- 所有公式只用 t 及之前 bar（PIT 安全）；最终统一 rank_pct 与既有因子同尺度。
- 向量化算子复用 wq_alpha_factors（含 sliding_window_view + einsum 实现）；
  新增 Alpha158 专属算子（slope/rsq/resi/argmax/argmin/quantile）亦向量化。
- 零新增 pip 依赖；不 vendoring 第三方代码。

面板缺字段处理（显式标注）：
- VWAP0（需 vwap）：**跳过**——面板无 vwap，amount 无法直接反推 vwap。
- CORR/CORD/VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD（需 volume）：
  用 amount 近似 volume，因子名后缀 [amt] 标注偏差。
  amount 与 volume 高度单调相关（amount = Σ price*volume），横截面 rank 近似有效，
  但量级/时序结构有偏差，仅作扩张候选，正 IC 才并入。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from train_next_open_rank_model import rank_pct
from wq_alpha_factors import (
    _rank_cs,
    _ts_mean,
    _ts_std,
    _ts_min,
    _ts_max,
    _ts_corr,
    _delta,
    _delay,
    _ts_rank,
    _decay_linear,
    _guard,
)

# Alpha158 滚动窗口（QLib 固定）
WINDOWS = [5, 10, 20, 30, 60]


# ----------------------------------------------------------------------------
# Alpha158 专属向量化算子（index=date, columns=symbol）
# ----------------------------------------------------------------------------
def _greater(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.maximum(a.values, b.values), index=a.index, columns=a.columns)


def _less(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.minimum(a.values, b.values), index=a.index, columns=a.columns)


def _ts_slope(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """trailing d 窗口内对时间索引的线性回归斜率（向量化）。"""
    arr = np.asarray(df.values, dtype=np.float32)
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    w = np.arange(d, dtype=np.float32) - (d - 1) / 2.0
    wden = float((w ** 2).sum())
    if T >= d:
        sw = sliding_window_view(arr, d, axis=0)            # (T-d+1, N, d)
        xmean = sw.mean(axis=2, keepdims=True)
        out[d - 1:] = (w[None, None, :] * (sw - xmean)).sum(axis=2) / wden
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _ts_rsq(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """trailing d 窗口内对时间索引线性回归的 R^2（向量化）。"""
    arr = np.asarray(df.values, dtype=np.float32)
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    w = np.arange(d, dtype=np.float32) - (d - 1) / 2.0
    wcnt = w - w.mean()
    wden = float((wcnt ** 2).sum())
    if T >= d:
        sw = sliding_window_view(arr, d, axis=0)
        xmean = sw.mean(axis=2, keepdims=True)
        xcnt = sw - xmean
        cov = (wcnt[None, None, :] * xcnt).sum(axis=2)
        varx = (xcnt ** 2).sum(axis=2)
        r = cov / np.sqrt(varx * wden + 1e-12)
        out[d - 1:] = r * r
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _ts_resi(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """trailing d 窗口内对时间索引线性回归的残差（取窗口末根 bar）。"""
    arr = np.asarray(df.values, dtype=np.float32)
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    w = np.arange(d, dtype=np.float32) - (d - 1) / 2.0
    wmean = float(w.mean())
    wcnt = w - wmean
    wden = float((wcnt ** 2).sum())
    if T >= d:
        sw = sliding_window_view(arr, d, axis=0)
        xmean = sw.mean(axis=2, keepdims=True)
        xcnt = sw - xmean
        slope = (wcnt[None, None, :] * xcnt).sum(axis=2) / wden
        pred_last = slope * (w[-1] - wmean) + xmean[:, :, 0]
        out[d - 1:] = sw[:, :, -1] - pred_last
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _ts_argmax_pos(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """trailing d 窗口内最大值的位置（0-based）。"""
    arr = np.asarray(df.values, dtype=np.float32)
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    if T >= d:
        sw = sliding_window_view(arr, d, axis=0)
        out[d - 1:] = np.argmax(sw, axis=2).astype(np.float32)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _ts_argmin_pos(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """trailing d 窗口内最小值的位置（0-based）。"""
    arr = np.asarray(df.values, dtype=np.float32)
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    if T >= d:
        sw = sliding_window_view(arr, d, axis=0)
        out[d - 1:] = np.argmin(sw, axis=2).astype(np.float32)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _ts_quantile(df: pd.DataFrame, d: int, q: float) -> pd.DataFrame:
    return df.rolling(d).quantile(q)


# ----------------------------------------------------------------------------
# Alpha158 因子集（纯 OHLC + amount 近似）
# ----------------------------------------------------------------------------
def build_alpha158_factors(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ret = _delta(close, 1)
    rng = (high - low).replace(0, np.nan)
    rng2 = rng + 1e-12
    amt_ret = _delta(amount, 1)
    log_amt = np.log(amount + 1.0)
    log_amt_chg = np.log(amount / (amount.shift(1) + 1e-12) + 1.0)
    up = (ret > 0).astype(np.float32)
    dn = (ret < 0).astype(np.float32)
    abs_ret = ret.abs()

    out: dict[str, pd.DataFrame] = {}

    # ---------- 1. K线基础特征 (9, 纯 OHLC) ----------
    out["a158_KMID"] = (close - open_px) / open_px
    out["a158_KLEN"] = rng / open_px
    out["a158_KMID2"] = (close - open_px) / rng2
    # KUP = (high - max(open,close)) / open : 上影线比例（正向）
    out["a158_KUP"] = (high - _greater(open_px, close)) / open_px
    out["a158_KUP2"] = (high - _greater(open_px, close)) / rng2
    # KLOW = (min(open,close) - low) / open : 下影线比例（正向）
    out["a158_KLOW"] = (_less(open_px, close) - low) / open_px
    out["a158_KLOW2"] = (_less(open_px, close) - low) / rng2
    out["a158_KSFT"] = (2.0 * close - high - low) / open_px
    out["a158_KSFT2"] = (2.0 * close - high - low) / rng2

    # ---------- 2. 价格特征 (3, 跳过 VWAP0) ----------
    out["a158_OPEN0"] = open_px / close
    out["a158_HIGH0"] = high / close
    out["a158_LOW0"] = low / close

    # ---------- 3. 滚动窗口技术指标 (29 类 × 5 窗 = 145) ----------
    for d in WINDOWS:
        # 3.1 价格趋势与动量
        out[f"a158_ROC{d}"] = _guard(close.shift(d) / close)
        out[f"a158_MA{d}"] = _ts_mean(close, d) / close
        out[f"a158_STD{d}"] = _ts_std(close, d) / close
        out[f"a158_BETA{d}"] = _ts_slope(close, d) / close
        out[f"a158_RSQR{d}"] = _ts_rsq(close, d)
        out[f"a158_RESI{d}"] = _ts_resi(close, d) / close
        # 3.2 价格位置
        out[f"a158_MAX{d}"] = _ts_max(high, d) / close
        out[f"a158_MIN{d}"] = _ts_min(low, d) / close
        out[f"a158_QTLU{d}"] = _ts_quantile(close, d, 0.8) / close
        out[f"a158_QTLD{d}"] = _ts_quantile(close, d, 0.2) / close
        out[f"a158_RANK{d}"] = _ts_rank(close, d)
        out[f"a158_RSV{d}"] = (close - _ts_min(low, d)) / (
            _ts_max(high, d) - _ts_min(low, d) + 1e-12
        )
        # 3.3 时间序列位置类
        out[f"a158_IMAX{d}"] = (_ts_argmax_pos(high, d) + 1.0) / d
        out[f"a158_IMIN{d}"] = (_ts_argmin_pos(low, d) + 1.0) / d
        out[f"a158_IMXD{d}"] = (_ts_argmax_pos(high, d) - _ts_argmin_pos(low, d)) / d
        # 3.4 价格-成交量关联 [amt]：用 amount 近似 volume
        out[f"a158_CORR{d}"] = _ts_corr(close, log_amt, d)
        out[f"a158_CORD{d}"] = _ts_corr(ret, log_amt_chg, d)
        # 3.5 涨跌统计类
        out[f"a158_CNTP{d}"] = up.rolling(d).mean()
        out[f"a158_CNTN{d}"] = dn.rolling(d).mean()
        out[f"a158_CNTD{d}"] = out[f"a158_CNTP{d}"] - out[f"a158_CNTN{d}"]
        # 3.6 RSI 类
        out[f"a158_SUMP{d}"] = up.rolling(d).sum() / (abs_ret.rolling(d).sum() + 1e-12)
        out[f"a158_SUMN{d}"] = dn.rolling(d).sum() / (abs_ret.rolling(d).sum() + 1e-12)
        out[f"a158_SUMD{d}"] = out[f"a158_SUMP{d}"] - out[f"a158_SUMN{d}"]
        # 3.7 成交量技术指标 [amt]：用 amount 近似 volume
        out[f"a158_VMA{d}"] = _ts_mean(amount, d) / (amount + 1e-12)
        out[f"a158_VSTD{d}"] = _ts_std(amount, d) / (amount + 1e-12)
        wvma_num = (abs_ret * amount).rolling(d).std()
        wvma_den = (abs_ret * amount).rolling(d).mean() + 1e-12
        out[f"a158_WVMA{d}"] = wvma_num / wvma_den
        amt_up = (amt_ret > 0).astype(np.float32)
        amt_dn = (amt_ret < 0).astype(np.float32)
        amt_abs = amt_ret.abs()
        out[f"a158_VSUMP{d}"] = amt_up.rolling(d).sum() / (amt_abs.rolling(d).sum() + 1e-12)
        out[f"a158_VSUMN{d}"] = amt_dn.rolling(d).sum() / (amt_abs.rolling(d).sum() + 1e-12)
        out[f"a158_VSUMD{d}"] = out[f"a158_VSUMP{d}"] - out[f"a158_VSUMN{d}"]

    # 统一 rank_pct，与既有因子同尺度（spearman IC 不变）
    return {name: rank_pct(frame) for name, frame in out.items()}
