"""吸收券商研报因子：19 篇 A 股量价/行为金融研报挖掘候选（model3 适配版）。

来源：见 outputs/broker_factor_mining_report.md（华泰/华西/光大/东方/开源/海通/招商/方正/方大/广发等）。
参考实现：自实现，不依赖任何研报附带代码，保证 PIT 安全。

设计纪律（与 wq_alpha_factors.py / alpha158_factors.py 一致）：
- 不改动生产因子集（PRODUCTION_FEATURE_NAMES / build_features）。
- 仅用面板既有字段 close/open/high/low/amount（volume 用 amount 近似；日收益由 close 推导；
  市场收益 r_m 用截面均值代）。
- 所有公式只用 t 及之前 bar 的数据（trailing 窗口），天然 PIT 安全。
- 统一 rank_pct，与既有因子同尺度，交给 walk-forward 的 select_positive(IC>0) 自适应选择。
- 因子命名 brk_ 前缀，避免与 wq_/a158_/exp_ 冲突。

候选分档（详见挖掘报告）：
- A 组 (YES, 纯 OHLC/amount 直供)：振幅切割动量/反转、理想振幅、成交额/价格收敛 ACF/PCF、
  凸显 STR、多空对比 BB-Total、惊恐/草木皆兵、球队硬币。
- B 组 (PARTIAL, 代理近似)：APB 买卖压力(典型价代VWAP)、成交额等分动量、量价背离族、
  PVCF 收敛(amount/均价代volume)。
- C1 方法论复用：纯真去相关（对 A3 理想振幅 / A4 成交额收敛 生成正交变体 brk_*_pure）。

符号约定：报告给出各因子已知 IC 符号；对已知为负 IC 的构造（A3/A6/A8/A9）已取反，
使因子方向与"正向预测 next_open 收益"一致，最大化被 select_positive(IC>0) 选入的概率。
（若取反方向判断失误，该因子仅不被选入，不构成污染。）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from train_next_open_rank_model import rank_pct
from wq_alpha_factors import (
    _rank_cs,
    _ts_corr,
    _guard,
)


# ----------------------------------------------------------------------------
# 研报专属算子（index=date, columns=symbol）
# ----------------------------------------------------------------------------
def _ma_convergence(x: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """多均线离散度（开源《均线收敛发散》）：-log(1+std(各 k 日均线))。

    离散度越低 = 均线越收敛 = 趋势越平稳。取负使"收敛"为正信号。
    """
    mas = [x.rolling(w).mean() for w in windows]
    stacked = np.stack([m.values for m in mas], axis=2)     # (T, N, k)
    std = np.nanstd(stacked, axis=2)
    return pd.DataFrame(-np.log1p(std), index=x.index, columns=x.columns)


def _amp_segment_sum(ret: pd.DataFrame, amp: pd.DataFrame, d: int, q: float,
                     high: bool) -> pd.DataFrame:
    """振幅切割分段收益加总（开源《如何构造动量》）。

    对每根 bar t，取 trailing 窗口内：
      - high=False：振幅位于最低 q 分位（低振幅日）的收益加总 -> 动量结构；
      - high=True ：振幅位于最高 (1-q) 分位（高振幅日）的收益加总 -> 反转结构。
    采用自适应局部阈值（每根 bar 用自身 trailing d 窗口的分位），等价捕捉低/高振幅日，
    且避免 3D 逐窗 nanquantile 的高开销。rolling 自动忽略 NaN；窗口未成形的前 d-1 行留 NaN。
    """
    thr = amp.rolling(d, min_periods=d).quantile(q)        # (T,N) 窗口内 q 分位阈值
    valid = thr.notna()
    mask = ((amp >= thr) if high else (amp <= thr)) & valid
    sel = ret.where(mask, 0.0)
    return sel.rolling(d, min_periods=d).sum()


def _price_tier_amp_diff(close: pd.DataFrame, amp: pd.DataFrame, d: int,
                         hi_q: float, lo_q: float) -> pd.DataFrame:
    """理想振幅 V(λ)=V_high−V_low（开源《振幅隐藏结构》）。

    对每根 bar t，按窗口内收盘价高低分两层（各取 λ 分位边界），分别求振幅均值，
    返回 high 层振幅 − low 层振幅。采用自适应局部分位（等价捕捉高/低价层），全 1D rolling 实现。
    """
    qh = close.rolling(d, min_periods=d).quantile(hi_q)
    ql = close.rolling(d, min_periods=d).quantile(lo_q)
    valid = qh.notna() & ql.notna()
    mh = (close >= qh) & valid
    ml = (close <= ql) & valid
    # 条件滚动均值：在 trailing d 窗口内，对落入该价格层（高/低价层）的日子的振幅取均值。
    # min_periods=1：窗口内只要存在至少 1 个落入该层的日子即可给出均值（层内日数本就稀疏）。
    vh = amp.where(mh).rolling(d, min_periods=1).mean()
    vl = amp.where(ml).rolling(d, min_periods=1).mean()
    return vh - vl


def _pure_decorrelate(frame: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    """C1 纯真去相关（东吴《纯真波动率》）：减掉前 lags 期自身均值，剔除跨期水平依赖。

    对波动率/振幅类因子做"去趋势"正交化，生成纯真版变体。
    廉价实现：residual = x_t − mean(x_{t-lags..t-1})。
    """
    trailing = frame.shift(1).rolling(lags, min_periods=lags).mean()
    return frame - trailing


def _salience_weighted_return(ret: pd.DataFrame, weight: pd.DataFrame,
                              win: int = 20) -> pd.DataFrame:
    """凸显/惊恐通用：权重 ω 加权的窗口内收益（salience-weighted return）。"""
    wret = (weight * ret).rolling(win).sum()
    wsum = weight.rolling(win).sum().replace(0, np.nan)
    return wret / wsum


# ----------------------------------------------------------------------------
# 研报因子全集（仅 OHLC + amount；PIT 安全；rank_pct 统一尺度）
# ----------------------------------------------------------------------------
def build_broker_mined_factors(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ret = close.pct_change(fill_method=None)
    amp = (high - low).replace(0, np.nan)                  # 当日振幅
    rng = (high - low).replace(0, np.nan)
    tp = (high + low + close) / 3.0                        # 典型价（VWAP 代理）
    r_m = ret.mean(axis=1)                                 # 截面市场收益代理 (T,)

    # 凸显权重 ω 与惊恐(非对称 dread)权重（均需 r_m 广播）
    rm_bc = r_m.values[:, None]
    salience_w = (ret - rm_bc).abs() / (ret.abs() + np.abs(rm_bc) + 0.1)
    dread_w = (rm_bc - ret).clip(lower=0.0) / ((rm_bc - ret).abs() + 0.1)

    out: dict[str, pd.DataFrame] = {}

    # ---- A1 振幅切割动量 A(λ)：近 160 日最低 70% 振幅日的收益加总（动量） ----
    a1 = _amp_segment_sum(ret, amp, 160, 0.70, high=False)
    out["brk_a1_ampcut_mom"] = a1

    # ---- A2 振幅切割反转 B：近 160 日最高 30% 振幅日的收益加总取反（反转） ----
    a2 = _amp_segment_sum(ret, amp, 160, 0.70, high=True)
    out["brk_a2_ampcut_rev"] = -a2

    # ---- A3 理想振幅 V(λ)=V_high−V_low（取反使正向；IC≈−0.067 -> 取反为正） ----
    a3 = _price_tier_amp_diff(close, amp, 20, 0.75, 0.25)
    out["brk_a3_ideal_amp"] = -a3
    # C1 纯真去相关变体（对 A3 原始信号先做去相关再取反）
    out["brk_a3_ideal_amp_pure"] = rank_pct(-_pure_decorrelate(a3, 6))

    # ---- A4 成交额收敛度 ACF：amount 六均线离散度（RankIC 10.30%，最优且无需代理） ----
    a4 = _ma_convergence(amount, [1, 5, 10, 20, 60, 120])
    out["brk_a4_acf"] = a4
    out["brk_a4_acf_pure"] = rank_pct(_pure_decorrelate(a4, 6))

    # ---- A5 价格收敛度 PCF：close 六均线离散度（RankIC 2.78%，偏弱但独特） ----
    out["brk_a5_pcf"] = _ma_convergence(close, [1, 5, 10, 20, 60, 120])

    # ---- A6 凸显 STR：对称凸显权重下的 salience-weighted return（IC≈−3.8% -> 取反） ----
    str_sal = _salience_weighted_return(ret, salience_w, 20)
    out["brk_a6_str"] = rank_pct(-str_sal)

    # ---- A7 多空对比总量 BB-Total：−Σ(C−L)/(H−C) 窗口求和（IC 0.063，正） ----
    bb = -((close - low) / (high - close).replace(0, np.nan)).rolling(20).sum()
    out["brk_a7_bbtotal"] = bb

    # ---- A8 惊恐/草木皆兵(OHLC 部分)：非对称 dread 权重 salience-weighted return（IC≈−8.9% -> 取反） ----
    dread_sal = _salience_weighted_return(ret, dread_w, 20)
    out["brk_a8_dread"] = rank_pct(-dread_sal)

    # ---- A9 球队硬币(波动翻转)：高波动翻转日取反转、低波动延续日取动量，求和取反（IC≈−9.67% -> 取反） ----
    flip = (ret * ret.shift(1)) < 0
    coin = (-ret.where(flip, 0.0))                         # 翻转日反转
    team = (ret.where(~flip, 0.0))                         # 延续日动量
    a9 = (coin + team).rolling(20).sum()
    out["brk_a9_cointeam"] = rank_pct(-a9)

    # ---- B1 APB 买卖压力：ln(mean(TP) / Σ(amount·TP)/Σ(amount))（RankIC 9.07%，正） ----
    mean_tp = tp.rolling(20).mean()
    wtd_tp = (amount * tp).rolling(20).sum() / amount.rolling(20).sum().replace(0, np.nan)
    out["brk_b1_apb"] = np.log(_guard(mean_tp / wtd_tp.replace(0, np.nan)))

    # ---- B2 成交额等分 K 线动量（代理）：流动性加权累积收益（amount 代 volume） ----
    out["brk_b2_amt_wtd_mom"] = (
        (amount * ret).rolling(20).sum() / amount.rolling(20).sum().replace(0, np.nan))

    # ---- B3 量价背离族（amount/均价 代 volume；排序/相关对量纲不敏感） ----
    amt_chg = amount.pct_change(fill_method=None)
    out["brk_b3_pv_div"] = -_ts_corr(_rank_cs(ret), _rank_cs(amt_chg), 20)   # 量价秩相关取反=背离
    out["brk_b3_ampvol"] = _ts_corr(_rank_cs(rng), _rank_cs(amount), 20)     # 振幅-量 同向

    # ---- B5 PVCF 收敛（PV=price×volume≈amount；用 close*amount 作价格加权收敛，区别于 A4） ----
    pv = close * amount
    out["brk_b5_pvcf"] = _ma_convergence(pv, [1, 5, 10, 20, 60, 120])

    # 统一 rank_pct，与既有因子同尺度（spearman IC 不变）
    return {name: rank_pct(frame) for name, frame in out.items()}
