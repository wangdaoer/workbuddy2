"""pit_universe.py — point-in-time（截至 t 日）股票池筛选框架。

核心纪律：任何"某标的在日期 t 是否合格"的判断，只能使用 ≤t 的信息。
绝对禁止用未来信息（未来收益、未来退市状态、修订后基本面、或"当前幸存者名单"）。

这是根治"筛选过的自选股"前视的唯一办法：在每个再平衡日 t，只用 ≤t 的数据
重算合格宇宙，而不是拿一张"当前好票"的静态名单去回测历史。

提供：
  - pit_presence / pit_suspension / pit_liquidity / pit_min_history / pit_eligible
        : 严格 ≤t 的资格掩码（全部由历史数据滚动得出，无前视）
  - static_survivor_mask
        : 反例——把"最后一天存在的标的"广播到所有历史日
          = 用 2026 年幸存者名单回测 2022 年（确定的前视错误写法）
  - detect_static_leak
        : 给定任意静态 watchlist，量化其在各历史日的前视污染度
  - selftest
        : 断言"某票 t+δ 退市，框架在 t 日仍判合格" → 证明无前视
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def pit_presence(close: pd.DataFrame) -> pd.DataFrame:
    """point-in-time 上市/未退市：t 日有收盘价即视为当时在市。"""
    return close.notna()


def pit_suspension(amount: pd.DataFrame) -> pd.DataFrame:
    """t 日成交额>0 视为正常交易（停牌/无成交则排除）。用 amount 而非 volume，
    以兼容只暴露 amount 的面板对象（如 production_soft_score.build_panel）。"""
    return amount.fillna(0.0) > 0


def pit_liquidity(amount: pd.DataFrame, thr: float = 1e8, lookback: int = 252,
                   shift: int = 1) -> pd.DataFrame:
    """trailing 流动性：滚动 median(amount) ≥ thr，严格因果（shift 避免用到 t 当日额）。"""
    roll = amount.rolling(lookback, min_periods=max(20, lookback // 4)).median()
    if shift:
        roll = roll.shift(shift)
    return roll >= thr


def pit_min_history(close: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    """上市满 min_days 才纳入，过滤新股上市初期噪声 / 涨跌停异常。"""
    if min_days <= 0:
        return pd.DataFrame(True, index=close.index, columns=close.columns)
    cnt = close.notna().cumsum()
    return cnt >= min_days


def pit_eligible(close, amount, volume=None, thr: float = 1e8, lookback: int = 252,
                 min_days: int = 60) -> pd.DataFrame:
    """组合 PIT 资格：在市 & 非停牌 & 流动性达标 & 上市满 min_days。全部 ≤t。
    `volume` 可选（兼容只给 amount 的面板）；停牌统一用 amount>0 判定。"""
    return (pit_presence(close) & pit_suspension(amount) &
            pit_liquidity(amount, thr, lookback) & pit_min_history(close, min_days))


def static_survivor_mask(close: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """反例 / 前视错误写法：取 as_of 日（默认最后一日）存在的标的，广播到所有历史日。
    等价于'用 2026 年的幸存者名单去回测 2022 年'——确定的前视。"""
    if as_of is None:
        as_of = close.index[-1]
    present_final = close.loc[as_of].notna()
    return pd.DataFrame(np.tile(present_final.values, (len(close.index), 1)),
                        index=close.index, columns=close.columns)


def detect_static_leak(close, amount, volume=None, static_list=None, thr: float = 1e8,
                       lookback: int = 252, min_days: int = 60) -> dict:
    """量化静态 watchlist 的前视污染度。

    对每个历史日 t，统计 static_list 中有多少标的'当时并不满足 PIT 资格'
    （未上市 / 已退市 / 停牌 / 不流动）——这部分就是'用未来信息筛出来的'。
    static_list 缺省时取面板末日存在的全样本，作为"典型幸存者 watchlist"示例。
    """
    if static_list is None:
        static_list = list(close.columns)
    sym_set = [s for s in static_list if s in close.columns]
    if not sym_set:
        raise ValueError("static_list 中没有命中面板的标的")
    pit = pit_eligible(close, amount, None, thr, lookback, min_days)
    pit_sub = pit.reindex(columns=sym_set).fillna(False)
    leak_by_date = (~pit_sub).mean(axis=1)
    return {
        "n_watchlist": len(sym_set),
        "mean_leak_ratio": float(leak_by_date.mean()),
        "max_leak_ratio": float(leak_by_date.max()),
        "min_leak_ratio": float(leak_by_date.min()),
        "first_date": str(close.index[0].date()),
        "last_date": str(close.index[-1].date()),
    }


def selftest() -> None:
    """断言框架无前视：退市信息不会泄漏到退市前的判断里。"""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    idx_a = dates[:130]                      # 股 A 只在前 130 个交易日存在，之后退市
    close = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    close.loc[idx_a, "A"] = 10.0
    close["B"] = 10.0
    amount = close * 1e7
    volume = close * 1e5
    mask = pit_eligible(close, amount, volume, thr=1e6, lookback=20, min_days=0)
    last_a = idx_a[-1]
    # 退市前一日，A 应合格（此时无从知晓未来退市）
    assert bool(mask.loc[last_a, "A"]) is True, "退市前一日应合格"
    # 退市后（无数据日），A 不合格
    after = dates[131]
    assert bool(mask.loc[after, "A"]) is False, "退市后数据缺失应不合格"
    # static_survivor 反例：用最后一日(只有 B)的名单广播，A 在前期被错误排除 = 前视
    sm = static_survivor_mask(close)
    assert bool(sm.loc[idx_a[0], "A"]) is False, "static 反例：前期 A 被未来名单排除=前视"
    print("[selftest] PIT 无前视 OK ; static_survivor 确为前视反例 OK")


if __name__ == "__main__":
    selftest()
