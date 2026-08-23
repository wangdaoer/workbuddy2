"""cost_model.py — 交易成本 / 滑点 / 换手 / 回撤 统一模型 (方案 B step4a).

提供:
  - apply_costs(): 把毛前向收益序列转净收益 (印花税+佣金+滑点+冲击)
  - turnover(): 两期持仓权重的变动率, 用于换手成本
  - max_drawdown(): 累计净值序列最大回撤
  - net_portfolio_path(): 组合逐期净值路径 (含成本), 供回撤/夏普统计

费率假设 (A股单边, 可在调用处覆盖):
  stamp_tax   = 0.0005  (印花税, 仅卖出)
  commission  = 0.00025 (佣金, 双边, 不足5元按5元/笔 — 此处按费率近似, 小单另计)
  slippage    = 动态: 按当日振幅估算 (低流动性/涨停跌停不可用)
  impact      = 0.0001 * sqrt(weight)  简化市场冲击 (大权重更贵)

涨停/跌停/停牌处理: 若买入日无 open 或当日一字板, 标记不可交易 (收益记 NaN, 不计入)。
"""
from __future__ import annotations

import pandas as pd

# 默认费率
STAMP_TAX = 0.0005
COMMISSION = 0.00025
IMPACT_K = 0.0001


def _slippage_from_range(open_p: float, high_p: float, low_p: float) -> float:
    """用当日 [open,high,low] 估算相对滑点 (振幅代理)。振幅越大滑点越高, 封顶 0.01。"""
    if open_p is None or pd.isna(open_p) or open_p == 0:
        return 0.01  # 无价格信息, 保守按最大滑点
    rng = (high_p - low_p) / open_p if high_p is not None and low_p is not None else 0.0
    # 滑点约为日内振幅的一半, 封顶 1%
    return min(0.01, max(0.0005, abs(rng) * 0.5))


def apply_costs(gross_ret: float, weight: float = 0.05, *,
                stamp: float = STAMP_TAX, comm: float = COMMISSION,
                slippage: float = 0.001, tradable: bool = True) -> float:
    """单笔毛收益 -> 净收益。tradable=False 时返回 NaN (不可交易, 不计入)。"""
    if not tradable or gross_ret is None or pd.isna(gross_ret):
        return float("nan")
    # 买入成本: 佣金 + 滑点 + 冲击
    buy_cost = comm + slippage + IMPACT_K * (weight ** 0.5)
    # 卖出成本: 印花税 + 佣金 + 滑点 + 冲击
    sell_cost = stamp + comm + slippage + IMPACT_K * (weight ** 0.5)
    net = (1.0 + gross_ret) * (1.0 - buy_cost) - sell_cost - 1.0
    return net


def turnover(weights_t0: pd.Series, weights_t1: pd.Series) -> float:
    """两期权重向量的单边换手率 (sum|w1-w0|)/2 的等价: 用重叠部分算变更。"""
    w0 = weights_t0.fillna(0.0)
    w1 = weights_t1.reindex(w0.index).fillna(0.0)
    return float((w0 - w1).abs().sum() / 2.0)


def max_drawdown(equity: pd.Series) -> float:
    """输入累计净值序列, 返回最大回撤 (负数, 如 -0.12 表示 -12%)。"""
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def net_portfolio_path(period_returns: pd.Series, weights: pd.Series | None = None,
                       *, costs_per_period: float | pd.Series = 0.0) -> pd.Series:
    """由逐期净收益构造累计净值 (起始=1.0)。

    period_returns: 每期组合毛/净收益序列 (index=期)
    costs_per_period: 每期额外成本 (如换手成本), 标量或序列
    """
    r = period_returns.copy()
    if isinstance(costs_per_period, (int, float)):
        costs_per_period = pd.Series(costs_per_period, index=r.index)
    else:
        costs_per_period = costs_per_period.reindex(r.index).fillna(0.0)
    net = r - costs_per_period
    equity = (1.0 + net.fillna(0.0)).cumprod()
    equity.name = "equity"
    return equity
