"""成本感知回测：在 gross 权益曲线上扣 A股双边成本，给 net 区间。

模型默认 freq=5（每 5 个交易日调仓一次），约 130+ 次调仓/全样本。
净收益 ≈ 毛收益 − 每次调仓成本，成本 = 双边费率 × 调仓换手率。

用法：
  python cost_aware_backtest.py
读取 outputs/production_soft_score/book_soft_equity.csv（strongD 全样本 gross）。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EQUITY = HERE / "outputs" / "production_soft_score" / "book_soft_equity.csv"
REBALANCE = 5  # Route C freq=5


def net_curve(gross: pd.Series, cost_per_rebal: float) -> pd.Series:
    r = gross.copy()
    # 每 REBALANCE 个交易日扣一次成本（首个调仓日在第 REBALANCE 日）
    for i in range(REBALANCE, len(r), REBALANCE):
        if i < len(r):
            r.iloc[i] = r.iloc[i] - cost_per_rebal
    return r


def stats(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    total = eq.iloc[-1] - 1
    yrs = len(ret) / 252
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    return dict(total=total, cagr=eq.iloc[-1] ** (1 / yrs) - 1, sharpe=sharpe, dd=dd)


def main() -> None:
    df = pd.read_csv(EQUITY, usecols=["date", "gross_return"])
    df["date"] = pd.to_datetime(df["date"])
    gross = df["gross_return"].dropna().reset_index(drop=True)
    base = stats(gross)
    print(f"样本: {df['date'].min().date()} ~ {df['date'].max().date()}  "
          f"交易日={len(gross)}  调仓次数≈{len(gross)//REBALANCE}")
    print(f"\n[毛收益 GROSS] 总收益 {base['total']*100:+.1f}%  年化 {base['cagr']*100:.1f}%  "
          f"Sharpe {base['sharpe']:.3f}  最大回撤 {base['dd']*100:.1f}%\n")

    print("净收益敏感性（行=双边费率, 列=调仓换手率）— 数值为 净总收益% / 净Sharpe / 净最大回撤%")
    costs = [0.0010, 0.0015, 0.0020]          # 0.10% / 0.15% / 0.20% 双边
    turns = [0.30, 0.50, 0.70]                 # 每次调仓换手 30%/50%/70%
    hdr = "费率\\换手 | " + " | ".join(f"{t*100:.0f}%" for t in turns)
    print(hdr)
    for c in costs:
        cells = []
        for t in turns:
            cost_per_rebal = c * t
            s = stats(net_curve(gross, cost_per_rebal))
            cells.append(f"{s['total']*100:+.1f}%/{s['sharpe']:.2f}/{s['dd']*100:.1f}%")
        print(f"{c*100:.2f}%     | " + " | ".join(cells))

    # 基准情景：0.15% 费率 × 50% 换手
    c, t = 0.0015, 0.50
    s = stats(net_curve(gross, c * t))
    print(f"\n[基准情景 0.15%×50%换手] 净总收益 {s['total']*100:+.1f}%  净年化 {s['cagr']*100:.1f}%  "
          f"净Sharpe {s['sharpe']:.3f}  净最大回撤 {s['dd']*100:.1f}%")
    print("注：换手率为假设（等权20只调仓，部分标的不变更）；真实净值以实盘成交回测为准。")


if __name__ == "__main__":
    main()
