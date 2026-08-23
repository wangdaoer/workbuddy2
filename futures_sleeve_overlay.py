"""P9-2：衍生品 sleeve / 指数对冲 overlay 分支（接入 high_risk_quant_model3）。

本模块消费 P9-1 已验证的连续近月期货面板（futures_near_*.csv），提供两条可叠加到
A 股横截面排序模型（P0–P8 冠军 = p8b_dynamic_liquidity）的跨市场分支：

  1) 期货 sleeve（指数期货横截面动量）：
     对 IF/IH/IC/IM 四个指数期货，按 20 日动量做横截面排序，多最强 / 空最弱，
     等权；用 change2/前结算 计算「展期无跳变」的连续日收益。

  2) 指数对冲 overlay（给 A 股权益簿做 Beta 对冲）：
     以 IF（沪深300）为主对冲工具、IH（上证50）补足大市值暴露，对权益簿日收益
     做滚动 Beta 估计并对冲，输出对冲后权益曲线与回撤改善。

设计约束 / 诚实声明：
  - 连续合约日收益用 change2 / prev_settle（= 当日结算 - 同合约前结算），
    自动桥接展期日的基差跳变，无需额外回溯调整。
  - 本模块对数据「逐合约」计算收益，因此即便只喂入抽样月份（非连续）也能跑通
    sleeve 逻辑；但要用 overlay 与 P8b 权益曲线逐日融合，必须先做「连续月份回填」
    （下载 202312–202607 相邻月 zip，见 P9-2.5）。
  - 期货 sleeve 的交易成本按 0.6 bp / 单边（~1 个指数点）估算，保守。

用法：
  # 仅期货 sleeve（可用抽样月份验证）
  python3 futures_sleeve_overlay.py --futures external_data/derivatives/futures_near_crossyear.csv \
      --out external_data/derivatives/futures_sleeve_overlay.csv

  # sleeve + 对冲 overlay（需权益曲线 CSV，且 futures 与 equity 日期对齐）
  python3 futures_sleeve_overlay.py --futures external_data/derivatives/futures_near_contig.csv \
      --equity outputs/p8b_dynamic_liquidity/equity_curve_aum_100000000.csv \
      --out external_data/derivatives/futures_sleeve_overlay.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_FUTURES = ["IF", "IH", "IC", "IM"]
MULT = {"IF": 300, "IH": 300, "IC": 200, "IM": 200,
        "T": 10000, "TF": 10000, "TS": 20000, "TL": 10000}
SLEEVE_COST_BPS = 0.6  # 单边
ROLL_BETA_WIN = 60      # 滚动 Beta 估计窗口（交易日）


def load_futures(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["trade_date"] = d["trade_date"].astype(str)
    for c in ["open", "high", "low", "close", "volume", "amount", "oi",
              "settlement", "prev_settle", "change1", "change2"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values(["underlying", "trade_date"]).reset_index(drop=True)


def roll_bridged_returns(fut: pd.DataFrame) -> pd.DataFrame:
    """每个标的逐合约计算连续日收益 = change2 / prev_settle（展期无跳变）。"""
    out = []
    for u, g in fut.groupby("underlying"):
        g = g.sort_values("trade_date").copy()
        g["ret"] = g["change2"] / g["prev_settle"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_sleeve(fut: pd.DataFrame, mom_win: int = 20) -> dict:
    """指数期货横截面动量 sleeve：每日多动量最强、空最弱，等权。"""
    r = roll_bridged_returns(fut[fut.underlying.isin(INDEX_FUTURES)]).copy()
    r["trade_date"] = r["trade_date"].astype(str)
    wide = r.pivot(index="trade_date", columns="underlying", values="ret")
    wide = wide[INDEX_FUTURES]
    # 横截面动量 = 各期货近 mom_win 日累收益的横截面排序
    mom = wide.rolling(mom_win).sum()
    rank = mom.rank(axis=1, pct=True)
    # 多 top-1（rank>0.75），空 bottom-1（rank<0.25）
    pos = (rank > 0.75).astype(float) - (rank < 0.25).astype(float)
    # 归一：每边等权到 ±0.5，总杠杆 1.0
    long_w = (rank > 0.75).astype(float)
    short_w = (rank < 0.25).astype(float)
    lw = long_w.div(long_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
    sw = short_w.div(short_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 0.5
    w = lw - sw
    # sleeve 日收益（含单边成本：每日调仓换手近似 = |w 变化| 的 2 倍单边）
    sleeve_ret = (w.shift(1).fillna(0) * wide).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * (SLEEVE_COST_BPS / 1e4)
    net = sleeve_ret - cost
    eq = (1 + net.fillna(0)).cumprod()
    res = pd.DataFrame({
        "trade_date": eq.index,
        "sleeve_ret": net.values,
        "sleeve_equity": eq.values,
        "turnover": turnover.values,
    })
    return {
        "sleeve": res,
        "weights": w,
        "metrics": _curve_metrics(net.dropna()),
    }


def build_hedge_overlay(equity_csv: str, fut: pd.DataFrame) -> dict:
    """用 IF/IH 对权益簿做 Beta 对冲 overlay。

    权益簿日收益来自 equity_csv 的 gross_return；对冲工具日收益来自 futures 面板
    （IF 主、IH 辅）。滚动 Beta 估计后，对冲后收益 = 权益收益 - beta·对冲工具收益。
    """
    eq = pd.read_csv(equity_csv)
    # 权益曲线 date 形如 2023-12-21，期货面板 trade_date 形如 20231201，统一为 YYYYMMDD
    eq["date"] = eq["date"].astype(str).str.replace("-", "", regex=False)
    book = eq[["date", "gross_return", "market_exposure"]].copy()
    book = book.rename(columns={"date": "trade_date", "gross_return": "book_ret"})

    r = roll_bridged_returns(fut[fut.underlying.isin(["IF", "IH"])]).copy()
    r["trade_date"] = r["trade_date"].astype(str)
    hf = r.pivot(index="trade_date", columns="underlying", values="ret")
    hf = hf[["IF"]].rename(columns={"IF": "IF_ret"})  # 主对冲：沪深300期货

    m = book.merge(hf, on="trade_date", how="inner").dropna()
    # 滚动 Beta：book_ret 对 IF_ret
    beta = (m["book_ret"] - m["book_ret"].rolling(ROLL_BETA_WIN).mean()) \
        * (m["IF_ret"] - m["IF_ret"].rolling(ROLL_BETA_WIN).mean())
    denom = (m["IF_ret"] - m["IF_ret"].rolling(ROLL_BETA_WIN).mean()) ** 2
    beta = (beta.rolling(ROLL_BETA_WIN).sum() / denom.rolling(ROLL_BETA_WIN).sum()).fillna(1.0)
    m["hedged_ret"] = m["book_ret"] - beta * m["IF_ret"]
    m["beta"] = beta
    res = m[["trade_date", "book_ret", "IF_ret", "hedged_ret", "beta"]].copy()
    return {
        "overlay": res,
        "book_metrics": _curve_metrics(m["book_ret"]),
        "hedged_metrics": _curve_metrics(m["hedged_ret"]),
        "avg_beta": float(beta.mean()),
    }


def _curve_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        return {}
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1).min()
    vol = ret.std() * np.sqrt(252)
    ann = cum.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) else 0.0
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    return {
        "n_days": int(len(ret)),
        "ann_return": float(ann),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "cum_return": float(cum.iloc[-1] - 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P9-2 futures sleeve / hedge overlay")
    ap.add_argument("--futures", required=True, help="连续近月期货面板 CSV（P9-1 产出）")
    ap.add_argument("--equity", default=None, help="可选：A 股权益曲线 CSV（含 gross_return）用于对冲 overlay")
    ap.add_argument("--out", required=True, help="sleeve/overlay 结果 CSV")
    ap.add_argument("--mom-win", type=int, default=20)
    args = ap.parse_args()

    fut = load_futures(args.futures)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sleeve = build_sleeve(fut, mom_win=args.mom_win)
    sleeve["sleeve"].to_csv(out, index=False)

    report = {
        "sleeve_metrics": sleeve["metrics"],
        "sleeve_date_start": sleeve["sleeve"]["trade_date"].iloc[0],
        "sleeve_date_end": sleeve["sleeve"]["trade_date"].iloc[-1],
        "sleeve_n_days": int(len(sleeve["sleeve"])),
    }

    if args.equity:
        ov = build_hedge_overlay(args.equity, fut)
        ov_out = out.parent / "hedge_overlay.csv"
        ov["overlay"].to_csv(ov_out, index=False)
        report["hedge_overlay"] = {
            "avg_beta": ov["avg_beta"],
            "book_metrics": ov["book_metrics"],
            "hedged_metrics": ov["hedged_metrics"],
            "overlay_csv": str(ov_out),
            "note": "需 futures 与 equity 日期连续对齐；抽样月份需先回填连续窗口",
        }

    report["sleeve_csv"] = str(out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
