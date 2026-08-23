"""validate_blend_leakfree.py — 无泄漏 per-as-of 混合前向回测 (方案 B step3 证据).

与 validate_blend_forward.py 的区别:
  1. TI 打分来自 leakfree_feed (每个 as-of 用该 as-of 专属 purge 后评分器, 见 build_leakfree_scorer_feed.py),
     而非全量拟合的 authoritative_scorer -> 杜绝 P0-3 前视泄漏.
  2. 显式区分两个窗口:
     - 调优/训练窗口 (默认 2025-09 ~ 2026-06): leakfree feed 覆盖, 用于扫 w_ti/w_horizon
     - 验证窗口 (默认 2026-07 ~ 2026-08): 完全不参与任何 as-of 的 fit, 纯 OOS 验证

前向收益口径沿用 validate_blend_forward: 取 as-of 日 close, 算 +20/+60 交易日(近似自然日偏移)收益均值与胜率.

用法:
  python validate_blend_leakfree.py --tune-window 2025-09-01 2026-06-01 --holdout-window 2026-07-01 2026-08-20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cost_model

ROOT = Path(__file__).resolve().parent
DEFAULT_TI60 = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "leakfree_feed_fwd60.csv"
DEFAULT_TI20 = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "leakfree_feed_fwd20.csv"
DEFAULT_NOR = ROOT / "outputs" / "production_soft_score" / "soft_score_feed.csv"
DEFAULT_NOR_LK = ROOT / "outputs" / "production_soft_score" / "leakfree_nor_feed.csv"
DEFAULT_PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool"


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.std(ddof=0) == 0 or len(s) < 2:
        return s * 0.0
    return (s - s.mean()) / s.std(ddof=0)


def _load_feed(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    # leakfree feed: 第一列是 asof (每个 as-of 一行, 列=symbol)
    df = pd.read_csv(path, parse_dates=["asof"]).set_index("asof").sort_index()
    return df


def _fwd(piv_close, piv_open, piv_high, piv_low, pidx, i, symset, h_days, *, exec_mode="next_open", cost=None):
    """前向收益。h_days=**自然日**(对齐 feed purge HOLD_NATURAL: 20日=28自然日, 60日=84自然日)。
    next_open: 决策日 i 收盘定候选, i+1 交易日开盘买, 持有 h_days 自然日后首个交易日开盘卖。
    P1-2: 接通动态滑点(买卖日振幅代理)与可交易性(一字板/停牌剔除, 留痕)。
    返回 (毛均值, 胜率, 净均值, 不可交易笔数)。
    """
    cols = [int(s) for s in symset if int(s) in piv_close.columns]
    if not cols:
        return (np.nan, np.nan, np.nan, 0)
    buy_i = i + 1 if exec_mode == "next_open" else i
    if buy_i >= len(pidx):
        return (np.nan, np.nan, np.nan, 0)
    buy_dt = pidx[buy_i]
    sell_target = buy_dt + pd.Timedelta(days=h_days)
    sell_i = int(pidx.searchsorted(sell_target, side="left"))
    if sell_i >= len(pidx):
        return (np.nan, np.nan, np.nan, 0)
    buy_px = (piv_open.loc[pidx[buy_i], cols] if exec_mode == "next_open"
              else piv_close.loc[pidx[buy_i], cols])
    sell_px = (piv_open.loc[pidx[sell_i], cols] if exec_mode == "next_open"
               else piv_close.loc[pidx[sell_i], cols])
    r = sell_px / buy_px - 1
    # 可交易性: 买卖日必须有价格且非一字板(high==low)/停牌(high/low 缺失)
    trad = r.notna() & buy_px.notna() & sell_px.notna()
    if piv_high is not None and piv_low is not None:
        b_hi = piv_high.loc[pidx[buy_i], cols]
        b_lo = piv_low.loc[pidx[buy_i], cols]
        s_hi = piv_high.loc[pidx[sell_i], cols]
        s_lo = piv_low.loc[pidx[sell_i], cols]
        trad &= b_hi.notna() & b_lo.notna() & (b_hi != b_lo)
        trad &= s_hi.notna() & s_lo.notna() & (s_hi != s_lo)
    n_untrad = int((~trad).sum())
    r = r[trad]
    if len(r) == 0:
        return (np.nan, np.nan, np.nan, n_untrad)
    mean_gross = r.mean()
    win = (r > 0).mean()
    if cost is None:
        net = mean_gross
    else:
        # P1-2: 逐票动态滑点(买卖日振幅代理, 取较大) + 冲击; tradable 已在上方过滤
        wgt = 1.0 / len(r)
        net_list = []
        for g, sym in zip(r.tolist(), r.index):
            if piv_high is not None and piv_low is not None:
                try:
                    b_sp = cost_model._slippage_from_range(
                        float(buy_px[sym]), float(piv_high.loc[pidx[buy_i], sym]),
                        float(piv_low.loc[pidx[buy_i], sym]))
                    s_sp = cost_model._slippage_from_range(
                        float(sell_px[sym]), float(piv_high.loc[pidx[sell_i], sym]),
                        float(piv_low.loc[pidx[sell_i], sym]))
                    sp = max(b_sp, s_sp)
                except Exception:  # noqa: BLE001
                    sp = 0.001
            else:
                sp = 0.001
            net_list.append(cost_model.apply_costs(g, weight=wgt, slippage=sp))
        net = float(np.nanmean(net_list))
    return (mean_gross, win, net, n_untrad)


def _eval_asof(d, ti60_col, ti20_col, nor_row, piv, piv_open, piv_high, piv_low, pidx, args):
    """对单个 as-of 做 horizon-aware 混合与前向收益, 返回字典."""
    ti60 = ti60_col.loc[d] if d in ti60_col.index else None
    ti20 = ti20_col.loc[d] if (ti20_col is not None and d in ti20_col.index) else None
    nor = nor_row.loc[d] if d in nor_row.index else None
    if ti60 is None or nor is None:
        return None
    cov = set(ti60[ti60.notna()].index) & set(nor[nor.notna()].index)
    if len(cov) < args.min_cov:
        return None
    sub = pd.DataFrame({"ti60": ti60, "nor": nor}).loc[list(cov)]
    if ti20 is not None and d in ti20_col.index:
        sub["ti20"] = ti20_col.loc[d]
        has20 = sub["ti20"].notna().any()
    else:
        has20 = False
    z_nor = _zscore(sub["nor"])
    z_ti60 = _zscore(sub["ti60"])
    bl_60d = args.w_ti * z_ti60 + args.w_nor * z_nor
    if has20:
        z_ti20 = _zscore(sub["ti20"])
        bl_20d = args.w_ti * z_ti20 + args.w_nor * z_nor
        bl = args.w_horizon * bl_60d + (1 - args.w_horizon) * bl_20d
    else:
        bl = bl_60d
    bl_top = set(bl.sort_values(ascending=False).head(args.top_n).index)
    ti_top = set(sub["ti60"].dropna().sort_values(ascending=False).head(args.top_n).index)
    nor_top = set(sub["nor"].dropna().sort_values(ascending=False).head(args.top_n).index)
    i = pidx.get_indexer([d])[0]
    if i < 0:
        return None
    m20, h20, n20, u20 = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, bl_top, 28,
                              exec_mode=args.exec_mode, cost=args.cost_mod)
    m60, h60, n60, u60 = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, bl_top, 84,
                              exec_mode=args.exec_mode, cost=args.cost_mod)
    tm20, th20, tn20, _ = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, ti_top, 28,
                               exec_mode=args.exec_mode, cost=args.cost_mod)
    tm60, th60, tn60, _ = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, ti_top, 84,
                               exec_mode=args.exec_mode, cost=args.cost_mod)
    nm20, nh20, nn20, _ = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, nor_top, 28,
                               exec_mode=args.exec_mode, cost=args.cost_mod)
    nm60, nh60, nn60, _ = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, nor_top, 84,
                               exec_mode=args.exec_mode, cost=args.cost_mod)
    return {"asof": d.date(), "cov": len(cov), "has20": has20,
            "bl_m20": m20, "bl_h20": h20, "bl_m60": m60, "bl_h60": h60,
            "bl_n20": n20, "bl_n60": n60, "bl_u20": u20, "bl_u60": u60,
            "ti_m20": tm20, "ti_h20": th20, "ti_m60": tm60, "ti_h60": th60,
            "ti_n20": tn20, "ti_n60": tn60,
            "nor_m20": nm20, "nor_h20": nh20, "nor_m60": nm60, "nor_h60": nh60,
            "nor_n20": nn20, "nor_n60": nn60}


def _run_window(name, dates, ti60_col, ti20_col, nor_row, piv, piv_open, piv_high, piv_low, pidx, args):
    rows = []
    for d in dates:
        r = _eval_asof(d, ti60_col, ti20_col, nor_row, piv, piv_open, piv_high, piv_low, pidx, args)
        if r:
            rows.append(r)
    if not rows:
        print(f"[{name}] 无有效 as-of")
        return
    df = pd.DataFrame(rows)
    # 组合逐期净值 (用混合 60日净收益), 算最大回撤
    net60 = df["bl_n60"].dropna()
    equity = cost_model.net_portfolio_path(net60) if len(net60) else pd.Series(dtype=float)
    mdd = cost_model.max_drawdown(equity) if len(equity) else float("nan")
    n_untrad20 = int(df["bl_u20"].sum()) if "bl_u20" in df else 0
    n_untrad60 = int(df["bl_u60"].sum()) if "bl_u60" in df else 0
    print(f"\n=== {name} 窗口 ({len(df)} as-of, min_cov={args.min_cov}, exec={args.exec_mode}) ===")
    print(f"  [毛] 混合 bl: 20日 {df['bl_m20'].mean()*100:+.2f}% | 60日 {df['bl_m60'].mean()*100:+.2f}% 胜率 {df['bl_h60'].mean()*100:.1f}%")
    print(f"  [净] 混合 bl: 20日 {df['bl_n20'].mean()*100:+.2f}% | 60日 {df['bl_n60'].mean()*100:+.2f}% | 组合最大回撤 {mdd*100:+.1f}%")
    print(f"  [不可成交剔除] 20日 {n_untrad20} 笔 / 60日 {n_untrad60} 笔（一字板/停牌）")
    print(f"  [毛] 仅TI60: 60日 {df['ti_m60'].mean()*100:+.2f}% | [净] {df['ti_n60'].mean()*100:+.2f}%")
    print(f"  [毛] 仅NOR:  60日 {df['nor_m60'].mean()*100:+.2f}% | [净] {df['nor_n60'].mean()*100:+.2f}%")
    print(f"  horizon-aware 占比: {df['has20'].mean()*100:.0f}% as-of 有 fwd20 维度")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="无泄漏 per-as-of 混合前向回测")
    ap.add_argument("--ti-feed-fwd60", default=str(DEFAULT_TI60))
    ap.add_argument("--ti-feed-fwd20", default=str(DEFAULT_TI20))
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR))
    ap.add_argument("--nor-feed-leakfree", default=str(DEFAULT_NOR_LK),
                    help="leakfree NOR feed (build_leakfree_nor_feed.py 产物, asof 列格式); "
                         "文件存在时优先使用, 否则回退 --nor-feed 全量版本")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--w-ti", type=float, default=0.9)
    ap.add_argument("--w-nor", type=float, default=0.1)
    ap.add_argument("--w-horizon", type=float, default=0.5)
    ap.add_argument("--min-cov", type=int, default=30)
    ap.add_argument("--tune-window", nargs=2, default=["2025-09-01", "2026-06-01"])
    ap.add_argument("--holdout-window", nargs=2, default=["2026-07-01", "2026-08-20"])
    ap.add_argument("--exec-mode", choices=("close", "next_open"), default="next_open",
                    help="执行时点: close=决策日收盘买收盘卖(旧口径); next_open=次日开盘买(对齐策略名)")
    ap.add_argument("--no-cost", action="store_true", help="不扣交易成本(仅毛收益)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    args.cost_mod = cost_model if not args.no_cost else None  # P1-2: --no-cost 真正生效

    ti60_col = _load_feed(args.ti_feed_fwd60)
    ti20_col = _load_feed(args.ti_feed_fwd20) if args.ti_feed_fwd20 else None
    # NOR: 优先 leakfree 版本 (asof 列格式, per-asof purge 重建), 否则回退全量版本
    lk = Path(args.nor_feed_leakfree)
    if lk.exists():
        nor = pd.read_csv(lk, parse_dates=["asof"]).set_index("asof").sort_index()
        print(f"[NOR] 使用 leakfree 版本: {lk} ({len(nor)} as-of)")
    else:
        nor = pd.read_csv(args.nor_feed, parse_dates=["date"]).set_index("date").sort_index()
        print(f"[NOR] leakfree 版本不存在, 回退全量: {args.nor_feed} ({len(nor)} 行)")
    if ti60_col.empty:
        raise SystemExit("缺少 leakfree_feed_fwd60.csv, 请先跑 build_leakfree_scorer_feed.py")

    # 价格面板 (close + open, 供 next_open 模式与成本滑点估算; high/low 供 P1-2 可交易性/动态滑点)
    usecols = ["date", "symbol", "close"]
    p = pd.read_csv(args.panel, usecols=usecols)
    common_syms = ti60_col.columns.intersection(nor.columns)
    p = p[p["symbol"].isin(int(c) for c in common_syms)]
    p["date"] = pd.to_datetime(p["date"])
    piv = p.pivot_table(index="date", columns="symbol", values="close").sort_index()
    pidx = piv.index
    if args.exec_mode == "next_open":
        po = pd.read_csv(args.panel, usecols=["date", "symbol", "open"])
        po = po[po["symbol"].isin(int(c) for c in common_syms)]
        po["date"] = pd.to_datetime(po["date"])
        piv_open = po.pivot_table(index="date", columns="symbol", values="open").sort_index()
    else:
        piv_open = piv
    # high/low pivot (P1-2: 一字板/停牌判定 + 动态滑点振幅)
    ph = pd.read_csv(args.panel, usecols=["date", "symbol", "high"])
    ph = ph[ph["symbol"].isin(int(c) for c in common_syms)]
    ph["date"] = pd.to_datetime(ph["date"])
    piv_high = ph.pivot_table(index="date", columns="symbol", values="high").sort_index()
    pl = pd.read_csv(args.panel, usecols=["date", "symbol", "low"])
    pl = pl[pl["symbol"].isin(int(c) for c in common_syms)]
    pl["date"] = pd.to_datetime(pl["date"])
    piv_low = pl.pivot_table(index="date", columns="symbol", values="low").sort_index()

    tw = (pd.to_datetime(args.tune_window[0]), pd.to_datetime(args.tune_window[1]))
    hw = (pd.to_datetime(args.holdout_window[0]), pd.to_datetime(args.holdout_window[1]))
    tune_dates = [d for d in ti60_col.index if tw[0] <= d <= tw[1]]
    hold_dates = [d for d in ti60_col.index if hw[0] <= d <= hw[1]]

    print(f"权重 w_ti={args.w_ti} w_nor={args.w_nor} w_horizon={args.w_horizon} "
          f"exec_mode={args.exec_mode} 成本={'开' if args.cost_mod else '关'}")
    tune_df = _run_window("调优/训练", tune_dates, ti60_col, ti20_col, nor, piv, piv_open, piv_high, piv_low, pidx, args)
    hold_df = _run_window("验证(Holdout, 纯OOS)", hold_dates, ti60_col, ti20_col, nor, piv, piv_open, piv_high, piv_low, pidx, args)

    out = Path(args.output_dir) / "leakfree_backtest.csv"
    frames = [f.assign(window=name) for f, name in [(tune_df, "tune"), (hold_df, "holdout")] if f is not None]
    if frames:
        pd.concat(frames).to_csv(out, index=False, encoding="utf-8")
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
