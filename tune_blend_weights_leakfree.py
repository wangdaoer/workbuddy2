"""tune_blend_weights_leakfree.py — 无泄漏混合权重 + 持有期口径修正 + 全轴准OOS (step5 补强).

修复 step3/4 的两处根因 bug:
  (1) 持有期口径错位: 旧 _fwd 把 h_nat 当**交易日**偏移, 使"20日"实为 ~40 自然日、"60日"~120 自然日,
      与 feed purge 的 HOLD_NATURAL(自然日 28/84) 不一致, 且导致 holdout 仅 1 个有效样本.
      本脚本 _fwd 改用**自然日**偏移, 与 feed purge 完全一致 (h_days=28/84 自然日).
  (2) 单点 OOS 不可信: 旧 G4 用 1 个 holdout as-of 判失败. 本脚本增加 --wfv 全轴 walk-forward 准OOS:
      前 60% as-of 调参选优 -> 后 40% as-of 作验证(它们各自 per-as-of 无泄漏, 且时间靠后接近真OOS分布),
      样本从 1 扩到几十个, 统计可信.

口径对齐 validate_blend_leakfree(均 next_open + cost_model 成本). w_nor = 1 - w_ti.

用法:
  python tune_blend_weights_leakfree.py                 # 全局网格扫描(tune/holdout, 自然日口径)
  python tune_blend_weights_leakfree.py --wfv          # 加全轴 walk-forward 准OOS 验证
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

H_DAYS_20 = 28   # 自然日, 对齐 feed HOLD_NATURAL["20"]
H_DAYS_60 = 84   # 自然日, 对齐 feed HOLD_NATURAL["60"]


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.std(ddof=0) == 0 or len(s) < 2:
        return s * 0.0
    return (s - s.mean()) / s.std(ddof=0)


def _load_feed(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["asof"]).set_index("asof").sort_index()


def _fwd(piv_close, piv_open, piv_high, piv_low, pidx, i, symset, h_days, *, exec_mode="next_open", cost=None):
    """前向收益, h_days = **自然日**(与 feed purge 的 HOLD_NATURAL 对齐).
    next_open: 决策日 i 收盘定候选, i+1 交易日开盘买, 持有 h_days 自然日后首个交易日开盘卖.
    P1-2: 接通动态滑点(买卖日振幅代理)与可交易性(一字板/停牌剔除, 留痕).
    返回 (毛均值, 胜率, 净均值, 不可交易笔数)."""
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
    r = (sell_px / buy_px - 1)
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


def _blend_top(d, ti60_col, ti20_col, nor_row, w_ti, w_horizon, top_n, min_cov):
    ti60 = ti60_col.loc[d] if d in ti60_col.index else None
    ti20 = ti20_col.loc[d] if (ti20_col is not None and d in ti20_col.index) else None
    nor = nor_row.loc[d] if d in nor_row.index else None
    if ti60 is None or nor is None:
        return (None, False, 0)
    cov = set(ti60[ti60.notna()].index) & set(nor[nor.notna()].index)
    if len(cov) < min_cov:
        return (None, False, len(cov))
    sub = pd.DataFrame({"ti60": ti60, "nor": nor}).loc[list(cov)]
    if ti20 is not None and d in ti20_col.index:
        sub["ti20"] = ti20_col.loc[d]
        has20 = sub["ti20"].notna().any()
    else:
        has20 = False
    w_nor = 1.0 - w_ti
    z_nor = _zscore(sub["nor"])
    z_ti60 = _zscore(sub["ti60"])
    bl_60d = w_ti * z_ti60 + w_nor * z_nor
    if has20:
        z_ti20 = _zscore(sub["ti20"])
        bl_20d = w_ti * z_ti20 + w_nor * z_nor
        bl = w_horizon * bl_60d + (1 - w_horizon) * bl_20d
    else:
        bl = bl_60d
    top = set(bl.sort_values(ascending=False).head(top_n).index)
    return (top, has20, len(cov))


def sweep_point(w_ti, w_horizon, dates, ti60, ti20, nor, piv, piv_open, piv_high, piv_low, pidx, args):
    n20s, n60s, w20s, untrads = [], [], [], []
    for d in dates:
        top, _, _ = _blend_top(d, ti60, ti20, nor, w_ti, w_horizon, args.top_n, args.min_cov)
        if top is None:
            continue
        i = pidx.get_indexer([d])[0]
        if i < 0:
            continue
        _, h20, n20, u20 = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, top, H_DAYS_20,
                                exec_mode=args.exec_mode, cost=args.cost_mod)
        _, h60, n60, u60 = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, top, H_DAYS_60,
                                exec_mode=args.exec_mode, cost=args.cost_mod)
        if not np.isnan(n20):
            n20s.append(n20)
        if not np.isnan(n60):
            n60s.append(n60)
        if not np.isnan(h20):
            w20s.append(h20)
        untrads.append((u20, u60))
    return (float(np.nanmean(n20s)) if n20s else np.nan,
            float(np.nanmean(n60s)) if n60s else np.nan,
            float(np.nanmean(w20s)) if w20s else np.nan,
            len(n20s),
            (sum(u for u, _ in untrads), sum(u for _, u in untrads)))


def build_grid(w_ti_vals, w_horizon_vals):
    return [(wt, wh) for wt in w_ti_vals for wh in w_horizon_vals]


def main() -> None:
    ap = argparse.ArgumentParser(description="无泄漏混合权重 + 持有期修正 + 全轴准OOS (step5 补强)")
    ap.add_argument("--ti-feed-fwd60", default=str(DEFAULT_TI60))
    ap.add_argument("--ti-feed-fwd20", default=str(DEFAULT_TI20))
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR))
    ap.add_argument("--nor-feed-leakfree", default=str(DEFAULT_NOR_LK),
                    help="leakfree NOR feed (asof 列格式); 存在时优先使用, 否则回退 --nor-feed")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--min-cov", type=int, default=30)
    ap.add_argument("--tune-window", nargs=2, default=["2025-09-01", "2026-06-01"])
    ap.add_argument("--holdout-window", nargs=2, default=["2026-07-01", "2026-08-20"])
    ap.add_argument("--exec-mode", choices=("close", "next_open"), default="next_open")
    ap.add_argument("--no-cost", action="store_true")
    ap.add_argument("--wfv", action="store_true", help="全轴 walk-forward 准OOS(前60%调参->后40%验证)")
    ap.add_argument("--wfv-split", type=float, default=0.6, help="walk-forward 训练占比")
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
        raise SystemExit("缺少 leakfree_feed_fwd60.csv")

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
    hold_dates = sorted([d for d in ti60_col.index if hw[0] <= d <= hw[1]])

    print(f"[口径修正] 持有期=自然日(20日=28自然日, 60日=84自然日, 对齐 feed purge)")
    print(f"[执行] exec_mode={args.exec_mode} 成本={'开' if not args.no_cost else '关'} "
          f"tune={len(tune_dates)}as-of holdout={len(hold_dates)}as-of")

    grid = build_grid([0.3, 0.5, 0.7, 0.9, 1.0], [0.0, 0.3, 0.5, 0.7, 1.0])
    rows = []
    untrad_tot = [0, 0]
    for (wt, wh) in grid:
        tn20, tn60, tw_, _, _ = sweep_point(wt, wh, tune_dates, ti60_col, ti20_col, nor, piv, piv_open,
                                            piv_high, piv_low, pidx, args)
        hn20, hn60, hw_, hn, untrad = sweep_point(wt, wh, hold_dates, ti60_col, ti20_col, nor, piv, piv_open,
                                                  piv_high, piv_low, pidx, args)
        untrad_tot[0] += untrad[0]
        untrad_tot[1] += untrad[1]
        rows.append({"w_ti": wt, "w_nor": round(1 - wt, 2), "w_horizon": wh,
                     "tune_n20": tn20, "tune_n60": tn60, "tune_win20": tw_,
                     "hold_n20": hn20, "hold_n60": hn60, "hold_win20": hw_, "hold_n": hn})
    sweep = pd.DataFrame(rows)
    print(f"  [不可成交剔除] 全部网格 holdout 20日 {untrad_tot[0]} 笔 / 60日 {untrad_tot[1]} 笔（一字板/停牌）")
    out = Path(args.output_dir) / "blend_weight_sweep.csv"
    sweep.to_csv(out, index=False, encoding="utf-8")
    print(f"\n-> 网格扫描(自然日口径) {out}")
    for _, r in sweep.iterrows():
        print(f"  w_ti={r.w_ti:.1f} w_hor={r.w_horizon:.1f} | tune n20={_pct(r.tune_n20)} n60={_pct(r.tune_n60)} "
              f"| holdout n20={_pct(r.hold_n20)}(n={int(r.hold_n)}) n60={_pct(r.hold_n60)}")

    cur = sweep[(sweep.w_ti == 0.9) & (sweep.w_horizon == 0.5)]
    if len(cur):
        r = cur.iloc[0]
        print(f"[当前生产权重 0.9/0.5] holdout n20={_pct(r.hold_n20)} n60={_pct(r.hold_n60)} (n={int(r.hold_n)})")
    valid = sweep.dropna(subset=["hold_n20"])
    best = valid.sort_values("hold_n20", ascending=False).iloc[0]
    print(f"[holdout n20 最优] w_ti={best.w_ti:.1f} w_hor={best.w_horizon:.1f} "
          f"holdout n20={_pct(best.hold_n20)} n60={_pct(best.hold_n60)} (n={int(best.hold_n)})")
    print(f"[稳健性] {(valid.hold_n20 > 0).sum()}/{len(valid)} 网格点 holdout n20 > 0")

    if args.wfv:
        print(f"\n=== 全轴 walk-forward 准OOS (训练占比 {args.wfv_split}) ===")
        all_dates = sorted(ti60_col.index)
        k = int(len(all_dates) * args.wfv_split)
        train_d, oos_d = all_dates[:k], all_dates[k:]
        # 训练集上选 tune 净20 最优权重
        best_w, best_v = None, -1e9
        for (wt, wh) in grid:
            tn20, _, _, _, _ = sweep_point(wt, wh, train_d, ti60_col, ti20_col, nor, piv, piv_open,
                                           piv_high, piv_low, pidx, args)
            if not np.isnan(tn20) and tn20 > best_v:
                best_v, best_w = tn20, (wt, wh)
        print(f"  训练集({len(train_d)}as-of)最优权重 w_ti={best_w[0]:.1f} w_hor={best_w[1]:.1f} "
              f"训练净20={_pct(best_v)}")
        # 用该权重评估 OOS(后40%, 各自无泄漏)
        o_n20, o_n60, o_w20, o_u20 = [], [], [], []
        for d in oos_d:
            top, _, _ = _blend_top(d, ti60_col, ti20_col, nor, best_w[0], best_w[1], args.top_n, args.min_cov)
            if top is None:
                continue
            i = pidx.get_indexer([d])[0]
            if i < 0:
                continue
            _, h20, n20, u20 = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, top, H_DAYS_20,
                                    exec_mode=args.exec_mode, cost=args.cost_mod)
            _, h60, n60, _ = _fwd(piv, piv_open, piv_high, piv_low, pidx, i, top, H_DAYS_60,
                                  exec_mode=args.exec_mode, cost=args.cost_mod)
            if not np.isnan(n20):
                o_n20.append(n20); o_w20.append(h20); o_u20.append(u20)
            if not np.isnan(n60):
                o_n60.append(n60)
        if o_n20:
            print(f"  [准OOS验证] 样本={len(o_n20)} as-of")
            print(f"    净20 = {_pct(np.nanmean(o_n20))}  胜率20 = {np.nanmean(o_w20)*100:.1f}%")
            print(f"    净60 = {_pct(np.nanmean(o_n60)) if o_n60 else 'NaN(超面板)'}")
            print(f"    不可成交剔除(20日) = {sum(o_u20)} 笔")
            print(f"    结论: {'转正' if np.nanmean(o_n20) > 0 else '仍为负 — 方案B无泄漏准OOS未通过'}")
            pd.DataFrame({"asof": [str(d.date()) for d in oos_d],
                          "n20": o_n20 + [np.nan] * (len(oos_d) - len(o_n20))}).to_csv(
                Path(args.output_dir) / "blend_wfv_oos.csv", index=False, encoding="utf-8")


def _pct(x):
    return f"{x*100:+.2f}%" if isinstance(x, (int, float)) and not np.isnan(x) else "NaN"


if __name__ == "__main__":
    main()
