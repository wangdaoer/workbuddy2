"""validate_blend_forward.py — 多 as-of 回测: 验证趋势点火(Pool B)与 next_open_rank 混合是否提升候选质量.

对一组 as-of 日期(双覆盖符号 >= --min-cov), 在每日常量:
  - TI  top-N  (trend_ignition feed 分数)
  - NOR top-N  (next_open_rank soft feed 分数)
  - BL  top-N  (双覆盖符号上 z-score 加权融合, w_ti/w_nor)
计算各池等权持有的前向 20/60 交易日收益, 跨 as-of 平均, 比较三池.

输入:
  --ti-feed     trend_ignition_feed.csv
  --nor-feed    next_open_rank soft feed
  --panel       价格面板 (date,symbol,close) 用于前向收益
  --top-n       每池取前 N (默认 20)
  --w-ti/--w-nor 加权权重 (默认 0.5/0.5)
  --min-cov     纳入回测的 as-of 需满足的双覆盖符号下限 (默认 50)
  --step        在合格 as-of 中每隔 step 个取样 (默认 4, 控制回测次数)
  --max-asof    限定 as-of <= 此日期 (默认 2026-06-30, 保证前向 60 日有数据)
  --output      回测明细 CSV (默认 outputs/high_return_v2/trend_ignition_daily_pool/blend_backtest.csv)

输出: 打印三池跨 as-of 的平均前向收益/胜率, 以及 blend 相对 NOR 的增量.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_TI = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "trend_ignition_feed.csv"
DEFAULT_NOR = ROOT / "outputs" / "production_soft_score" / "soft_score_feed.csv"
DEFAULT_PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "blend_backtest.csv"


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.std(ddof=0) == 0 or len(s) < 2:
        return s * 0.0
    return (s - s.mean()) / s.std(ddof=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="混合候选池多 as-of 前向回测")
    ap.add_argument("--ti-feed", default=str(DEFAULT_TI))
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR))
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--w-ti", type=float, default=0.5)
    ap.add_argument("--w-nor", type=float, default=0.5)
    ap.add_argument("--min-cov", type=int, default=50)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--max-asof", default="2026-06-30")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    ti = pd.read_csv(args.ti_feed, parse_dates=["date"]).set_index("date").sort_index()
    nor = pd.read_csv(args.nor_feed, parse_dates=["date"]).set_index("date").sort_index()
    cd = ti.index.intersection(nor.index)
    cs = ti.columns.intersection(nor.columns)
    ti = ti.loc[cd, cs]
    nor = nor.loc[cd, cs]

    max_asof = pd.Timestamp(args.max_asof)
    # 候选 as-of: 双覆盖 >= min-cov 且 <= max_asof
    cov_cnt = (ti.notna() & nor.notna()).sum(axis=1)
    asof_candidates = cov_cnt[(cov_cnt >= args.min_cov) & (cd <= max_asof)].index
    asof_list = asof_candidates[:: max(1, args.step)]
    print(f"合格 as-of(双覆盖>={args.min_cov}) {len(asof_candidates)} 个, 取样 {len(asof_list)} 个回测")

    # 价格 pivot (date x symbol), 限常用符号 + 前向窗口够用
    p = pd.read_csv(args.panel, usecols=["date", "symbol", "close"])
    p = p[p["symbol"].isin(int(c) for c in cs)]
    p["date"] = pd.to_datetime(p["date"])
    piv = p.pivot_table(index="date", columns="symbol", values="close").sort_index()
    pidx = piv.index

    rows = []
    for d in asof_list:
        i = pidx.get_indexer([d])[0]
        if i < 0 or i + 60 >= len(pidx):
            continue
        ti_row = ti.loc[d]
        nor_row = nor.loc[d]
        cov_syms = set(ti_row[ti_row.notna()].index) & set(nor_row[nor_row.notna()].index)
        if len(cov_syms) < args.min_cov:
            continue
        ti_top = set(ti_row.dropna().sort_values(ascending=False).head(args.top_n).index)
        nor_top = set(nor_row.dropna().sort_values(ascending=False).head(args.top_n).index)
        sub = pd.DataFrame({"ti": ti_row, "nor": nor_row}).loc[list(cov_syms)]
        z = args.w_ti * _zscore(sub["ti"]) + args.w_nor * _zscore(sub["nor"])
        bl_top = set(z.sort_values(ascending=False).head(args.top_n).index)

        def fwd(symset, h):
            cols = [int(s) for s in symset if int(s) in piv.columns]
            if not cols:
                return (pd.NA, pd.NA)
            r = piv.loc[pidx[i + h], cols] / piv.loc[pidx[i], cols] - 1
            r = r.dropna()
            if len(r) == 0:
                return (pd.NA, pd.NA)
            return (r.mean(), (r > 0).mean())

        ti_m20, ti_h20 = fwd(ti_top, 20)
        nor_m20, nor_h20 = fwd(nor_top, 20)
        bl_m20, bl_h20 = fwd(bl_top, 20)
        ti_m60, ti_h60 = fwd(ti_top, 60)
        nor_m60, nor_h60 = fwd(nor_top, 60)
        bl_m60, bl_h60 = fwd(bl_top, 60)
        rows.append({
            "asof": d.date(), "cov": len(cov_syms),
            "ti_m20": ti_m20, "nor_m20": nor_m20, "bl_m20": bl_m20,
            "ti_m60": ti_m60, "nor_m60": nor_m60, "bl_m60": bl_m60,
            "ti_h20": ti_h20, "nor_h20": nor_h20, "bl_h20": bl_h20,
            "ti_h60": ti_h60, "nor_h60": nor_h60, "bl_h60": bl_h60,
        })

    res = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False, encoding="utf-8")

    n = len(res)
    print(f"\n=== 多 as-of 回测汇总 (n={n} 个 as-of, top-{args.top_n}) ===")
    for h, mcol, hcol in ((20, "m20", "h20"), (60, "m60", "h60")):
        tm = res[f"ti_{mcol}"].mean(); nm = res[f"nor_{mcol}"].mean(); bm = res[f"bl_{mcol}"].mean()
        th = res[f"ti_{hcol}"].mean(); nh = res[f"nor_{hcol}"].mean(); bh = res[f"bl_{hcol}"].mean()
        print(f"  前向 {h} 日: TI mean={tm*100:6.2f}% hit={th*100:5.1f}% | "
              f"NOR mean={nm*100:6.2f}% hit={nh*100:5.1f}% | "
              f"BL mean={bm*100:6.2f}% hit={bh*100:5.1f}%")
    print(f"  BL vs NOR (前向20日 均值差): {(res['bl_m20']-res['nor_m20']).mean()*100:+.2f}%")
    print(f"  BL vs NOR (前向60日 均值差): {(res['bl_m60']-res['nor_m60']).mean()*100:+.2f}%")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
