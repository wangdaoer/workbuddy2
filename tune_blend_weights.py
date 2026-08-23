"""tune_blend_weights.py — horizon-aware 加权调优: 在 60 个 as-of 回测上扫 w_ti 网格.

对每个 as-of 日, 先算好双覆盖符号的 z_ti / z_nor (与权重无关, 只算一次),
再扫 w_ti in GRID (w_nor=1-w_ti): blended_z = w_ti*z_ti + w_nor*z_nor -> top-N,
算该池等权前向 20/60 日收益, 跨 as-of 平均. 找使 (前向20日均值 + 前向60日均值)
最大的 w_ti, 并校验其相对等权(w=0.5)与两单池的稳健性.

输入同 validate_blend_forward.py (--ti-feed/--nor-feed/--panel/--top-n/--min-cov/--step/--max-asof).
产物: 打印权重网格表 + 最优权重结论; 明细写 blend_weight_sweep.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_TI = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "trend_ignition_feed.csv"
DEFAULT_NOR = ROOT / "outputs" / "production_soft_score" / "soft_score_feed.csv"
DEFAULT_PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "blend_weight_sweep.csv"

GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.std(ddof=0) == 0 or len(s) < 2:
        return s * 0.0
    return (s - s.mean()) / s.std(ddof=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="混合权重调优")
    ap.add_argument("--ti-feed", default=str(DEFAULT_TI))
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR))
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--top-n", type=int, default=20)
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
    cov_cnt = (ti.notna() & nor.notna()).sum(axis=1)
    asof_candidates = cov_cnt[(cov_cnt >= args.min_cov) & (cd <= max_asof)].index
    asof_list = asof_candidates[:: max(1, args.step)]
    print(f"合格 as-of {len(asof_candidates)} 个, 取样 {len(asof_list)} 个回测")

    p = pd.read_csv(args.panel, usecols=["date", "symbol", "close"])
    p = p[p["symbol"].isin(int(c) for c in cs)]
    p["date"] = pd.to_datetime(p["date"])
    piv = p.pivot_table(index="date", columns="symbol", values="close").sort_index()
    pidx = piv.index

    # 预计算每个 as-of 的双覆盖符号 z_ti/z_nor (与权重无关)
    pre = []
    for d in asof_list:
        i = pidx.get_indexer([d])[0]
        if i < 0 or i + 60 >= len(pidx):
            continue
        ti_row = ti.loc[d]
        nor_row = nor.loc[d]
        cov = ti_row.notna() & nor_row.notna()
        syms = cov[cov].index
        if len(syms) < args.min_cov:
            continue
        zt = _zscore(ti_row[syms])
        zn = _zscore(nor_row[syms])
        pre.append((i, syms, zt, zn))

    # 扫权重: 每个 as-of 在给定 w 下算 top-N 池的前向收益
    sweep_rows = []
    for w in GRID:
        m20_list, m60_list, h20_list, h60_list = [], [], [], []
        for (i, syms, zt, zn) in pre:
            blended = w * zt + (1 - w) * zn
            top = blended.sort_values(ascending=False).head(args.top_n).index
            cols = [int(s) for s in top if int(s) in piv.columns]
            if not cols:
                continue
            r20 = piv.loc[pidx[i + 20], cols] / piv.loc[pidx[i], cols] - 1
            r60 = piv.loc[pidx[i + 60], cols] / piv.loc[pidx[i], cols] - 1
            r20 = r20.dropna(); r60 = r60.dropna()
            if len(r20): m20_list.append(r20.mean()); h20_list.append((r20 > 0).mean())
            if len(r60): m60_list.append(r60.mean()); h60_list.append((r60 > 0).mean())
        sweep_rows.append({
            "w_ti": w, "w_nor": round(1 - w, 2),
            "mean_fwd20": pd.Series(m20_list).mean(),
            "mean_fwd60": pd.Series(m60_list).mean(),
            "hit20": pd.Series(h20_list).mean(),
            "hit60": pd.Series(h60_list).mean(),
        })
    sweep = pd.DataFrame(sweep_rows)
    sweep["obj"] = (sweep["mean_fwd20"] + sweep["mean_fwd60"]) / 2  # 双 horizon 等权目标
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output, index=False, encoding="utf-8")

    n = len(pre)
    print(f"\n=== 权重网格 (n={n} as-of, top-{args.top_n}) ===")
    print(f"{'w_ti':>5} {'w_nor':>5} {'fwd20%':>8} {'fwd60%':>8} {'hit20%':>7} {'hit60%':>7} {'obj%':>7}")
    for _, r in sweep.iterrows():
        print(f"{r['w_ti']:5.1f} {r['w_nor']:5.1f} {r['mean_fwd20']*100:8.2f} {r['mean_fwd60']*100:8.2f} "
              f"{r['hit20']*100:7.1f} {r['hit60']*100:7.1f} {r['obj']*100:7.2f}")
    best = sweep.loc[sweep["obj"].idxmax()]
    eq = sweep[sweep["w_ti"] == 0.5].iloc[0]
    print(f"\n最优 w_ti={best['w_ti']:.1f}/w_nor={best['w_nor']:.1f}: fwd20={best['mean_fwd20']*100:.2f}% "
          f"fwd60={best['mean_fwd60']*100:.2f}% obj={best['obj']*100:.2f}%")
    print(f"等权 w=0.5: fwd20={eq['mean_fwd20']*100:.2f}% fwd60={eq['mean_fwd60']*100:.2f}% obj={eq['obj']*100:.2f}%")
    print(f"提升 obj: {(best['obj']-eq['obj'])*100:+.2f}pp  | 相对等权是否更优: {best['obj']>eq['obj']}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
