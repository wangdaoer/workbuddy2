"""blend_candidate_pools.py — 将 trend_ignition 独立候选池 B 与 next_open_rank 的 soft feed 混合.

两份并行候选池:
  - Pool A (next_open_rank): soft_score_feed.csv 的全市场每日 alpha feed.
  - Pool B (trend_ignition): trend_ignition_feed.csv (本仓库 trend_ignition_daily_pool.py 产出).

混合方法:
  intersection : 在 as-of 日取两池各自 top-N, 取交集 -> 双信号共振的高置信候选.
  weighted     : 在 as-of 日对两 feed 做横截面 z-score, blended = w_ti*z_ti + w_nor*z_nor, 排序.
                 仅在两 feed 同时非 NaN 的符号上计算 (要求双信号覆盖).

输入:
  --ti-feed      trend_ignition_feed.csv (默认 outputs/high_return_v2/trend_ignition_daily_pool/)
  --nor-feed     next_open_rank soft feed (默认 outputs/production_soft_score/soft_score_feed.csv)
  --asof         blend 日期 (默认 auto: 最近一个有 >= --min-active 个 TI 活跃信号的日期)
  --top-n        各池取前 N (默认 20)
  --w-ti/--w-nor 加权权重 (默认 0.5 / 0.5)
  --method       intersection|weighted|both (默认 both)
  --min-active   auto 模式下 TI 活跃符号最少数量 (默认 20)
  --output-dir   默认 outputs/high_return_v2/trend_ignition_daily_pool/

产物: blended_pool.csv (as-of 日的混合候选: symbol, ti_score, nor_score, blended_score, rank, in_intersection)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_TI = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "trend_ignition_feed.csv"
DEFAULT_TI_FWD20 = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool" / "trend_ignition_feed_fwd20.csv"
DEFAULT_NOR = ROOT / "outputs" / "production_soft_score" / "soft_score_feed.csv"
DEFAULT_OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool"


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.std(ddof=0) == 0 or len(s) < 2:
        return s * 0.0
    return (s - s.mean()) / s.std(ddof=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="trend_ignition 与 next_open_rank 候选池混合（horizon-aware 双评分）")
    ap.add_argument("--ti-feed", default=str(DEFAULT_TI), help="主 TI feed (fwd60 horizon)")
    ap.add_argument("--ti-feed-fwd20", default=str(DEFAULT_TI_FWD20),
                    help="短 horizon TI feed (fwd20)；传空字符串禁用第二维度")
    ap.add_argument("--nor-feed", default=str(DEFAULT_NOR))
    ap.add_argument("--asof", default="auto", help="blend 日期或 auto")
    ap.add_argument("--asof-mode", choices=("max-resonance", "latest"), default="max-resonance",
                    help="auto 模式选日逻辑: max-resonance=共振最强(研究演示); latest=最近一个有双覆盖的日期(生产)")
    ap.add_argument("--w-ti", type=float, default=0.9, help="trend_ignition 权重 (权重扫描最优 0.9)")
    ap.add_argument("--w-nor", type=float, default=0.1, help="next_open_rank 权重 (权重扫描最优 0.1)")
    ap.add_argument("--w-horizon", type=float, default=0.5,
                    help="双 horizon 综合权重: blended = w_horizon*blend_60d + (1-w_horizon)*blend_20d")
    ap.add_argument("--method", choices=("intersection", "weighted", "both"), default="both")
    ap.add_argument("--min-active", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=20, help="各池取前 N")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--run-id", default=None, help="绑定本轮 run 的唯一 id；缺省用 asof+时间戳生成")
    ap.add_argument("--min-blend-coverage", type=int, default=30,
                    help="P1 加固：能同时拿到 TI60+NOR+TI20 计算最终 horizon-aware 分的符号数下限；"
                         "低于此值 fail-closed 禁止晋级 latest")
    ap.add_argument("--write-manifest", action="store_true", default=True,
                    help="输出 manifest.json 绑定 run_id/asof/输入哈希/参数/输出哈希（默认开）")
    args = ap.parse_args()

    use_fwd20 = bool(args.ti_feed_fwd20)
    ti = pd.read_csv(args.ti_feed, parse_dates=["date"]).set_index("date").sort_index()
    ti_fwd20 = pd.read_csv(args.ti_feed_fwd20, parse_dates=["date"]).set_index("date").sort_index() if use_fwd20 else None
    nor = pd.read_csv(args.nor_feed, parse_dates=["date"]).set_index("date").sort_index()

    # 对齐共同日期与符号（以主 TI feed 为基准）
    common_dates = ti.index.intersection(nor.index)
    common_syms = ti.columns.intersection(nor.columns)
    ti = ti.loc[common_dates, common_syms]
    nor = nor.loc[common_dates, common_syms]
    if use_fwd20:
        # fwd20 feed 可能与主 feed 日期/符号略有差异，按共同部分对齐
        common_dates20 = ti_fwd20.index.intersection(ti.index)
        common_syms20 = ti_fwd20.columns.intersection(common_syms)
        ti_fwd20 = ti_fwd20.loc[common_dates20, common_syms20]

    # 选 as-of: 优先选"共振最强"(双覆盖最多)的日期, 且两池覆盖均 >= min_active
    if args.asof == "auto":
        ti_cnt = ti.notna().sum(axis=1)
        nor_cnt = nor.notna().sum(axis=1)
        inter_cnt = (ti.notna() & nor.notna()).sum(axis=1)
        both_ok = (ti_cnt >= args.min_active) & (nor_cnt >= args.min_active)
        if args.asof_mode == "latest":
            dates_ok = both_ok[both_ok].index
            if dates_ok.empty:
                asof = common_dates[-1]
                print(f"[auto/latest] 无日期满足双覆盖>=min-active={args.min_active}, 退回末日 {asof.date()}")
            else:
                asof = dates_ok[-1]
                print(f"[auto/latest] 选最近有双覆盖日 as-of={asof.date()} "
                      f"(TI活跃 {int(ti_cnt.loc[asof])}, NOR活跃 {int(nor_cnt.loc[asof])}, 共振 {int(inter_cnt.loc[asof])})")
        else:
            cand = inter_cnt[both_ok]
            if cand.empty:
                asof = common_dates[-1]
                print(f"[auto] 无日期满足双覆盖>=min-active={args.min_active}, 退回末日 {asof.date()}")
            else:
                # 取共振最强; 若并列取最近一日
                best = cand.max()
                asof = cand[cand == best].index[-1]
                print(f"[auto] 选 as-of={asof.date()} (共振符号 {int(best)}, "
                      f"TI活跃 {int(ti_cnt.loc[asof])}, NOR活跃 {int(nor_cnt.loc[asof])})")
    else:
        asof = pd.Timestamp(args.asof)
        if asof not in common_dates:
            raise SystemExit(f"asof {asof.date()} 不在共同日期内")
    print(f"  共同日期 {common_dates.min().date()}..{common_dates.max().date()} | 共同符号 {len(common_syms)}")

    ti_row = ti.loc[asof]
    nor_row = nor.loc[asof]
    ti_na = set(ti_row[ti_row.notna()].index)
    nor_na = set(nor_row[nor_row.notna()].index)
    cov_syms = ti_na & nor_na
    # 三方 coverage (P1 加固)：TI60∩NOR / TI20∩NOR / 三者全交集
    ti20_na = (set(ti_fwd20.loc[asof][ti_fwd20.loc[asof].notna()].index)
               if (use_fwd20 and asof in ti_fwd20.index) else set())
    cov_ti60_nor = cov_syms
    cov_ti20_nor = ti20_na & nor_na
    cov_all = cov_ti60_nor & ti20_na
    print(f"  as-of {asof.date()}: TI60 非NaN {len(ti_na)}, NOR 非NaN {len(nor_na)}, "
          f"TI60∩NOR(共振) {len(cov_ti60_nor)}")
    if use_fwd20:
        print(f"     TI20 非NaN {len(ti20_na)}, TI20∩NOR {len(cov_ti20_nor)}, "
              f"三者全交集(可算最终分) {len(cov_all)}")
        if len(cov_all) < args.min_blend_coverage:
            raise SystemExit(
                f"[fail-closed] as-of {asof.date()} 三者全交集仅 {len(cov_all)} < "
                f"门槛 {args.min_blend_coverage}；horizon-aware 最终分覆盖率不足，禁止晋级 latest。")

    rows = []
    for sym in common_syms:
        rec = {"symbol": sym, "ti_score": ti_row.get(sym), "nor_score": nor_row.get(sym)}
        if use_fwd20 and sym in ti_fwd20.columns:
            rec["ti_score_fwd20"] = ti_fwd20.loc[asof, sym] if asof in ti_fwd20.index else pd.NA
        else:
            rec["ti_score_fwd20"] = pd.NA
        rows.append(rec)
    out = pd.DataFrame(rows)

    # intersection: 各池 top-N 取交集（以主 ti_score 为准）
    ti_top = set(out.loc[out["ti_score"].notna()].sort_values("ti_score", ascending=False).head(args.top_n)["symbol"])
    nor_top = set(out.loc[out["nor_score"].notna()].sort_values("nor_score", ascending=False).head(args.top_n)["symbol"])
    inter = ti_top & nor_top
    out["in_intersection"] = out["symbol"].isin(inter)

    # weighted: 双覆盖符号上 z-score 混合；horizon-aware 双评分合成
    if args.method in ("weighted", "both"):
        sub = out[out["symbol"].isin(cov_syms)].copy().set_index("symbol")
        z_nor = _zscore(sub["nor_score"])
        z_ti60 = _zscore(sub["ti_score"])
        blend_60d = args.w_ti * z_ti60 + args.w_nor * z_nor
        if use_fwd20:
            z_ti20 = _zscore(sub["ti_score_fwd20"])
            blend_20d = args.w_ti * z_ti20 + args.w_nor * z_nor
            # horizon-aware 综合：双 horizon 互补，等权（可配 --w-horizon）
            w = args.w_horizon
            blended = (w * blend_60d + (1 - w) * blend_20d)
        else:
            blend_20d = blend_60d
            blended = blend_60d
        out = out.set_index("symbol")
        out["blended_score"] = blended
        out["blended_score_60d"] = blend_60d
        out["blended_score_20d"] = blend_20d
        out = out.reset_index()
        out["blended_rank"] = out["blended_score"].rank(ascending=False, method="min")
    else:
        out["blended_score"] = pd.NA
        out["blended_score_60d"] = pd.NA
        out["blended_score_20d"] = pd.NA
        out["blended_rank"] = pd.NA

    # 排序输出
    sort_col = "blended_score" if args.method in ("weighted", "both") else "ti_score"
    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    out.insert(0, "asof", asof.date())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blended_path = out_dir / "blended_pool.csv"
    out.to_csv(blended_path, index=False, encoding="utf-8")

    # ---- run 绑定 manifest (P0-1 加固) ----
    def _sha(p: Path) -> str:
        if not Path(p).exists():
            return "MISSING"
        h = hashlib.sha256()
        h.update(Path(p).read_bytes())
        return h.hexdigest()[:16]

    run_id = args.run_id or f"blend_{asof.date()}_{datetime.now().strftime('%H%M%S')}"
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asof": str(asof.date()),
        "asof_mode": args.asof_mode,
        "method": args.method,
        "horizon_aware": use_fwd20,
        "params": {
            "w_ti": args.w_ti, "w_nor": args.w_nor, "w_horizon": args.w_horizon,
            "top_n": args.top_n, "min_active": args.min_active,
            "min_blend_coverage": args.min_blend_coverage,
        },
        "inputs": {
            "ti_feed": str(args.ti_feed), "ti_feed_sha": _sha(args.ti_feed),
            "ti_feed_fwd20": str(args.ti_feed_fwd20), "ti_feed_fwd20_sha": _sha(args.ti_feed_fwd20),
            "nor_feed": str(args.nor_feed), "nor_feed_sha": _sha(args.nor_feed),
        },
        "coverage": {
            "ti60_nor": len(cov_ti60_nor), "ti20_nor": len(cov_ti20_nor),
            "all_three": len(cov_all) if use_fwd20 else None,
        },
        "output": {"blended_pool": str(blended_path), "blended_pool_sha": _sha(blended_path)},
        "note": ("权重 w_ti/w_nor/w_horizon 沿用泄漏回测调优结果；经 step5 无泄漏验证方案 B 纯 OOS "
                 "确凿为负（holdout 净20=-10.49% n=3、全轴准OOS 净20=-8.06% n=14，2026-08-23 口径），"
                 "已隔离、不晋级实盘，权重保持冻结不再替换。解除隔离须满足报告§五前置条件"
                 "（删 .QUARANTINE 标记 + 统计可信正 OOS 样本≥10 且均值>0 + 全部门控通过）。"),
    }
    if args.write_manifest:
        mpath = out_dir / "blend_manifest.json"
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  manifest -> {mpath}")

    print(f"\n=== 混合结果 (as-of {asof.date()}, method={args.method}, horizon-aware={use_fwd20}) ===")
    print(f"  TI top-{args.top_n} 与 NOR top-{args.top_n} 交集: {len(inter)} 只 -> {sorted(inter)[:20]}")
    if args.method in ("weighted", "both"):
        topb = out.dropna(subset=["blended_score"]).head(args.top_n)
        print(f"  weighted 前 {args.top_n} (blended_score):")
        for _, r in topb.iterrows():
            extra = f" 60d={r['blended_score_60d']:.2f} 20d={r['blended_score_20d']:.2f}" if use_fwd20 else ""
            print(f"    {r['symbol']}  ti={r['ti_score']:.4f} nor={r['nor_score']:.4f} "
                  f"blended={r['blended_score']:.3f}{extra} inter={bool(r['in_intersection'])}")
    print(f"\nsaved -> {out_dir / 'blended_pool.csv'}")


if __name__ == "__main__":
    main()
