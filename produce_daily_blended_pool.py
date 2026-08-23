"""produce_daily_blended_pool.py — 方案 B 每日生产编排器 (trend_ignition × next_open_rank 混合).

把两条独立链路封装成一个每日步骤:
  1) (可选) 重建延伸到 live 的 TI feed: build_live_trend_ignition_feed.py
  2) 混合: blend_candidate_pools.py --asof-mode latest (最近一个有双覆盖的日期, 生产口径)
  3) 把 blended_pool.csv 整理为可直接被 overlay/日报/盘前简报消费的产物:
       blended_pool_latest.csv  (权威最新混合名单, 带 as-of)
       blended_pool_latest.md   (人读 top-N 表, 供盘前简报拼接)

用法 (每日 19:00 自动化步骤):
  python produce_daily_blended_pool.py --refresh-ti-feed
  (首次/面板更新后加 --refresh-ti-feed; 日常只跑混合即可)

产物默认 outputs/high_return_v2/trend_ignition_daily_pool/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # P2: 跟随当前解释器，避免硬编码作者机器 Python 路径
OUT = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"步骤失败: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="方案 B 每日混合编排器")
    ap.add_argument("--refresh-ti-feed", action="store_true", help="先重建 live TI feed (面板更新后必加)")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--w-ti", type=float, default=0.9)
    ap.add_argument("--w-nor", type=float, default=0.1)
    ap.add_argument("--min-active", type=int, default=20)
    ap.add_argument("--min-blend-coverage", type=int, default=30,
                    help="P1 加固：三者全交集覆盖率门槛，低于此 fail-closed 不晋级 latest")
    ap.add_argument("--output-dir", default=str(OUT))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_ti_feed:
        run([PY, "build_live_trend_ignition_feed.py"])

    run([PY, "blend_candidate_pools.py",
         "--asof-mode", "latest",
         "--top-n", str(args.top_n),
         "--w-ti", str(args.w_ti),
         "--w-nor", str(args.w_nor),
         "--min-active", str(args.min_active),
         "--min-blend-coverage", str(args.min_blend_coverage),
         "--output-dir", str(out_dir)])

    # 整理产物
    pool = pd.read_csv(out_dir / "blended_pool.csv")
    asof = pool["asof"].iloc[0] if "asof" in pool.columns else "unknown"
    has_fwd20 = "blended_score_20d" in pool.columns and pool["blended_score_20d"].notna().any()

    # ---- P0-1 fail-closed: 必须绑定同一 run 的 manifest，覆盖率不足禁止晋级 latest ----
    manifest_path = out_dir / "blend_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"[fail-closed] 缺少 blend_manifest.json（run 未绑定），拒绝写 latest。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cov_all = manifest.get("coverage", {}).get("all_three")
    min_cov = args.min_blend_coverage
    if has_fwd20 and cov_all is not None and cov_all < min_cov:
        raise SystemExit(
            f"[fail-closed] 三者全交集覆盖率 {cov_all} < 门槛 {min_cov}，"
            f"horizon-aware 最终分不足，拒绝晋级 latest（保留 blended_pool.csv 供研究）。")

    blended = pool.dropna(subset=["blended_score"]).sort_values("blended_score", ascending=False).head(args.top_n)

    latest_csv = out_dir / "blended_pool_latest.csv"
    blended.to_csv(latest_csv, index=False, encoding="utf-8")

    horizon_note = ("horizon-aware 双评分（fwd60 + fwd20 互补合成）；" if has_fwd20
                    else "单 horizon（fwd60）；")
    coverage_note = (f"TI60∩NOR={manifest['coverage']['ti60_nor']} 只，"
                     f"TI20∩NOR={manifest['coverage']['ti20_nor']} 只，"
                     f"三者全交集(可算最终分)={cov_all} 只。"
                     if has_fwd20 else
                     f"TI60∩NOR={manifest['coverage']['ti60_nor']} 只。")
    lines = [
        f"# 趋势点火 × next_open_rank 混合候选池 (as-of {asof})",
        "",
        f"> run_id: `{manifest['run_id']}` · 生成: {manifest['generated_at']}",
        f"> 混合权重 w_ti={args.w_ti} / w_nor={args.w_nor}；前 {args.top_n} 名。{horizon_note}",
        f"> 覆盖率: {coverage_note}（三者全交集偏小，最终分样本有限，结论谨慎）",
        "> 研究信号，非投资建议。双信号加权融合；权重沿用泄漏回测调优，经 step5 无泄漏验证"
        "方案 B 纯 OOS 确凿为负，已隔离、不晋级实盘，权重冻结不再替换（解除条件见 blend_manifest note）。",
        "",
        "| 排名 | 代码 | TI评分 | NOR评分 | 混合分 | 双覆盖 |",
        "|---|---|---|---|---|---|",
    ]
    for i, (_, r) in enumerate(blended.iterrows(), 1):
        lines.append(f"| {i} | {r['symbol']} | {r['ti_score']:.4f} | {r['nor_score']:.4f} | "
                     f"{r['blended_score']:.3f} | {'✓' if r.get('in_intersection') else ''} |")
    lines += ["", f"_生成: produce_daily_blended_pool.py @ {asof} · run {manifest['run_id']}_"]
    (out_dir / "blended_pool_latest.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✅ 方案 B 每日混合完成 as-of={asof} (run {manifest['run_id']})")
    print(f"  双覆盖候选 {len(pool.dropna(subset=['blended_score']))} 只, 输出 top-{args.top_n}")
    print(f"  覆盖率 TI60∩NOR={manifest['coverage']['ti60_nor']} / 三者全交集={cov_all}")
    print(f"  -> {latest_csv}")
    print(f"  -> {out_dir / 'blended_pool_latest.md'}")


if __name__ == "__main__":
    main()
