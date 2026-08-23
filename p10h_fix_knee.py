"""P10h 修正：重算容量拐点（修复插值公式 gl/gh 反置 bug），从已存 metrics.json 重算，无需重跑回测。

正确插值：在 avg_gross 跨 0.6 的两个相邻 AUM 间做 log-AUM 线性插值：
  t = (0.6 - g_{i-1}) / (g_i - g_{i-1})   # 从 a_{i-1} 指向 a_i 的分数
  knee = a_{i-1} * (a_i / a_{i-1})^t
  knee_sharpe / knee_impact 同样按 t 插值
若 g0 < 0.6 -> 容量 < 最小 AUM（记 knee=最小 AUM 下界）；若全程 >=0.6 -> knee=最大 AUM（上界）。
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "p10h_capacity_engineering"
METRICS = OUT / "metrics.json"
DEPLOY_FLOOR = 0.6


def recompute_knee(row):
    aums = [r["aum"] for r in row["aum_sweep"]]
    gs = [r["avg_gross"] for r in row["aum_sweep"]]
    shps = [r["sharpe"] for r in row["aum_sweep"]]
    imps = [r["impact_share"] for r in row["aum_sweep"]]
    if gs[0] < DEPLOY_FLOOR:
        knee = aums[0]; t = 0.0; i = 0
    else:
        i = None
        for k in range(1, len(gs)):
            if gs[k] < DEPLOY_FLOOR:
                i = k; break
        if i is None:
            knee = aums[-1]; t = 1.0; i = len(gs) - 1
        else:
            t = (DEPLOY_FLOOR - gs[i-1]) / (gs[i] - gs[i-1])
            knee = aums[i-1] * (aums[i] / aums[i-1]) ** t
    knee_sharpe = float(np.interp(knee, aums, shps)) if i == 0 else \
        (shps[i-1] + t * (shps[i] - shps[i-1]))
    knee_impact = float(np.interp(knee, aums, imps)) if i == 0 else \
        (imps[i-1] + t * (imps[i] - imps[i-1]))
    # 边界情况用最近点
    if i == 0:
        knee_sharpe = shps[0]; knee_impact = imps[0]
    if i == len(gs) - 1:
        knee_sharpe = shps[-1]; knee_impact = imps[-1]
    return knee, round(float(knee_sharpe), 4), round(float(knee_impact), 4)


def main():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    for r in m["configs"]:
        knee, kshp, kimp = recompute_knee(r)
        r["capacity_knee_aum"] = knee
        r["knee_sharpe"] = kshp
        r["knee_impact_share"] = kimp

    viable = [r for r in m["configs"] if r["knee_sharpe"] >= 0.8 and r["knee_impact_share"] <= 0.5]
    viable.sort(key=lambda r: r["capacity_knee_aum"], reverse=True)
    best = viable[0] if viable else max(m["configs"], key=lambda r: r["capacity_knee_aum"])
    m["recommended_config"] = best["config"]
    m["recommended_capacity_knee_aum"] = best["capacity_knee_aum"]
    m["recommended_knee_sharpe"] = best["knee_sharpe"]
    m["recommended_knee_impact_share"] = best["knee_impact_share"]
    m["knee_interpolation_fixed"] = True
    METRICS.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== P10h 修正后容量前沿（部署率>=0.6 的拐点 AUM）===")
    print("config(top_n/w/p)        knee_AUM    knee_shp  knee_impact")
    for r in m["configs"]:
        c = r["config"]
        print(f"  {c['top_n']:>2d}/{c['w']:.2f}/{c['p']:.2f}        "
              f"{r['capacity_knee_aum']/1e8:>7.2f}e8   {r['knee_sharpe']:.3f}     {r['knee_impact_share']:.3f}")
    print(f"\n推荐配置: {best['config']}  容量拐点≈{best['capacity_knee_aum']/1e8:.2f}e8  "
          f"knee_sharpe={best['knee_sharpe']} impact={best['knee_impact_share']}")

    # 图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    keys = [(r["config"]["top_n"], r["config"]["w"], r["config"]["p"]) for r in m["configs"]]
    knees = [r["capacity_knee_aum"]/1e8 for r in m["configs"]]
    colors = ["#1f77b4" if r["config"]["p"] == 0.01 else "#ff7f0e" for r in m["configs"]]
    xlabels = [f"{c[0]}/{c[2]:.0%}" for c in keys]
    ax1.bar(range(len(knees)), knees, color=colors)
    ax1.set_xticks(range(len(knees)))
    ax1.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Capacity knee (AUM, e8)")
    ax1.set_title("Book capacity by config (blue=1% ADV, orange=2% ADV)\ntop_n/w/p")
    ax1.axhline(1.0, color="gray", ls=":", lw=1.0, label="baseline ~1e8")
    ax1.legend(fontsize=8)

    # avg_gross vs AUM for 4 key configs
    sel = {(20,0.04,0.01),(20,0.04,0.02),(40,0.04,0.01),(40,0.04,0.02)}
    for r in m["configs"]:
        c = r["config"]
        if (c["top_n"],c["w"],c["p"]) in sel:
            aums=[s["aum"]/1e8 for s in r["aum_sweep"]]
            gs=[s["avg_gross"] for s in r["aum_sweep"]]
            lbl=f"top_n={c['top_n']} p={c['p']:.0%}"
            ax2.plot(aums, gs, "o-", label=lbl)
    ax2.axhline(0.6, color="red", ls="--", lw=1.0, label="0.6 deploy floor")
    ax2.set_xlabel("AUM (e8)"); ax2.set_ylabel("avg gross exposure")
    ax2.set_title("Deployment vs AUM (selected configs)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capacity_frontier.png", dpi=130)
    print("saved", OUT / "capacity_frontier.png")


if __name__ == "__main__":
    main()
