"""P10d 辅助：绘制累计净收益曲线（4 变体），标注 4 等分窗口与 W4 阴影。"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "p10c_ensemble"
AUM = 1_000_000.0
VARIANTS = ["incumbent", "mlp_only", "ensemble_ew", "ensemble_icw"]
COLORS = {"incumbent": "#888888", "mlp_only": "#d62728", "ensemble_ew": "#1f77b4", "ensemble_icw": "#ff7f0e"}

df = None
for v in VARIANTS:
    d = pd.read_csv(OUT / f"equity_aum_{int(AUM)}_{v}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d["cum"] = (1 + d["gross_return"] - d["cost"]).cumprod()
    if df is None:
        df = d[["date", "cum"]].rename(columns={"cum": v})
    else:
        df = df.merge(d[["date", "cum"]].rename(columns={"cum": v}), on="date")

n = len(df)
edges = [0, n // 4, n // 2, 3 * n // 4, n]
fig, ax = plt.subplots(figsize=(11, 5.5))
for v in VARIANTS:
    ax.plot(df["date"], df[v], label=v, color=COLORS[v], lw=1.6)
ax.axvspan(df["date"].iloc[edges[3]], df["date"].iloc[edges[4] - 1], color="red", alpha=0.08, label="W4 (latest adverse window)")
for k in range(1, 4):
    ax.axvline(df["date"].iloc[edges[k]], color="gray", ls="--", lw=0.8, alpha=0.6)
ax.set_title(f"P10d: cumulative net return (AUM=1e6, shaded=W4 latest adverse window)", fontsize=11)
ax.set_ylabel("cumulative NAV")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "p10d_cumret_1e6.png", dpi=130)
print("saved", OUT / "p10d_cumret_1e6.png")
