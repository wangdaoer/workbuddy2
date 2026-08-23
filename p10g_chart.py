"""P10g chart (English titles).

Panel A: combined portfolio equity (hk_passive + risk_parity) p8b baseline vs p10e_soft upgrade.
Panel B: book capacity profile (avg gross exposure vs AUM) under 1% ADV cap - soft vs p8b incumbent.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
P10G = HERE / "outputs" / "p10g_production_integration"
OUT = P10G / "p10g_portfolio_cumret.png"


def main():
    # Panel A
    a = pd.read_csv(P10G / "portfolio_p8b_baseline_hk_passive_risk_parity.csv")
    b = pd.read_csv(P10G / "portfolio_p10e_soft_hk_passive_risk_parity.csv")
    a["d"] = pd.to_datetime(a["td"], format="%Y%m%d")
    b["d"] = pd.to_datetime(b["td"], format="%Y%m%d")
    a = a.sort_values("d").reset_index(drop=True)
    b = b.sort_values("d").reset_index(drop=True)

    # Panel B capacity
    g = json.loads((P10G / "metrics.json").read_text(encoding="utf-8"))
    soft = g["book_capacity"]["capacity_sweep"]
    p6 = json.loads(Path(HERE / "outputs/p6b_capacity_at_scale/metrics.json").read_text(encoding="utf-8"))
    p8b = p6["aum_sweep"]
    sa = [r["aum"] for r in soft]
    sg = [r["avg_gross_exposure"] for r in soft]
    pa = [r["aum"] for r in p8b]
    pg = [r.get("avg_gross_exposure") for r in p8b]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(a["d"], a["equity"] / a["equity"].iloc[0], label="p8b book (baseline)", color="#888", ls="--", lw=1.8)
    ax1.plot(b["d"], b["equity"] / b["equity"].iloc[0], label="p10e_soft book (upgrade)", color="#2ca02c", lw=1.8)
    ax1.set_title("Combined portfolio (hk_passive + risk_parity)\nbook: p8b -> p10e_soft")
    ax1.set_ylabel("Growth of 1.0")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(sa, sg, "o-", color="#2ca02c", label="p10e_soft (regime-gated MLP)")
    ax2.plot(pa, pg, "s--", color="#888", label="p8b incumbent (linear)")
    ax2.axhline(0.6, color="red", ls=":", lw=1.0, label="75% deploy threshold")
    ax2.set_xscale("log")
    ax2.set_xticks(sa)
    ax2.set_xticklabels([f"{int(x/1e6)}M" if x < 1e8 else f"{int(x/1e8)}e8" for x in sa])
    ax2.set_xlabel("Book AUM (CNY, log)")
    ax2.set_ylabel("Avg gross exposure (deployment)")
    ax2.set_title("Book capacity profile (1% ADV cap)\nsoft has higher alpha, same capacity knee")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("saved", OUT)


if __name__ == "__main__":
    main()
