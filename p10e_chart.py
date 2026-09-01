"""P10e chart (English titles to avoid CJK font warnings).

Panel A: cumulative net return (AUM=1e6) incumbent vs mlp_only vs ensemble_ew vs p10e_soft_causal,
         with W4 (panel days 660-880, the alpha-decay window) shaded.
Panel B: regime signal = 60d trailing rank-IC of MLP score, with 0 line and the alpha-dead threshold.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix, daily_ic
from execution_rules import next_open_return_label

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
P10C = HERE / "outputs" / "p10c_ensemble"
P10E = HERE / "outputs" / "p10e_regime_gated"
MAX_ABS = 0.22
SCORES = np.load(P10E / "linear_mlp_scores.npz")
EDGES = [0, 220, 440, 660, 880]


def cumret(csv, init=1_000_000.0):
    eq = pd.read_csv(csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    net = (eq["gross_return"] - eq["cost"]).values
    eqv = init * np.cumprod(1 + net)
    return eq["date"].values, eqv / init - 1


def main():
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    symbols = list(close.columns)
    mlp = pd.DataFrame(SCORES["mlp"], index=close.index, columns=symbols)
    mlp_ic = daily_ic({"mlp": mlp}, label)["mlp"]
    trailing = mlp_ic.rolling(60, min_periods=40).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    series = [
        ("incumbent", P10C / "equity_aum_1000000_incumbent.csv", "#888888", "--"),
        ("mlp_only", P10C / "equity_aum_1000000_mlp_only.csv", "#d62728", "-"),
        ("ensemble_ew", P10C / "equity_aum_1000000_ensemble_ew.csv", "#1f77b4", "-."),
        ("p10e_soft_causal", P10E / "equity_aum_1000000_p10e_soft_causal.csv", "#2ca02c", "-"),
    ]
    d0 = None
    for name, p, c, ls in series:
        d, r = cumret(p)
        if d0 is None:
            d0 = pd.DatetimeIndex(d)
        ax1.plot(d, r, label=name, color=c, ls=ls, lw=1.8)
    w4_start = close.index[EDGES[3]]
    w4_end = close.index[EDGES[4] - 1]
    ax1.axvspan(w4_start, w4_end, color="orange", alpha=0.15,
                label="W4 (alpha-decay window)")
    ax1.set_ylabel("Cumulative net return")
    ax1.set_title("P10e: regime-gated MLP (soft, causal) vs baselines — AUM=1e6")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    tr = trailing.reindex(d0)
    ax2.plot(d0, tr.values, color="#9467bd", lw=1.5, label="MLP trailing 60d IC")
    ax2.axhline(0.0, color="black", lw=1.0)
    ax2.axhline(0.03, color="green", ls=":", lw=1.0, label="soft health threshold (0.03)")
    ax2.axvspan(w4_start, w4_end, color="orange", alpha=0.15)
    ax2.set_ylabel("Trailing IC")
    ax2.set_xlabel("Date")
    ax2.set_title("Regime signal: MLP rank-IC quality (drops in W4 -> gate leans defensive)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = P10E / "p10e_causal_cumret_1e6.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
