"""P10e fix: recompute W1-W4 OOS decomposition with correct trading-calendar edges.

The original p10e_regime_gated.py split by full-panel 880-day edges, but equity curves
only start trading after the warm-up (~day 253, ~627 rows). That made net.iloc[660:880]
out-of-range -> W4=nan, and shifted W1/W2/W3 by one quarter.

Fix: reindex each net-return series onto the full panel date axis (warm-up filled 0),
then split on the same panel quartiles as P10d (0,220,440,660,880). W1 (0-220) is warm-up
with no trades -> reported n/a.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
P10C = HERE / "outputs" / "p10c_ensemble"
P10E = HERE / "outputs" / "p10e_regime_gated"
MAX_ABS = 0.22
TRAIN_DAYS = 252
AUMS = [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]
EDGES = [0, 220, 440, 660, 880]


def sharpe_like(net):
    net = np.asarray(net, float)
    if len(net) < 6 or net.std() == 0:
        return float("nan")
    return float(net.mean() / net.std() * np.sqrt(252))


def load_net(date_index, csv):
    eq = pd.read_csv(csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    net = (eq["gross_return"] - eq["cost"]).values
    s = pd.Series(0.0, index=date_index)
    s.loc[eq["date"].values] = net
    return s


def main():
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    date_index = close.index
    print(f"panel days={len(date_index)}  trade start ~day {TRAIN_DAYS+1}")

    paths = {
        "incumbent": P10C,
        "mlp_only": P10C,
        "ensemble_ew": P10C,
        "p10e_switch": P10E,
        "p10e_defense": P10E,
        "p10e_soft": P10E,
    }

    for aum in AUMS:
        ai = int(aum)
        nets = {}
        for v, d in paths.items():
            p = d / f"equity_aum_{ai}_{v}.csv"
            if p.exists():
                nets[v] = load_net(date_index, p)
        print(f"\n=== AUM={aum:,.0f}  (W1=warm-up n/a) ===")
        print("variant".ljust(14) + "".join(f"W{k+1}_shp".rjust(11) for k in range(4))
              + "".join(f"W{k+1}_ret".rjust(11) for k in range(4)))
        for v in ["incumbent", "mlp_only", "ensemble_ew", "p10e_switch", "p10e_defense", "p10e_soft"]:
            if v not in nets:
                continue
            s = nets[v]
            shp = ""
            ret = ""
            for k in range(4):
                seg = s.iloc[EDGES[k]:EDGES[k + 1]].values
                seg = seg[~np.isnan(seg)]
                if k == 0 or len(seg) < 6 or np.allclose(seg, 0):
                    shp += f"{'n/a':>11}"
                    ret += f"{np.prod(1+seg)-1:>+11.3f}"
                else:
                    shp += f"{sharpe_like(seg):>11.3f}"
                    ret += f"{np.prod(1+seg)-1:>+11.3f}"
            print(v.ljust(14) + shp + ret)

    m = json.loads((P10E / "metrics.json").read_text(encoding="utf-8"))
    print("\nregime dead% by panel quartile:", m["regime_dead_pct"])


if __name__ == "__main__":
    main()
