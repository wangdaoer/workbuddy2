"""P10g-production monitoring chart (English titles).

Plots the MLP trailing rank-IC (causal, 60d) over the last ~250 trading days with the
healthy/decaying/dead thresholds, shades dead regions, and marks the latest asof point.
Demonstrates the regime gate triggering de-risk at the live edge (as of 2026-07-23: dead).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix, daily_ic
from execution_rules import next_open_return_label
from p10c_ensemble import TRAIN_DAYS, build_features

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
P10E = HERE / "outputs" / "p10e_regime_gated"
SCORES_NPZ = P10E / "linear_mlp_scores.npz"
OUT_DIR = HERE / "outputs" / "production_soft_score"
IC_WIN, IC_MIN, THR_HI, THR_LO = 60, 40, 0.03, 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P10g-production regime monitor chart (daily-refreshable)."
    )
    parser.add_argument("--panel", default=str(PANEL), help="日更面板 CSV（含 close/open 等）")
    parser.add_argument("--scores-npz", default=str(SCORES_NPZ),
                        help="p10e_regime_gated 的 linear_mlp_scores.npz（外部前提产物）")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="图表输出目录")
    parser.add_argument("--asof-date", default=None, help="YYYY-MM-DD，用于文件名令牌；缺省取数据末日")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    panel_path = Path(args.panel)
    scores_npz = Path(args.scores_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = (
        pd.Timestamp(args.asof_date).strftime("%Y%m%d")
        if args.asof_date else "latest"
    )
    out = output_dir / f"regime_monitor_chart_{token}.png"

    raw = load_prices(panel_path, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), 0.22)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), 0.22)
    label = next_open_return_label(open_px, max_abs_daily_return=0.22)
    symbols = list(close.columns)
    d = np.load(scores_npz, allow_pickle=True)
    mlp = pd.DataFrame(d["mlp"], index=close.index, columns=symbols)
    mlp_ic = daily_ic({"mlp": mlp}, label)["mlp"]
    # P0-3 残留修复 (2026-08-23): label[t]=open[t+2]/open[t+1]-1 在 t+2 开盘才成熟,
    # 因果 trailing 须用 shift(2); 原 shift(1) 泄漏 1 天 (与 production_soft_score 一致).
    trailing = mlp_ic.shift(2).rolling(IC_WIN, min_periods=IC_MIN).mean()

    s = trailing.iloc[-250:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(s.index, s.values, color="#9467bd", lw=1.6, label="MLP trailing 60d IC (causal)")
    ax.axhline(THR_HI, color="green", ls=":", lw=1.0, label="healthy threshold (0.03)")
    ax.axhline(THR_LO, color="red", ls="--", lw=1.0, label="dead threshold (0.0)")
    dead = s <= THR_LO
    if dead.any():
        ax.fill_between(s.index, s.values, THR_LO, where=dead.values, color="red", alpha=0.20)
    # mark asof
    ax.scatter([s.index[-1]], [s.values[-1]], color="black", zorder=5, s=40)
    ax.annotate(f"asof {s.index[-1].date()}\nIC={s.values[-1]:.4f} -> DEAD\nde-risk gross to 0.40",
                (s.index[-1], s.values[-1]), xytext=(-120, 20), textcoords="offset points",
                fontsize=9, bbox=dict(boxstyle="round", fc="yellow", alpha=0.6))
    ax.set_title("Production regime monitor: MLP trailing rank-IC (live edge)")
    ax.set_ylabel("Trailing IC")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

