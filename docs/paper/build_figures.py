from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parent
FIGURE_DIR = PAPER_DIR / "figures"
PROFILE_PATH = PAPER_DIR / "appendix" / "figure_data_profile.json"
MODEL_DIR = (
    ROOT
    / "outputs"
    / "high_return_v2"
    / "next_open_rank_model_stock_focus_lev093_marketfilter_bench20260722_20220101_20260722"
)
EQUITY_PATH = MODEL_DIR / "equity_curve.csv"
WEIGHTS_PATH = MODEL_DIR / "rolling_feature_weights.csv"
INITIAL_CAPITAL = 1_000_000.0


FACTOR_NAMES_CN = {
    "momentum_5": "5日动量",
    "momentum_20": "20日动量",
    "momentum_60": "60日动量",
    "reversal_5": "5日反转",
    "breakout_20": "20日突破",
    "distance_ma20": "距MA20",
    "volatility_20": "20日波动",
    "liquidity_20": "20日流动性",
    "intraday_return": "日内收益",
    "close_position": "收盘位置",
    "strong_pullback_20_5": "20日强势回调",
    "strong_pullback_60_5": "60日强势回调",
    "breakout_pullback_20_5": "突破后回调",
    "anti_chase_intraday": "日内反追高",
    "liquid_pullback": "流动性回调",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    gray = ROOT / "tmp" / "paper_figures_gray" / f"{stem}_gray.png"
    gray.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    with Image.open(png) as image:
        image.convert("L").save(gray)
    return {"png": str(png), "pdf": str(pdf), "grayscale_preview": str(gray)}


def load_equity() -> pd.DataFrame:
    equity = pd.read_csv(EQUITY_PATH, parse_dates=["date"])
    required = {
        "date",
        "equity",
        "gross_return",
        "cost",
        "gross_exposure",
        "market_exposure",
    }
    missing = required.difference(equity.columns)
    if missing:
        raise ValueError(f"equity curve missing columns: {sorted(missing)}")
    equity = equity.sort_values("date").drop_duplicates("date", keep="last")
    equity["net_return"] = equity["gross_return"] - equity["cost"]
    equity["nav"] = equity["equity"] / INITIAL_CAPITAL
    running_max = np.maximum.accumulate(np.r_[1.0, equity["nav"].to_numpy()])[1:]
    equity["drawdown"] = equity["nav"] / running_max - 1.0
    return equity


def figure_equity_drawdown(equity: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 4.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
        constrained_layout=True,
    )
    blue = "#2166AC"
    red = "#B2182B"
    axes[0].plot(equity["date"], equity["nav"], color=blue, linewidth=1.5)
    axes[0].axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("归一化净值")
    axes[0].set_title("严格次日开盘生产冠军的净值与回撤")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)

    axes[1].fill_between(
        equity["date"],
        equity["drawdown"] * 100.0,
        0,
        color=red,
        alpha=0.28,
        linewidth=0,
    )
    axes[1].plot(
        equity["date"], equity["drawdown"] * 100.0, color=red, linewidth=1.0
    )
    axes[1].set_ylabel("回撤 (%)")
    axes[1].set_xlabel("交易日期")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    return save_figure(fig, "figure_1_equity_drawdown")


def figure_calendar_and_exposure(equity: pd.DataFrame) -> dict[str, str]:
    annual = (
        equity.assign(year=equity["date"].dt.year)
        .groupby("year")["net_return"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    exposure_counts = (
        equity["market_exposure"].round(2).value_counts().sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.15), constrained_layout=True)
    annual_colors = ["#2166AC" if value >= 0 else "#B2182B" for value in annual]
    bars = axes[0].bar(
        [str(year) for year in annual.index],
        annual.to_numpy() * 100.0,
        color=annual_colors,
        width=0.62,
    )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_title("年度净收益集中度")
    axes[0].set_ylabel("净收益 (%)")
    axes[0].set_xlabel("2026 为截至 7 月 22 日")
    annual_percent = annual.to_numpy() * 100.0
    axes[0].set_ylim(float(annual_percent.min() - 3.0), float(annual_percent.max() + 4.0))
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    for bar, value in zip(bars, annual_percent):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + (1.0 if value >= 0 else -1.4),
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    exposure_colors = ["#B2182B", "#D9A441", "#2166AC"]
    labels = [f"{int(value * 100)}%" for value in exposure_counts.index]
    bars = axes[1].bar(
        labels,
        exposure_counts.to_numpy(),
        color=exposure_colors[: len(exposure_counts)],
        width=0.58,
    )
    axes[1].set_title("市场风险目标暴露的会话分布")
    axes[1].set_ylabel("交易会话数")
    axes[1].set_xlabel("目标暴露 (非实际仓位)")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    for bar, value in zip(bars, exposure_counts.to_numpy()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(exposure_counts.max() * 0.015, 1),
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return save_figure(fig, "figure_2_calendar_exposure")


def figure_weight_heatmap() -> dict[str, str]:
    weights = pd.read_csv(WEIGHTS_PATH, parse_dates=["date"]).set_index("date")
    weights = weights.apply(pd.to_numeric, errors="coerce")
    factor_order = [factor for factor in FACTOR_NAMES_CN if factor in weights.columns]
    matrix = weights[factor_order].T.to_numpy()
    finite = np.abs(matrix[np.isfinite(matrix)])
    bound = float(np.quantile(finite, 0.98)) if finite.size else 0.1
    bound = max(bound, 0.05)

    fig, ax = plt.subplots(figsize=(6.4, 4.35), constrained_layout=True)
    norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(range(len(factor_order)))
    ax.set_yticklabels([FACTOR_NAMES_CN[factor] for factor in factor_order])
    tick_count = min(7, len(weights))
    tick_positions = np.linspace(0, len(weights) - 1, tick_count, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [weights.index[position].strftime("%Y-%m") for position in tick_positions],
        rotation=30,
        ha="right",
    )
    ax.set_title("滚动训练得到的因子权重 (仅表示打分倾向)")
    ax.set_xlabel("重训日期")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("归一化平均 RankIC 权重", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    return save_figure(fig, "figure_3_rolling_factor_weights")


def write_profile(equity: pd.DataFrame, outputs: dict[str, dict[str, str]]) -> None:
    weights = pd.read_csv(WEIGHTS_PATH)
    profile = {
        "equity_source": str(EQUITY_PATH),
        "weights_source": str(WEIGHTS_PATH),
        "equity_rows": int(len(equity)),
        "equity_start": equity["date"].min().strftime("%Y-%m-%d"),
        "equity_end": equity["date"].max().strftime("%Y-%m-%d"),
        "equity_missing_values": {
            column: int(value) for column, value in equity.isna().sum().items()
        },
        "weight_rows": int(len(weights)),
        "weight_factor_count": int(max(len(weights.columns) - 1, 0)),
        "max_drawdown_recomputed": float(equity["drawdown"].min()),
        "annual_net_returns": {
            str(year): float(np.prod(1.0 + group["net_return"]) - 1.0)
            for year, group in equity.groupby(equity["date"].dt.year)
        },
        "market_exposure_session_counts": {
            f"{float(exposure):.2f}": int(count)
            for exposure, count in equity["market_exposure"]
            .round(2)
            .value_counts()
            .sort_index()
            .items()
        },
        "figure_outputs": outputs,
    }
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    configure_style()
    equity = load_equity()
    outputs = {
        "figure_1": figure_equity_drawdown(equity),
        "figure_2": figure_calendar_and_exposure(equity),
        "figure_3": figure_weight_heatmap(),
    }
    write_profile(equity, outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
