"""Analyze personal brokerage executions and derive strategy overlay evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workspace_paths import personal_trades_file


SOURCE_DEFAULT = personal_trades_file()
PANEL_DEFAULT = Path("data_panel_history_main_chinext_20220101_20260629.csv")
OUTPUT_DEFAULT = Path("outputs/personal_trade_review_20260629")


RAW_COLUMNS = [
    "trade_date_raw",
    "trade_time",
    "symbol",
    "name",
    "operation",
    "quantity",
    "trade_id",
    "price",
    "gross_amount",
    "cash_balance",
    "position_balance",
    "cash_flow",
    "commission",
    "stamp_tax",
    "misc_fee",
    "current_amount",
    "contract_id",
    "occurrence_quantity",
    "turnover_amount",
    "net_commission",
    "regulatory_fee",
    "transfer_fee",
    "market",
]


def clean_symbol(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else text


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def load_broker_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", encoding="gbk", dtype=str, engine="python")
    raw = raw.iloc[:, : len(RAW_COLUMNS)].copy()
    raw.columns = RAW_COLUMNS
    for col in raw.columns:
        raw[col] = raw[col].astype(str).str.strip().str.replace("\r", "", regex=False)
    raw = raw.replace({"nan": np.nan, "": np.nan})

    out = raw.copy()
    out["symbol"] = out["symbol"].map(clean_symbol)
    out["trade_date"] = pd.to_datetime(out["trade_date_raw"], format="%Y%m%d", errors="coerce")
    out["trade_dt"] = pd.to_datetime(
        out["trade_date"].dt.strftime("%Y-%m-%d") + " " + out["trade_time"].fillna("00:00:00"),
        errors="coerce",
    )
    numeric_cols = [
        "quantity",
        "price",
        "gross_amount",
        "cash_balance",
        "position_balance",
        "cash_flow",
        "commission",
        "stamp_tax",
        "misc_fee",
        "current_amount",
        "occurrence_quantity",
        "turnover_amount",
        "net_commission",
        "regulatory_fee",
        "transfer_fee",
    ]
    for col in numeric_cols:
        out[col] = to_num(out[col])
    op = out["operation"].fillna("")
    out["side"] = np.select(
        [op.str.contains("买入", regex=False), op.str.contains("卖出", regex=False)],
        ["buy", "sell"],
        default="other",
    )
    out["abs_quantity"] = out["quantity"].abs()
    out["fees"] = out[["commission", "stamp_tax", "misc_fee", "regulatory_fee", "transfer_fee"]].fillna(0).sum(axis=1)
    out["year"] = out["trade_date"].dt.year
    out["month"] = out["trade_date"].dt.to_period("M").astype(str)
    out["weekday"] = out["trade_date"].dt.day_name()
    out["time_bucket"] = out["trade_dt"].map(time_bucket)
    return out.sort_values(["trade_dt", "contract_id"], kind="mergesort").reset_index(drop=True)


def time_bucket(dt: pd.Timestamp | float) -> str:
    if not isinstance(dt, pd.Timestamp) or pd.isna(dt):
        return "unknown"
    minute = dt.hour * 60 + dt.minute
    if minute < 9 * 60 + 30:
        return "preopen_or_afterhours"
    if minute < 10 * 60:
        return "09:30-10:00"
    if minute < 11 * 60 + 30:
        return "10:00-11:30"
    if minute < 13 * 60:
        return "lunch_or_afterhours"
    if minute < 14 * 60 + 30:
        return "13:00-14:30"
    if minute <= 15 * 60:
        return "14:30-15:00"
    return "afterhours"


def build_round_trips(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lots: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    rows: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    stock_trades = trades[trades["side"].isin(["buy", "sell"])].copy()

    for _, row in stock_trades.iterrows():
        symbol = row["symbol"]
        qty = float(abs(row["quantity"])) if pd.notna(row["quantity"]) else 0.0
        if qty <= 0 or pd.isna(row["price"]):
            continue
        if row["side"] == "buy":
            total_cost = abs(float(row["cash_flow"])) if pd.notna(row["cash_flow"]) and row["cash_flow"] < 0 else float(row["gross_amount"] + row["fees"])
            if not np.isfinite(total_cost) or total_cost <= 0:
                unmatched.append(
                    {
                        "symbol": symbol,
                        "name": row["name"],
                        "buy_dt": row["trade_dt"],
                        "buy_qty": qty,
                        "buy_price": float(row["price"]),
                        "reason": "buy_has_non_positive_cost",
                    }
                )
                continue
            lots[symbol].append(
                {
                    "buy_dt": row["trade_dt"],
                    "buy_date": row["trade_date"],
                    "buy_time": row["trade_time"],
                    "symbol": symbol,
                    "name": row["name"],
                    "remaining_qty": qty,
                    "buy_qty": qty,
                    "buy_price": float(row["price"]),
                    "buy_cost_per_share": total_cost / qty,
                    "buy_gross_amount": float(row["gross_amount"]),
                    "buy_total_cost": total_cost,
                    "buy_fees": float(row["fees"]),
                    "buy_time_bucket": row["time_bucket"],
                    "buy_market": row["market"],
                    "buy_contract_id": row["contract_id"],
                }
            )
            continue

        sell_net_total = float(row["cash_flow"]) if pd.notna(row["cash_flow"]) and row["cash_flow"] > 0 else float(row["gross_amount"] - row["fees"])
        sell_net_per_share = sell_net_total / qty
        remaining_sell = qty
        if not lots[symbol]:
            unmatched.append(
                {
                    "symbol": symbol,
                    "name": row["name"],
                    "sell_dt": row["trade_dt"],
                    "sell_qty": remaining_sell,
                    "sell_price": float(row["price"]),
                    "reason": "sell_without_prior_buy_in_file",
                }
            )
            continue

        while remaining_sell > 1e-9:
            if not lots[symbol]:
                unmatched.append(
                    {
                        "symbol": symbol,
                        "name": row["name"],
                        "sell_dt": row["trade_dt"],
                        "sell_qty": remaining_sell,
                        "sell_price": float(row["price"]),
                        "reason": "sell_exceeds_prior_buys_in_file",
                    }
                )
                break
            lot = lots[symbol][0]
            matched_qty = min(remaining_sell, float(lot["remaining_qty"]))
            buy_cost_per_share = float(lot["buy_cost_per_share"])
            if not np.isfinite(buy_cost_per_share) or buy_cost_per_share <= 0:
                unmatched.append(
                    {
                        "symbol": symbol,
                        "name": lot["name"],
                        "buy_dt": lot["buy_dt"],
                        "remaining_qty": lot["remaining_qty"],
                        "buy_price": lot["buy_price"],
                        "reason": "matched_buy_has_non_positive_cost",
                    }
                )
                lots[symbol].popleft()
                continue
            pnl = matched_qty * (sell_net_per_share - buy_cost_per_share)
            ret = sell_net_per_share / buy_cost_per_share - 1.0
            holding_days = (row["trade_date"] - lot["buy_date"]).days
            rows.append(
                {
                    "symbol": symbol,
                    "name": lot["name"],
                    "buy_dt": lot["buy_dt"],
                    "buy_date": lot["buy_date"],
                    "buy_time": lot["buy_time"],
                    "buy_time_bucket": lot["buy_time_bucket"],
                    "buy_price": lot["buy_price"],
                    "buy_cost_per_share": buy_cost_per_share,
                    "sell_dt": row["trade_dt"],
                    "sell_date": row["trade_date"],
                    "sell_time": row["trade_time"],
                    "sell_time_bucket": row["time_bucket"],
                    "sell_price": float(row["price"]),
                    "sell_net_per_share": sell_net_per_share,
                    "matched_qty": matched_qty,
                    "gross_buy_amount": matched_qty * float(lot["buy_price"]),
                    "net_sell_amount": matched_qty * sell_net_per_share,
                    "pnl": pnl,
                    "return_pct": ret,
                    "holding_days": holding_days,
                    "buy_contract_id": lot["buy_contract_id"],
                    "sell_contract_id": row["contract_id"],
                }
            )
            lot["remaining_qty"] = float(lot["remaining_qty"]) - matched_qty
            remaining_sell -= matched_qty
            if float(lot["remaining_qty"]) <= 1e-9:
                lots[symbol].popleft()

    open_rows = []
    for symbol, symbol_lots in lots.items():
        for lot in symbol_lots:
            open_rows.append(
                {
                    "symbol": symbol,
                    "name": lot["name"],
                    "buy_dt": lot["buy_dt"],
                    "remaining_qty": lot["remaining_qty"],
                    "buy_price": lot["buy_price"],
                    "buy_cost_per_share": lot["buy_cost_per_share"],
                    "reason": "open_lot_not_sold_by_file_end",
                }
            )
    unmatched.extend(open_rows)
    return pd.DataFrame(rows), pd.DataFrame(unmatched)


def enrich_with_market(round_trips: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    if round_trips.empty or not panel_path.exists():
        return round_trips
    panel = pd.read_csv(panel_path, parse_dates=["date"], dtype={"symbol": str})
    panel["symbol"] = panel["symbol"].map(clean_symbol)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "amount"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    grouped = panel.groupby("symbol", sort=False)
    panel["prev_close"] = grouped["close"].shift(1)
    panel["ret_5_pre"] = grouped["close"].pct_change(5, fill_method=None).shift(1)
    panel["ret_20_pre"] = grouped["close"].pct_change(20, fill_method=None).shift(1)
    panel["ret_60_pre"] = grouped["close"].pct_change(60, fill_method=None).shift(1)
    panel["ma20_pre"] = grouped["close"].transform(lambda s: s.rolling(20).mean().shift(1))
    panel["ma60_pre"] = grouped["close"].transform(lambda s: s.rolling(60).mean().shift(1))
    panel["ma120_pre"] = grouped["close"].transform(lambda s: s.rolling(120).mean().shift(1))
    panel["dist_ma20_pre"] = panel["prev_close"] / (panel["ma20_pre"] + 1e-12) - 1.0
    panel["dist_ma60_pre"] = panel["prev_close"] / (panel["ma60_pre"] + 1e-12) - 1.0
    panel["dist_ma120_pre"] = panel["prev_close"] / (panel["ma120_pre"] + 1e-12) - 1.0
    panel["avg_amount_20_pre"] = grouped["amount"].transform(lambda s: s.replace(0, np.nan).rolling(20).median().shift(1))

    buy_key = panel[
        [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "ret_5_pre",
            "ret_20_pre",
            "ret_60_pre",
            "dist_ma20_pre",
            "dist_ma60_pre",
            "dist_ma120_pre",
            "avg_amount_20_pre",
        ]
    ].rename(
        columns={
            "date": "buy_date",
            "open": "buy_day_open",
            "high": "buy_day_high",
            "low": "buy_day_low",
            "close": "buy_day_close",
        }
    )
    out = round_trips.merge(buy_key, on=["symbol", "buy_date"], how="left")
    rng = (out["buy_day_high"] - out["buy_day_low"]).replace(0, np.nan)
    out["entry_day_position"] = (out["buy_price"] - out["buy_day_low"]) / (rng + 1e-12)
    out["entry_gap_from_prev_close"] = out["buy_price"] / (out["prev_close"] + 1e-12) - 1.0
    out["entry_above_ma20_pre"] = out["dist_ma20_pre"].gt(0)
    out["entry_above_ma60_pre"] = out["dist_ma60_pre"].gt(0)

    indexed = {sym: grp.set_index("date").sort_index() for sym, grp in panel.groupby("symbol")}
    mfe = []
    mae = []
    close_to_sell = []
    for row in out.itertuples(index=False):
        hist = indexed.get(row.symbol)
        if hist is None:
            mfe.append(np.nan)
            mae.append(np.nan)
            close_to_sell.append(np.nan)
            continue
        window = hist.loc[(hist.index >= row.buy_date) & (hist.index <= row.sell_date)]
        if window.empty:
            mfe.append(np.nan)
            mae.append(np.nan)
            close_to_sell.append(np.nan)
            continue
        cost = float(row.buy_cost_per_share)
        mfe.append(float(window["high"].max() / (cost + 1e-12) - 1.0))
        mae.append(float(window["low"].min() / (cost + 1e-12) - 1.0))
        close_to_sell.append(float(window["close"].iloc[-1] / (cost + 1e-12) - 1.0))
    out["mfe_pct"] = mfe
    out["mae_pct"] = mae
    out["close_to_sell_return_pct"] = close_to_sell
    out["giveback_from_mfe_pct"] = out["mfe_pct"] - out["return_pct"]
    out["exit_efficiency"] = np.where(out["mfe_pct"].gt(0), out["return_pct"] / out["mfe_pct"], np.nan)
    return out


def weighted_mean(frame: pd.DataFrame, value: str, weight: str = "gross_buy_amount") -> float:
    valid = frame[[value, weight]].dropna()
    if valid.empty or valid[weight].abs().sum() <= 0:
        return float("nan")
    return float(np.average(valid[value], weights=valid[weight].abs()))


def summarize(round_trips: pd.DataFrame, trades: pd.DataFrame, unmatched: pd.DataFrame) -> dict[str, object]:
    stock_trades = trades[trades["side"].isin(["buy", "sell"])]
    wins = round_trips[round_trips["pnl"] > 0]
    losses = round_trips[round_trips["pnl"] < 0]
    gross_win = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["pnl"].sum()) if not losses.empty else 0.0
    total_pnl = float(round_trips["pnl"].sum()) if not round_trips.empty else 0.0
    return {
        "source_rows": int(len(trades)),
        "period_start": str(trades["trade_date"].min().date()) if trades["trade_date"].notna().any() else None,
        "period_end": str(trades["trade_date"].max().date()) if trades["trade_date"].notna().any() else None,
        "operation_counts": trades["operation"].value_counts(dropna=False).to_dict(),
        "stock_execution_count": int(len(stock_trades)),
        "buy_execution_count": int((stock_trades["side"] == "buy").sum()),
        "sell_execution_count": int((stock_trades["side"] == "sell").sum()),
        "unique_symbols_traded": int(stock_trades["symbol"].nunique()),
        "matched_round_trips": int(len(round_trips)),
        "unmatched_lots": int(len(unmatched)),
        "realized_pnl": total_pnl,
        "gross_profit": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": gross_win / abs(gross_loss) if gross_loss < 0 else None,
        "win_rate": float((round_trips["pnl"] > 0).mean()) if not round_trips.empty else None,
        "avg_return_pct": float(round_trips["return_pct"].mean()) if not round_trips.empty else None,
        "weighted_return_pct": weighted_mean(round_trips, "return_pct") if not round_trips.empty else None,
        "median_return_pct": float(round_trips["return_pct"].median()) if not round_trips.empty else None,
        "avg_holding_days": float(round_trips["holding_days"].mean()) if not round_trips.empty else None,
        "median_holding_days": float(round_trips["holding_days"].median()) if not round_trips.empty else None,
        "same_day_share": float((round_trips["holding_days"] == 0).mean()) if not round_trips.empty else None,
        "le_3_day_share": float((round_trips["holding_days"] <= 3).mean()) if not round_trips.empty else None,
        "fee_total": float(stock_trades["fees"].sum()),
        "fee_to_stock_turnover": float(stock_trades["fees"].sum() / stock_trades["gross_amount"].abs().sum())
        if stock_trades["gross_amount"].abs().sum() > 0
        else None,
    }


def bucket_tables(round_trips: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if round_trips.empty:
        return out
    rt = round_trips.copy()
    rt["holding_bucket"] = pd.cut(
        rt["holding_days"],
        bins=[-1, 0, 1, 3, 10, 30, 10_000],
        labels=["0d", "1d", "2-3d", "4-10d", "11-30d", ">30d"],
    )
    rt["entry_momentum_bucket"] = pd.cut(
        rt["ret_20_pre"],
        bins=[-10, -0.2, -0.05, 0.05, 0.2, 10],
        labels=["20d<-20%", "-20%~-5%", "-5%~+5%", "+5%~+20%", ">+20%"],
    )
    rt["entry_position_bucket"] = pd.cut(
        rt["entry_day_position"],
        bins=[-1, 0.25, 0.5, 0.75, 2],
        labels=["low_quarter", "mid_low", "mid_high", "high_quarter"],
    )

    def agg(group_cols: list[str]) -> pd.DataFrame:
        g = rt.groupby(group_cols, observed=False)
        return g.agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: float((x > 0).mean())),
            avg_ret=("return_pct", "mean"),
            median_ret=("return_pct", "median"),
            avg_holding_days=("holding_days", "mean"),
            avg_mfe=("mfe_pct", "mean"),
            avg_mae=("mae_pct", "mean"),
        ).reset_index()

    out["by_holding_bucket"] = agg(["holding_bucket"])
    out["by_buy_time_bucket"] = agg(["buy_time_bucket"])
    out["by_sell_time_bucket"] = agg(["sell_time_bucket"])
    out["by_entry_momentum_bucket"] = agg(["entry_momentum_bucket"])
    out["by_entry_position_bucket"] = agg(["entry_position_bucket"])
    out["by_symbol"] = (
        rt.groupby(["symbol", "name"], observed=False)
        .agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: float((x > 0).mean())),
            avg_ret=("return_pct", "mean"),
            total_buy_amount=("gross_buy_amount", "sum"),
            avg_holding_days=("holding_days", "mean"),
            avg_mfe=("mfe_pct", "mean"),
            avg_mae=("mae_pct", "mean"),
        )
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    return out


def make_figures(round_trips: pd.DataFrame, tables: dict[str, pd.DataFrame], out_dir: Path) -> list[str]:
    paths: list[str] = []
    if round_trips.empty:
        return paths
    plt.style.use("seaborn-v0_8-whitegrid")

    rt = round_trips.copy()
    rt["sell_month"] = pd.to_datetime(rt["sell_date"]).dt.to_period("M").astype(str)
    monthly = rt.groupby("sell_month", observed=False)["pnl"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = np.where(monthly["pnl"] >= 0, "#4C78A8", "#E45756")
    ax.bar(monthly["sell_month"], monthly["pnl"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Monthly Realized P&L from Matched Trades")
    ax.set_xlabel("Sell month")
    ax.set_ylabel("P&L")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    p = out_dir / "monthly_realized_pnl.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4))
    clipped = rt["return_pct"].clip(-0.3, 0.3)
    ax.hist(clipped, bins=50, color="#72B7B2", edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_title("Closed Trade Return Distribution, Clipped at +/-30%")
    ax.set_xlabel("Matched lot return")
    ax.set_ylabel("Count")
    fig.tight_layout()
    p = out_dir / "return_distribution.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    table = tables.get("by_holding_bucket")
    if table is not None and not table.empty:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        labels = table["holding_bucket"].astype(str)
        ax1.bar(labels, table["pnl"], color=np.where(table["pnl"] >= 0, "#4C78A8", "#E45756"))
        ax1.axhline(0, color="#333333", linewidth=0.8)
        ax1.set_title("P&L by Holding Period Bucket")
        ax1.set_xlabel("Holding period")
        ax1.set_ylabel("P&L")
        fig.tight_layout()
        p = out_dir / "holding_bucket_pnl.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4))
    sample = rt.dropna(subset=["mfe_pct", "return_pct"])
    ax.scatter(sample["mfe_pct"], sample["return_pct"], s=18, alpha=0.55, color="#F58518")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axline((0, 0), slope=1, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_title("Realized Return vs Maximum Favorable Excursion")
    ax.set_xlabel("MFE during holding")
    ax.set_ylabel("Realized return")
    fig.tight_layout()
    p = out_dir / "realized_vs_mfe.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))
    return paths


def pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{value:.2%}"


def money(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{value:,.2f}"


def write_report(
    out_dir: Path,
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
    round_trips: pd.DataFrame,
    unmatched: pd.DataFrame,
    figure_paths: list[str],
) -> Path:
    best_symbols = tables.get("by_symbol", pd.DataFrame()).head(10)
    worst_symbols = tables.get("by_symbol", pd.DataFrame()).tail(10).sort_values("pnl")
    lines = [
        "# Personal trade habit review",
        "",
        "This report is research-only and is based on matched buy/sell executions from the brokerage export.",
        "",
        "## Dataset",
        f"- Period: {summary['period_start']} to {summary['period_end']}",
        f"- Source rows: {summary['source_rows']}",
        f"- Stock executions: {summary['stock_execution_count']} buys/sells across {summary['unique_symbols_traded']} symbols",
        f"- Matched lots: {summary['matched_round_trips']}; unmatched/open lots: {summary['unmatched_lots']}",
        "",
        "## Core behavior metrics",
        f"- Realized P&L on matched lots: {money(summary['realized_pnl'])}",
        f"- Win rate: {pct(summary['win_rate'])}",
        f"- Profit factor: {summary['profit_factor']:.2f}" if summary["profit_factor"] is not None else "- Profit factor: NA",
        f"- Average return: {pct(summary['avg_return_pct'])}; weighted return: {pct(summary['weighted_return_pct'])}; median return: {pct(summary['median_return_pct'])}",
        f"- Median holding days: {summary['median_holding_days']}; same-day share: {pct(summary['same_day_share'])}; <=3 day share: {pct(summary['le_3_day_share'])}",
        f"- Total explicit fees: {money(summary['fee_total'])}; fee / turnover: {pct(summary['fee_to_stock_turnover'])}",
        "",
        "## Best symbol clusters",
        best_symbols.to_markdown(index=False) if not best_symbols.empty else "NA",
        "",
        "## Worst symbol clusters",
        worst_symbols.to_markdown(index=False) if not worst_symbols.empty else "NA",
        "",
        "## Bucket tables",
    ]
    for name in [
        "by_holding_bucket",
        "by_buy_time_bucket",
        "by_entry_momentum_bucket",
        "by_entry_position_bucket",
    ]:
        table = tables.get(name)
        lines.extend(["", f"### {name}", table.to_markdown(index=False) if table is not None and not table.empty else "NA"])
    lines.extend(["", "## Figure files"])
    lines.extend([f"- {p}" for p in figure_paths])
    lines.extend(
        [
            "",
            "## Notes",
            "- FIFO matching excludes sells where the original buy happened before the exported period.",
            "- MFE/MAE use local daily OHLC data when available; they are diagnostics, not tradable intraday facts.",
            "- The brokerage file is GBK tab-separated text with an .xls extension.",
        ]
    )
    report_path = out_dir / "personal_trade_review.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze personal brokerage trade history.")
    parser.add_argument("--source", default=str(SOURCE_DEFAULT))
    parser.add_argument("--panel", default=str(PANEL_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = load_broker_file(Path(args.source))
    round_trips, unmatched = build_round_trips(trades)
    round_trips = enrich_with_market(round_trips, Path(args.panel))
    summary = summarize(round_trips, trades, unmatched)
    tables = bucket_tables(round_trips)
    figure_paths = make_figures(round_trips, tables, out_dir)

    trades.to_csv(out_dir / "trades_clean.csv", index=False, encoding="utf-8-sig")
    round_trips.to_csv(out_dir / "round_trips.csv", index=False, encoding="utf-8-sig")
    unmatched.to_csv(out_dir / "unmatched_lots.csv", index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = write_report(out_dir, summary, tables, round_trips, unmatched, figure_paths)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
