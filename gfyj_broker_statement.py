"""广发易淘金 PC 版对账单 -> 个人行为叠加层 symbol-history。

输入：C:/Users/86176/Desktop/广发易淘金PC版-普通对账单结果查询.csv（UTF-8-sig，字段带引号+制表符）
  列：业务日期,发生时间,流水序号,资金账号,证券代码,证券名称,业务标志名称,成交数量(买+/卖-),
      成交价格,净佣金,印花税,过户费,证管费,经手费,其他费,清算金额(买-/卖+),货币名称,委托编号,应计利息

处理：
  1) 归一化 trades（cash_flow=清算金额带符号、gross_amount=成交金额price*qty、fees=费用合计、
     side=证券买入/卖出 → buy/sell、market 按 6 开头=SH 否则 SZ）。
  2) 复用 analyze_personal_trades.build_round_trips（FIFO 配对）→ 已实现 round trips。
  3) 每票聚合：trades=已实现配对次数, pnl=已实现盈亏, win_rate=盈利配对占比,
     avg_ret, avg_holding_days；avg_mfe/avg_mae 无逐日数据置 NaN。
  4) 与名单关注度合并：有成交的票用成交统计；仅关注的票 trades=关注天数、pnl=NaN（诚实惰性）。

用法：
    python gfyj_broker_statement.py [--csv 路径] [--output outputs/watchlist_audit/combined_symbol_history.csv]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_personal_trades import build_round_trips, clean_symbol, time_bucket  # noqa: E402

DEFAULT_CSV = Path(r"C:\Users\86176\Desktop\广发易淘金PC版-普通对账单结果查询.csv")
DEFAULT_OUT = Path("outputs/watchlist_audit/combined_symbol_history.csv")
WATCHLIST_HISTORY = Path("outputs/watchlist_audit/watchlist_symbol_history.csv")


def parse_gfyj(path: Path) -> pd.DataFrame:
    records = []
    with io.open(path, encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = [h.strip().strip("\t") for h in next(reader)]
        for rec in reader:
            d = {}
            for i, col in enumerate(header):
                d[col] = (rec[i].strip().strip("\t") if i < len(rec) else "")
            records.append(d)
    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit(f"对账单为空: {path}")

    df["trade_date"] = pd.to_datetime(df["业务日期"], errors="coerce")
    df["trade_time"] = df["发生时间"].fillna("00:00:00")
    df["trade_dt"] = pd.to_datetime(
        df["trade_date"].dt.strftime("%Y-%m-%d") + " " + df["trade_time"], errors="coerce"
    )
    df["symbol"] = df["证券代码"].map(clean_symbol)
    df["name"] = df["证券名称"].fillna("")
    op = df["业务标志名称"].fillna("")
    df["side"] = np.select(
        [op.str.contains("买入", regex=False), op.str.contains("卖出", regex=False)],
        ["buy", "sell"],
        default="other",
    )
    for col in ["成交数量", "成交价格", "清算金额", "净佣金", "印花税", "过户费", "证管费", "经手费", "其他费"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["quantity"] = df["成交数量"]
    df["price"] = df["成交价格"]
    df["cash_flow"] = df["清算金额"]  # 买负/卖正，带费用
    df["gross_amount"] = (df["price"] * df["quantity"]).abs()  # 成交金额（不含费）
    fee_cols = ["净佣金", "印花税", "过户费", "证管费", "经手费", "其他费"]
    df["fees"] = df[fee_cols].fillna(0.0).sum(axis=1)
    df["market"] = np.where(df["symbol"].str.startswith("6"), "SH", "SZ")
    df["contract_id"] = df["流水序号"].fillna("").astype(str)
    df["time_bucket"] = df["trade_dt"].map(time_bucket)
    return df.sort_values(["trade_dt", "contract_id"], kind="mergesort").reset_index(drop=True)


def broker_symbol_stats(trades: pd.DataFrame) -> pd.DataFrame:
    rt, unmatched = build_round_trips(trades)
    print(f"[broker] 成交记录={len(trades)} 已实现配对={len(rt)} 未配对={len(unmatched)} "
          f"未配对原因: {unmatched['reason'].value_counts().to_dict() if not unmatched.empty else {}}")
    if rt.empty:
        stats = pd.DataFrame(
            columns=["symbol", "trades", "pnl", "win_rate", "avg_ret", "avg_holding_days", "avg_mfe", "avg_mae"]
        )
    else:
        g = rt.groupby("symbol").agg(
            trades=("pnl", "size"),
            pnl=("pnl", "sum"),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
            avg_ret=("return_pct", "mean"),
            avg_holding_days=("holding_days", "mean"),
        ).reset_index()
        g["avg_mfe"] = np.nan
        g["avg_mae"] = np.nan
        stats = g[["symbol", "trades", "pnl", "win_rate", "avg_ret", "avg_holding_days", "avg_mfe", "avg_mae"]]
        stats["win_rate"] = stats["win_rate"].round(4)
        stats["pnl"] = stats["pnl"].round(2)
        stats["avg_ret"] = stats["avg_ret"].round(4)
        stats["avg_holding_days"] = stats["avg_holding_days"].round(1)
        stats["symbol"] = stats["symbol"].astype(str).str.zfill(6)
        stats = stats.sort_values(["trades", "pnl"], ascending=[False, False]).reset_index(drop=True)
    return stats


def merge_with_watchlist(broker: pd.DataFrame) -> pd.DataFrame:
    if WATCHLIST_HISTORY.exists():
        watch = pd.read_csv(WATCHLIST_HISTORY, dtype={"symbol": str})[["symbol", "trades"]]
        watch = watch.rename(columns={"trades": "watch_days"})
    else:
        watch = pd.DataFrame(columns=["symbol", "watch_days"])
    traded = set(broker["symbol"])
    rows = []
    for _, r in broker.iterrows():
        rows.append({**r.to_dict(), "watch_days": int(watch.loc[watch["symbol"] == r["symbol"], "watch_days"].sum())})
    for _, r in watch.iterrows():
        if r["symbol"] not in traded:
            rows.append(
                {
                    "symbol": r["symbol"],
                    "trades": int(r["watch_days"]),  # 仅关注：用关注天数当 trades（触发不了习惯层，pnl=NaN）
                    "pnl": np.nan,
                    "win_rate": np.nan,
                    "avg_ret": np.nan,
                    "avg_holding_days": np.nan,
                    "avg_mfe": np.nan,
                    "avg_mae": np.nan,
                    "watch_days": int(r["watch_days"]),
                }
            )
    out = pd.DataFrame(rows)
    if "watch_days" in out:
        out = out.drop(columns=["watch_days"])
    out = out.sort_values(["trades", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="广发易淘金对账单 → 个人行为叠加 symbol-history")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    trades = parse_gfyj(Path(args.csv))
    print(f"[broker] 日期范围 {trades['trade_date'].min().date()}..{trades['trade_date'].max().date()} "
          f"涉及 {trades['symbol'].nunique()} 只, 买卖次数 {trades['side'].value_counts().to_dict()}")

    stats = broker_symbol_stats(trades)
    combined = merge_with_watchlist(stats)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False, encoding="utf-8")
    print(f"[broker] -> {out} ({len(combined)} 只; 有成交统计 {len(stats)} 只, 仅关注 {len(combined)-len(stats)} 只)")
    if not stats.empty:
        print("[broker] 已实现盈亏 top3 盈利: ")
        top = stats[stats["pnl"] > 0].sort_values("pnl", ascending=False).head(3)
        for _, r in top.iterrows():
            print(f"    {r['symbol']} pnl={r['pnl']} trades={r['trades']} win_rate={r['win_rate']}")
        loss = stats[stats["pnl"] <= 0].sort_values("pnl").head(3)
        if not loss.empty:
            print("[broker] 亏损 top3: ")
            for _, r in loss.iterrows():
                print(f"    {r['symbol']} pnl={r['pnl']} trades={r['trades']}")
        print(f"[broker] 总已实现盈亏={stats['pnl'].sum():.2f}")


if __name__ == "__main__":
    main()
