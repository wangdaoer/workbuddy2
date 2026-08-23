"""Adapter: 用户每日自选股名单 -> apply_personal_trade_overlay 的 symbol-history 契约。

背景（2026-08-05）：
  apply_personal_trade_overlay 的个人行为层本来吃"券商成交记录"(personal_trades.xls)，
  经 analyze_personal_trades 提炼成每票 trades/pnl/win_rate。用户没给成交记录，
  只有每日 ths_money_flow 自选名单。本适配器把名单映射为 symbol-history：
    trades  = 该票出现在每日名单的天数（关注度）
    pnl/win_rate/avg_* = NaN（无成交数据 -> 交易习惯层诚实惰性，不编造）
  输出契约列与 apply_overlay 的 rename 完全一致（symbol, trades, pnl, win_rate,
  avg_ret, avg_holding_days, avg_mfe, avg_mae）。

另产出 names JSON（代码->名称），供候选 CSV 的 stock_name 列做 ST 过滤。

用法：
    python watchlist_symbol_history.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from watchlist_leak_audit import load_daily_files  # noqa: E402

OUT = Path("outputs/watchlist_audit")


def main() -> None:
    files, per_day, names = load_daily_files()
    print(f"[adapter] 名单 {len(per_day)} 天, 涉及 {len(names)} 只")

    all_codes = sorted({code for s in per_day.values() for code in s})
    rows = []
    for code in all_codes:
        trades = sum(1 for s in per_day.values() if code in s)
        rows.append(
            {
                "symbol": code,
                "trades": trades,
                "pnl": None,          # 无券商成交记录 -> 交易习惯层惰性
                "win_rate": None,
                "avg_ret": None,
                "avg_holding_days": None,
                "avg_mfe": None,
                "avg_mae": None,
            }
        )
    history = pd.DataFrame(rows).sort_values(["trades", "symbol"], ascending=[False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUT / "watchlist_symbol_history.csv", index=False, encoding="utf-8")
    (OUT / "watchlist_names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # load_name_map 认 CSV（CODE_COLUMNS/NAME_COLUMNS：symbol + 名称）
    name_frame = pd.DataFrame(
        {"symbol": sorted(names.keys()), "名称": [names[k] for k in sorted(names.keys())]}
    )
    name_frame.to_csv(OUT / "watchlist_names.csv", index=False, encoding="utf-8")
    print(f"[adapter] -> {OUT / 'watchlist_symbol_history.csv'} "
          f"({len(history)} 只; 关注度 top5: "
          + ", ".join(history.head(5)["symbol"].tolist()) + ")")


if __name__ == "__main__":
    main()
