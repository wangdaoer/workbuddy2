"""生成"关注度≥N 天"的聚焦 watchlist PIT 掩码（3b 实验）。

背景：用户名单从 06-22 的 185 只膨胀到 08-05 的 283 只/日，很多票只看过 1-2 天。
本脚本把 watchlist_pit_mask.parquet 里"累计关注天数 < N"的票整列置 False，
产出聚焦掩码到 outputs/watchlist_audit_focus{N}/watchlist_pit_mask.parquet，
供 production_soft_score --universe watchlist --watchlist-mask-dir ... 消费。

用法：
    python build_focused_mask.py [--min-days 5] [--out outputs/watchlist_audit_focus5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SRC_MASK = Path("outputs/watchlist_audit/watchlist_pit_mask.parquet")
HISTORY = Path("outputs/watchlist_audit/watchlist_symbol_history.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-days", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(f"outputs/watchlist_audit_focus{args.min_days}")

    mask = pd.read_parquet(SRC_MASK)
    mask.index = pd.to_datetime(mask.index)
    hist = pd.read_csv(HISTORY, dtype={"symbol": str})
    watch_days = dict(zip(hist["symbol"].astype(str).str.zfill(6), hist["trades"], strict=False))

    focused = mask.copy()
    kept, dropped = [], []
    for col in mask.columns:
        code = str(col)
        days = watch_days.get(code, 0)
        if days < args.min_days:
            focused[col] = False
            dropped.append(code)
        else:
            kept.append(code)

    out.mkdir(parents=True, exist_ok=True)
    focused.to_parquet(out / "watchlist_pit_mask.parquet")
    total_days = int(focused.sum(axis=1).loc[mask.index.max()])
    print(f"[focus{args.min_days}] 末日 {mask.index.max().date()} 合格 {total_days} 只 "
          f"(保留 {len(kept)} 只关注≥{args.min_days}天, 剔除 {len(dropped)} 只) -> {out / 'watchlist_pit_mask.parquet'}")


if __name__ == "__main__":
    main()
