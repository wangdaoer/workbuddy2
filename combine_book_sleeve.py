"""P9 收尾：把 A 股权益簿（P8b）与衍生品 sleeve 按不同权重融合，量化跨市场分散价值。

输入：
  - equity_csv : P8b 权益曲线（含 gross_return, date=YYYY-MM-DD）
  - sleeve_csv : P9-2.5 生成的 futures_sleeve_overlay.csv（含 sleeve_ret, trade_date=YYYYMMDD）
输出：
  - combined_book_sleeve.csv : 各分配比例下的融合日收益与权益
  - 终端打印：相关性与各分配下的年化/波动/夏普/最大回撤
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", required=True)
    ap.add_argument("--sleeve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alloc", default="1.0,0.8,0.5",
                    help="sleeve 权重扫描点（book 权重=1-sleeve），逗号分隔")
    args = ap.parse_args()
    sleeve_w = [float(x) for x in args.alloc.split(",")]

    eq = pd.read_csv(args.equity)
    eq["date"] = eq["date"].astype(str).str.replace("-", "", regex=False)
    book = eq[["date", "gross_return"]].rename(columns={"date": "trade_date", "gross_return": "book_ret"})

    sl = pd.read_csv(args.sleeve)
    sl["trade_date"] = sl["trade_date"].astype(str)
    sl = sl[["trade_date", "sleeve_ret"]]

    m = book.merge(sl, on="trade_date", how="inner").dropna()
    m = m.sort_values("trade_date").reset_index(drop=True)
    corr = m["book_ret"].corr(m["sleeve_ret"])

    rows = []
    summary = []
    for ws in sleeve_w:
        wb = 1.0 - ws
        comb = wb * m["book_ret"] + ws * m["sleeve_ret"]
        cum = (1 + comb.fillna(0)).cumprod()
        vol = comb.std() * np.sqrt(252)
        ann = cum.iloc[-1] ** (252 / len(comb)) - 1
        sharpe = comb.mean() / comb.std() * np.sqrt(252)
        peak = cum.cummax(); dd = (cum / peak - 1).min()
        summary.append({
            "sleeve_weight": ws, "book_weight": round(wb, 3),
            "ann_return": round(float(ann), 4), "ann_vol": round(float(vol), 4),
            "sharpe": round(float(sharpe), 4), "max_drawdown": round(float(dd), 4),
        })
        tmp = m[["trade_date"]].copy()
        tmp["sleeve_weight"] = ws
        tmp["combined_ret"] = comb.values
        tmp["combined_equity"] = cum.values
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    rep = {
        "n_overlap_days": int(len(m)),
        "corr_book_sleeve": round(float(corr), 4),
        "allocations": summary,
        "combined_csv": str(args.out),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
