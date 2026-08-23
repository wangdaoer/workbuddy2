"""P9-4 跨两岸（A 股 + 港股通 + 期货）容量与分散重测。

输入：
  - book  : P8b 权益曲线（含 gross_return, date=YYYY-MM-DD）
  - sleeve: P9-2.5 futures_sleeve_overlay.csv（sleeve_ret, trade_date=YYYYMMDD）
  - hk    : P9-4 hk_connect_index.csv（ret, date=YYYY-MM-DD，流动性加权港股通指数）
  - p9cap : P9-3 p9_capacity_retest.json（A 股簿 + 期货 sleeve 容量上限）
  - hkliq : P9-4 hk_connect_liquidity.json（港股通 ADV）

产出：
  - cross_strait.json：相关性、融合组合、跨市场合计容量
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

FX_HKD_TO_CNY = 0.92


def ymd(s):
    return str(s).replace("-", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--sleeve", required=True)
    ap.add_argument("--hk-index", required=True)
    ap.add_argument("--p9-cap", required=True)
    ap.add_argument("--hk-liq", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eq = pd.read_csv(args.book); eq["date"] = eq["date"].astype(str).str.replace("-", "", regex=False)
    book = eq[["date", "gross_return"]].rename(columns={"date": "trade_date", "gross_return": "book_ret"})

    sl = pd.read_csv(args.sleeve); sl["trade_date"] = sl["trade_date"].astype(str)
    sl = sl[["trade_date", "sleeve_ret"]].rename(columns={"sleeve_ret": "sleeve_ret"})

    hk = pd.read_csv(args.hk_index); hk["trade_date"] = hk["date"].astype(str).str.replace("-", "", regex=False)
    hk = hk[["trade_date", "ret"]].rename(columns={"ret": "hk_ret"})

    m = book.merge(sl, on="trade_date").merge(hk, on="trade_date").dropna().sort_values("trade_date")
    corr = m[["book_ret", "sleeve_ret", "hk_ret"]].corr()

    # 融合组合扫描（三类收益流等权/定制分配）
    allocs = {
        "book_only":   {"book": 1.0, "sleeve": 0.0, "hk": 0.0},
        "book+sleeve_8020": {"book": 0.8, "sleeve": 0.2, "hk": 0.0},
        "three_way_602020": {"book": 0.6, "sleeve": 0.2, "hk": 0.2},
        "three_way_505025": {"book": 0.5, "sleeve": 0.25, "hk": 0.25},
        "hk_only":     {"book": 0.0, "sleeve": 0.0, "hk": 1.0},
    }
    comb = {}
    for name, w in allocs.items():
        r = w["book"] * m["book_ret"] + w["sleeve"] * m["sleeve_ret"] + w["hk"] * m["hk_ret"]
        cum = (1 + r.fillna(0)).cumprod()
        vol = r.std() * np.sqrt(252)
        ann = cum.iloc[-1] ** (252 / len(r)) - 1
        sharpe = r.mean() / r.std() * np.sqrt(252)
        peak = cum.cummax(); dd = (cum / peak - 1).min()
        comb[name] = {"weights": w, "ann_return": round(float(ann), 4),
                      "ann_vol": round(float(vol), 4), "sharpe": round(float(sharpe), 4),
                      "max_drawdown": round(float(dd), 4)}

    # —— 跨市场合计容量 ——
    p9 = json.loads(Path(args.p9_cap).read_text())
    hkliq = json.loads(Path(args.hk_liq).read_text())
    ashare_cap = p9.get("ashare_capacity_ceiling_aum")
    futures_cap = p9.get("derivative_sleeve_capacity_ceiling_aum")
    # 港股通容量（流动性池估算，与 P8b 同口径：1% ADV 参与、年化换手 ~1.2）
    total_adv_cny = hkliq["total_adv_cny_per_day"]           # CNY/日
    annual_adv_cny = total_adv_cny * 252
    participation = 0.01
    annual_turnover = 1.2
    hk_cap = annual_adv_cny * participation / annual_turnover

    report = {
        "overlap_days": int(len(m)),
        "date_start": m["trade_date"].iloc[0],
        "date_end": m["trade_date"].iloc[-1],
        "correlation": {
            "book_sleeve": round(float(corr.loc["book_ret", "sleeve_ret"]), 4),
            "book_hk": round(float(corr.loc["book_ret", "hk_ret"]), 4),
            "sleeve_hk": round(float(corr.loc["sleeve_ret", "hk_ret"]), 4),
        },
        "combined_portfolios": comb,
        "capacity_aum_cny": {
            "ashare_book_p8b": ashare_cap,
            "derivative_futures_sleeve": futures_cap,
            "hk_connect_estimate": round(float(hk_cap), 0),
            "combined_cross_market": (round(float(ashare_cap or 0)) + round(float(futures_cap or 0)) + round(float(hk_cap))),
            "assumptions": {
                "hk_participation": participation, "hk_annual_turnover": annual_turnover,
                "hk_total_adv_cny_per_day": round(float(total_adv_cny), 0),
                "fx_hkd_to_cny": FX_HKD_TO_CNY,
            },
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
