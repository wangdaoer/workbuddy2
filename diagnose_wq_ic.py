"""诊断：在真实合并面板上对 wq_alpha 因子算 IC，与既有因子对照。

只读分析脚本（不写任何产物）。输出各因子的 mean IC / IR / 正值占比，按 mean IC 排序，
并单独标出 wq_ 因子，判断哪些有增量信号。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import production_soft_score as ps
import train_next_open_rank_model as tm
import factor_expansion as fe
import wq_alpha_factors as wq

PANEL = Path("external_data/daily-market-data/data_panel.csv")


def main():
    P = ps.build_panel(PANEL)
    close, open_px, high, low, amount = P["close"], P["open_px"], P["high"], P["low"], P["amount"]
    label = P["label"]
    print(f"面板: {close.shape[0]} 交易日 × {close.shape[1]} 标的 | 标签末日 {label.index[-1].date()}")

    base = tm.build_features(close, open_px, high, low, amount)
    exp = fe.build_features_expanded(close, open_px, high, low, amount)
    wqf = wq.build_wq_alpha_factors(close, open_px, high, low, amount)
    print(f"因子数: base={len(base)} exp={len(exp)} wq={len(wqf)}")

    merged = {**base, **exp, **wqf}
    ic = tm.daily_ic(merged, label)            # date × factor, 每日横截面 spearman
    ic = ic.dropna(how="all")

    rows = []
    for name in ic.columns:
        s = ic[name].dropna()
        if len(s) < 30:
            continue
        rows.append({
            "factor": name,
            "group": "wq" if name.startswith("wq_") else ("exp" if name in exp else "base"),
            "mean_ic": s.mean(),
            "ir": s.mean() / s.std() if s.std() > 0 else np.nan,
            "pct_pos": (s > 0).mean(),
            "n": len(s),
        })
    tbl = pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)

    print("\n=== 全因子 IC 排序（top/bottom 各 8）===")
    with pd.option_context("display.max_rows", 20, "display.width", 120):
        print(tbl.to_string(index=False,
              formatters={"mean_ic": "{:+.4f}".format, "ir": "{:+.3f}".format,
                          "pct_pos": "{:.2%}".format}))

    print("\n=== 仅 wq_ 因子 ===")
    wq_tbl = tbl[tbl.group == "wq"].sort_values("mean_ic", ascending=False)
    print(wq_tbl.to_string(index=False,
          formatters={"mean_ic": "{:+.4f}".format, "ir": "{:+.3f}".format,
                      "pct_pos": "{:.2%}".format}))

    n_wq_pos = (wq_tbl["mean_ic"] > 0).sum()
    print(f"\nwq 因子中 mean IC > 0 的个数: {n_wq_pos}/{len(wq_tbl)}")
    print(f"wq 因子平均 |mean IC|: {wq_tbl['mean_ic'].abs().mean():.4f}")


if __name__ == "__main__":
    main()
