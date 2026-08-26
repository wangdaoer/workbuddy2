"""诊断：在真实合并面板上对 broker_mined 因子算 IC，与既有因子对照。

只读分析脚本（不写任何产物）。输出各因子的 mean IC / IR / 正值占比，按 mean IC 排序，
并单独标出 brk_ 因子，判断哪些有增量信号（供 wf_incremental_broker 的 select_positive 预筛）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import production_soft_score as ps
import train_next_open_rank_model as tm
import factor_expansion as fe
import wq_alpha_factors as wq
import broker_mined_factors as brk

PANEL = Path("external_data/daily-market-data/data_panel.csv")


def main():
    P = ps.build_panel(PANEL)
    close, open_px, high, low, amount = P["close"], P["open_px"], P["high"], P["low"], P["amount"]
    label = P["label"]
    print(f"面板: {close.shape[0]} 交易日 × {close.shape[1]} 标的 | 标签末日 {label.index[-1].date()}")

    base = tm.build_features(close, open_px, high, low, amount)
    exp = fe.build_features_expanded(close, open_px, high, low, amount)
    wqf = wq.build_wq_alpha_factors(close, open_px, high, low, amount)
    brkf = brk.build_broker_mined_factors(close, open_px, high, low, amount)
    print(f"因子数: base={len(base)} exp={len(exp)} wq={len(wqf)} brk={len(brkf)}")

    merged = {**base, **exp, **wqf, **brkf}
    ic = tm.daily_ic(merged, label)            # date × factor, 每日横截面 spearman
    ic = ic.dropna(how="all")

    rows = []
    for name in ic.columns:
        s = ic[name].dropna()
        if len(s) < 30:
            continue
        rows.append({
            "factor": name,
            "group": ("brk" if name.startswith("brk_") else
                      "a158" if name.startswith("a158_") else
                      "wq" if name.startswith("wq_") else
                      "exp" if name in exp else "base"),
            "mean_ic": s.mean(),
            "ir": s.mean() / s.std() if s.std() > 0 else np.nan,
            "pct_pos": (s > 0).mean(),
            "n": len(s),
        })
    tbl = pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)

    def grp(name: str) -> str:
        if name.startswith("brk_"):
            return "brk"
        if name.startswith("wq_"):
            return "wq"
        if name.startswith("exp"):
            return "exp"
        return "base"
    tbl["group"] = tbl["factor"].map(grp)

    print("\n=== 全因子 IC 排序（top/bottom 各 8）===")
    with pd.option_context("display.max_rows", 20, "display.width", 140):
        print(tbl.to_string(index=False,
              formatters={"mean_ic": "{:+.4f}".format, "ir": "{:+.3f}".format,
                          "pct_pos": "{:.2%}".format}))

    print("\n=== 仅 brk_ 因子（按 mean IC 降序）===")
    brk_tbl = tbl[tbl.group == "brk"].sort_values("mean_ic", ascending=False)
    print(brk_tbl.to_string(index=False,
          formatters={"mean_ic": "{:+.4f}".format, "ir": "{:+.3f}".format,
                      "pct_pos": "{:.2%}".format}))
    n_pos = (brk_tbl["mean_ic"] > 0).sum()
    print(f"\nbrk 因子中 mean IC > 0 的个数: {n_pos}/{len(brk_tbl)}")
    print(f"brk 因子平均 |mean IC|: {brk_tbl['mean_ic'].abs().mean():.4f}")
    print(f"brk 因子平均 |IR|: {brk_tbl['ir'].abs().mean():.3f}" if brk_tbl['ir'].notna().any() else "")

    # 与既有族重叠度提示：brk 因子在 select_positive(IC>0) 下有望入选者列出
    sel = brk_tbl[brk_tbl["mean_ic"] > 0]["factor"].tolist()
    print(f"\n预计经 select_positive(IC>0) 入选的 brk 因子（mean IC>0）：{sel}")


if __name__ == "__main__":
    main()
