"""Alpha158 因子 IC 诊断（只读，不改生产文件）。

对 alpha158_factors.build_alpha158_factors 的全部因子跑 daily_ic，
输出按 mean IC 排序的全表 + 仅 alpha158 因子表，标出正 IC 候选。

运行时机：等 wq walk-forward（wf_incremental_wq.py）跑完释放 CPU 后再跑，
避免与重算力任务争用。

产出：outputs/diagnose_alpha158_ic/report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_next_open_rank_model as tm
import alpha158_factors as a158
from production_soft_score import build_panel

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = ROOT / "outputs" / "diagnose_alpha158_ic"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    P = build_panel(PANEL)
    close, open_px, high, low, amount = (
        P["close"], P["open_px"], P["high"], P["low"], P["amount"],
    )
    label = P["label"]

    print(f"build alpha158 factors ... panel={close.shape}")
    fac = a158.build_alpha158_factors(close, open_px, high, low, amount)
    print(f"alpha158 factors={len(fac)}")

    ic = tm.daily_ic(fac, label)
    mean_ic = ic.mean().sort_values(ascending=False)
    ir = (ic.mean() / (ic.std() + 1e-12))
    pct_pos = (ic > 0).mean() * 100

    rows = []
    for name in mean_ic.index:
        rows.append({
            "factor": name,
            "mean_ic": float(mean_ic[name]),
            "ir": float(ir[name]),
            "pct_positive": float(pct_pos[name]),
        })
    pos = [r for r in rows if r["mean_ic"] > 0]
    neg = [r for r in rows if r["mean_ic"] <= 0]

    report = {
        "n_factors": len(fac),
        "n_positive_ic": len(pos),
        "positive_ic_factors": pos,
        "all_sorted": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("=" * 72)
    print(f"Alpha158 IC 诊断：{len(fac)} 因子，{len(pos)} 个正 IC")
    print("=" * 72)
    print("正 IC 因子（按 mean IC 降序）：")
    print(f"{'factor':22s} {'meanIC':>9s} {'IR':>8s} {'%pos':>7s}")
    for r in pos[:30]:
        print(f"{r['factor']:22s} {r['mean_ic']:+.4f} {r['ir']:+.3f} {r['pct_positive']:6.1f}%")
    if len(pos) > 30:
        print(f"... 其余 {len(pos) - 30} 个正 IC 因子见 report.json")
    print("-" * 72)
    print("最差 10 个（负 IC）：")
    for r in neg[:10]:
        print(f"{r['factor']:22s} {r['mean_ic']:+.4f} {r['ir']:+.3f} {r['pct_positive']:6.1f}%")
    print(f"\n-> {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
