"""walk-forward 增量验证：base+exp+wq(53) vs base+exp+wq+alpha158(210)。

只读诊断：不改动任何生产文件。与 wf_incremental_wq.py 衔接——
wq 那层已并入 base+exp，本脚本在"已含 wq"的宇宙上隔离 alpha158 的增量贡献。

两路使用同一套生产回测配置（pit 诚实宇宙 / sqrt 冲击 / 万三佣金 / 印花税 / top_n=40 / rebalance=5），
仅因子集不同，故 equity delta 纯归因 alpha158 因子吸收。

产出：
- outputs/wf_incremental_alpha158/report.json
- outputs/wf_incremental_alpha158/equity_A.csv / equity_B.csv
- outputs/wf_incremental_alpha158/weights_B.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import train_next_open_rank_model as tm
import factor_expansion as fe
from production_soft_score import build_panel
import pit_universe as pit
from p10c_ensemble import TRAIN_DAYS, MTH, RETRAIN, IMPACT_REF

PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = ROOT / "outputs" / "wf_incremental_alpha158"
OUT.mkdir(parents=True, exist_ok=True)

COMMISSION_BPS = 3.0
IMPACT_BPS = 0.7
STAMP_TAX_BPS = 5.0
BOOK_TOP_N = 40
BOOK_MAX_W = 0.04
BOOK_ADV_P = 0.02
REBALANCE = 5

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("build_panel ...")
P = build_panel(PANEL)
close = P["close"]
open_px = P["open_px"]
high = P["high"]
low = P["low"]
amount = P["amount"]
label = P["label"]
mkt_exp = P["market_exposure"]
log(f"panel={close.shape}  t={time.time() - t0:.1f}s")

backtest_mask = pit.pit_eligible(close, amount, None, thr=1e8, lookback=252, min_days=60)


def run_wf(features: dict, tag: str):
    ic = tm.daily_ic(features, label)
    eq, wdf, _ = tm.run_walk_forward(
        close=close, open_px=open_px, features=features, label=label, ic=ic,
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=BOOK_TOP_N,
        rebalance_frequency=REBALANCE, max_position_weight=BOOK_MAX_W, leverage=1.0,
        commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS, stamp_tax_bps=STAMP_TAX_BPS,
        impact_model="sqrt", impact_ref_participation=IMPACT_REF,
        max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=mkt_exp,
        initial_capital=1e8, max_training_horizon=MTH, feature_directions=None,
        amount=amount, feature_selection=None, max_daily_amount_participation=BOOK_ADV_P,
        liquid_mask=backtest_mask,
    )
    m = tm.calculate_walk_forward_metrics(eq, 1e8)
    return eq, wdf, m


# ---------- A: base + expansion + wq ----------
log("build features A (base+exp+wq) ...")
feA = fe.build_features_expanded_wq(close, open_px, high, low, amount)
log(f"A features={len(feA)}  t={time.time() - t0:.1f}s")
log("WALK-FORWARD A ...")
eqA, wA, mA = run_wf(feA, "A")
log(f"A done: sharpe={mA.get('sharpe_like'):.3f} ret={mA.get('total_return'):.4f} "
    f"dd={mA.get('max_drawdown'):.4f}  t={time.time() - t0:.1f}s")

# ---------- B: base + expansion + wq + alpha158 ----------
log("build features B (base+exp+wq+alpha158) ...")
feB = fe.build_features_expanded_wq158(close, open_px, high, low, amount)
log(f"B features={len(feB)}  t={time.time() - t0:.1f}s")
log("WALK-FORWARD B ...")
eqB, wB, mB = run_wf(feB, "B")
log(f"B done: sharpe={mB.get('sharpe_like'):.3f} ret={mB.get('total_return'):.4f} "
    f"dd={mB.get('max_drawdown'):.4f}  t={time.time() - t0:.1f}s")

# ---------- 增量对比 ----------
a158_names = [n for n in feB if n.startswith("a158_")]
a158_w = wB[a158_names]
a158_stats = {
    n: {
        "mean_weight": float(a158_w[n].mean()),
        "pct_positive": float((a158_w[n] > 0).mean() * 100),
        "mean_abs_weight": float(a158_w[n].abs().mean()),
    }
    for n in a158_names
}
icB = tm.daily_ic(feB, label)
a158_ic = {n: float(icB[n].mean()) for n in a158_names}

non_a158 = [n for n in feB if not n.startswith("a158_")]
non_a158_abs = wB[non_a158].abs().mean().sum()
a158_abs_total = float(pd.Series({n: a158_stats[n]["mean_abs_weight"] for n in a158_names}).sum())
a158_share = float(a158_abs_total / (non_a158_abs + a158_abs_total)) if (non_a158_abs + a158_abs_total) > 0 else 0.0

# 与 wq 层结论联动：A 路（已含 wq）的指标即 wq 层 B 路
report = {
    "config": {
        "universe": "pit", "train_days": TRAIN_DAYS, "retrain": RETRAIN,
        "rebalance": REBALANCE, "top_n": BOOK_TOP_N, "max_w": BOOK_MAX_W,
        "adv_p": BOOK_ADV_P, "commission_bps": COMMISSION_BPS,
        "impact_bps": IMPACT_BPS, "stamp_bps": STAMP_TAX_BPS,
        "feature_selection": None, "n_A": len(feA), "n_B": len(feB),
    },
    "A_base_exp_wq": {k: mA.get(k) for k in (
        "sharpe_like", "total_return", "annualized_return", "max_drawdown",
        "avg_turnover", "avg_positions_count", "trade_days")},
    "B_base_exp_wq_a158": {k: mB.get(k) for k in (
        "sharpe_like", "total_return", "annualized_return", "max_drawdown",
        "avg_turnover", "avg_positions_count", "trade_days")},
    "delta": {k: (mB.get(k) - mA.get(k)) for k in (
        "sharpe_like", "total_return", "annualized_return", "max_drawdown",
        "avg_turnover", "avg_positions_count")},
    "a158_portfolio_weight_share": a158_share,
    "a158_factor_weight_stats": a158_stats,
    "a158_factor_mean_ic": a158_ic,
}

(OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
eqA.to_csv(OUT / "equity_A.csv", index=False)
eqB.to_csv(OUT / "equity_B.csv", index=False)
wB.to_csv(OUT / "weights_B.csv", index=False)

print("\n" + "=" * 64)
print("WALK-FORWARD 增量验证（alpha158 因子吸收）")
print("=" * 64)
print(f"A base+exp+wq (n={len(feA)}): sharpe={mA['sharpe_like']:.3f}  "
      f"ret={mA['total_return']:.3f}  dd={mA['max_drawdown']:.3f}")
print(f"B +a158      (n={len(feB)}): sharpe={mB['sharpe_like']:.3f}  "
      f"ret={mB['total_return']:.3f}  dd={mB['max_drawdown']:.3f}")
print(f"DELTA               : sharpe={report['delta']['sharpe_like']:+.3f}  "
      f"ret={report['delta']['total_return']:+.3f}  dd={report['delta']['max_drawdown']:+.3f}")
print(f"\nalpha158 因子在 B 投资组合权重中的平均占比（|weight|）: {a158_share * 100:.1f}%")
print("-" * 64)
print("alpha158 因子权重/IC（按 |mean_weight| 降序，前 30）")
print(f"{'factor':30s} {'wmean':>9s} {'pctPos':>7s} {'IC':>8s}")
for n in sorted(a158_names, key=lambda x: -abs(a158_stats[x]["mean_weight"]))[:30]:
    s = a158_stats[n]
    print(f"{n:30s} {s['mean_weight']:+.4f} {s['pct_positive']:6.1f}% {a158_ic[n]:+.4f}")
print("-" * 64)
print(f"total time {time.time() - t0:.1f}s ; outputs -> {OUT}")
