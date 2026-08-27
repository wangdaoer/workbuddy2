"""walk-forward 增量验证（决策相关版）：base+exp+broker(50) vs base+exp+broker+alpha158(~207)。

只读诊断：不改动任何生产文件。

为什么需要这一版：
- 当前生产默认栈 = base+exp+broker（WQ 是 opt-in，不在默认）；
  之前 wf_incremental_alpha158.py 测的是 base+exp+wq 栈，对"是否并入生产默认"针对性有偏差。
- 本脚本直接在生产默认栈上隔离 alpha158 的边际贡献，Δ 即为"并入默认"的决策指标。

⚠️ 关键纪律（沿用 WQ 教训）：feature_selection 必须用生产同款 select_positive(IC>0)，
   绝不用 feature_selection=None（后者强制全部因子入选，会虚高假阳性）。

产出：
- outputs/wf_incremental_alpha158_brokerstack/report.json
- outputs/wf_incremental_alpha158_brokerstack/equity_A.csv / equity_B.csv
- outputs/wf_incremental_alpha158_brokerstack/weights_B.csv
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
import alpha158_factors as a158
from production_soft_score import build_panel
import pit_universe as pit
from p10c_ensemble import TRAIN_DAYS, MTH, RETRAIN, IMPACT_REF, select_positive

PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = ROOT / "outputs" / "wf_incremental_alpha158_brokerstack"
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
        amount=amount, feature_selection=select_positive, max_daily_amount_participation=BOOK_ADV_P,
        liquid_mask=backtest_mask,
    )
    m = tm.calculate_walk_forward_metrics(eq, 1e8)
    return eq, wdf, m


# ---------- A: base + expansion + broker（= 当前生产默认栈） ----------
log("build features A (base+exp+broker) ...")
feA = fe.build_features_expanded_broker(close, open_px, high, low, amount)
log(f"A features={len(feA)}  t={time.time() - t0:.1f}s")
log("WALK-FORWARD A (select_positive) ...")
eqA, wA, mA = run_wf(feA, "A")
log(f"A done: sharpe={mA.get('sharpe_like'):.3f} ret={mA.get('total_return'):.4f} "
    f"dd={mA.get('max_drawdown'):.4f}  t={time.time() - t0:.1f}s")

# ---------- B: base + expansion + broker + alpha158（并入默认候选） ----------
log("build features B (base+exp+broker+alpha158) ...")
a158_feats = a158.build_alpha158_factors(close, open_px, high, low, amount)
feB = {**feA, **a158_feats}
log(f"B features={len(feB)}  t={time.time() - t0:.1f}s")
log("WALK-FORWARD B (select_positive) ...")
eqB, wB, mB = run_wf(feB, "B")
log(f"B done: sharpe={mB.get('sharpe_like'):.3f} ret={mB.get('total_return'):.4f} "
    f"dd={mB.get('max_drawdown'):.4f}  t={time.time() - t0:.1f}s")

# ---------- 增量对比（仅取实际入选列，规避 select_positive 排除） ----------
a158_names = [n for n in feB if n.startswith("a158_") and n in wB.columns]
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

non_a158 = [n for n in feB if not n.startswith("a158_") and n in wB.columns]
non_a158_abs = wB[non_a158].abs().mean().sum()
a158_abs_total = float(pd.Series({n: a158_stats[n]["mean_abs_weight"] for n in a158_names}).sum())
a158_share = float(a158_abs_total / (non_a158_abs + a158_abs_total)) if (non_a158_abs + a158_abs_total) > 0 else 0.0

report = {
    "config": {
        "universe": "pit", "train_days": TRAIN_DAYS, "retrain": RETRAIN,
        "rebalance": REBALANCE, "top_n": BOOK_TOP_N, "max_w": BOOK_MAX_W,
        "adv_p": BOOK_ADV_P, "commission_bps": COMMISSION_BPS,
        "impact_bps": IMPACT_BPS, "stamp_bps": STAMP_TAX_BPS,
        "feature_selection": "select_positive(IC>0)  # 生产同款，非 None",
        "stack": "base+exp+broker (生产默认栈)",
        "n_A": len(feA), "n_B": len(feB),
    },
    "A_base_exp_broker": {k: mA.get(k) for k in (
        "sharpe_like", "total_return", "annualized_return", "max_drawdown",
        "avg_turnover", "avg_positions_count", "trade_days")},
    "B_base_exp_broker_a158": {k: mB.get(k) for k in (
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
print("WALK-FORWARD 增量验证（alpha158 on 生产默认栈 broker，select_positive）")
print("=" * 64)
print(f"A base+exp+broker       (n={len(feA)}): sharpe={mA['sharpe_like']:.3f}  "
      f"ret={mA['total_return']:.3f}  dd={mA['max_drawdown']:.3f}")
print(f"B +a158                 (n={len(feB)}): sharpe={mB['sharpe_like']:.3f}  "
      f"ret={mB['total_return']:.3f}  dd={mB['max_drawdown']:.3f}")
print(f"DELTA                         : sharpe={report['delta']['sharpe_like']:+.3f}  "
      f"ret={report['delta']['total_return']:+.3f}  dd={report['delta']['max_drawdown']:+.3f}")
print(f"\nalpha158 因子在 B 投资组合权重中的平均占比（|weight|）: {a158_share * 100:.1f}%")
print("-" * 64)
print("alpha158 因子权重/IC（按 |mean_weight| 降序，前 20）")
print(f"{'factor':30s} {'wmean':>9s} {'pctPos':>7s} {'IC':>8s}")
for n in sorted(a158_names, key=lambda x: -abs(a158_stats[x]["mean_weight"]))[:20]:
    s = a158_stats[n]
    print(f"{n:30s} {s['mean_weight']:+.4f} {s['pct_positive']:6.1f}% {a158_ic[n]:+.4f}")
print("-" * 64)
print(f"total time {time.time() - t0:.1f}s ; outputs -> {OUT}")
