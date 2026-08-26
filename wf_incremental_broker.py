"""walk-forward 增量验证：base+exp(29) vs base+exp+broker。

只读诊断：不改动任何生产文件。
- A: factor_expansion.build_features_expanded            -> 29 因子（生产基线）
- B: factor_expansion.build_features_expanded_broker     -> 29 + brk 因子（券商研报吸收候选）

两路使用同一套生产回测配置（pit 诚实宇宙 / sqrt 冲击 / 万三佣金 / 印花税 / top_n=40 / rebalance=5），
仅因子集不同，故 equity delta 纯归因 brk 因子吸收。

⚠️ 关键纪律（沿用 WQ 教训）：feature_selection 必须用生产同款 select_positive(IC>0)
   自适应机制，绝不用 feature_selection=None（后者强制全部因子入选，会虚高假阳性）。
   brk 中仅 mean IC>0 的因子经 select_positive 进入训练/回测，与真实生产路径一致。

产出：
- outputs/wf_incremental_broker/report.json
- outputs/wf_incremental_broker/equity_A.csv / equity_B.csv
- outputs/wf_incremental_broker/weights_B.csv  （B 中每个因子逐再平衡日权重，用于统计 brk 因子占比）
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
from p10c_ensemble import TRAIN_DAYS, MTH, RETRAIN, IMPACT_REF, select_positive

PANEL = ROOT / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = ROOT / "outputs" / "wf_incremental_broker"
OUT.mkdir(parents=True, exist_ok=True)

# --- 生产回测配置（与 production_soft_score.produce_book_score 回测路径一致）---
COMMISSION_BPS = 3.0
IMPACT_BPS = 0.7
STAMP_TAX_BPS = 5.0
BOOK_TOP_N = 40
BOOK_MAX_W = 0.04
BOOK_ADV_P = 0.02
REBALANCE = 5  # Route C 降频甜区（production_soft_score.REBALANCE）

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

# PIT 诚实宇宙（与生产 universe='pit' 一致）
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


# ---------- A: base + expansion（生产基线） ----------
log("build features A (base+exp) ...")
feA = fe.build_features_expanded(close, open_px, high, low, amount)
log(f"A features={len(feA)}  t={time.time() - t0:.1f}s")
log("WALK-FORWARD A (select_positive) ...")
eqA, wA, mA = run_wf(feA, "A")
log(f"A done: sharpe={mA.get('sharpe_like'):.3f} ret={mA.get('total_return'):.4f} "
    f"dd={mA.get('max_drawdown'):.4f}  t={time.time() - t0:.1f}s")

# ---------- B: base + expansion + broker（券商研报吸收候选） ----------
log("build features B (base+exp+broker) ...")
feB = fe.build_features_expanded_broker(close, open_px, high, low, amount)
log(f"B features={len(feB)}  t={time.time() - t0:.1f}s")
log("WALK-FORWARD B (select_positive) ...")
eqB, wB, mB = run_wf(feB, "B")
log(f"B done: sharpe={mB.get('sharpe_like'):.3f} ret={mB.get('total_return'):.4f} "
    f"dd={mB.get('max_drawdown'):.4f}  t={time.time() - t0:.1f}s")

# ---------- 增量对比 ----------
# select_positive 只把 mean IC>0 的因子纳入训练/回测，故 wB 列仅含被选中的因子；
# 那些 mean IC<=0 的 brk 因子（如 brk_a7_bbtotal/brk_a8_dread/brk_b2_amt_wtd_mom/brk_b3_ampvol）
# 不会出现在 wB 中——这正是 select_positive 的预期行为，按实际入选列过滤即可。
brk_names = [n for n in feB if n.startswith("brk_") and n in wB.columns]
brk_w = wB[brk_names]
brk_stats = {
    n: {
        "mean_weight": float(brk_w[n].mean()),
        "pct_positive": float((brk_w[n] > 0).mean() * 100),
        "mean_abs_weight": float(brk_w[n].abs().mean()),
    }
    for n in brk_names
}
icB = tm.daily_ic(feB, label)
brk_ic = {n: float(icB[n].mean()) for n in brk_names}

# A 中（非 brk）因子的权重统计，便于看 brk 贡献占比（仅取实际入选列）
non_brk = [n for n in feB if not n.startswith("brk_") and n in wB.columns]
non_brk_abs = wB[non_brk].abs().mean().sum()
brk_abs_total = float(pd.Series({n: brk_stats[n]["mean_abs_weight"] for n in brk_names}).sum())
brk_share = float(brk_abs_total / (non_brk_abs + brk_abs_total)) if (non_brk_abs + brk_abs_total) > 0 else 0.0

report = {
    "config": {
        "universe": "pit", "train_days": TRAIN_DAYS, "retrain": RETRAIN,
        "rebalance": REBALANCE, "top_n": BOOK_TOP_N, "max_w": BOOK_MAX_W,
        "adv_p": BOOK_ADV_P, "commission_bps": COMMISSION_BPS,
        "impact_bps": IMPACT_BPS, "stamp_bps": STAMP_TAX_BPS,
        "feature_selection": "select_positive(IC>0)  # 生产同款，非 None",
        "n_A": len(feA), "n_B": len(feB),
    },
    "A_base_exp": {k: mA.get(k) for k in (
        "sharpe_like", "total_return", "annualized_return", "max_drawdown",
        "avg_turnover", "avg_positions_count", "trade_days")},
    "B_base_exp_broker": {k: mB.get(k) for k in (
        "sharpe_like", "total_return", "annualized_return", "max_drawdown",
        "avg_turnover", "avg_positions_count", "trade_days")},
    "delta": {k: (mB.get(k) - mA.get(k)) for k in (
        "sharpe_like", "total_return", "annualized_return", "max_drawdown",
        "avg_turnover", "avg_positions_count")},
    "broker_portfolio_weight_share": brk_share,
    "broker_factor_weight_stats": brk_stats,
    "broker_factor_mean_ic": brk_ic,
}
(OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
eqA.to_csv(OUT / "equity_A.csv", index=False)
eqB.to_csv(OUT / "equity_B.csv", index=False)
wB.to_csv(OUT / "weights_B.csv", index=False)

# ---------- 打印 ----------
print("\n" + "=" * 64)
print("WALK-FORWARD 增量验证（broker 因子吸收，select_positive 真实配置）")
print("=" * 64)
print(f"A base+exp          (n={len(feA)}): sharpe={mA['sharpe_like']:.3f}  "
      f"ret={mA['total_return']:.3f}  dd={mA['max_drawdown']:.3f}")
print(f"B +broker           (n={len(feB)}): sharpe={mB['sharpe_like']:.3f}  "
      f"ret={mB['total_return']:.3f}  dd={mB['max_drawdown']:.3f}")
print(f"DELTA                      : sharpe={report['delta']['sharpe_like']:+.3f}  "
      f"ret={report['delta']['total_return']:+.3f}  dd={report['delta']['max_drawdown']:+.3f}")
print(f"\nbrk 因子在 B 投资组合权重中的平均占比（|weight|）: {brk_share * 100:.1f}%")
print("-" * 64)
print("brk 因子权重/IC（按 |mean_weight| 降序）")
print(f"{'factor':30s} {'wmean':>9s} {'pctPos':>7s} {'IC':>8s}")
for n in sorted(brk_names, key=lambda x: -abs(brk_stats[x]["mean_weight"])):
    s = brk_stats[n]
    print(f"{n:30s} {s['mean_weight']:+.4f} {s['pct_positive']:6.1f}% {brk_ic[n]:+.4f}")
print("-" * 64)
print(f"total time {time.time() - t0:.1f}s ; outputs -> {OUT}")
