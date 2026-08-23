"""demo_pit.py — 用真实数据量化"自选股前视"对 sharpe 的虚高。

复用已缓存的生产 soft 分（outputs/production_soft_score/soft_score_feed.csv）与
真实面板，在【同一套 Route C 成本模型】下对比两种股票池：

  (A) PIT 宇宙      : 每个再平衡日 t 只用 ≤t 信息算合格宇宙（pit_universe.pit_eligible）
  (B) 静态幸存者宇宙 : 把"最后一天存在的标的"广播到所有历史日（=用未来名单回测，前视错误写法）

差异 (B)-(A) 即"筛选后的自选股"若用未来信息筛选，会在回测里虚高多少。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from production_soft_score import build_panel, PANEL
from train_next_open_rank_model import (
    run_walk_forward, daily_ic, calculate_walk_forward_metrics, MIN_LIVE_SYMBOLS,
)
from p10c_ensemble import TRAIN_DAYS, MTH, RETRAIN, LIQ_LOOKBACK, THR
import pit_universe as pit

HERE = Path(__file__).resolve().parent
AUM = 100_000_000.0


def _run(mask):
    P = build_panel(PANEL)
    soft = pd.read_csv(HERE / "outputs" / "production_soft_score" / "soft_score_feed.csv",
                       index_col=0, parse_dates=True)
    soft = soft.reindex(index=P["label"].index, columns=P["symbols"])
    ic = daily_ic({"soft": soft}, P["label"])
    eq, _, _ = run_walk_forward(
        close=P["close"], open_px=P["open_px"], features={"soft": soft}, label=P["label"],
        ic=ic, train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=40,
        rebalance_frequency=5, max_position_weight=0.04, leverage=1.0,
        commission_bps=3.0, impact_bps=0.7, stamp_tax_bps=5.0, impact_model="sqrt",
        impact_ref_participation=0.01, max_buy_open_gap=0.06, limit_buffer=0.995,
        market_exposure=P["market_exposure"], initial_capital=AUM, max_training_horizon=MTH,
        amount=P["amount"], max_daily_amount_participation=0.02, liquid_mask=mask,
    )
    m = calculate_walk_forward_metrics(eq, AUM)
    return m


def main():
    P = build_panel(PANEL)
    pit_mask = pit.pit_eligible(P["close"], P["amount"], None,
                                thr=THR, lookback=LIQ_LOOKBACK, min_days=60)
    static_mask = pit.static_survivor_mask(P["close"])

    print("=== PIT 宇宙（正确写法，≤t 信息）===")
    m_pit = _run(pit_mask)
    print(f"  total={m_pit['total_return']:+.4f}  sharpe={m_pit['sharpe_like']:.3f}  "
          f"dd={m_pit['max_drawdown']:.3f}")

    print("=== 静态幸存者宇宙（前视错误写法，用未来名单）===")
    m_static = _run(static_mask)
    print(f"  total={m_static['total_return']:+.4f}  sharpe={m_static['sharpe_like']:.3f}  "
          f"dd={m_static['max_drawdown']:.3f}")

    gap = m_static["sharpe_like"] - m_pit["sharpe_like"]
    print(f"\n=== 前视虚高（静态 - PIT）===\nsharpe 虚高: {gap:+.3f}  "
          f"({(gap / max(abs(m_pit['sharpe_like']), 1e-9) * 100):+.1f}% vs PIT)")

    # 静态名单前视污染度（若把"最后一天名单"当 watchlist 用）
    leak = pit.detect_static_leak(P["close"], P["amount"], None,
                                  list(P["close"].columns), thr=THR,
                                  lookback=LIQ_LOOKBACK, min_days=60)
    print(f"\n=== 静态名单前视污染度（以全市场末日名单为 watchlist 示例）===\n"
          f"  watchlist 规模={leak['n_watchlist']}  平均泄漏比例={leak['mean_leak_ratio']:.3f}  "
          f"最大={leak['max_leak_ratio']:.3f} 最小={leak['min_leak_ratio']:.3f}")


if __name__ == "__main__":
    main()
