"""P9-5.1：降换手主动版港股通横截面信号。

P9-5 主动信号容量塌到 ~1 亿，根因是日换手 0.785（rebalance_frequency=1）。
本脚本保持 P9-5 同口径（同因子/标签/流动性筛选），仅降低再平衡频率，
扫描 AUM 找新容量上限，并复算与 A 股簿相关性 + 三路融合。

核心问题：降换手能否把港股通主动信号容量从 ~1 亿抬升到数十/数百亿？
代价：信号本为短 horizon（次日开盘收益），降频可能损 Sharpe——测出来。

rebalance_frequency 扫描：1(日,基线来自P9-5) / 5(周) / 10(双周) / 21(月)。
每个频率 AUM 扫描：1亿/10亿/100亿/500亿/1000亿/5000亿。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import next_open_return_label
from run_backtest import pivot_prices, prepare_prices
from train_next_open_rank_model import (
    build_features,
    calculate_walk_forward_metrics,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    run_walk_forward,
)

HERE = Path(__file__).resolve().parent
NORM = HERE / "external_data" / "daily-market-data-tdx" / "hk_connect" / "normalized"
OUT = HERE / "outputs" / "p9_5_1_low_turnover"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
TRAIN_DAYS = 252
MTH = 1
LIQ_LOOKBACK = 126
THR = 1e8
RETRAIN = 20
TOP_N = 20
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
AUMS = [1e8, 1e9, 1e10, 5e10, 1e11, 5e11]
FREQS = [5, 10, 21]   # 周/双周/月；日频(1)基线取自 P9-5 metrics.json


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def decompose_cost(eq: pd.DataFrame) -> dict:
    turnover = float(eq["turnover"].sum())
    total_cost = float(eq["cost"].sum())
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {"total_cost": total_cost, "commission_cost": commission,
            "impact_cost": impact, "impact_share": impact / total_cost if total_cost > 0 else 0.0}


def main() -> None:
    files = sorted(NORM.glob("ths_hk_connect_*.csv"))
    print(f"拼接港股通面板：{len(files)} 日文件 …")
    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw = prepare_prices(long, None, None, strict_validation=False)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    print(f"面板：{close.shape[0]} 交易日 × {close.shape[1]} 标的")

    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=60).median()
    liquid_mask = (roll_med >= THR).fillna(False)
    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    ic = daily_ic(features, label)
    market_exposure = load_market_exposure(None, close.index, ma_window=120,
        risk_off_drawdown_20d=-0.08, below_ma_exposure=0.60, crash_exposure=0.0)

    # 载入 P9-5 日频基线（rebalance_frequency=1）
    p95 = json.loads(Path("outputs/p9_5_hk_connect_signal/metrics.json").read_text(encoding="utf-8"))
    all_results = {"freq_1_baseline_from_p95": p95["aum_sweep"]}

    for freq in FREQS:
        print(f"\n########## rebalance_frequency = {freq} ##########")
        rows = []
        eq_at_1e8 = None
        for aum in AUMS:
            eq, _, _ = run_walk_forward(
                close=close, open_px=open_px, features=features, label=label, ic=ic,
                train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
                rebalance_frequency=freq, max_position_weight=0.04, leverage=1.0,
                commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
                impact_model="sqrt", impact_ref_participation=IMPACT_REF,
                max_buy_open_gap=0.06, limit_buffer=0.995,
                market_exposure=market_exposure, initial_capital=aum,
                max_training_horizon=MTH, feature_directions=None, amount=amount,
                feature_selection=select_positive, max_daily_amount_participation=None,
                liquid_mask=liquid_mask,
            )
            m = calculate_walk_forward_metrics(eq, aum)
            eq.to_csv(OUT / f"equity_freq{freq}_aum_{int(aum)}.csv", index=False, encoding="utf-8")
            dec = decompose_cost(eq)
            rows.append({
                "aum": aum, "total_return": m.get("total_return"),
                "annualized_return": m.get("annualized_return"),
                "max_drawdown": m.get("max_drawdown"), "sharpe_like": m.get("sharpe_like"),
                "avg_turnover": m.get("avg_turnover"),
                "avg_gross_exposure": m.get("avg_gross_exposure"),
                "avg_positions_count": m.get("avg_positions_count"),
                "impact_cost": dec["impact_cost"], "impact_cost_share": dec["impact_share"],
            })
            if aum == 1e8:
                eq_at_1e8 = eq
            print(f"  AUM={aum:>14,.0f} ann={m.get('annualized_return'):.4f} "
                  f"sharpe={m.get('sharpe_like'):.3f} dd={m.get('max_drawdown'):.3f} "
                  f"turn={m.get('avg_turnover'):.4f} impact_share={dec['impact_share']:.3f}")
        cap = next((r["aum"] for r in rows if r["impact_cost_share"] >= 0.5), None)
        all_results[f"freq_{freq}"] = {"aum_sweep": rows, "capacity_ceiling_aum": cap,
                                        "equity_1e8_path": str(OUT / f"equity_freq{freq}_aum_100000000.csv")}
        # 保存 1亿权益曲线用于相关性/融合
        if eq_at_1e8 is not None:
            eq_at_1e8.to_csv(OUT / f"signal_freq{freq}_aum_1e8.csv", index=False, encoding="utf-8")

    # ---- 相关性 + 三路融合（用各频率 1亿权益曲线，最小成本污染）----
    book = pd.read_csv("outputs/p8b_dynamic_liquidity/equity_curve_aum_100000000.csv")
    book["td"] = book["date"].astype(str).str.replace("-", "", regex=False)
    book = book[["td", "gross_return"]].rename(columns={"gross_return": "book_ret"})
    sl = pd.read_csv("external_data/derivatives/futures_sleeve_overlay.csv")
    sl["td"] = sl["trade_date"].astype(str)
    sl = sl[["td", "sleeve_ret"]]

    blends = {}
    for freq in FREQS:
        hk = pd.read_csv(OUT / f"signal_freq{freq}_aum_1e8.csv")
        hk["td"] = hk["date"].astype(str).str.replace("-", "", regex=False)
        hk = hk[["td", "gross_return"]].rename(columns={"gross_return": "hk_ret"})
        m = book.merge(sl, on="td").merge(hk, on="td").dropna().sort_values("td")
        corr = m[["book_ret", "sleeve_ret", "hk_ret"]].corr()
        def stats(r):
            cum = (1 + r.fillna(0)).cumprod(); v = r.std() * np.sqrt(252)
            a = cum.iloc[-1] ** (252 / len(r)) - 1; s = r.mean() / r.std() * np.sqrt(252)
            d = (cum / cum.cummax() - 1).min()
            return round(float(a), 4), round(float(v), 4), round(float(s), 4), round(float(d), 4)
        b = {}
        for name, (wb, ws, wh) in {"book_only": (1, 0, 0), "book+sleeve_8020": (0.8, 0.2, 0),
                                    "three_way_602020": (0.6, 0.2, 0.2),
                                    "three_way_505025": (0.5, 0.25, 0.25)}.items():
            r = wb * m["book_ret"] + ws * m["sleeve_ret"] + wh * m["hk_ret"]
            a, v, s, d = stats(r); b[name] = {"ann": a, "vol": v, "sharpe": s, "mdd": d}
        blends[f"freq_{freq}"] = {
            "corr": {"book_hk": round(float(corr.loc["book_ret", "hk_ret"]), 4),
                     "sleeve_hk": round(float(corr.loc["sleeve_ret", "hk_ret"]), 4)},
            "blends": b,
        }
        print(f"\n[freq={freq}] book<->hk={blends[f'freq_{freq}']['corr']['book_hk']:+.4f} "
              f"three_way_505025 sharpe={b['three_way_505025']['sharpe']:.3f}")

    all_results["blends_by_freq"] = blends
    (OUT / "metrics.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P9-5.1 降换手主动版 ===")
    print(json.dumps({k: v for k, v in all_results.items() if k != "freq_1_baseline_from_p95"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
