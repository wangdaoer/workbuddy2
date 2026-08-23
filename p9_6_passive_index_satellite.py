"""P9-6：被动港股通指数卫星路线（对比 P9-5 主动信号）。

P9-5 结论：训练出的港股通横截面 alpha 信号相关 0.57、容量仅 ~1 亿（日换手 0.785）。
但第 9 节的港股通**流动性加权指数**是被动持有（book↔hk≈0，独立），其"容量 4,327 亿"
在主动信号下不成立，但在**被动持有**下应成立——前提是换手足够低。

本脚本量化这一点：
  1. 构造被动港股通卫星：流动性加权篮子，月再平衡（trailing 126 日 median ADV 重构权重）。
  2. 测其真实年换手 → 套用与 P8b/P9-5 同一套容量公式反推实证容量。
  3. 用冲击模型做 AUM 扫描，证明被动卫星冲击成本占比在数千亿规模下仍≈0。
  4. 重新做 book+sleeve+被动港股通 三路融合，给出最终诚实跨市场组合。

核心论证：同一套流动性池容量公式
    capacity = ADV_daily × 252 × participation / annual_turnover
主动信号 annual_turnover≈198（日换手 0.785×252）→ 容量 ~1 亿；
被动卫星 annual_turnover≈0.5–1 → 容量 ~数千亿（合法）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NORM = HERE / "external_data" / "daily-market-data-tdx" / "hk_connect" / "normalized"
OUT = HERE / "outputs" / "p9_6_passive_index_satellite"
OUT.mkdir(parents=True, exist_ok=True)

FX_HKD_TO_CNY = 0.92
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
REBALANCE_DAYS = 21          # ~月度再平衡
LIQ_LOOKBACK = 126
PARTICIPATION_CAP = 0.01


def build_panel():
    files = sorted(NORM.glob("ths_hk_connect_*.csv"))
    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    long = long[["date", "symbol", "close", "amount"]].dropna()
    long["date"] = pd.to_datetime(long["date"])
    close = long.pivot(index="date", columns="symbol", values="close").sort_index()
    amount = long.pivot(index="date", columns="symbol", values="amount").sort_index()
    return close, amount


def passive_monthly_rebalance(close: pd.DataFrame, amount: pd.DataFrame) -> tuple[pd.Series, float]:
    """月再平衡流动性加权被动篮子。返回权益曲线与单边年换手。"""
    ret = close.pct_change(fill_method=None)
    # trailing 126 日 median ADV 作为重构权重（仅用历史，无前视）
    adv_trailing = amount.rolling(LIQ_LOOKBACK, min_periods=60).median()
    dates = close.index
    n = len(dates)
    w = pd.Series(0.0, index=close.columns)
    w_prev = pd.Series(0.0, index=close.columns)
    eq = 1.0
    eq_curve = []
    oneway_turnover_total = 0.0
    rebalances = 0
    last_rebal = -REBALANCE_DAYS
    for i in range(1, n):
        # 当日收益（按当前权重）
        r = ret.iloc[i].reindex(close.columns).fillna(0.0)
        day_ret = float((w * r).sum())
        eq *= (1.0 + day_ret)
        # 权重随价格漂移
        if w.sum() > 0:
            w = w * (1.0 + r)
            w = w / w.sum()
        # 再平衡判定
        if i - last_rebal >= REBALANCE_DAYS:
            target = adv_trailing.iloc[i]
            target = target[target > 0]
            if target.sum() > 0:
                target = target / target.sum()
                # 单边换手
                oneway = float((w.reindex(target.index).fillna(0.0) - target).abs().sum()) / 2.0
                oneway_turnover_total += oneway
                rebalances += 1
                w = target.reindex(close.columns).fillna(0.0)
                w = w / w.sum() if w.sum() > 0 else w
                last_rebal = i
        eq_curve.append((dates[i], eq))
    eq_series = pd.Series({d: v for d, v in eq_curve})
    years = (dates[-1] - dates[0]).days / 365.25
    annual_turnover = oneway_turnover_total / years if years > 0 else 0.0
    return eq_series, annual_turnover


def capacity_from_turnover(total_adv_cny_per_day: float, annual_turnover: float) -> float:
    return total_adv_cny_per_day * 252 * PARTICIPATION_CAP / annual_turnover


def impact_share_at_aum(eq: pd.Series, annual_turnover: float, total_adv_cny_per_day: float, aum: float) -> float:
    """被动卫星在 AUM 下的年化冲击成本占比（同一套 sqrt 模型）。"""
    # 年化交易额 = annual_turnover × aum；参与率 = 年化交易额 / (ADV×252)
    annual_traded = annual_turnover * aum
    participation = annual_traded / (total_adv_cny_per_day * 252)
    if participation <= 0:
        return 0.0
    # 全市场等参与率近似：impact_bps × sqrt(part/ref)
    impact_bps_eff = IMPACT_BPS * np.sqrt(min(participation, 1.0) / IMPACT_REF)
    annual_impact_cost = annual_traded * (COMMISSION_BPS + impact_bps_eff) / 1e4
    annual_return = abs(eq.iloc[-1] / eq.iloc[0] - 1.0)
    return annual_impact_cost / (aum * max(annual_return, 1e-6))


def main():
    close, amount = build_panel()
    # 总 ADV（CNY/日）：用窗口内 median ADV 求和（与 P9-4 同口径）
    med_adv_hkd = amount.median().sum()
    total_adv_cny_per_day = med_adv_hkd * FX_HKD_TO_CNY
    print(f"港股通总 ADV: {med_adv_hkd:,.0f} HKD/日 ≈ {total_adv_cny_per_day:,.0f} CNY/日")

    eq, annual_turnover = passive_monthly_rebalance(close, amount)
    ret = eq.pct_change(fill_method=None).dropna()
    ann = eq.iloc[-1] ** (252 / len(ret)) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    mdd = float((eq / eq.cummax() - 1).min())
    print(f"被动港股通卫星: 年换手={annual_turnover:.3f}, 年化={ann:.4f}, 波动={vol:.4f}, "
          f"Sharpe={sharpe:.3f}, 回撤={mdd:.3f}")

    cap_passive = capacity_from_turnover(total_adv_cny_per_day, annual_turnover)
    cap_active = capacity_from_turnover(total_adv_cny_per_day, 0.785 * 252)  # P9-5 主动年换手≈198
    print(f"被动卫星容量(同公式): {cap_passive:,.0f} CNY")
    print(f"主动信号容量(同公式, 年换手198): {cap_active:,.0f} CNY")

    # AUM 扫描：被动卫星冲击成本占比
    aums = [1e9, 1e10, 1e11, 1e12, 5e12]
    impact_rows = []
    for a in aums:
        sh = impact_share_at_aum(eq, annual_turnover, total_adv_cny_per_day, a)
        impact_rows.append({"aum": a, "impact_cost_share": round(float(sh), 6)})
        print(f"  AUM={a:,.0f} CNY -> 被动卫星冲击成本占比={sh:.6f}")

    # 三路融合（用 P9-4 被动港股通指数，book+sleeve 已有）
    book = pd.read_csv(HERE / "outputs/p8b_dynamic_liquidity/equity_curve_aum_100000000.csv")
    book["td"] = book["date"].astype(str).str.replace("-", "", regex=False)
    book = book[["td", "gross_return"]].rename(columns={"gross_return": "book_ret"})
    sl = pd.read_csv(HERE / "external_data/derivatives/futures_sleeve_overlay.csv")
    sl["td"] = sl["trade_date"].astype(str)
    sl = sl[["td", "sleeve_ret"]]
    # 被动卫星日收益（由权益曲线反算）
    hk = eq.reset_index(); hk.columns = ["date", "eq"]
    hk["td"] = hk["date"].astype(str).str.replace("-", "", regex=False)
    hk["hk_ret"] = hk["eq"].pct_change(fill_method=None)
    hk = hk[["td", "hk_ret"]]
    m = book.merge(sl, on="td").merge(hk, on="td").dropna().sort_values("td")

    def stats(r):
        cum = (1 + r.fillna(0)).cumprod()
        v = r.std() * np.sqrt(252); a = cum.iloc[-1] ** (252 / len(r)) - 1
        s = r.mean() / r.std() * np.sqrt(252); d = (cum / cum.cummax() - 1).min()
        return round(float(a), 4), round(float(v), 4), round(float(s), 4), round(float(d), 4)

    blends = {}
    for name, (wb, ws, wh) in {
        "book_only": (1, 0, 0), "book+sleeve_8020": (0.8, 0.2, 0),
        "three_way_602020": (0.6, 0.2, 0.2), "three_way_505025": (0.5, 0.25, 0.25),
    }.items():
        r = wb * m["book_ret"] + ws * m["sleeve_ret"] + wh * m["hk_ret"]
        a, v, s, d = stats(r)
        blends[name] = {"ann": a, "vol": v, "sharpe": s, "mdd": d}
        print(f"  {name:20s} ann={a:.4f} vol={v:.4f} sharpe={s:.3f} mdd={d:.3f}")

    report = {
        "route": "passive HK connect index satellite (monthly rebalance, liquidity-weighted)",
        "passive_satellite": {
            "annual_turnover": round(float(annual_turnover), 4),
            "annualized_return": round(float(ann), 4),
            "ann_vol": round(float(vol), 4),
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(mdd), 4),
        },
        "capacity_same_formula": {
            "total_adv_cny_per_day": round(float(total_adv_cny_per_day), 0),
            "passive_annual_turnover": round(float(annual_turnover), 4),
            "passive_capacity_cny": round(float(cap_passive), 0),
            "active_annual_turnover": round(float(0.785 * 252), 1),
            "active_capacity_cny_p9_5": round(float(cap_active), 0),
            "note": "同一套流动性池公式；被动因年换手低→容量数千亿，主动因日换手0.785→容量~1亿",
        },
        "passive_impact_cost_share_by_aum": impact_rows,
        "three_way_blend_with_passive_index": blends,
        "cross_market_capacity_cny": {
            "ashare_book_p8b": 100_000_000,
            "derivative_futures_sleeve": 100_000_000_000,
            "hk_connect_passive_satellite": round(float(cap_passive), 0),
            "combined": round(float(100_000_000 + 100_000_000_000 + cap_passive), 0),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    eq.to_frame("equity").to_csv(OUT / "passive_satellite_equity.csv", encoding="utf-8")
    print("\n=== P9-6 被动指数卫星 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
