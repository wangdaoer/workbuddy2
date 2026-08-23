"""P9-7：生产化跨市场组合构建 + 风险预算再平衡。

合并三条收益流（A 股簿 p8b / 期货 sleeve / 港股通 leg），用**滚动窗口**
（walk-forward，避免前视）估计权重，对比多种配置方案，输出可部署组合。

港股通 leg 提供两版：
  - 被动指数卫星（P9-6，用户选定最终组合）
  - 主动月频信号（P9-5.1，替代主动路线）

配置方案：
  - fixed_502525 : 固定 50/25/25（P9-6 基线）
  - equal        : 等权 33/33/33
  - risk_parity  : 逆波动加权（滚动 126 日）
  - min_variance : 最小方差（滚动 126 日，多空截断为多头）
  - max_sharpe   : 最大夏普（滚动 126 日，多空截断为多头，略乐观）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "p9_7_production_portfolio"
OUT.mkdir(parents=True, exist_ok=True)
# P10g-production: 生产化 regime 门控 MLP 的 book 权益（p8b 兼容格式），由 production_soft_score.py 产出
PROD_BOOK = HERE / "outputs" / "production_soft_score" / "book_soft_equity.csv"

EST_WIN = 126          # 滚动估计窗口（交易日）
ANN = 252


def load_streams(book_source: str = "p8b") -> dict[str, pd.DataFrame]:
    # A 股簿（book）：默认 p8b incumbent；P10g 起支持生产化 regime 门控 MLP (soft)
    if book_source == "soft":
        if not PROD_BOOK.exists():
            raise FileNotFoundError(f"生产 book 权益缺失：{PROD_BOOK}；请先运行 production_soft_score.py")
        b = pd.read_csv(PROD_BOOK)
        book_name = "p10e_soft (regime-gated MLP, production)"
    else:
        b = pd.read_csv(HERE / "outputs/p8b_dynamic_liquidity/equity_curve_aum_100000000.csv")
        book_name = "p8b incumbent (linear)"
    b["td"] = b["date"].astype(str).str.replace("-", "", regex=False)
    book = b[["td", "gross_return"]].rename(columns={"gross_return": "book_ret"})

    # 期货 sleeve，sleeve_ret
    s = pd.read_csv(HERE / "external_data/derivatives/futures_sleeve_overlay.csv")
    s["td"] = s["trade_date"].astype(str)
    sleeve = s[["td", "sleeve_ret"]].rename(columns={"sleeve_ret": "sleeve_ret"})

    # 港股通被动卫星（P9-6）：equity 增长指数 → 日收益
    hkp = pd.read_csv(HERE / "outputs/p9_6_passive_index_satellite/passive_satellite_equity.csv")
    hkp.columns = ["date", "equity"]
    hkp["td"] = hkp["date"].astype(str).str.replace("-", "", regex=False)
    hkp = hkp.assign(hk_ret=hkp["equity"].pct_change()).dropna()[["td", "hk_ret"]]

    # 港股通主动月频（P9-5.1 freq_21）
    hka = pd.read_csv(HERE / "outputs/p9_5_1_low_turnover/signal_freq21_aum_1e8.csv")
    hka["td"] = hka["date"].astype(str).str.replace("-", "", regex=False)
    hka = hka[["td", "gross_return"]].rename(columns={"gross_return": "hk_ret"})

    return {"book": book, "sleeve": sleeve, "hk_passive": hkp, "hk_active": hka}


def stats(r: pd.Series) -> dict:
    r = r.fillna(0.0)
    cum = (1 + r).cumprod()
    ann = float(cum.iloc[-1] ** (ANN / len(r)) - 1)
    vol = float(r.std() * np.sqrt(ANN))
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    mdd = float((cum / cum.cummax() - 1).min())
    return {"annualized_return": round(ann, 4), "ann_vol": round(vol, 4),
            "sharpe": round(sharpe, 4), "max_drawdown": round(mdd, 4),
            "n_days": int(len(r))}


def weights_inv_vol(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))
    w = 1.0 / vol
    return w / w.sum()


def weights_quadratic(cov: np.ndarray, mu: np.ndarray | None, ridge: float = 1e-8) -> np.ndarray:
    n = cov.shape[0]
    c = cov + ridge * np.eye(n)
    try:
        inv = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(c)
    if mu is None:           # 最小方差
        raw = inv @ np.ones(n)
    else:                    # 最大夏普
        raw = inv @ mu
    raw = np.clip(raw, 0, None)          # 多头约束
    if raw.sum() <= 0:
        return np.ones(n) / n
    return raw / raw.sum()


def build_portfolio(panel: pd.DataFrame, cols: list[str]) -> dict:
    R = panel[cols].fillna(0.0).to_numpy()          # (T, 3)
    T = len(R)
    schemes = {
        "fixed_502525": np.array([0.50, 0.25, 0.25]),
        "equal": np.array([1/3, 1/3, 1/3]),
    }
    # 滚动窗口动态权重
    dyn = {"risk_parity": [], "min_variance": [], "max_sharpe": []}
    w_hist = []
    for t in range(T):
        if t < EST_WIN:
            w = schemes["equal"]          # 预热期用等权
        else:
            win = R[t - EST_WIN:t]
            cov = np.cov(win, rowvar=False)
            mu = win.mean(0)
            dyn["risk_parity"].append(weights_inv_vol(cov))
            dyn["min_variance"].append(weights_quadratic(cov, None))
            dyn["max_sharpe"].append(weights_quadratic(cov, mu))
            w = dyn["max_sharpe"][-1]     # 默认占位，下面逐方案重算
        w_hist.append(w)
    # 重新逐方案构造组合收益（fixed/equal 用固定权重；动态用各自权重序列）
    out = {}
    eq_curves = {}
    for name, w_fixed in schemes.items():
        pr = R @ w_fixed
        out[name] = stats(pd.Series(pr))
        eq_curves[name] = (1 + pd.Series(pr)).cumprod().values
    for name in dyn:
        wmat = np.array([schemes["equal"]] * EST_WIN + dyn[name])  # 预热等权
        pr = (R * wmat).sum(1)
        out[name] = stats(pd.Series(pr))
        eq_curves[name] = (1 + pd.Series(pr)).cumprod().values
    return out, eq_curves, cols


def main(book_source: str = "p8b") -> None:
    st = load_streams(book_source)
    book = st["book"]; sleeve = st["sleeve"]
    book_name = "p10e_soft (regime-gated MLP)" if book_source == "soft" else "p8b incumbent (linear)"
    results = {}
    for leg_name, leg in [("hk_passive", st["hk_passive"]), ("hk_active", st["hk_active"])]:
        panel = book.merge(sleeve, on="td").merge(leg, on="td").dropna().sort_values("td").reset_index(drop=True)
        cols = ["book_ret", "sleeve_ret", "hk_ret"]
        metr, eqs, _ = build_portfolio(panel, cols)
        # 相关性
        corr = panel[cols].corr()
        results[leg_name] = {
            "n_days": int(len(panel)),
            "first": panel["td"].iloc[0],
            "last": panel["td"].iloc[-1],
            "corr": {
                "book_sleeve": round(float(corr.loc["book_ret", "sleeve_ret"]), 4),
                "book_hk": round(float(corr.loc["book_ret", "hk_ret"]), 4),
                "sleeve_hk": round(float(corr.loc["sleeve_ret", "hk_ret"]), 4),
            },
            "schemes": metr,
        }
        # 存权益曲线（前缀区分 book 来源）
        for name, eq in eqs.items():
            pd.DataFrame({"td": panel["td"], "equity": eq}).to_csv(
                OUT / f"portfolio_{book_source}_{leg_name}_{name}.csv", index=False, encoding="utf-8")
        print(f"\n===== book={book_name} | HK leg = {leg_name} (n={len(panel)}) =====")
        print(f"  corr: book_sleeve={results[leg_name]['corr']['book_sleeve']:+.3f} "
              f"book_hk={results[leg_name]['corr']['book_hk']:+.3f} sleeve_hk={results[leg_name]['corr']['sleeve_hk']:+.3f}")
        for name, m in metr.items():
            print(f"  {name:14s} ann={m['annualized_return']:.4f} vol={m['ann_vol']:.4f} "
                  f"sharpe={m['sharpe']:.3f} mdd={m['max_drawdown']:.4f}")

    # 推荐生产组合：被动 leg + 风险平价（波动加权，降波动提 Sharpe）
    rec = results["hk_passive"]["schemes"]["risk_parity"]
    payload = {
        "meta": {
            "title": "P9-7 生产化跨市场组合",
            "asof": results["hk_passive"]["last"],
            "est_window": EST_WIN,
            "weight_method": "rolling-window (walk-forward, no look-ahead)",
            "book_source": book_source,
            "book_name": book_name,
            "components": {
                "ashare_book": book_name + " (A股横截面)",
                "futures_sleeve": "P9-3 指数期货动量 sleeve (~1000亿容量)",
                "hk_connect_leg": "P9-6 被动指数卫星 (用户选定, ~5141亿容量)",
            },
        },
        "results_by_hk_leg": results,
        "recommended_production_portfolio": {
            "hk_leg": "hk_passive (P9-6, 用户选定)",
            "weight_scheme": "risk_parity (滚动126日逆波动)",
            "book_source": book_source,
            "metrics": rec,
            "rationale": "被动 leg 波动最高(25%)，风险平价自动降权港股通、升权低波动的 sleeve，"
                         "在不牺牲分散的前提下降波动、提 Sharpe。book 已升级为 P10e regime 门控 MLP。"
                         if book_source == "soft" else
                         "被动 leg 波动最高(25%)，风险平价自动降权港股通、升权低波动的 sleeve，"
                         "在不牺牲分散的前提下降波动、提 Sharpe。",
            "rebalance_schedule": {
                "ashare_book": "每日 walk-forward 调仓（production_soft_score 引擎）" if book_source == "soft"
                               else "每日 walk-forward 调仓（p8b 引擎）",
                "futures_sleeve": "每日 20 日动量再平衡（futures_sleeve_overlay）",
                "hk_passive": "每月再平衡（流动性加权，年换手 0.73）",
            },
            "combined_capacity_cny": 614187961937,  # book 容量可忽略，合并容量基本不变
        },
    }
    (OUT / f"metrics_{book_source}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== P9-7 推荐生产组合（book={book_name}, 被动 leg + 风险平价）===")
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-source", choices=["p8b", "soft"], default="p8b",
                    help="A股簿打分来源：p8b=incumbent 线性 / soft=生产化 regime 门控 MLP")
    args = ap.parse_args()
    main(args.book_source)
