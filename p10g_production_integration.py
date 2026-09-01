"""P10g：把 P10e 的 regime 门控 MLP（p10e_soft 因果版）接入 P9-7 生产组合并重新估算容量。

P9-7 把三条收益流（A 股簿 p8b / 期货 sleeve / 港股通 leg）用滚动窗口配权，构成可部署组合；
其中 A 股簿（book）用的是 P8b  incumbent（线性）。P10e 证明 regime 门控 MLP（连续混合 soft）
在同样成本/约束下夏普更高、且能防御 W4 衰减，且前视泄漏验证稳健。

P10g 做两件事：
  1) 组合集成：把 book 流从 p8b 换成 p10e_soft_causal（同为 AUM=1e8、无参与度上限，对等替换），
     重算 5 种配权方案，与 p8b 基线对比组合级指标。
  2) 容量重算：按 P6b 口径（max_daily_amount_participation=0.01）对 soft 分重跑 AUM 扫描
     {1e7,1e8,5e8,1e9}，测 capacity_blocked_buy_weight / avg_gross_exposure / impact_share，
     给出 book 的独立容量，并据此更新组合合并容量。

复用 p10e 已存的 linear_mlp_scores.npz（跳过 7-min 构建），只重跑带约束回测（约 3 分钟）。
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from execution_rules import next_open_return_label
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import (
    MIN_LIVE_SYMBOLS, build_features, calculate_walk_forward_metrics, clean_matrix,
    daily_ic, load_market_exposure, run_walk_forward,
)
from p10c_ensemble import (
    select_positive, build_linear_mlp_scores, MLP, MIN_LIVE_SYMBOLS as _M,
    TRAIN_DAYS, MTH, LIQ_LOOKBACK, THR as LIQ_THR, RETRAIN,
)

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
P10E = HERE / "outputs" / "p10e_regime_gated"
SCORES_NPZ = P10E / "linear_mlp_scores.npz"
OUT = HERE / "outputs" / "p10g_production_integration"
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"

MAX_ABS = 0.22
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
TOP_N = 20
REBALANCE = 1
EST_WIN = 126
ANN = 252
BOOK_AUM = 100_000_000.0
IC_WIN = 60
IC_MIN = 40
THR_HI = 0.03
THR_LO = 0.0
CAP_PARTICIPATION = 0.01
AUMS_CAP = [10_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]


def log(msg):
    ts = __import__("time").strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---- P9-7 portfolio machinery (copied, self-contained) ----
def stats(r):
    r = r.fillna(0.0)
    cum = (1 + r).cumprod()
    ann = float(cum.iloc[-1] ** (ANN / len(r)) - 1)
    vol = float(r.std() * np.sqrt(ANN))
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    mdd = float((cum / cum.cummax() - 1).min())
    return {"annualized_return": round(ann, 4), "ann_vol": round(vol, 4),
            "sharpe": round(sharpe, 4), "max_drawdown": round(mdd, 4), "n_days": int(len(r))}


def weights_inv_vol(cov):
    vol = np.sqrt(np.diag(cov))
    w = 1.0 / vol
    return w / w.sum()


def weights_quadratic(cov, mu, ridge=1e-8):
    n = cov.shape[0]
    c = cov + ridge * np.eye(n)
    try:
        inv = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(c)
    if mu is None:
        raw = inv @ np.ones(n)
    else:
        raw = inv @ mu
    raw = np.clip(raw, 0, None)
    if raw.sum() <= 0:
        return np.ones(n) / n
    return raw / raw.sum()


def build_portfolio(panel, cols):
    R = panel[cols].fillna(0.0).to_numpy()
    T = len(R)
    schemes = {"fixed_502525": np.array([0.50, 0.25, 0.25]),
               "equal": np.array([1/3, 1/3, 1/3])}
    dyn = {"risk_parity": [], "min_variance": [], "max_sharpe": []}
    w_hist = []
    for t in range(T):
        if t < EST_WIN:
            w = schemes["equal"]
        else:
            win = R[t - EST_WIN:t]
            cov = np.cov(win, rowvar=False)
            mu = win.mean(0)
            dyn["risk_parity"].append(weights_inv_vol(cov))
            dyn["min_variance"].append(weights_quadratic(cov, None))
            dyn["max_sharpe"].append(weights_quadratic(cov, mu))
            w = dyn["max_sharpe"][-1]
        w_hist.append(w)
    out, eq_curves = {}, {}
    for name, w_fixed in schemes.items():
        pr = R @ w_fixed
        out[name] = stats(pd.Series(pr))
        eq_curves[name] = (1 + pd.Series(pr)).cumprod().values
    for name in dyn:
        wmat = np.array([schemes["equal"]] * EST_WIN + dyn[name])
        pr = (R * wmat).sum(1)
        out[name] = stats(pd.Series(pr))
        eq_curves[name] = (1 + pd.Series(pr)).cumprod().values
    return out, eq_curves, cols


# ---- load streams (book source configurable) ----
def _book_ret(csv_path):
    b = pd.read_csv(csv_path)
    b["td"] = b["date"].astype(str).str.replace("-", "", regex=False)
    return b[["td", "gross_return"]].rename(columns={"gross_return": "book_ret"})


def load_streams(book_csv):
    """加载组合集成所需收益流。book 为必需；sleeve/hk_passive/hk_active 卫星腿仅在
    其产物文件存在时加载（日更里 p9_5/p9_6 等研究产物可能缺失，缺失则跳过该腿）。"""
    book = _book_ret(book_csv)
    streams = {"book": book}
    sleeve_path = HERE / "external_data/derivatives/futures_sleeve_overlay.csv"
    if sleeve_path.exists():
        s = pd.read_csv(sleeve_path)
        s["td"] = s["trade_date"].astype(str)
        streams["sleeve"] = s[["td", "sleeve_ret"]]
    hkp_path = HERE / "outputs/p9_6_passive_index_satellite/passive_satellite_equity.csv"
    if hkp_path.exists():
        hkp = pd.read_csv(hkp_path)
        hkp.columns = ["date", "equity"]
        hkp["td"] = hkp["date"].astype(str).str.replace("-", "", regex=False)
        hkp = hkp.assign(hk_ret=hkp["equity"].pct_change()).dropna()[["td", "hk_ret"]]
        streams["hk_passive"] = hkp
    hka_path = HERE / "outputs/p9_5_1_low_turnover/signal_freq21_aum_1e8.csv"
    if hka_path.exists():
        hka = pd.read_csv(hka_path)
        hka["td"] = hka["date"].astype(str).str.replace("-", "", regex=False)
        hka = hka[["td", "gross_return"]].rename(columns={"gross_return": "hk_ret"})
        streams["hk_active"] = hka
    return streams


def build_soft_causal_scores(linear_score, mlp_score, label, symbols):
    mlp_ic = daily_ic({"mlp": mlp_score}, label)["mlp"]
    # P0-3 残留修复 (2026-08-23): label[t]=open[t+2]/open[t+1]-1 在 t+2 开盘才成熟,
    # 因果 trailing 须用 shift(2); 原 shift(1) 泄漏 1 天 (与 production_soft_score 一致).
    trailing = mlp_ic.shift(2).rolling(IC_WIN, min_periods=IC_MIN).mean()
    adv = ((THR_HI - trailing) / (THR_HI - THR_LO)).clip(0, 1).fillna(1.0)
    soft = adv.values[:, None] * linear_score.values + (1 - adv.values[:, None]) * mlp_score.values
    return pd.DataFrame(soft, index=label.index, columns=symbols)


def capacity_stress(close, open_px, features, label, amount, market_exposure, liquid_mask_aligned, symbols, feat_arrays, label_arr, scores_npz=SCORES_NPZ):
    scores_npz = Path(scores_npz)
    if scores_npz.exists():
        log("载入已存 linear/mlp 分（跳过构建）")
        d = np.load(scores_npz, allow_pickle=True)
        linear_score = pd.DataFrame(d["linear"], index=label.index, columns=symbols)
        mlp_score = pd.DataFrame(d["mlp"], index=label.index, columns=symbols)
    else:
        log("构建 linear + mlp 分...")
        linear_score, mlp_score = build_linear_mlp_scores(features, label, symbols, feat_arrays, label_arr, liquid_mask_aligned)
        np.savez(scores_npz, linear=linear_score.values, mlp=mlp_score.values)
    soft = build_soft_causal_scores(linear_score, mlp_score, label, symbols)
    ic_run = daily_ic({"soft": soft}, label)
    rows = []
    for aum in AUMS_CAP:
        eq, _, _ = run_walk_forward(
            close=close, open_px=open_px, features={"soft": soft}, label=label, ic=ic_run,
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
            rebalance_frequency=REBALANCE, max_position_weight=0.04, leverage=1.0,
            commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
            impact_model="sqrt", impact_ref_participation=IMPACT_REF,
            max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=market_exposure,
            initial_capital=aum, max_training_horizon=MTH, feature_directions=None,
            amount=amount, feature_selection=None, max_daily_amount_participation=CAP_PARTICIPATION,
            liquid_mask=None,
        )
        m = calculate_walk_forward_metrics(eq, aum)
        cap_blocked = float(eq["capacity_blocked_buy_weight"].sum())
        cap_sessions = int((eq["capacity_limited_symbols"] > 0).sum())
        avg_gross = float(eq["gross_exposure"].mean())
        turnover = float(eq["turnover"].sum())
        commission = turnover * COMMISSION_BPS / 1e4
        total_cost = float(eq["cost"].sum())
        impact = total_cost - commission
        impact_share = float(impact / total_cost) if total_cost > 0 else 0.0
        rows.append({
            "aum": aum, "participation": CAP_PARTICIPATION,
            "total_return": round(float(m.get("total_return")), 4),
            "annualized_return": round(float(m.get("annualized_return")), 4),
            "max_drawdown": round(float(m.get("max_drawdown")), 4),
            "sharpe_like": round(float(m.get("sharpe_like")), 4),
            "avg_gross_exposure": round(avg_gross, 4),
            "impact_cost_share": round(impact_share, 4),
            "capacity_blocked_buy_weight_total": round(cap_blocked, 3),
            "capacity_limited_sessions": cap_sessions,
        })
        log(f"  cap AUM={aum:,.0f}: shp={m.get('sharpe_like'):.3f} avg_gross={avg_gross:.3f} "
            f"impact_share={impact_share:.3f} cap_blocked_w={cap_blocked:.2f} sessions={cap_sessions}")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P10g: p10e_soft regime-gated MLP 接入 P9-7 生产组合 + 容量重算（每日可刷新）。"
    )
    parser.add_argument("--panel", default=str(PANEL),
                        help="日更面板 CSV（含 close/open/high/low/amount）")
    parser.add_argument(
        "--book-csv",
        default=str(HERE / "outputs" / "production_soft_score" / "book_soft_equity.csv"),
        help="当前生产簿收益 CSV（默认日更新鲜簿 book_soft_equity.csv，含 date/gross_return）",
    )
    parser.add_argument(
        "--baseline-book-csv",
        default=str(P10E / f"equity_aum_{int(BOOK_AUM)}_p10e_soft_causal.csv"),
        help="基线对比簿（默认 p10e soft 权益；缺失则跳过基线对比）",
    )
    parser.add_argument("--scores-npz", default=str(SCORES_NPZ),
                        help="p10e_regime_gated 的 linear_mlp_scores.npz（容量段外部前提；缺失则跳过容量段）")
    parser.add_argument("--output-dir", default=str(OUT), help="metrics.json / portfolio_*.csv 输出目录")
    parser.add_argument("--asof-date", default=None, help="YYYY-MM-DD，写入 metrics.json 的 asof_date")
    parser.add_argument("--skip-capacity", action="store_true", help="跳过 ~3min 容量压力测试")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    RUN_LOG.write_text("", encoding="utf-8")
    log("=== P10g 启动：p10e_soft 接入 P9-7 生产组合 + 容量重算 ===")
    t0 = __import__("time").time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # panel + label + exposure (for capacity stress)
    raw = load_prices(Path(args.panel), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    symbols = list(close.columns)
    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    market_exposure = load_market_exposure(None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08, below_ma_exposure=0.60, crash_exposure=0.0)
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= LIQ_THR).fillna(False)
    liquid_mask_aligned = liquid_mask.reindex(index=close.index, columns=close.columns).fillna(False)
    feat_arrays = {k: fr.values for k, fr in features.items()}
    label_arr = label.values

    # ---- (1) capacity stress on soft ----
    book_capacity = None
    cap_rows: list[dict] = []
    scores_npz = Path(args.scores_npz)
    if args.skip_capacity:
        log("跳过容量重算（--skip-capacity）")
    elif not scores_npz.exists():
        log(f"容量段前提缺失（{scores_npz.name} 不存在），跳过容量重算")
    else:
        log("容量重算（P6b 口径, participation=0.01）...")
        cap_rows = capacity_stress(close, open_px, features, label, amount, market_exposure,
                                   liquid_mask_aligned, symbols, feat_arrays, label_arr,
                                   scores_npz=str(scores_npz))
        # book capacity = largest AUM with avg_gross >= 0.75*0.8 and impact_share <= 0.60
        TARGET_GROSS = 0.8
        for r in cap_rows:
            if r["avg_gross_exposure"] >= 0.75 * TARGET_GROSS and r["impact_cost_share"] <= 0.60:
                book_capacity = r["aum"]
        log(f"book capacity (avg_gross>=0.6 & impact_share<=0.6) ≈ {book_capacity:,.0f}")

    # ---- (2) portfolio integration: 主簿（日更新鲜簿）+ 可选基线 ----
    book_specs = [("production_book", Path(args.book_csv))]
    baseline = Path(args.baseline_book_csv)
    if baseline.exists():
        book_specs.append(("p8b_baseline", baseline))
    else:
        log(f"基线簿 {baseline.name} 缺失，跳过基线对比")

    portfolio_results = {}
    primary = load_streams(Path(args.book_csv))
    required_legs = ("sleeve", "hk_passive", "hk_active")
    if all(leg in primary for leg in required_legs):
        log("组合集成：主簿 + sleeve + hk_passive + hk_active")
        for tag, book_csv in book_specs:
            st = load_streams(book_csv)
            log(f"  [{tag}] book={Path(book_csv).name}")
            res = {}
            for leg_name, leg in [("hk_passive", st["hk_passive"]), ("hk_active", st["hk_active"])]:
                panel = st["book"].merge(st["sleeve"], on="td").merge(leg, on="td").dropna().sort_values("td").reset_index(drop=True)
                cols = ["book_ret", "sleeve_ret", "hk_ret"]
                metr, eqs, _ = build_portfolio(panel, cols)
                corr = panel[cols].corr()
                res[leg_name] = {"n_days": int(len(panel)), "first": panel["td"].iloc[0], "last": panel["td"].iloc[-1],
                                 "corr": {"book_sleeve": round(float(corr.loc["book_ret", "sleeve_ret"]), 4),
                                          "book_hk": round(float(corr.loc["book_ret", "hk_ret"]), 4),
                                          "sleeve_hk": round(float(corr.loc["sleeve_ret", "hk_ret"]), 4)},
                                 "schemes": metr}
                for name, eq in eqs.items():
                    pd.DataFrame({"td": panel["td"], "equity": eq}).to_csv(
                        out_dir / f"portfolio_{tag}_{leg_name}_{name}.csv", index=False, encoding="utf-8")
            portfolio_results[tag] = res
            rec = res["hk_passive"]["schemes"]["risk_parity"]
            log(f"    hk_passive risk_parity: shp={rec['sharpe']:.3f} vol={rec['ann_vol']:.3f} mdd={rec['max_drawdown']:.4f}")
    else:
        present = [leg for leg in required_legs if leg in primary]
        log(f"卫星腿缺失（仅 {present}），跳过组合集成；仅容量段产出")

    # combined capacity (inherit P9-7 passive-leg口径: sleeve~1000亿, hk~5141亿)
    sleeve_cap = 100_000_000_000.0
    hk_cap = 514_100_000_000.0
    combined_capacity = (book_capacity or BOOK_AUM) + sleeve_cap + hk_cap

    payload = {
        "asof_date": args.asof_date,
        "meta": {
            "title": "P10g: p10e_soft regime-gated MLP 接入 P9-7 生产组合",
            "book_upgrade": "p8b incumbent (linear) -> p10e_soft causal continuous-blend MLP",
            "book_aum_for_portfolio": BOOK_AUM,
            "est_window": EST_WIN,
            "capacity_participation": CAP_PARTICIPATION,
            "capacity_skipped": bool(args.skip_capacity) or (not scores_npz.exists()),
            "portfolio_skipped": not all(leg in primary for leg in required_legs),
        },
        "book_capacity": {"criterion": "avg_gross_exposure>=0.6 AND impact_cost_share<=0.60",
                          "estimated_book_capacity_cny": book_capacity,
                          "capacity_sweep": cap_rows},
        "portfolio_comparison": portfolio_results,
        "combined_capacity_cny": combined_capacity,
        "combined_capacity_inherit_basis": {
            "book_capacity_cny": book_capacity, "futures_sleeve_cny": sleeve_cap,
            "hk_passive_cny": hk_cap},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"=== P10g 完成 ({__import__('time').time()-t0:.0f}s) ===")
    log(f"book capacity ≈ {book_capacity:,.0f}; combined ≈ {combined_capacity:,.0f}")


if __name__ == "__main__":
    main()
