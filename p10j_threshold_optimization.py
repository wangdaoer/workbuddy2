"""P10j：阈值自适应寻优 —— 优化 regime × 容量联合调度的阈值 (THR_HI, THR_LO, FLOOR)。

P10i 用 P10e 固定阈值 (THR_HI=0.03, THR_LO=0.0, FLOOR=0.40)。本脚本用 walk-forward 校准/验证
把阈值寻优出来，并加入「自适应阈值」变体（阈值随 trailing IC 的滚动分布移动）。

防前视（沿用 P10e 的纪律）：
  - trailing IC 本身因果（shift(1)），阈值只作用其上。
  - 寻优在 CALIBRATION 窗口（前 70%）上选阈值，再应用到 VALIDATION 窗口（后 30%，含 asof
    2026-07-23 的 dead 段）做 OOS 评估。VALIDATION 不参与选阈值 -> 无泄漏。
  - 自适应变体用 rolling 统计（仅用过去），天然无泄漏。

复用 production_soft_score 的面板/因果 soft 分（linear_mlp_scores.npz）与 joint_schedule。
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_next_open_rank_model import calculate_walk_forward_metrics
from production_soft_score import (
    build_panel, causal_soft_blend, SCORES_NPZ, joint_schedule,
    assert_regime_thresholds_consistent,
    BOOK_AUM, BOOK_TOP_N, BOOK_MAX_W, BOOK_ADV_P, BOOK_KNEE_REF, MIN_DEAD_TOP_N,
    COMMISSION_BPS, IMPACT_BPS, IMPACT_REF, REBALANCE, TRAIN_DAYS, MTH, RETRAIN,
)
from train_next_open_rank_model import run_walk_forward, daily_ic  # 引擎（含 top_n_schedule）

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = HERE / "outputs" / "p10j_threshold_optimization"
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"

# P10i 固定基线
BASE_THR_HI, BASE_THR_LO, BASE_FLOOR = 0.03, 0.0, 0.40
FIXED = (BASE_THR_HI, BASE_THR_LO, BASE_FLOOR)
CALIB_FRAC = 0.70


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_schedule(P, soft, ic_run, trailing, thr_hi, thr_lo, floor, base_top_n=BOOK_TOP_N,
                 soft_thr_hi=None, soft_thr_lo=None):
    """用给定阈值跑联合调度（top_n 按容量余量动态），返回 equity。

    soft_thr_hi/soft_thr_lo 是构造 soft 分时 causal_soft_blend 实际使用的阈值，必须与 gross 闸门
    阈值 (thr_hi/thr_lo) 一致；否则触发 P10j 阈值一致性守卫（fail-fast）。默认等于闸门阈值。
    这一守卫专门拦截 P10j 曾出现的跨函数错配：main 用模块 THR_HI 混 soft、run_schedule 用扫到的
    另一 THR_HI 当闸门。
    """
    if soft_thr_hi is None:
        soft_thr_hi = thr_hi
    if soft_thr_lo is None:
        soft_thr_lo = thr_lo
    # P10j 阈值一致性守卫：soft 混合阈值 必须 == gross 闸门阈值（fail-fast，绝不静默放行）
    assert_regime_thresholds_consistent(soft_thr_hi, soft_thr_lo, thr_hi, thr_lo, context="p10j.run_schedule")
    gross, tn_sched, dead_tn, headroom = joint_schedule(
        trailing, BOOK_AUM, BOOK_KNEE_REF, base_top_n=base_top_n,
        thr_hi=thr_hi, thr_lo=thr_lo, floor=floor)
    joint_mkt = pd.Series(
        np.minimum(P["market_exposure"].reindex(gross.index).fillna(1.0).values, gross.values),
        index=gross.index)
    eq, _, _ = run_walk_forward(
        close=P["close"], open_px=P["open_px"], features={"soft": soft}, label=P["label"], ic=ic_run,
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=base_top_n,
        rebalance_frequency=REBALANCE, max_position_weight=BOOK_MAX_W, leverage=1.0,
        commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
        impact_model="sqrt", impact_ref_participation=IMPACT_REF,
        max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=joint_mkt,
        initial_capital=BOOK_AUM, max_training_horizon=MTH, feature_directions=None,
        amount=P["amount"], feature_selection=None, max_daily_amount_participation=BOOK_ADV_P,
        liquid_mask=None, top_n_schedule=tn_sched,
    )
    return eq


def adaptive_schedule(trailing, floor, win=252, k=0.5, base_top_n=BOOK_TOP_N,
                      min_dead=MIN_DEAD_TOP_N, knee_ref=BOOK_KNEE_REF, aum=BOOK_AUM):
    """自适应阈值：THR_HI/THR_LO 取 trailing IC 滚动分布 (mu ± k·sd)，仅用过去 -> 无泄漏。"""
    t = trailing.fillna(0.0)
    mu = t.rolling(win, min_periods=120).mean()
    sd = t.rolling(win, min_periods=120).std()
    thr_hi = (mu + k * sd)
    thr_lo = (mu - k * sd)
    gross = np.where(t <= thr_lo, floor,
             np.where(t < thr_hi, floor + (1 - floor) * (t - thr_lo) / (thr_hi - thr_lo), 1.0))
    gross = pd.Series(gross, index=trailing.index)
    headroom = float(np.clip(1.0 - aum / knee_ref, 0.0, 1.0))
    dead_tn = int(np.clip(round(base_top_n - (base_top_n - min_dead) * headroom), min_dead, base_top_n))
    tn = np.where(t <= thr_lo, dead_tn,
          np.where(t < thr_hi, dead_tn + (base_top_n - dead_tn) * (t - thr_lo) / (thr_hi - thr_lo),
                   base_top_n))
    tn = pd.Series(np.round(tn).astype(int), index=trailing.index)
    return gross, tn, dead_tn


def run_adaptive(P, soft, ic_run, trailing, floor, win=252, k=0.5):
    gross, tn_sched, dead_tn = adaptive_schedule(trailing, floor, win=win, k=k)
    joint_mkt = pd.Series(
        np.minimum(P["market_exposure"].reindex(gross.index).fillna(1.0).values, gross.values),
        index=gross.index)
    eq, _, _ = run_walk_forward(
        close=P["close"], open_px=P["open_px"], features={"soft": soft}, label=P["label"], ic=ic_run,
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=BOOK_TOP_N,
        rebalance_frequency=REBALANCE, max_position_weight=BOOK_MAX_W, leverage=1.0,
        commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
        impact_model="sqrt", impact_ref_participation=IMPACT_REF,
        max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=joint_mkt,
        initial_capital=BOOK_AUM, max_training_horizon=MTH, feature_directions=None,
        amount=P["amount"], feature_selection=None, max_daily_amount_participation=BOOK_ADV_P,
        liquid_mask=None, top_n_schedule=tn_sched,
    )
    return eq, dead_tn


def eval_windows(eq_d, tr_eq, ret, cal_mask, val_mask, thr_lo):
    """返回 full / calib / val 的 sharpe 与 dead 期间累计亏损。

    注意：ret 必须是规范日收益 net = gross_return - cost（与 calculate_walk_forward_metrics /
    sharpe_like 同口径），不能用 equity.pct_change()（含预热首日跳变，会虚高且不自洽）。
    """
    def _sharpe(mask):
        r = ret[mask]
        if len(r) < 5 or not np.isfinite(r.std(ddof=1)) or r.std(ddof=1) == 0:
            return None
        return float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    def _dead_loss(mask):
        d = eq_d[mask]
        dead = (tr_eq.reindex(d) <= thr_lo).fillna(False).values
        if dead.sum() == 0:
            return 0.0
        return float(np.prod(1 + ret[mask][dead]) - 1)
    return {
        "full_sharpe": _sharpe(np.ones(len(ret), dtype=bool)),
        "cal_sharpe": _sharpe(cal_mask),
        "val_sharpe": _sharpe(val_mask),
        "cal_dead_loss": _dead_loss(cal_mask),
        "val_dead_loss": _dead_loss(val_mask),
    }


def main():
    RUN_LOG.write_text("", encoding="utf-8")
    log("=== P10j 阈值自适应寻优 ===")
    t0 = time.time()
    P = build_panel(PANEL)
    d = np.load(SCORES_NPZ, allow_pickle=True)
    linear_score = pd.DataFrame(d["linear"], index=P["label"].index, columns=P["symbols"])
    mlp_score = pd.DataFrame(d["mlp"], index=P["label"].index, columns=P["symbols"])
    soft, trailing, adv, alpha_dead, mlp_ic = causal_soft_blend(
        linear_score, mlp_score, P["label"], P["symbols"])
    ic_run = daily_ic({"soft": soft}, P["label"])

    # 日期拆分（校准 70% / 验证 30%，验证含 dead 尾段）
    dates = list(trailing.index)
    n = len(dates)
    split = int(n * CALIB_FRAC)
    cal_dates = set(dates[:split])
    val_dates = set(dates[split:])
    log(f"面板 {n} 日，校准 {split} / 验证 {n-split}（验证含 asof dead 段）")

    # 网格
    GRID = []
    for thi in (0.02, 0.03, 0.04, 0.05, 0.06):
        for fl in (0.25, 0.30, 0.35, 0.40, 0.50):
            GRID.append((thi, 0.0, fl))          # THR_LO 固定 0（dead=负 IC，绝对规则）；并向 0.05/0.06 延伸以验证 0.04 是否真为边界最优
    for tlo in (-0.01, 0.01):
        GRID.append((0.03, tlo, 0.40))            # THR_LO 敏感性
    log(f"网格 {len(GRID)} 组合 + 自适应变体")

    results = []
    for (thi, tlo, fl) in GRID:
        # 关键：soft 混合与 gross 闸门必须用【同一个】THR_HI/THR_LO 一起扫，否则出现
        # 混合阈值与闸门阈值错配的脏结果（早期脏跑即因此：soft 用 0.03、闸门用 0.04）。
        soft_i, trailing_i, adv_i, alpha_dead_i, mlp_ic_i = causal_soft_blend(
            linear_score, mlp_score, P["label"], P["symbols"], thr_hi=thi, thr_lo=tlo)
        ic_run_i = daily_ic({"soft": soft_i}, P["label"])
        # soft 混合与 gross 闸门用同一 (thi, tlo) —— 一致性守卫在 run_schedule 内校验
        eq = run_schedule(P, soft_i, ic_run_i, trailing_i, thi, tlo, fl,
                         soft_thr_hi=thi, soft_thr_lo=tlo)
        eq_d = pd.to_datetime(eq["date"])
        ret = (eq["gross_return"] - eq["cost"]).fillna(0.0).values
        tr_eq = trailing.reindex(eq_d)
        cal_mask = eq_d.isin(cal_dates).values
        val_mask = eq_d.isin(val_dates).values
        m = eval_windows(eq_d, tr_eq, ret, cal_mask, val_mask, tlo)
        m.update({"thr_hi": thi, "thr_lo": tlo, "floor": fl})
        results.append(m)
        log(f"  hi={thi:.3f} lo={tlo:+.3f} fl={fl:.2f} | "
            f"full={m['full_sharpe']:.3f} cal={m['cal_sharpe']:.3f} val={m['val_sharpe']:.3f} | "
            f"val_dead_loss={m['val_dead_loss']:+.3f}")

    # 自适应变体（floor 用 0.40 基准）
    eqA, dead_tnA = run_adaptive(P, soft, ic_run, trailing, 0.40)
    eq_dA = pd.to_datetime(eqA["date"])
    retA = (eqA["gross_return"] - eqA["cost"]).fillna(0.0).values
    tr_eqA = trailing.reindex(eq_dA)
    cal_maskA = eq_dA.isin(cal_dates).values
    val_maskA = eq_dA.isin(val_dates).values
    mA = eval_windows(eq_dA, tr_eqA, retA, cal_maskA, val_maskA, 0.0)
    mA.update({"thr_hi": "adaptive", "thr_lo": "adaptive", "floor": 0.40, "dead_top_n": dead_tnA})
    log(f"  自适应(k=0.5,win=252) | full={mA['full_sharpe']:.3f} cal={mA['cal_sharpe']:.3f} "
        f"val={mA['val_sharpe']:.3f} | val_dead_loss={mA['val_dead_loss']:+.3f}")

    # 选优：在验证窗口 OOS 上，取 val_sharpe 最高且 dead 亏损不差于固定基线（亏损更不负或相等）的组合。
    # 注意 dead_loss 为负（亏损）；"不恶化" = val_dead_loss >= 基线（更不负=更好，更负=更差）。
    fixed_row = next(r for r in results if (r["thr_hi"], r["thr_lo"], r["floor"]) == FIXED)
    base_val_loss = fixed_row["val_dead_loss"]
    base_val_shp = fixed_row["val_sharpe"]
    log(f"固定基线 (0.03/0/0.40): val_sharpe={base_val_shp:.3f} val_dead_loss={base_val_loss:+.3f}")
    candidates = [r for r in results if r["val_dead_loss"] >= base_val_loss - 1e-9]
    best = max(candidates, key=lambda r: r["val_sharpe"]) if candidates else max(results, key=lambda r: r["val_sharpe"])
    log(f"推荐阈值: hi={best['thr_hi']:.3f} lo={best['thr_lo']:+.3f} fl={best['floor']:.2f} | "
        f"val_sharpe={best['val_sharpe']:.3f} val_dead_loss={best['val_dead_loss']:+.3f}")

    out = {
        "fixed_baseline": fixed_row,
        "best_grid": best,
        "adaptive": mA,
        "grid": results,
        "calib_frac": CALIB_FRAC,
        "n_days": n, "split": split,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"耗时 {out['elapsed_min']} 分钟；指标: {OUT / 'metrics.json'}")
    make_chart(results, fixed_row, best, mA, OUT)


def make_chart(results, fixed_row, best, mA, OUT):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # full sharpe heat-ish: bar by combo
    labels = [f"{r['thr_hi']:.2f}/{r['thr_lo']:+.2f}/{r['floor']:.2f}" for r in results]
    full = [r["full_sharpe"] for r in results]
    val = [r["val_sharpe"] for r in results]
    x = np.arange(len(results))
    axes[0].bar(x - 0.2, full, 0.4, label="full", color="#1f77b4")
    axes[0].bar(x + 0.2, val, 0.4, label="val(OOS)", color="#ff7f0e")
    axes[0].axhline(fixed_row["full_sharpe"], color="gray", ls=":", label="fixed full")
    axes[0].axhline(fixed_row["val_sharpe"], color="red", ls=":", label="fixed val")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[0].set_title("Sharpe by threshold combo"); axes[0].legend(fontsize=7)

    # val dead loss
    axes[1].bar(x, [r["val_dead_loss"] for r in results], color="#2ca02c")
    axes[1].axhline(fixed_row["val_dead_loss"], color="red", ls=":", label="fixed")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[1].set_title("Validation dead-period loss"); axes[1].legend(fontsize=7)

    # best vs fixed vs adaptive summary
    names = ["fixed", "best(grid)", "adaptive"]
    fsh = [fixed_row["full_sharpe"], best["full_sharpe"], mA["full_sharpe"]]
    vsh = [fixed_row["val_sharpe"], best["val_sharpe"], mA["val_sharpe"]]
    xx = np.arange(3)
    axes[2].bar(xx - 0.2, fsh, 0.4, label="full", color="#1f77b4")
    axes[2].bar(xx + 0.2, vsh, 0.4, label="val(OOS)", color="#ff7f0e")
    axes[2].set_xticks(xx); axes[2].set_xticklabels(names, fontsize=9)
    axes[2].set_title("fixed vs optimized vs adaptive"); axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_optimization.png", dpi=130)
    log(f"图表: {OUT / 'threshold_optimization.png'}")


if __name__ == "__main__":
    main()
