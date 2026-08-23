"""P10i：regime × 容量 联合调度实验。

P10h 报告第 10 节可选项 2 —— 在 dead 状态下不仅降 gross，还按容量余量动态下调 top_n（用更少名数
降低冲击敏感度）。本脚本把 P10e 的 regime 监控信号变成「可执行的部署调度」，并对比三种模式：

  A) static      : 当前生产行为。top_n=40 固定，market_exposure=风险偏好(risk_off) 仅。
                   （注意：当前生产簿实际上【没有】在回测里执行 dead->0.40 降权，0.40 仅是告警。）
  B) regime-gross: 仅把 regime 的 gross 降权写进回测（dead->0.40，decaying 插值），top_n 仍 40。
  C) joint       : regime->gross 降权 + top_n 按容量余量动态下调（dead 时收窄到 20~40 之间）。

关键实现：
  - gross 轴：market_exposure 改为 min(risk_off, regime_gross_scale) 的逐日序列（引擎原生支持）。
  - top_n 轴：run_walk_forward 新增 top_n_schedule 逐日序列（已向后兼容，默认 None=原行为）。
  - 容量余量耦合：headroom = clip(1 - AUM/knee_ref, 0, 1)；dead_top_n = base - (base-min_dead)*headroom
    -> 容量充裕(AUM<<knee)时 dead 可砍到 min_dead(冲击最敏感)，容量紧张(AUM~knee)时不砍(避免爆容量)。

复用 production_soft_score 的面板/因果 soft 分（linear_mlp_scores.npz）。
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import (
    build_features, calculate_walk_forward_metrics, clean_matrix,
    daily_ic, load_market_exposure, run_walk_forward,
)
from p10c_ensemble import TRAIN_DAYS, MTH, LIQ_LOOKBACK, THR as LIQ_THR, RETRAIN
from production_soft_score import build_panel, causal_soft_blend, SCORES_NPZ

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
OUT = HERE / "outputs" / "p10i_joint_scheduling"
OUT.mkdir(parents=True, exist_ok=True)

MAX_ABS = 0.22
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
REBALANCE = 1
BOOK_AUM = 100_000_000.0
# regime 参数（与 P10e/P10g 一致，因果 shift(1)）
IC_WIN = 60
IC_MIN = 40
THR_HI = 0.03
THR_LO = 0.0
EXPOSURE_FLOOR = 0.40
# 容量工程（P10h）推荐配置参考
BASE_TOP_N = 40
MIN_DEAD_TOP_N = 20
W = 0.04
ADV_P = 0.02
KNEE_REF = 451_469_068.0  # P10h 推荐配置容量拐点（top_n=40/ADV=2%）


def joint_schedule(trailing: pd.Series, aum: float, knee_ref: float,
                   base_top_n: int = BASE_TOP_N, min_dead_top_n: int = MIN_DEAD_TOP_N,
                   thr_hi: float = THR_HI, thr_lo: float = THR_LO, floor: float = EXPOSURE_FLOOR):
    """返回 (gross_scale, top_n_schedule, dead_top_n, headroom)。"""
    t = trailing.fillna(0.0)
    # gross 轴：dead->floor；decaying 在 (floor,1) 间插值；healthy->1
    gross = np.where(t <= thr_lo, floor,
             np.where(t < thr_hi, floor + (1 - floor) * (t - thr_lo) / (thr_hi - thr_lo), 1.0))
    gross = pd.Series(gross, index=trailing.index)
    # 容量余量
    headroom = float(np.clip(1.0 - aum / knee_ref, 0.0, 1.0))
    # dead 时 top_n：容量充裕->min_dead（冲击最敏感），紧张->base（不砍）
    dead_tn = int(round(base_top_n - (base_top_n - min_dead_top_n) * headroom))
    dead_tn = int(np.clip(dead_tn, min_dead_top_n, base_top_n))
    # top_n 调度：dead->dead_tn；decaying 插值；healthy->base
    tn = np.where(t <= thr_lo, dead_tn,
          np.where(t < thr_hi, dead_tn + (base_top_n - dead_tn) * (t - thr_lo) / (thr_hi - thr_lo),
                   base_top_n))
    tn = pd.Series(np.round(tn).astype(int), index=trailing.index)
    return gross, tn, dead_tn, headroom


def run_mode(P, soft, ic_run, top_n, top_n_schedule, mkt_exp):
    eq, _, _ = run_walk_forward(
        close=P["close"], open_px=P["open_px"], features={"soft": soft}, label=P["label"], ic=ic_run,
        train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=top_n,
        rebalance_frequency=REBALANCE, max_position_weight=W, leverage=1.0,
        commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
        impact_model="sqrt", impact_ref_participation=IMPACT_REF,
        max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=mkt_exp,
        initial_capital=BOOK_AUM, max_training_horizon=MTH, feature_directions=None,
        amount=P["amount"], feature_selection=None, max_daily_amount_participation=ADV_P,
        liquid_mask=None, top_n_schedule=top_n_schedule,
    )
    return eq


def mode_metrics(eq, aum):
    m = calculate_walk_forward_metrics(eq, aum)
    avg_gross = float(eq["gross_exposure"].mean())
    turnover = float(eq["turnover"].sum())
    commission = turnover * COMMISSION_BPS / 1e4
    total_cost = float(eq["cost"].sum())
    impact_share = float((total_cost - commission) / total_cost) if total_cost > 0 else 0.0
    avg_topn = float(eq["positions_count"].mean())
    return {
        "sharpe": round(float(m.get("sharpe_like")), 4),
        "total_return": round(float(m.get("total_return")), 4),
        "max_drawdown": round(float(m.get("max_drawdown")), 4),
        "avg_gross": round(avg_gross, 4),
        "avg_top_n": round(avg_topn, 2),
        "turnover": round(turnover, 1),
        "impact_share": round(impact_share, 4),
        "n_days": int(len(eq)),
    }


def regime_split_ret(eq, status):
    """按 regime 状态拆分逐日净收益，返回 dead/decaying/healthy 子样本的累计与波动。"""
    d = pd.to_datetime(eq["date"])
    ret = eq["equity"].pct_change().fillna(0.0).values
    s = status.reindex(d).fillna("healthy").values
    out = {}
    for lab in ["dead", "decaying", "healthy"]:
        mask = s == lab
        r = ret[mask]
        if len(r) > 1 and r.std() > 0:
            cum = float(np.prod(1 + r) - 1)
            sharpe = float(r.mean() / r.std() * np.sqrt(252))
        else:
            cum = float(np.prod(1 + r) - 1) if len(r) else 0.0
            sharpe = 0.0
        out[lab] = {"n_days": int(mask.sum()), "cum_return": round(cum, 4), "sharpe": round(sharpe, 3)}
    return out


def main():
    print("=== P10i regime × 容量 联合调度实验 ===")
    P = build_panel(PANEL)
    d = np.load(SCORES_NPZ, allow_pickle=True)
    linear_score = pd.DataFrame(d["linear"], index=P["label"].index, columns=P["symbols"])
    mlp_score = pd.DataFrame(d["mlp"], index=P["label"].index, columns=P["symbols"])
    soft, trailing, adv, alpha_dead, mlp_ic = causal_soft_blend(
        linear_score, mlp_score, P["label"], P["symbols"])
    ic_run = daily_ic({"soft": soft}, P["label"])
    risk_off = P["market_exposure"]

    # regime 状态（因果）
    def _cls(v):
        if pd.isna(v):
            return "insufficient_history"
        if v > THR_HI:
            return "healthy"
        if v > THR_LO:
            return "decaying"
        return "dead"
    status = trailing.apply(_cls)

    gross, tn_sched, dead_tn, headroom = joint_schedule(trailing, BOOK_AUM, KNEE_REF)
    joint_mkt = pd.Series(np.minimum(risk_off.reindex(gross.index).fillna(1.0).values, gross.values),
                          index=gross.index)

    print(f"  AUM={BOOK_AUM/1e8:.0f}e8  knee_ref={KNEE_REF/1e8:.2f}e8  headroom={headroom:.3f}  "
          f"dead_top_n={dead_tn}")
    print(f"  dead 天数={int((status=='dead').sum())}  decaying={int((status=='decaying').sum())}  "
          f"healthy={int((status=='healthy').sum())}")
    print(f"  top_n 调度范围: {int(tn_sched.min())}~{int(tn_sched.max())}  平均 {tn_sched.mean():.1f}")

    # 模式 A: static（当前生产）
    eqA = run_mode(P, soft, ic_run, BASE_TOP_N, None, risk_off)
    # 模式 B: regime-gross only
    eqB = run_mode(P, soft, ic_run, BASE_TOP_N, None, joint_mkt)
    # 模式 C: joint
    eqC = run_mode(P, soft, ic_run, BASE_TOP_N, tn_sched, joint_mkt)

    mA = mode_metrics(eqA, BOOK_AUM)
    mB = mode_metrics(eqB, BOOK_AUM)
    mC = mode_metrics(eqC, BOOK_AUM)
    print("\n模式对比 @ AUM=1e8:")
    print(f"  {'mode':<14}{'sharpe':>8}{'total':>8}{'mdd':>8}{'avgGross':>9}{'avgTopN':>8}{'impact':>8}")
    for name, m in [("A static", mA), ("B reg-gross", mB), ("C joint", mC)]:
        print(f"  {name:<14}{m['sharpe']:>8.3f}{m['total_return']:>8.3f}{m['max_drawdown']:>8.3f}"
              f"{m['avg_gross']:>9.3f}{m['avg_top_n']:>8.1f}{m['impact_share']:>8.3f}")

    # regime 拆分
    sA = regime_split_ret(eqA, status)
    sB = regime_split_ret(eqB, status)
    sC = regime_split_ret(eqC, status)
    print("\nregime 子样本净收益（dead 期间是关键）：")
    print(f"  {'mode':<14}{'dead_ret':>10}{'dead_shp':>9}{'dec_ret':>9}{'hlth_ret':>9}")
    for name, s in [("A static", sA), ("B reg-gross", sB), ("C joint", sC)]:
        print(f"  {name:<14}{s['dead']['cum_return']:>10.3f}{s['dead']['sharpe']:>9.3f}"
              f"{s['decaying']['cum_return']:>9.3f}{s['healthy']['cum_return']:>9.3f}")

    # 保存
    result = {
        "aum": BOOK_AUM, "knee_ref": KNEE_REF, "headroom": round(headroom, 4),
        "dead_top_n": dead_tn, "dead_days": int((status == "dead").sum()),
        "decaying_days": int((status == "decaying").sum()),
        "healthy_days": int((status == "healthy").sum()),
        "modes": {"A_static": mA, "B_regime_gross": mB, "C_joint": mC},
        "regime_split": {"A_static": sA, "B_regime_gross": sB, "C_joint": sC},
    }
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    idxA = pd.to_datetime(eqA["date"]); eqA_v = eqA["equity"].values / eqA["equity"].iloc[0]
    idxB = pd.to_datetime(eqB["date"]); eqB_v = eqB["equity"].values / eqB["equity"].iloc[0]
    idxC = pd.to_datetime(eqC["date"]); eqC_v = eqC["equity"].values / eqC["equity"].iloc[0]
    ax1.plot(idxA, eqA_v, label=f"A static (shp={mA['sharpe']:.2f})", color="#888888", lw=1.2)
    ax1.plot(idxB, eqB_v, label=f"B reg-gross (shp={mB['sharpe']:.2f})", color="#1f77b4", lw=1.2)
    ax1.plot(idxC, eqC_v, label=f"C joint (shp={mC['sharpe']:.2f})", color="#ff7f0e", lw=1.4)
    # dead shading
    dead_dates = pd.to_datetime(status[status == "dead"].index)
    for dd in dead_dates:
        ax1.axvline(dd, color="red", alpha=0.04, lw=0.5)
    ax1.set_title("Book equity: static vs regime-gross vs joint scheduling (AUM=1e8)")
    ax1.set_ylabel("cum return (norm)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(gross.index, gross.values, label="regime gross scale", color="#1f77b4", lw=1.0)
    ax2.plot(tn_sched.index, tn_sched.values / 40.0, label="top_n schedule (/40)", color="#ff7f0e", lw=1.0)
    ax2.plot(risk_off.index, risk_off.values, label="risk_off exposure", color="#888888", lw=0.8, ls="--")
    ax2.axhline(EXPOSURE_FLOOR, color="red", ls=":", lw=1.0, label="0.40 floor")
    ax2.set_title("Joint schedule signals (gross scale + top_n/40 + risk_off)")
    ax2.set_ylabel("scale / norm"); ax2.legend(fontsize=8, loc="lower left"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "joint_scheduling.png", dpi=130)
    print(f"\n图表: {OUT / 'joint_scheduling.png'}")
    print(f"指标: {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
