"""P10h：容量工程 —— 放大 A 股簿（soft）可部署容量的参数权衡扫描。

P10g 结论：在 1% ADV 参与度上限 + top_n=20 下，book 真实可部署容量仅 ~1e7-1e8（avg_gross
在 1e8 已降至 0.53）。本脚本固定 p10e_soft 打分，扫描三组合杆寻找容量-夏普前沿：

  - top_n          : 持仓名数（更多名 = 更多可部署流动性 aggregate）
  - max_position   : 单票权重上限（更高 = 单票可承载更多资金）
  - participation  : ADV 参与度上限（更高 = 单票可承载 2x 资金，但冲击成本上升）

对每个配置在 AUM ∈ {1e8,5e8,1e9,3e9} 跑 walk-forward，测：
  avg_gross（部署率，目标 0.8）、sharpe、impact_share、capacity_blocked_buy_weight。
定位「部署率 >= 0.6（即 avg_gross>=0.6）」的容量拐点 AUM_knee，并报告该点的 sharpe / impact_share。
目标：找到在可接受夏普(>=0.8)与冲击(<0.5)下、容量最大的配置。

复用 production_soft_score 的面板构建 + 因果 soft 分（linear_mlp_scores.npz），仅改回测参数。
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import (
    build_features, calculate_walk_forward_metrics, clean_matrix,
    daily_ic, load_market_exposure, run_walk_forward,
)
from p10c_ensemble import (
    TRAIN_DAYS, MTH, LIQ_LOOKBACK, THR as LIQ_THR, RETRAIN,
)
from production_soft_score import build_panel, causal_soft_blend, SCORES_NPZ

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = HERE / "outputs" / "p10h_capacity_engineering"
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"

MAX_ABS = 0.22
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
REBALANCE = 1
AUMS = [100_000_000.0, 500_000_000.0, 1_000_000_000.0, 3_000_000_000.0]
DEPLOY_TARGET = 0.8
DEPLOY_FLOOR = 0.6  # 容量拐点判定：avg_gross >= 0.6 视为充分部署

CONFIGS = [
    {"top_n": 20, "w": 0.04, "p": 0.01},
    {"top_n": 20, "w": 0.06, "p": 0.01},
    {"top_n": 20, "w": 0.04, "p": 0.02},
    {"top_n": 20, "w": 0.06, "p": 0.02},
    {"top_n": 40, "w": 0.04, "p": 0.01},
    {"top_n": 40, "w": 0.06, "p": 0.01},
    {"top_n": 40, "w": 0.04, "p": 0.02},
    {"top_n": 40, "w": 0.06, "p": 0.02},
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    RUN_LOG.write_text("", encoding="utf-8")
    log("=== P10h 容量工程扫描 ===")
    t0 = time.time()
    P = build_panel(PANEL)
    # 复用已存分数
    import numpy as np
    d = np.load(SCORES_NPZ, allow_pickle=True)
    linear_score = pd.DataFrame(d["linear"], index=P["label"].index, columns=P["symbols"])
    mlp_score = pd.DataFrame(d["mlp"], index=P["label"].index, columns=P["symbols"])
    soft, trailing, adv, alpha_dead, mlp_ic = causal_soft_blend(
        linear_score, mlp_score, P["label"], P["symbols"])
    ic_run = daily_ic({"soft": soft}, P["label"])

    results = []
    for cfg in CONFIGS:
        log(f"\n--- top_n={cfg['top_n']} w={cfg['w']} p={cfg['p']} ---")
        row = {"config": cfg, "aum_sweep": []}
        for aum in AUMS:
            eq, _, _ = run_walk_forward(
                close=P["close"], open_px=P["open_px"], features={"soft": soft}, label=P["label"], ic=ic_run,
                train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=cfg["top_n"],
                rebalance_frequency=REBALANCE, max_position_weight=cfg["w"], leverage=1.0,
                commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
                impact_model="sqrt", impact_ref_participation=IMPACT_REF,
                max_buy_open_gap=0.06, limit_buffer=0.995, market_exposure=P["market_exposure"],
                initial_capital=aum, max_training_horizon=MTH, feature_directions=None,
                amount=P["amount"], feature_selection=None, max_daily_amount_participation=cfg["p"],
                liquid_mask=None,
            )
            m = calculate_walk_forward_metrics(eq, aum)
            avg_gross = float(eq["gross_exposure"].mean())
            turnover = float(eq["turnover"].sum())
            commission = turnover * COMMISSION_BPS / 1e4
            total_cost = float(eq["cost"].sum())
            impact_share = float((total_cost - commission) / total_cost) if total_cost > 0 else 0.0
            cap_blocked = float(eq["capacity_blocked_buy_weight"].sum())
            cap_sessions = int((eq["capacity_limited_symbols"] > 0).sum())
            rec = {"aum": aum, "sharpe": round(float(m.get("sharpe_like")), 4),
                   "total_return": round(float(m.get("total_return")), 4),
                   "avg_gross": round(avg_gross, 4), "impact_share": round(impact_share, 4),
                   "cap_blocked": round(cap_blocked, 2), "cap_sessions": cap_sessions}
            row["aum_sweep"].append(rec)
            log(f"  AUM={aum:>13,.0f}: shp={rec['sharpe']:.3f} avg_gross={avg_gross:.3f} "
                f"impact_share={impact_share:.3f} cap_blocked={cap_blocked:.1f} sess={cap_sessions}")
        # 容量拐点：avg_gross 跨 0.6 的 AUM（log 插值）
        gs = [r["avg_gross"] for r in row["aum_sweep"]]
        aums = [r["aum"] for r in row["aum_sweep"]]
        knee = None
        knee_sharpe = None
        knee_impact = None
        if gs[0] >= DEPLOY_FLOOR:
            # 找第一次跌破 0.6 的相邻对
            for i in range(1, len(gs)):
                if gs[i] < DEPLOY_FLOOR:
                    # 在 aums[i-1],aums[i] 间插值
                    lo, hi = np.log(aums[i-1]), np.log(aums[i])
                    gl, gh = gs[i-1], gs[i]
                    frac = (DEPLOY_FLOOR - gh) / (gl - gh)
                    knee = float(np.exp(lo + frac * (hi - lo)))
                    sf = np.log(row["aum_sweep"][i-1]["sharpe"]); hf = np.log(row["aum_sweep"][i]["sharpe"])
                    knee_sharpe = round(float(np.exp(sf + frac * (hf - sf))), 4)
                    ish = row["aum_sweep"][i-1]["impact_share"]; ihh = row["aum_sweep"][i]["impact_share"]
                    knee_impact = round(ish + frac * (ihh - ish), 4)
                    break
            if knee is None:
                knee = aums[-1]  # 全程 >=0.6，容量 >= 最大 AUM
                knee_sharpe = row["aum_sweep"][-1]["sharpe"]
                knee_impact = row["aum_sweep"][-1]["impact_share"]
        else:
            knee = aums[0]  # 起点就 <0.6，容量 < 最小 AUM
            knee_sharpe = row["aum_sweep"][0]["sharpe"]
            knee_impact = row["aum_sweep"][0]["impact_share"]
        row["capacity_knee_aum"] = knee
        row["knee_sharpe"] = knee_sharpe
        row["knee_impact_share"] = knee_impact
        log(f"  >> 容量拐点(AUM@avg_gross>=0.6) ≈ {knee:,.0f}  (knee sharpe={knee_sharpe}, impact={knee_impact})")
        results.append(row)

    # 推荐：容量最大且 knee_sharpe>=0.8 且 knee_impact<=0.5
    viable = [r for r in results if r["knee_sharpe"] >= 0.8 and r["knee_impact_share"] <= 0.5]
    viable.sort(key=lambda r: r["capacity_knee_aum"], reverse=True)
    best = viable[0] if viable else max(results, key=lambda r: r["capacity_knee_aum"])

    payload = {
        "method": "fixed p10e_soft score; sweep top_n x max_position_weight x ADV participation; "
                  "capacity knee = AUM where avg_gross>=0.6",
        "aum_grid": AUMS,
        "configs": results,
        "capacity_frontier": [
            {"config": r["config"], "capacity_knee_aum": r["capacity_knee_aum"],
             "knee_sharpe": r["knee_sharpe"], "knee_impact_share": r["knee_impact_share"]}
            for r in results
        ],
        "recommended_config": best["config"],
        "recommended_capacity_knee_aum": best["capacity_knee_aum"],
        "recommended_knee_sharpe": best["knee_sharpe"],
        "recommended_knee_impact_share": best["knee_impact_share"],
        "note": "baseline (top_n=20,w=0.04,p=0.01) capacity knee from P10g ~1e7-1e8",
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"=== P10h 完成 ({time.time()-t0:.0f}s) ===")
    log(f"推荐配置: {best['config']}  容量拐点≈{best['capacity_knee_aum']:,.0f}  "
        f"knee_sharpe={best['knee_sharpe']} impact={best['knee_impact_share']}")


if __name__ == "__main__":
    main()
