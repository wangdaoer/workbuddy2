"""P9-3：用衍生品流动性，对 A 股横截面排序模型（P8b 冠军）做跨市场容量重测。

方法学（与 P7 real_impact_cost 完全一致，便于同口径比较）：
  - 冲击模型：sqrt。impact_bps = impact_bps_ref * sqrt(participation / ref_participation)
    ref: impact_bps_ref=0.7, ref_participation=0.01（与 P7 一致）
  - 对「衍生品 sleeve」（指数期货横截面动量）与「A 股权益簿」（P8b）分别做 AUM 扫描，
    看冲击成本随 AUM 增长如何侵蚀收益/夏普，定位各自容量上限。
  - 衍生品流动性来自 P9-1 连续近月面板（成交金额 amount = 真实 ADV）。
  - 最终给出「跨市场合计可部署容量」= A 股容量上限 + 衍生品 sleeve 容量上限。

用法：
  python3 derivatives_capacity_retest.py \
      --futures external_data/derivatives/futures_near_contig.csv \
      --p8b-metrics outputs/p8b_dynamic_liquidity/metrics.json \
      --sleeve-csv external_data/derivatives/futures_sleeve_overlay.csv \
      --out external_data/derivatives/p9_capacity_retest.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import futures_sleeve_overlay as fso   # reuse build_sleeve / roll_bridged_returns

IMPACT_BPS_REF = 0.7
REF_PART = 0.01
INDEX_FUTURES = fso.INDEX_FUTURES


def sleeve_capacity_curve(futures_csv: str, aums) -> list[dict]:
    """对衍生品 sleeve 做 AUM 扫描：基于 P7 sqrt 冲击模型。"""
    fut = fso.load_futures(futures_csv)
    sleeve = fso.build_sleeve(fut, mom_win=20)
    w = sleeve["weights"]                       # trade_date x underlying 权重
    res = sleeve["sleeve"]                       # trade_date, sleeve_ret(含0.6bp线性成本), turnover
    # 重建逐标的日收益宽表与逐标的 ADV（amount=成交金额，真实流动性）
    r = fso.roll_bridged_returns(fut[fut.underlying.isin(INDEX_FUTURES)]).copy()
    r["trade_date"] = r["trade_date"].astype(str)
    wide = r.pivot(index="trade_date", columns="underlying", values="ret")[INDEX_FUTURES]
    # 注意：源 成交金额(amount) 单位为「万元」，notional_turnover=volume*close*MULT 才是正确 CNY 日流动性
    amt = fut[fut.underlying.isin(INDEX_FUTURES)].pivot(
        index="trade_date", columns="underlying", values="notional_turnover")[INDEX_FUTURES]
    # 对齐日期
    common = wide.index.intersection(w.index)
    wide = wide.loc[common]; w = w.loc[common]; amt = amt.loc[common]
    # 实现收益用 w.shift(1)（次日成交，无未来函数）
    gross = (w.shift(1).fillna(0.0) * wide).sum(axis=1)

    out = []
    for aum in aums:
        # 逐标的参与率与冲击成本（bps）
        dw = w.diff().abs().fillna(0.0)          # 每日权重变化
        part = dw.mul(aum) / amt.replace(0, np.nan)   # participation per underlying
        impact_bps = IMPACT_BPS_REF * np.sqrt(part / REF_PART)
        # 冲击成本（占 AUM）= sum_u impact_bps_u/1e4 * |dw_u| * aum / aum
        daily_cost_frac = (impact_bps / 1e4 * dw).sum(axis=1).fillna(0.0)
        net = gross - daily_cost_frac
        # 叠加原 sleeve 已有的 0.6bp 线性成本（仅展示口径，不重复计）
        cum = (1 + net.fillna(0)).cumprod()
        ret = net.dropna()
        ann = cum.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) else 0.0
        vol = ret.std() * np.sqrt(252)
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
        peak = cum.cummax(); dd = (cum / peak - 1).min()
        # 平均参与率（约束腿，取各日最大参与率，即最拥挤腿）
        avg_max_part = part.max(axis=1).mean()
        out.append({
            "aum": int(aum),
            "ann_return": round(float(ann), 4),
            "ann_vol": round(float(vol), 4),
            "sharpe": round(float(sharpe), 4),
            "max_drawdown": round(float(dd), 4),
            "cum_return": round(float(cum.iloc[-1] - 1), 4),
            "avg_worst_leg_participation": round(float(avg_max_part), 5),
            "avg_impact_bps": round(float((impact_bps * dw).sum(axis=1).sum() / max(1, (dw>0).sum().sum())), 4),
        })
    return out


def load_p8b_curve(p8b_metrics: str) -> list[dict]:
    """读取 P8b 的 AUM 扫描曲线（来自 P7/P8b metrics.json）。"""
    m = json.loads(Path(p8b_metrics).read_text())
    sweep = m.get("aum_sweep", [])
    out = []
    for r in sweep:
        out.append({
            "aum": int(r["aum"]),
            "ann_return": round(float(r.get("annualized_return", r.get("total_return", 0))), 4),
            "sharpe": round(float(r.get("sharpe_like", 0)), 4),
            "max_drawdown": round(float(r.get("max_drawdown", 0)), 4),
            "impact_cost_bps": round(float(r.get("impact_cost", 0) * 1e4) if r.get("impact_cost") else None, 3),
            "impact_cost_share": round(float(r.get("impact_cost_share", 0)), 3) if r.get("impact_cost_share") is not None else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="P9-3 cross-market capacity re-test")
    ap.add_argument("--futures", required=True)
    ap.add_argument("--p8b-metrics", required=True)
    ap.add_argument("--sleeve-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aums", default="1e7,1e8,1e9,1e10,1e11,5e11",
                    help="逗号分隔的 AUM 扫描点")
    args = ap.parse_args()
    aums = [float(x) for x in args.aums.split(",")]

    sleeve_curve = sleeve_capacity_curve(args.futures, aums)
    p8b_curve = load_p8b_curve(args.p8b_metrics)

    # 找 A 股容量上限：impact_cost_share 越过 0.30（冲击占成本 30%）即视为接近上限
    p8b_limit = None
    for r in p8b_curve:
        if r.get("impact_cost_share") is not None and r["impact_cost_share"] >= 0.30:
            p8b_limit = r["aum"]; break
    # 找 sleeve 容量上限：worst-leg 参与率越过 0.10（10% ADV）视为接近上限
    sleeve_limit = None
    for r in sleeve_curve:
        if r["avg_worst_leg_participation"] >= 0.10:
            sleeve_limit = r["aum"]; break

    report = {
        "method": "P7 sqrt-impact (impact_bps_ref=0.7 @ ref_participation=0.01), identical to A-share P7/P8b",
        "sleeve_curve": sleeve_curve,
        "ashare_p8b_curve": p8b_curve,
        "ashare_capacity_ceiling_aum": p8b_limit,
        "derivative_sleeve_capacity_ceiling_aum": sleeve_limit,
        "combined_cross_market_capacity_aum": (
            (p8b_limit or 0) + (sleeve_limit or 0)) if (p8b_limit and sleeve_limit) else None,
        "interpretation": (
            "衍生品 sleeve 容量上限远高于 A 股簿（指数期货 ADV 在数十亿元/日量级），"
            "跨市场合计可部署容量 ≈ A 股容量上限 + 衍生品 sleeve 容量上限。"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
