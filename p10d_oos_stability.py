"""P10d：MLP 架构升级的滚动样本外（OOS）稳定性验证。

P10c 发现 MLP-only 全样本显著优于线性 incumbent，但「后半段收益」为负且 MLP 后半段
跑输 incumbent——疑似对早期 regime 过拟合。本脚本复用 P10c 已保存的 16 条权益曲线
（4 AUM × 4 变体），做时间分段分解，判断优势是否前置：

  - 按自然年（2023/2024/2025/2026）
  - 按 4 个等分时间窗（W1..W4，自早至晚）
  - 前半段 vs 后半段 Sharpe

若为「前置过拟合」：mlp_only/ensemble_ew 的 Sharpe 优势应集中在 W1/W2（早期），
在 W3/W4（后期）消失甚至转负、跑输 incumbent。
若为「真实泛化」：优势应在各窗口（尤其后期）持续。

指标口径与 calculate_walk_forward_metrics 一致：sharpe_like(净收益序列)，
年/窗内总收益 = prod(1+净收益)-1。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_backtest import sharpe_like

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "p10c_ensemble"
REP = HERE.parent / "跨市场OOS稳定性_P10d_报告.md"

AUMS = [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]
VARIANTS = ["incumbent", "mlp_only", "ensemble_ew", "ensemble_icw"]
SPLIT = 440  # 与 P10c 一致的后半段切分


def load_net(aum: float, variant: str) -> pd.DataFrame:
    p = OUT / f"equity_aum_{int(aum)}_{variant}.csv"
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["net"] = df["gross_return"] - df["cost"]
    return df


def seg_stats(net: pd.Series) -> dict:
    total = float(np.prod(1.0 + net) - 1.0)
    shr = float(sharpe_like(net)) if len(net) > 5 else float("nan")
    return {"total_return": total, "sharpe": shr, "n_days": int(len(net))}


def main() -> None:
    report = {"by_year": {}, "by_quarter_block": {}, "first_vs_second_half": {}}
    print("=" * 100)
    print("P10d：MLP 架构升级的滚动 OOS 稳定性验证（基于 P10c 已存权益曲线）")
    print("=" * 100)

    for aum in AUMS:
        print(f"\n########## AUM = {aum:,.0f} ##########")
        dfs = {v: load_net(aum, v) for v in VARIANTS}

        # 按年
        print("\n--- 按自然年 Sharpe（行=变体，列=年）---")
        years = sorted({d.year for v in VARIANTS for d in dfs[v]["date"]})
        yr_tbl = {}
        for v in VARIANTS:
            row = {}
            for y in years:
                sub = dfs[v][dfs[v]["date"].dt.year == y]["net"]
                row[str(y)] = seg_stats(sub)
            yr_tbl[v] = row
        # 打印
        hdr = "variant".ljust(12) + "".join(f"{y:>16}" for y in years)
        print(hdr)
        for v in VARIANTS:
            line = v.ljust(12)
            for y in years:
                s = yr_tbl[v][str(y)]
                line += f"{s['sharpe']:>16.3f}"
            print(line)
        print("  (总收益口径，mlp-incumbent 年度 Sharpe 差)：")
        for y in years:
            diff = yr_tbl["mlp_only"][str(y)]["sharpe"] - yr_tbl["incumbent"][str(y)]["sharpe"]
            print(f"    {y}: mlp-incumbent Sharpe Δ = {diff:+.3f}")
        report["by_year"][str(aum)] = yr_tbl

        # 4 等分窗口
        print("\n--- 按 4 等分时间窗 Sharpe（W1 最早 … W4 最晚）---")
        n = len(dfs["incumbent"])
        edges = [0, n // 4, n // 2, 3 * n // 4, n]
        w_tbl = {}
        for v in VARIANTS:
            row = {}
            for k in range(4):
                sub = dfs[v]["net"].iloc[edges[k]:edges[k + 1]]
                row[f"W{k+1}"] = seg_stats(sub)
            w_tbl[v] = row
        hdr = "variant".ljust(12) + "".join(f"{f'W{k+1}':>16}" for k in range(4))
        print(hdr)
        for v in VARIANTS:
            line = v.ljust(12)
            for k in range(4):
                line += f"{w_tbl[v][f'W{k+1}']['sharpe']:>16.3f}"
            print(line)
        print("  mlp-incumbent 窗口 Sharpe 差（W1→W4）：",
              " ".join(f"{w_tbl['mlp_only'][f'W{k+1}']['sharpe']-w_tbl['incumbent'][f'W{k+1}']['sharpe']:+.3f}" for k in range(4)))
        print("  ew-incumbent  窗口 Sharpe 差（W1→W4）：",
              " ".join(f"{w_tbl['ensemble_ew'][f'W{k+1}']['sharpe']-w_tbl['incumbent'][f'W{k+1}']['sharpe']:+.3f}" for k in range(4)))
        report["by_quarter_block"][str(aum)] = w_tbl

        # 前半 vs 后半
        print("\n--- 前半段 vs 后半段（切分点 index=440）---")
        hs = {}
        for v in VARIANTS:
            first = seg_stats(dfs[v]["net"].iloc[:SPLIT])
            second = seg_stats(dfs[v]["net"].iloc[SPLIT:])
            hs[v] = {"first_half": first, "second_half": second}
            print(f"  {v:12s} 前半 Sharpe={first['sharpe']:+.3f}(tot={first['total_return']:+.3f})  "
                  f"后半 Sharpe={second['sharpe']:+.3f}(tot={second['total_return']:+.3f})")
        print("  mlp-incumbent 前半Δ={:+.3f}  后半Δ={:+.3f}".format(
            hs["mlp_only"]["first_half"]["sharpe"] - hs["incumbent"]["first_half"]["sharpe"],
            hs["mlp_only"]["second_half"]["sharpe"] - hs["incumbent"]["second_half"]["sharpe"]))
        print("  ew -incumbent 前半Δ={:+.3f}  后半Δ={:+.3f}".format(
            hs["ensemble_ew"]["first_half"]["sharpe"] - hs["incumbent"]["first_half"]["sharpe"],
            hs["ensemble_ew"]["second_half"]["sharpe"] - hs["incumbent"]["second_half"]["sharpe"]))
        report["first_vs_second_half"][str(aum)] = hs

    (OUT / "p10d_oos_stability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P10d 分析完成，见 outputs/p10c_ensemble/p10d_oos_stability.json ===")


if __name__ == "__main__":
    main()
