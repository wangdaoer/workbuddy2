"""每日面板增量刷新 —— 根治自动化"只检查不更新面板"的空转问题。

流程：
  1) 从当前 data_panel.csv 提取全市场 universe（最权威的"我们要覆盖的标的集"）
  2) 腾讯公开接口摄取近期日线到 --end-date（默认今天）
  3) merge_panel_sources 合并进 data_panel.csv（Tdx 历史在前、腾讯近期在后覆盖）

供每日自动化在 production_soft_score 出分【之前】调用。
用法：
  python refresh_panel_daily.py [--end-date YYYY-MM-DD] [--count 5]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
Tdx_DIR = HERE / "external_data/daily-market-data-tdx/ths_exports/normalized"
WESTOCK_DIR = HERE / "external_data/daily-market-data-westock/ths_exports/normalized"
PANEL = HERE / "external_data/daily-market-data/data_panel.csv"
PY = sys.executable


MIN_ROWS_FLOOR = 1000      # 单个交易日的绝对行数下限（正常 ~5200）
COLLAPSE_RATIO = 0.5       # 相对刷新前塌陷到此比例以下即判定损坏


def build_universe(out_file: Path) -> int:
    """从 data_panel 历史所有 symbol 取并集（最全的市场范围）。"""
    df = pd.read_csv(PANEL, usecols=["symbol"], dtype={"symbol": str})
    syms = sorted(df["symbol"].dropna().astype(str).str.zfill(6).unique())
    out_file.write_text("\n".join(syms), encoding="utf-8")
    return len(syms)


def recent_date_counts(panel: Path, lookback: int = 12) -> dict[str, int]:
    """最近 lookback 个交易日的行数快照，用于刷新前后对比。"""
    if not panel.exists():
        return {}
    vc = pd.read_csv(panel, usecols=["date"])["date"].value_counts()
    return {d: int(vc[d]) for d in sorted(vc.index)[-lookback:]}


def assert_no_collapse(before: dict[str, int], after: dict[str, int]) -> None:
    """fail-closed 防塌陷护栏。

    背景(2026-08-07)：ingest 曾无条件覆盖当日文件，`--count 1` 重拉时
    ths_hs_a_share_2026-08-05.csv 被 5192 行覆盖成 1 行，导致 08-05/08-06
    特征全 NaN、模型尾端无分。ingest 已改为按 symbol 归并，这里再加一道
    面板层校验，任何一天行数塌陷即中止，避免自动化静默产出垃圾。
    """
    problems = []
    for d, n_after in sorted(after.items()):
        n_before = before.get(d)
        if n_before and n_after < n_before * COLLAPSE_RATIO:
            problems.append(f"{d}: {n_before} -> {n_after} 行（塌陷 >50%）")
        elif n_after < MIN_ROWS_FLOOR:
            problems.append(f"{d}: 仅 {n_after} 行（低于下限 {MIN_ROWS_FLOOR}）")
    if problems:
        print("[refresh][FAIL] 面板数据完整性校验未通过：")
        for p in problems:
            print("   -", p)
        print("   建议：重跑 ingest（已按 symbol 归并，可自愈）后再合并；勿在此状态下出分。")
        raise SystemExit(2)
    print(f"[refresh] 完整性校验通过（最近 {len(after)} 个交易日行数正常）")


def main() -> None:
    ap = argparse.ArgumentParser(description="每日面板增量刷新：摄取腾讯近期日线并合并进 data_panel.csv")
    ap.add_argument("--end-date", default=date.today().strftime("%Y-%m-%d"),
                    help="摄取结束日，默认今天")
    ap.add_argument("--count", type=int, default=5,
                    help="向接口请求的 K 线根数（覆盖窗口），默认 5")
    args = ap.parse_args()

    uni = WESTOCK_DIR / "_universe_autogen.txt"
    n = build_universe(uni)
    print(f"[refresh] universe={n} symbols, end-date={args.end_date}")

    before = recent_date_counts(PANEL)

    print("[refresh] 1/2 腾讯摄取 ...")
    subprocess.run(
        [PY, "ingest_westock_recent.py", "--universe", str(uni),
         "--count", str(args.count), "--end-date", args.end_date],
        check=True, cwd=HERE,
    )

    print("[refresh] 2/2 合并面板 ...")
    subprocess.run(
        [PY, "merge_panel_sources.py",
         "--source-dirs", str(Tdx_DIR), str(WESTOCK_DIR),
         "--output", str(PANEL)],
        check=True, cwd=HERE,
    )

    after = recent_date_counts(PANEL)
    assert_no_collapse(before, after)

    last = pd.read_csv(PANEL, usecols=["date"]).iloc[-1]["date"]
    print(f"[refresh] done. panel last date -> {last}")


if __name__ == "__main__":
    main()
