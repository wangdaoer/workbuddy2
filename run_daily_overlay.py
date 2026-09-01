"""每日生产命令：全市场 PIT 出票 → 个人行为/名单叠加 → 日报。

前置（每日收盘后先跑一次，重建/刷新分数 npz）：
    python production_soft_score.py --universe pit --panel external_data/daily-market-data/data_panel.csv \\
        --asof-date YYYY-MM-DD --token YYYYMMDD --monitor --output-dir outputs/ab_pit
（npz 由 score_cache 内容寻址自动重建，面板没变则命中缓存）

默认会先**刷新名单产物**（D:/codex/outputs/stock-analysis-dashboard/input 每日更新的
ths_money_flow_*.xls）：
    1) watchlist_leak_audit.py    重新解析名单 → 重建 PIT 掩码（含 07-02 全市场文件护栏）
    2) watchlist_symbol_history.py 名单 → symbol-history（关注天数）+ names
    3) gfyj_broker_statement.py    广发对账单 + 名单 → combined_symbol_history（行为层数据）
然后用最新产物做候选 + 叠加（秒级）：
    python run_daily_overlay.py [--no-refresh] [--asof 2026-08-05]
产出：
    outputs/watchlist_audit/full_candidates.csv         全市场有分候选（末有分日）
    outputs/watchlist_audit/full_overlay_calibrated.csv  叠加后组合
    outputs/watchlist_audit/full_overlay_calibrated.md   中文日报
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = Path("outputs/watchlist_audit")
RULES = "watchlist_overlay_rules_calibrated.json"
NAMES_SOURCE = str(OUT / "watchlist_names.csv")
SYMBOL_HISTORY = str(OUT / "combined_symbol_history.csv")


def refresh_watchlist_artifacts(py: str) -> None:
    """重新解析每日名单并对账单，刷新掩码/symbol-history/names/combined（名单目录每日更新）。"""
    steps = [
        ("watchlist_leak_audit.py", []),          # 掩码 + 名单审计（读 4.89M 面板，~1-2min）
        ("watchlist_symbol_history.py", []),      # symbol-history + names
        ("gfyj_broker_statement.py", []),         # combined（对账单 + 名单关注度）
    ]
    for i, (script, extra) in enumerate(steps, start=1):
        print(f"[0.{i}/3] 刷新名单产物: {script} ...")
        subprocess.run([py, script, *extra], check=True, cwd=HERE)


def main() -> None:
    parser = argparse.ArgumentParser(description="每日：全市场出票 + 个人行为叠加")
    # 2026-08-31 修复：生产入口(production_soft_score) broker 并入后默认文件名带 _broker 后缀，
    # 旧 _pit.npz 自 08-26 起停更（903 天 vs 面板 907 天 shape 错）。此处同步指向权威文件。
    parser.add_argument("--scores-npz", default="outputs/p10e_regime_gated/linear_mlp_scores_pit_broker.npz")
    parser.add_argument("--asof", default=None, help="仅用于命名日报（默认取 npz 最后有分日）")
    parser.add_argument("--no-refresh", action="store_true",
                        help="跳过名单产物刷新（名单未更新时省 1-2 分钟）")
    args = parser.parse_args()

    npz = Path(args.scores_npz)
    if not npz.exists():
        raise SystemExit(f"缺失 {npz}：请先跑 production_soft_score --universe pit 生成分数后再执行")

    py = sys.executable
    cand_csv = OUT / "full_candidates.csv"
    overlay_csv = OUT / "full_overlay_calibrated.csv"

    if not args.no_refresh:
        refresh_watchlist_artifacts(py)
    else:
        print("[0] 跳过名单产物刷新（--no-refresh）")

    print("[1/2] 全市场候选 ...")
    subprocess.run(
        [py, "build_watchlist_candidates.py", "--mode", "full",
         "--scores-npz", str(npz), "--out", str(cand_csv)],
        check=True, cwd=HERE,
    )
    print("[2/2] 行为叠加（校准规则）...")
    subprocess.run(
        [py, "apply_personal_trade_overlay.py",
         "--candidates", str(cand_csv),
         "--symbol-history", SYMBOL_HISTORY,
         "--rules", RULES,
         "--names-source", NAMES_SOURCE,
         "--output", str(overlay_csv),
         "--reselect-top-n", "20",
         "--selection-mode", "conservative_fill"],
        check=True, cwd=HERE,
    )
    print("[3/3] 策略体检（数据健康 + 断路器）...")
    subprocess.run(
        [py, "strategy_health_check.py",
         "--overlay-csv", str(overlay_csv),
         "--candidates-csv", str(cand_csv)],
        check=True, cwd=HERE,
    )
    print(f"\n完成：日报 -> {overlay_csv.with_suffix('.md')}")


if __name__ == "__main__":
    main()
