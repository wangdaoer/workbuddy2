"""Best-effort enrichment: 把趋势起爆影子评分回流进当日合并优先观察清单。

这是纯研究性后置步骤：把 score_trend_ignition_shadow.py 生成的
trend_ignition_shadow_scores.csv 中的评分列合并回 merged_daily_outputs 写出的
merged_priority_watchlist_{token}.csv（及其 _cn 版本）。不改变任何排序/权重/信号。

任何异常都只记录警告并退出 0，绝不让主流程失败。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import merged_daily_outputs as mdo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich merged priority watchlist with trend ignition shadow scores (best-effort)."
    )
    parser.add_argument("--priority-csv", required=True)
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--asof-date", required=True)
    args = parser.parse_args()
    try:
        paths = mdo.enrich_priority_watchlist_with_trend_ignition(
            Path(args.priority_csv), Path(args.scores_csv), args.asof_date
        )
    except Exception as exc:  # noqa: BLE001 - 研究性富集，绝不让主流程失败
        print(f"ENRICH_WARNING: trend ignition enrichment skipped: {exc}", file=sys.stderr)
        return
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
