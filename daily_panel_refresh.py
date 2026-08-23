"""每日行情面板增量刷新（纯接口路径，不依赖 THS 客户端导出）。

用途：每个交易日收盘后执行（建议 18:40），完成
  1) ingest_westock_recent（并发拉取腾讯日线，补最近增量）
  2) merge_panel_sources（合并 tdx 历史 + westock 增量 → data_panel.csv）
使 19:00 的量化每日自动化直接拿到含当日数据的新面板。

窗口逻辑：
  - end   = --end-date（默认今天）
  - start = westock 已有数据末日 +1（更精确）；若 westock 目录为空则用 --fallback-start（默认取 tdx 末日）
并发：--workers（默认 16，配合限流退避与启动抖动）。

用法：
  python daily_panel_refresh.py [--end-date YYYY-MM-DD] [--workers 16]
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TDX_DIR = HERE / "external_data/daily-market-data-tdx/ths_exports/normalized"
WESTOCK_DIR = HERE / "external_data/daily-market-data-westock/ths_exports/normalized"
PANEL = HERE / "external_data/daily-market-data/data_panel.csv"


def _last_date_in(dirpath: Path, pattern: str) -> str | None:
    files = sorted(glob.glob(str(dirpath / pattern)))
    if not files:
        return None
    # 文件名 ths_hs_a_share_YYYY-MM-DD.csv，取最后一个是最大日期（文件名字典序=日期序）
    return files[-1].split("_")[-1].replace(".csv", "")


def main() -> int:
    ap = argparse.ArgumentParser(description="每日行情面板增量刷新（纯接口路径）")
    ap.add_argument("--end-date", default=date.today().strftime("%Y-%m-%d"), help="结束日，默认今天")
    ap.add_argument("--workers", type=int, default=16, help="并发线程数（默认 16）")
    ap.add_argument("--fallback-start", default=None,
                    help="westock 无数据时的起始日（默认取 tdx 末日+1）")
    args = ap.parse_args()

    # 窗口起点：westock 已有末日 +1（避免重拉已有日期）
    westock_last = _last_date_in(WESTOCK_DIR, "ths_hs_a_share_*.csv")
    tdx_last = _last_date_in(TDX_DIR, "ths_hs_a_share_*.csv")
    if westock_last:
        start = (date.fromisoformat(westock_last) + timedelta(days=1)).strftime("%Y-%m-%d")
    elif args.fallback_start:
        start = args.fallback_start
    elif tdx_last:
        start = (date.fromisoformat(tdx_last) + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        print("无法确定起始日：westock 与 tdx 目录均无数据，请显式 --fallback-start")
        return 1

    end = args.end_date
    print(f"[panel-refresh] window {start}..{end} (westock_last={westock_last}, tdx_last={tdx_last})")

    # 跳过：窗口无效（end < start 或 end 非交易日由接口自然判定）
    if end < start:
        print("[panel-refresh] end < start，无新交易日，跳过。")
        return 0

    py = sys.executable
    steps = [
        [py, str(HERE / "ingest_westock_recent.py"),
         "--from-tdx-dir", str(TDX_DIR),
         "--start-date", start, "--end-date", end,
         "--workers", str(args.workers)],
        [py, str(HERE / "merge_panel_sources.py"),
         "--source-dirs", str(TDX_DIR), str(WESTOCK_DIR),
         "--output", str(PANEL)],
    ]
    for i, step in enumerate(steps, 1):
        print(f"[panel-refresh] step {i}/{len(steps)}: {Path(step[1]).name}")
        r = subprocess.run(step, cwd=str(HERE))
        if r.returncode != 0:
            print(f"[panel-refresh] 步骤失败 rc={r.returncode}: {Path(step[1]).name}")
            return r.returncode

    print("[panel-refresh] 完成：面板已刷新（如需确认，检查 data_panel.csv 最后日期）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
