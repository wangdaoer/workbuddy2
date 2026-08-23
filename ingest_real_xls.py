"""真实同花顺 .xls 接入适配器（P0 真实数据验证用）。

把上传的真实 ths_hs_a_share_YYYY-MM-DD.xls（tab 分隔 / gb18030 / 中文列名）
通过 ths_daily_data.normalize_daily_market_file 归一化为规范 CSV：
    date, symbol, open, high, low, close, volume, amount,
    turnover_rate, market_cap, main_net_inflow, main_net_volume_ratio
写入 build_data_panel 期望的 normalized 目录，并清掉之前的合成样本。

随后运行 build_data_panel 构建真实面板，输出质量报告。
（注：仅 3 天样本不足以跑 walk-forward 真实基线，此处只验证"接入链路"。）
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ths_daily_data import normalize_daily_market_file

HERE = Path(__file__).resolve().parent
NORMALIZED_DIR = HERE / "external_data" / "daily-market-data" / "ths_exports" / "normalized"
UPLOAD_DIR = Path("/root/uploads")
REAL_OUTPUT_DIR = HERE / "outputs" / "p0_real"
PANEL_CSV = REAL_OUTPUT_DIR / "data_panel.csv"

FILE_RE = re.compile(r"ths_hs_a_share_(\d{4}-\d{2}-\d{2})\.xls$", re.IGNORECASE)


def _clear_synthetic() -> int:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    old = list(NORMALIZED_DIR.glob("ths_hs_a_share_*.csv"))
    for p in old:
        p.unlink()
    return len(old)


def main() -> None:
    files = sorted(p for p in UPLOAD_DIR.glob("*ths_hs_a_share_*.xls") if FILE_RE.search(p.name))
    if not files:
        raise SystemExit(f"在 {UPLOAD_DIR} 未找到 ths_hs_a_share_*.xls")

    cleared = _clear_synthetic()
    print(f"[ingest] 已清除 {cleared} 个旧合成样本，开始接入 {len(files)} 个真实 .xls")

    report_rows = []
    for f in files:
        m = FILE_RE.search(f.name)
        trade_date = m.group(1)
        normalized, sources = normalize_daily_market_file(f, trade_date)
        out = NORMALIZED_DIR / f"ths_hs_a_share_{trade_date}.csv"
        normalized.to_csv(out, index=False, encoding="utf-8")

        valid_vol = pd.to_numeric(normalized["volume"], errors="coerce").gt(0).sum()
        valid_close = pd.to_numeric(normalized["close"], errors="coerce").gt(0).sum()
        report_rows.append(
            {
                "file": f.name,
                "date": trade_date,
                "rows": len(normalized),
                "valid_close": int(valid_close),
                "valid_volume": int(valid_vol),
                "col_sources": sources,
            }
        )
        flag = "" if valid_vol == valid_close else "  <-- 缺 volume/amount，训练面板将丢弃该日"
        print(
            f"[ingest] {f.name}: 行={len(normalized)} "
            f"有效close={valid_close} 有效volume={valid_vol}{flag}"
        )

    # build_data_panel
    REAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== build_data_panel.py (真实数据) ===")
    p = subprocess.run(
        [sys.executable, "build_data_panel.py", "--output", str(PANEL_CSV)],
        cwd=HERE, capture_output=True, text=True,
    )
    print(p.stdout.strip())
    if p.returncode != 0:
        print(p.stderr.strip())
        raise RuntimeError("build_data_panel 失败")

    panel = pd.read_csv(PANEL_CSV)
    print("\n=== 真实面板质量报告 ===")
    print(f"总行数      : {len(panel)}")
    print(f"交易日数    : {panel['date'].nunique()}")
    print(f"标的数      : {panel['symbol'].nunique()}")
    print(f"日期覆盖    : {sorted(panel['date'].unique())}")
    print(f"列          : {list(panel.columns)}")
    print("\n各文件接入详情:")
    for r in report_rows:
        print(f"  {r['file']}: rows={r['rows']} valid_vol={r['valid_volume']}/{r['valid_close']}")

    # 写出报告
    (REAL_OUTPUT_DIR / "ingest_report.json").write_text(
        pd.DataFrame(report_rows).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[done] 真实面板 -> {PANEL_CSV}")


if __name__ == "__main__":
    main()
