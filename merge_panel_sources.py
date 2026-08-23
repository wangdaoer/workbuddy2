"""合并多个"逐日归一化面板"源为统一 data_panel.csv。

用于把 通达信(Tdx) 历史 + 腾讯自选股(Westock) 近期 两套
ths_hs_a_share_YYYY-MM-DD.csv 合并成一份长表面板，供
build_data_panel.py / train_next_open_rank_model.py 消费。

合并规则：
  - 扫描每个源目录下所有 ths_hs_a_share_*.csv（也兼容 snapshot_all_*.csv /
    *market_snapshot.csv，按 build_data_panel 的列契约读取）。
  - 全部纵向拼接后，按 (date, symbol) 去重：后出现(命令行靠后)的源覆盖
    靠前的源 —— 因此调用时把 Tdx 放前面、腾讯自选股放后面，
    腾讯自选股在重叠日期(通常是拼接边界)胜出，保证最新价优先。
  - 输出列固定为 date,symbol,open,high,low,close,volume,amount，按
    (date, symbol) 排序。

P0-4 增强（2026-08-05，移植 model4 update_model_panel_from_daily_data.py 的两点价值）：
  - 合并后 fail-closed 交易日历校验（复用 model3_data_pipeline/trading_calendar.py）：
    基准 = 所有扫描文件的交易日(去周末) ∪ 合并结果的工作日；校验
    invalid_panel_dates(周末/节假日行) 与 missing_panel_dates(摄取缺口)。
    校验不过会抛错中止 —— 宁可在合并期拦截，不在 889×5910 训练时才炸。
  - 写 JSON 质量报告（默认 <output>.quality.json）：覆盖度 + 日历审计 + amount 汇总。
  - 合并输出本身保持与之前字节级一致（列/排序/优先级不变）。

用法：
  python3 merge_panel_sources.py \
      --source-dirs external_data/daily-market-data-tdx/ths_exports/normalized \
                     external_data/daily-market-data-westock/ths_exports/normalized \
      --output external_data/daily-market-data/data_panel.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from trading_calendar import trading_session_audit  # noqa: E402  (model3 根目录原生模块)

EXPECTED_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
    cols = set(df.columns.str.lower())
    # 与 build_data_panel._read_panel_csv 同构的列契约
    if {"date", "open", "high", "low", "close"}.issubset(cols):
        out = pd.DataFrame({
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "symbol": df["symbol"].astype(str).str.strip(),
            "open": pd.to_numeric(df.get("open"), errors="coerce"),
            "high": pd.to_numeric(df.get("high"), errors="coerce"),
            "low": pd.to_numeric(df.get("low"), errors="coerce"),
            "close": pd.to_numeric(df.get("close"), errors="coerce"),
            "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
            "amount": pd.to_numeric(df.get("amount"), errors="coerce"),
        })
        return out
    raise ValueError(f"不支持的 schema: {path.name} -> {list(df.columns)}")


def _file_date_from_name(path: Path) -> pd.Timestamp | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return pd.Timestamp(m.group(1)) if m else None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge(source_dirs: list[str], output: Path, start: str | None, end: str | None,
          quality_report: Path | None = None) -> dict:
    frames: list[pd.DataFrame] = []
    scanned = 0
    file_dates: list[pd.Timestamp] = []
    for d in source_dirs:
        dd = Path(d)
        if not dd.exists():
            print(f"[warn] 源目录不存在，跳过: {dd}")
            continue
        files = sorted(
            glob.glob(str(dd / "ths_hs_a_share_*.csv"))
            + glob.glob(str(dd / "snapshot_all_*.csv"))
            + glob.glob(str(dd / "*market_snapshot.csv"))
        )
        for f in files:
            try:
                frames.append(_read_one(Path(f)))
                scanned += 1
                fd = _file_date_from_name(Path(f))
                if fd is not None:
                    file_dates.append(fd)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 跳过无法解析的文件 {f}: {e}")

    if not frames:
        raise RuntimeError("没有任何可用面板文件，合并中止。")

    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["date", "symbol", "close"]).copy()
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    long["symbol"] = long["symbol"].astype(str)
    # 去重：靠后的源覆盖靠前的（调用顺序决定优先级）
    long = long.drop_duplicates(["date", "symbol"], keep="last")
    if start:
        long = long[long["date"] >= start]
    if end:
        long = long[long["date"] <= end]
    long = long.sort_values(["date", "symbol"]).reset_index(drop=True)
    long = long[EXPECTED_COLS]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output, index=False, encoding="utf-8")

    # --- P0-4：fail-closed 交易日历校验（基准 = 扫描文件交易日 ∪ 合并结果工作日） ---
    merged_dates = pd.to_datetime(long["date"], errors="coerce").dropna()
    expected = sorted(
        set(d for d in file_dates if d.dayofweek < 5)
        | set(merged_dates[merged_dates.dt.dayofweek < 5])
    )
    benchmark_csv = output.parent / f"{output.stem}.benchmark_sessions.csv"
    pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in expected]}).to_csv(
        benchmark_csv, index=False
    )
    calendar_audit = trading_session_audit(merged_dates, benchmark_csv)
    if not calendar_audit["passed"]:
        raise RuntimeError(
            "合并结果交易日历校验失败(宁在此拦截，不在训练期才炸): "
            f"invalid={calendar_audit['invalid_panel_dates'][:10]} "
            f"missing={calendar_audit['missing_panel_dates'][:10]}"
        )

    # --- P0-4：JSON 质量报告 ---
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dirs": source_dirs,
        "files_scanned": scanned,
        "rows": int(len(long)),
        "days": int(long["date"].nunique()),
        "symbols": int(long["symbol"].nunique()),
        "date_start": long["date"].min(),
        "date_end": long["date"].max(),
        "output": str(output),
        "calendar_audit": calendar_audit,
    }
    qpath = Path(quality_report) if quality_report is not None else output.with_name(
        f"{output.stem}.quality.json"
    )
    _write_json_atomic(qpath, report)
    report["quality_report"] = str(qpath)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple normalized panel sources into one data_panel.csv.")
    ap.add_argument("--source-dirs", nargs="+", required=True, help="源目录(多个)，靠后的优先级高(覆盖)")
    ap.add_argument("--output", default="external_data/daily-market-data/data_panel.csv", help="合并输出路径")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--quality-report", default=None, help="JSON 质量报告路径(默认 <output>.quality.json)")
    args = ap.parse_args()

    report = merge(args.source_dirs, Path(args.output), args.start_date, args.end_date,
                   Path(args.quality_report) if args.quality_report else None)
    print("合并完成:")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
