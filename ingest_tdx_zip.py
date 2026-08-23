"""把通达信 .day 归档 zip（如 hsjday.zip，沪深京合一）接入量化模型的逐日
归一化面板格式，使 build_data_panel + train_next_open_rank_model 能直接吃
**真实 A 股历史数据**。

与标准 TDX 拆分式（shlday.zip/szlday.zip）不同，hsjday.zip 是沪深京合一的
自定义命名 zip，且内含多市场 .day + 指数聚合文件。本脚本：

1. 遍历 zip 内所有 *.day 成员，按文件名推导市场与代码
   （sh/sz/bj 前缀优先；纯 6 位代码按首位推断：6/9→SH, 0/3→SZ, 8/4/92→BJ）。
2. 排除指数：
   - 非数字代码（shlday/szlday/bjlday/shzsday/szzsday 等聚合指数文件）。
   - 主要指数代码（上证综指 000001、沪深300 000300、深成指 399001、创业板指
     399006 等）——这些点位在几千，会与股票价格（~10–300）混在一起破坏归一化。
3. 用代码库已有 tdx_day_source.parse_tdx_day_bytes 解析（32 字节/条，价格 /100）。
4. 转成逐日 THS 归一化 CSV：ths_hs_a_share_YYYY-MM-DD.csv
   （date,symbol,open,high,low,close,volume,amount），写入独立的数据根，
   不污染合成数据目录。
5. 打印覆盖与单位自检（交易日范围、标的数、市场分布、amount/volume 单位推断）。

用法：
  python3 ingest_tdx_zip.py --zip hsjday.zip [--limit-days 880] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from tdx_day_source import parse_tdx_day_bytes


def _basename(name: str) -> str:
    """剥离 zip 成员内的目录前缀（兼容 Windows 反斜杠与 Unix 斜杠）。

    hsjday.zip 内成员形如 'sh\\\\lday\\\\sh000001.day'，在 Linux 下反斜杠是
    文件名的一部分而非路径分隔符，必须先按 / 与 \\\\ 切到最末一段再处理。
    """
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def infer_market_symbol(name: str) -> tuple[str, str] | None:
    """从 .day 文件名推导 (market, symbol)。返回 None 表示跳过（含指数聚合文件）。"""
    low = _basename(name).lower()
    if low.endswith(".day"):
        low = low[:-4]
    if len(low) >= 8 and low[:2] in ("sh", "sz", "bj"):
        market, code = low[:2].upper(), low[2:8]
    elif len(low) == 6 and low.isdigit():
        code = low
        if code[0] in ("6", "9"):
            market = "SH"
        elif code[0] in ("0", "3"):
            market = "SZ"
        elif code[0] in ("8", "4") or code.startswith("92"):
            market = "BJ"
        else:
            market = "SZ"
    else:
        return None
    # 仅接受 6 位数字代码的标的；非数字（lday/zsday 等聚合指数）直接跳过
    if not code.isdigit() or len(code) != 6:
        return None
    return market, code


def keep_stock(market: str, code: str) -> bool:
    """仅保留普通 A 股股票代码段，排除指数/基金/债券/可转债/B股等非股票标的。

    hsjday.zip 内含大量非股票 .day（如 12xxxx 指数/基金、5xxxxx 基金、000xxx
    上证系列指数、399xxx 深证系列指数、900xxx B股 等），其价格量级（指数几千点、
    基金/退市股 0.01）会破坏归一化与截面排序，必须按代码段过滤。
    """
    if not (code.isdigit() and len(code) == 6):
        return False
    if market == "SH":
        # 主板 600-605、科创板 688-689（排除 000xxx 指数 / 123xxx 基金指数 /
        # 900xxx B股 / 5xxxxx 基金 / 11xxxx/12xxxx 指数等）
        return code[:3] in ("600", "601", "602", "603", "604", "605", "688", "689")
    if market == "SZ":
        # 主板 000-003、创业板 300-301（排除 399xxx 指数 / 1xxxxx 基金等）
        return code[:3] in ("000", "001", "002", "003", "300", "301")
    if market == "BJ":
        # 北交所 83/87/88/89 开头及 920 开头（排除 4xxxxx 老三板、其它段基金等）
        return code[:2] in ("83", "87", "88", "89") or code[:3] == "920"
    return False



def _iter_stock_members(z: ZipFile):
    """生成 (member, market, symbol)，跳过非股票（含指数/基金/垃圾代码）。内存有界。"""
    for member in sorted(z.namelist()):
        if not member.lower().endswith(".day"):
            continue
        ms = infer_market_symbol(member)
        if ms is None:
            yield member, None, None
            continue
        market, symbol = ms
        if not keep_stock(market, symbol):
            yield member, "NONSTOCK", symbol
            continue
        yield member, market, symbol


def ingest(zip_path: Path, out_dir: Path, limit_days: int | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    skipped = 0
    skipped_index = 0
    members = 0

    with ZipFile(zip_path) as z:
        members_list = [m for m in z.namelist() if m.lower().endswith(".day")]

        # —— 第一遍（轻量）：仅收集全局交易日集合，确定日期窗口下界（避免全量驻留内存） ——
        cutoff = None
        if limit_days is not None:
            all_dates: set[str] = set()
            for member, market, symbol in _iter_stock_members(z):
                if market is None or market == "NONSTOCK":
                    continue
                try:
                    df = parse_tdx_day_bytes(
                        z.read(member), symbol=symbol, market=market, asset_type="stock",
                        source=f"{zip_path.name}!{member}",
                    )
                except Exception:  # noqa: BLE001
                    continue
                if not df.empty:
                    all_dates.update(df["date"].tolist())
            if all_dates:
                cutoff = sorted(all_dates)[-limit_days]
            del all_dates

        # —— 第二遍（带窗口过滤）：只保留 cutoff 之后的记录，逐成员 append 小 frame ——
        for member, market, symbol in _iter_stock_members(z):
            members += 1
            if market is None:
                skipped += 1
                continue
            if market == "NONSTOCK":
                skipped_index += 1
                continue
            try:
                df = parse_tdx_day_bytes(
                    z.read(member), symbol=symbol, market=market, asset_type="stock",
                    source=f"{zip_path.name}!{member}",
                )
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            if df.empty:
                continue
            if cutoff is not None:
                df = df[df["date"] >= cutoff]
            if not df.empty:
                frames.append(df)

    if not frames:
        raise RuntimeError(f"zip 内未解析出任何股票 .day 记录: {zip_path}")

    long = pd.concat(frames, ignore_index=True)
    del frames
    long = long[(long["close"] > 0) & (long["open"] > 0) & (long["volume"] > 0)]
    long = long.drop_duplicates(["date", "symbol"], keep="last")
    long["date"] = pd.to_datetime(long["date"])

    # —— 价格合理性符号过滤：剔除中位价异常（指数/基金/退市垃圾）的标的 ——
    # 真实 A 股 2022-2026 普通股票价格大致在 [0.5, 5000]；中位 close 超出此区间者
    # 几乎必为非股票（指数几千点 / 垃圾股 0.01），会污染截面排序，整只剔除。
    med_close = long.groupby("symbol")["close"].median()
    bad_sym = med_close[(med_close < 0.5) | (med_close > 5000.0)].index
    n_price_filtered = int(len(bad_sym))
    if n_price_filtered:
        long = long[~long["symbol"].isin(bad_sym)]
    price_filtered = n_price_filtered

    written = 0
    for d, grp in long.groupby(long["date"].dt.strftime("%Y-%m-%d")):
        out = grp[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]].copy()
        out.to_csv(out_dir / f"ths_hs_a_share_{d}.csv", index=False, encoding="utf-8")
        written += 1

    sample = long.sample(min(2000, len(long)), random_state=1)
    avg_px = sample[["open", "high", "low", "close"]].mean(axis=1)
    ratio = (sample["amount"] / (sample["volume"] * avg_px)).median()
    vol_unit = "股(shares)" if 0.5 < ratio < 5 else ("手(100股)" if 50 < ratio < 200 else f"未知(k≈{ratio:.1f})")

    coverage = {
        "zip": zip_path.name,
        "day_files_in_zip": members,
        "skipped_nonstock": skipped,
        "skipped_index_or_fund": skipped_index,
        "price_filtered_symbols": price_filtered,
        "rows": int(len(long)),
        "n_days": int(long["date"].nunique()),
        "n_symbols": int(long["symbol"].nunique()),
        "date_start": long["date"].min().strftime("%Y-%m-%d"),
        "date_end": long["date"].max().strftime("%Y-%m-%d"),
        "per_market": long.groupby("market")["symbol"].nunique().to_dict(),
        "daily_csv_written": written,
        "out_dir": str(out_dir),
        "volume_unit_guess": vol_unit,
        "amount_over_volume_x_avgprice_median": round(float(ratio), 3),
    }
    return coverage


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest TDX .day zip into per-day THS-normalized panel.")
    ap.add_argument("--zip", required=True, help="hsjday.zip 路径")
    ap.add_argument("--out-dir",
                    default="external_data/daily-market-data-tdx/ths_exports/normalized",
                    help="逐日归一化 CSV 输出目录（独立于合成数据根）")
    ap.add_argument("--limit-days", type=int, default=880,
                    help="仅保留最近 N 个交易日（默认 880≈3.5年）；0/不传=全量")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out_dir = Path(args.out_dir)
    limit = None if args.limit_days in (0,) else args.limit_days

    cov = ingest(zip_path, out_dir, limit)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
