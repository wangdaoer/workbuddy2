"""把港股通 TDX 归档 ggtday.zip 接入量化模型（P9-4 港股通扩展）。

与 A 股 hsjday.zip 的关键差异（已实测确认，不能复用 parse_tdx_day_bytes）：
  - 命名：M#CCCCC.day，M=31(沪港通港股通)/49(深港通港股通)，CCCCC 为 5 位港股代码。
  - 记录布局（32 字节）：date(int32) + open/high/low/close(float32，已是 HKD 价格，
    非 int×100) + amount(float32，HKD 成交金额) + volume(int32，手) + reserved(int32)。
    => 用 "<ifffffii" 解析；A 股 parse_tdx_day_bytes 的 /100 会把价格压到 1/100，错。
  - 金额单位为 HKD（与 A 股 CNY 不同，容量比较时按 FX≈0.92 折算 CNY）。

产出：
  1) hk_connect_long.csv       —— 合并长表(date,symbol,market,open,high,low,close,volume,amount)，
                                  供后续港股通横截面信号重训使用（与 A 股面板同 schema）。
  2) hk_connect_index.csv      —— 流动性加权港股通日收益指数（与 A 股簿 / 期货 sleeve 同窗口对齐）。
  3) hk_connect_liquidity.json —— 覆盖、标的数、各标的 ADV(窗口内日均成交金额 HKD/CNY)、龙头。

用法：
  python3 ingest_ggtday_zip.py --zip ggtday.zip \
      --out-dir external_data/daily-market-data-tdx/hk_connect \
      --window-start 2023-12-01 --window-end 2026-07-23
"""
from __future__ import annotations

import argparse
import json
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# 港股通 .day 记录：date(i) + o,h,l,c(f) + amount(f) + volume(i) + reserved(i) = 32B
TDX_HK = struct.Struct("<ifffffii")
FX_HKD_TO_CNY = 0.92  # 近似：1 HKD ≈ 0.92 CNY（2024–2026 区间），仅用于跨市场容量同口径比较


def parse_hk_day(raw: bytes) -> list[tuple]:
    """解析单个港股通 .day 字节流，返回 [(date_str, o,h,l,c, amount, volume), ...]。"""
    if len(raw) % TDX_HK.size != 0:
        return []
    out = []
    for off in range(0, len(raw), TDX_HK.size):
        date_i, o, h, l, c, amt, vol, _ = TDX_HK.unpack_from(raw, off)
        ds = str(int(date_i))
        if len(ds) != 8:
            continue
        y, m, d = ds[:4], ds[4:6], ds[6:8]
        try:
            date_str = f"{y}-{m}-{d}"
        except Exception:  # noqa: BLE001
            continue
        out.append((date_str, float(o), float(h), float(l), float(c), float(amt), int(vol)))
    return out


def ingest(zip_path: Path, out_dir: Path, window_start: str, window_end: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []          # 合并长表缓冲（窗口内）
    per_day = defaultdict(list)  # date_str -> list of row-dict（供生成逐日归一化文件）
    adv = {}                # code -> 窗口内日均 amount(HKD)
    name_amount = {}        # code -> 累计 amount（窗口内）
    name_dates = {}         # code -> list[(date, close)]（窗口内，用于指数）
    n_files = 0
    n_records = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as z:
        day_members = [n for n in z.namelist() if n.lower().endswith(".day")]
        for member in day_members:
            base = member.replace("\\", "/").rsplit("/", 1)[-1]
            if "#" not in base:
                skipped += 1
                continue
            market, code = base.split("#", 1)
            code = code.rsplit(".", 1)[0]
            if not code.isdigit() or len(code) != 5:
                skipped += 1
                continue
            n_files += 1
            recs = parse_hk_day(z.read(member))
            if not recs:
                skipped += 1
                continue
            # 转 df 并做基本质量过滤
            df = pd.DataFrame(recs, columns=["date", "open", "high", "low", "close", "amount", "volume"])
            df = df[(df["close"] > 0) & (df["open"] > 0) & (df["volume"] > 0)]
            if df.empty:
                continue
            # 价格合理性（港股通多为大市值，中位价落在 [0.5, 5000]；越界者多为垃圾/指数残留）
            med = df["close"].median()
            if med < 0.5 or med > 5000.0:
                skipped += 1
                continue
            df = df[(df["date"] >= window_start) & (df["date"] <= window_end)]
            if df.empty:
                continue
            n_records += len(df)
            df = df.sort_values("date")
            # 写入长表缓冲
            for _, r in df.iterrows():
                row = {
                    "date": r["date"], "symbol": code, "market": market,
                    "open": r["open"], "high": r["high"], "low": r["low"],
                    "close": r["close"], "volume": r["volume"], "amount": r["amount"],
                }
                long_rows.append(row)
                per_day[r["date"]].append(row)
            # ADV 累计
            name_amount[code] = name_amount.get(code, 0.0) + df["amount"].sum()
            # 指数用：日期->收盘
            name_dates.setdefault(code, []).extend(df[["date", "close"]].itertuples(index=False, name=None))

    if not long_rows:
        raise RuntimeError("ggtday.zip 未解析出任何有效港股通记录")

    long = pd.DataFrame(long_rows)
    del long_rows
    long.to_csv(out_dir / "hk_connect_long.csv", index=False, encoding="utf-8")

    # 逐日归一化 CSV（与 A 股 ths_hs_a_share_* 同 schema，便于后续港股通信号重训）
    norm_dir = out_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    for d, rows in per_day.items():
        grp = pd.DataFrame(rows)[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
        grp.to_csv(norm_dir / f"ths_hk_connect_{d}.csv", index=False, encoding="utf-8")
    del per_day

    # ADV（窗口内日均成交金额 HKD）
    n_days_per_name = long.groupby("symbol")["date"].nunique()
    span_days = max(1, (pd.to_datetime(window_end) - pd.to_datetime(window_start)).days)
    adv_hkd = {c: name_amount[c] / max(1, n_days_per_name.get(c, 1)) for c in name_amount}
    total_adv_hkd = sum(adv_hkd.values())
    top = sorted(adv_hkd.items(), key=lambda kv: kv[1], reverse=True)[:30]

    # —— 流动性加权港股通日收益指数（streaming 累加，内存有界） ——
    # 权重 = 该标的窗口内日均 amount(HKD)
    w_sum = defaultdict(float)
    wr_sum = defaultdict(float)
    for code, pairs in name_dates.items():
        w = adv_hkd.get(code, 0.0)
        if w <= 0:
            continue
        pairs = sorted(pairs)
        for i in range(1, len(pairs)):
            d0, c0 = pairs[i - 1]
            d1, c1 = pairs[i]
            if c0 > 0:
                ret = c1 / c0 - 1.0
                w_sum[d1] += w
                wr_sum[d1] += w * ret
    idx_rows = []
    for d in sorted(w_sum):
        r = wr_sum[d] / w_sum[d] if w_sum[d] > 0 else 0.0
        idx_rows.append((d, r))
    idx = pd.DataFrame(idx_rows, columns=["date", "ret"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["idx"] = (1 + idx["ret"].fillna(0)).cumprod()
    idx.to_csv(out_dir / "hk_connect_index.csv", index=False)

    ret = idx["ret"].dropna()
    cum = idx["idx"].iloc[-1]
    vol = ret.std() * np.sqrt(252)
    ann = cum ** (252 / len(ret)) - 1 if len(ret) else 0.0
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
    peak = idx["idx"].cummax()
    dd = (idx["idx"] / peak - 1).min()

    coverage = {
        "zip": zip_path.name,
        "window": [window_start, window_end],
        "day_files_parsed": n_files,
        "day_files_skipped": skipped,
        "records_in_window": int(n_records),
        "n_symbols": int(long["symbol"].nunique()),
        "markets": long.groupby("market")["symbol"].nunique().to_dict(),
        "date_start": long["date"].min(),
        "date_end": long["date"].max(),
        "total_adv_hkd_per_day": round(float(total_adv_hkd), 2),
        "total_adv_cny_per_day": round(float(total_adv_hkd * FX_HKD_TO_CNY), 2),
        "index_ann_return": round(float(ann), 4),
        "index_ann_vol": round(float(vol), 4),
        "index_sharpe": round(float(sharpe), 4),
        "index_max_drawdown": round(float(dd), 4),
        "index_cum_return": round(float(cum - 1), 4),
        "fx_hkd_to_cny": FX_HKD_TO_CNY,
        "top30_by_adv_hkd": [(c, round(v, 0)) for c, v in top],
        "outputs": {
            "long": str(out_dir / "hk_connect_long.csv"),
            "index": str(out_dir / "hk_connect_index.csv"),
            "normalized_dir": str(norm_dir),
        },
    }
    (out_dir / "hk_connect_liquidity.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2))
    return coverage


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest HK-connect ggtday.zip (float32 OHLC) into model panel.")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out-dir", default="external_data/daily-market-data-tdx/hk_connect")
    ap.add_argument("--window-start", default="2023-12-01")
    ap.add_argument("--window-end", default="2026-07-23")
    args = ap.parse_args()
    cov = ingest(Path(args.zip), Path(args.out_dir), args.window_start, args.window_end)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
