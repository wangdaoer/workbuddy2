"""把腾讯自选股（腾讯财经行情）近期日线接入量化模型的逐日归一化面板格式。

设计目标：与 ingest_tdx_zip.py 产出**完全同构**的逐日 CSV ——
    ths_hs_a_share_YYYY-MM-DD.csv
    (date, symbol, open, high, low, close, volume, amount)
其中 symbol 为纯 6 位代码（不带 sh/sz/bj 前缀），与 Tdx 历史面板保持一致，
便于后续 merge_panel_sources.py 直接按 (date, symbol) 拼接。

数据源：腾讯财经公开 K 线接口（无需鉴权，走公网）
    https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={mkt}{code},day,{start},{end},{count}
返回 data.{mkt}{code}.day = [[date, open, close, high, low, volume], ...]
（注意腾讯顺序：open/close/high/low，与常规 OHLC 不同）

universe 三种来源（任选其一）：
    1) --symbols sh600000,sz000001,600519   （可直接带/不带市场前缀）
    2) --universe symbols.txt                （每行一个，可带/不带前缀）
    3) --from-tdx-dir <Tdx归一化目录>         （取 Tdx 面板中最近 N 天的全部标的，
                                                自动只补这些标的的近期数据）

窗口：--start-date / --end-date；end 默认=今天，start 默认 = start 前推 count 个交易日。
本脚本只负责"近期尾巴"，历史主体由 ingest_tdx_zip.py 从 hsjday.zip 产出。

amount 处理：腾讯 K 线不含成交额，按 volume(手)*100*均价 估算（元），
             仅用于容量/冲击类特征；默认训练不依赖 amount，缺失亦可。

用法示例：
  # 样本冒烟
  python3 ingest_westock_recent.py --symbols sh600000,sz000001,600519 --count 20

  # 接在 Tdx 历史之后，自动只补 Tdx 覆盖的标的
  python3 ingest_tdx_zip.py --zip hsjday.zip
  python3 ingest_westock_recent.py --from-tdx-dir external_data/daily-market-data-tdx/ths_exports/normalized
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import urllib.request

KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={param}"
UA = {"User-Agent": "Mozilla/5.0 (compatible; quant-ingest/1.0)"}
DEFAULT_WORKERS = 16


# ----------------------------------------------------------------------------
# symbol 规范化：统一成 (market_prefix, plain_6digit)
# ----------------------------------------------------------------------------
def market_prefix(code: str) -> str:
    code = code.strip()
    if code[:2].lower() in ("sh", "sz", "bj"):
        return code[:2].lower()
    digits = code[-6:] if len(code) >= 6 else code
    if not digits.isdigit():
        raise ValueError(f"无法识别的代码: {code}")
    first = digits[0]
    if first in ("6", "9"):
        return "sh"
    if first in ("0", "3"):
        return "sz"
    if first in ("8", "4") or digits.startswith("92"):
        return "bj"
    return "sh"


def plain_symbol(code: str) -> str:
    code = code.strip().lower()
    if code[:2] in ("sh", "sz", "bj"):
        code = code[2:]
    return code.zfill(6)


def resolve_universe(args) -> list[str]:
    """返回纯 6 位代码列表（去重，保序）。"""
    symbols: list[str] = []
    if args.symbols:
        for raw in args.symbols.split(","):
            raw = raw.strip()
            if raw:
                symbols.append(plain_symbol(raw))
    if args.universe:
        p = Path(args.universe)
        text = p.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip().strip(",")
            if line and not line.startswith("#"):
                # 支持 csv 首列或纯代码
                first = line.split(",")[0].strip()
                if first:
                    symbols.append(plain_symbol(first))
    if args.from_tdx_dir:
        d = Path(args.from_tdx_dir)
        files = sorted(glob.glob(str(d / "ths_hs_a_share_*.csv")))
        if not files:
            raise FileNotFoundError(f"Tdx 归一化目录无 ths_hs_a_share_*.csv: {d}")
        # 取最近 args.tdx_lookback 个文件，并集所有 symbol
        recent = files[-args.tdx_lookback:]
        seen: set[str] = set()
        for f in recent:
            df = pd.read_csv(f, usecols=["symbol"], dtype={"symbol": str})
            for s in df["symbol"].dropna().unique():
                if s not in seen:
                    seen.add(str(s))
                    symbols.append(str(s))
    if not symbols:
        raise ValueError("未解析到任何标的，请检查 --symbols / --universe / --from-tdx-dir")
    # 去重保序
    out, seen = [], set()
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ----------------------------------------------------------------------------
# 拉取 + 解析
# ----------------------------------------------------------------------------
def _parse_day_arr(arr) -> dict | None:
    """腾讯 day 数组: [date, open, close, high, low, volume]"""
    if not arr or len(arr) < 6:
        return None
    try:
        d = str(arr[0])[:10]
        # 兼容 2026-01-05 / 20260105
        if "-" not in d:
            d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return {
            "date": d,
            "open": float(arr[1]),
            "close": float(arr[2]),
            "high": float(arr[3]),
            "low": float(arr[4]),
            "volume": float(arr[5]),
        }
    except (ValueError, TypeError):
        return None


def fetch_one(prefixed: str, start: str, end: str, count: int, retries: int = 3,
              jitter: float = 0.0) -> list[dict]:
    """拉取单标的日线；jitter>0 时首请求前先随机休眠，避免并发启动瞬间集体触发限流。"""
    if jitter > 0:
        time.sleep(random.uniform(0.0, jitter))
    param = f"{prefixed},day,{start},{end},{count}"
    url = KLINE_URL.format(param=param)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
            node = (data.get("data") or {}).get(prefixed)
            if not node:
                return []
            rows = node.get("day") or node.get("qfqday") or []
            out = []
            for arr in rows:
                rec = _parse_day_arr(arr)
                if rec:
                    out.append(rec)
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.0 * (attempt + 1) * (attempt + 1))  # 指数退避: 1s/4s/9s
    print(f"  [warn] {prefixed} 拉取失败: {last_err}")
    return []


def ingest(symbols: list[str], start: str, end: str, count: int, out_dir: Path,
           workers: int = DEFAULT_WORKERS) -> dict:
    """并发拉取全部标的日线，汇总为逐日 CSV。

    并发仅用于网络 I/O（GIL 下对 per_day 的 append 原子安全）；写文件阶段回退串行，
    输出与单线程版字节级一致（列序/排序/amount 估算不变）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_day: dict[str, list[dict]] = {}
    n_rows = 0
    n_ok = 0
    t0 = time.time()
    if workers and workers > 1:
        # 并发模式：启动抖动避免集体限流；主线程顺序收集结果
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(fetch_one, f"{market_prefix(sym)}{sym}", start, end, count,
                          jitter=0.5): sym
                for sym in symbols
            }
            for i, f in enumerate(as_completed(futs), 1):
                sym = futs[f]
                recs = f.result()
                if recs:
                    n_ok += 1
                    for rec in recs:
                        rec["symbol"] = sym
                        per_day.setdefault(rec["date"], []).append(rec)
                        n_rows += 1
                if i % 1000 == 0:
                    print(f"  [ingest] {i}/{len(symbols)} 完成 ({time.time()-t0:.0f}s)")
    else:
        # 单线程模式（保留原行为）
        for sym in symbols:
            prefixed = f"{market_prefix(sym)}{sym}"
            recs = fetch_one(prefixed, start, end, count)
            if recs:
                n_ok += 1
                for rec in recs:
                    rec["symbol"] = sym
                    per_day.setdefault(rec["date"], []).append(rec)
                    n_rows += 1
            time.sleep(0.05)  # 礼貌限速
    written = 0
    for d, rows in sorted(per_day.items()):
        df = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        avg_px = (df[["open", "high", "low", "close"]].mean(axis=1))
        # volume 为手(100股)，折算成交额(元) = 手*100*均价
        df["amount"] = (df["volume"] * 100 * avg_px).round(2)
        df = df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        out_path = out_dir / f"ths_hs_a_share_{d}.csv"
        # 修复(2026-08-07)：原为无条件整体覆盖，导致"部分抓取"会摧毁已完整的日文件
        # （实例：--count 1 重拉时仅个别停牌票的最后一根落在 08-05，
        #  ths_hs_a_share_2026-08-05.csv 由 5192 行被覆盖成 1 行，进而让 08-05/08-06
        #  特征全 NaN、模型尾端无分）。改为按 symbol 归并：新数据优先，旧数据补齐。
        if out_path.exists():
            try:
                old = pd.read_csv(out_path, dtype={"symbol": str})
                old["symbol"] = old["symbol"].astype(str).str.zfill(6)
                old = old.reindex(columns=df.columns)
                n_old = len(old)
                df = pd.concat([old, df], ignore_index=True)
                df = df.drop_duplicates(subset=["date", "symbol"], keep="last")
                df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
                if len(df) > n_old:
                    print(f"  [ingest] {d}: 归并旧文件 {n_old} 行 -> {len(df)} 行")
            except Exception as exc:  # 旧文件损坏则退回直接写新数据
                print(f"  [ingest][warn] {out_path.name} 归并失败，改为覆盖写: {exc}")
        df.to_csv(out_path, index=False, encoding="utf-8")
        written += 1
    return {
        "symbols_requested": len(symbols),
        "symbols_ok": n_ok,
        "rows": n_rows,
        "daily_csv_written": written,
        "date_start": min(per_day) if per_day else None,
        "date_end": max(per_day) if per_day else None,
        "workers": workers,
        "elapsed_sec": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest recent Tencent (腾讯自选股) daily K-line into per-day THS-normalized panel.")
    ap.add_argument("--symbols", default=None, help="逗号分隔代码，可带/不带 sh/sz/bj 前缀，如 sh600000,sz000001,600519")
    ap.add_argument("--universe", default=None, help="标的清单文件，每行一个代码")
    ap.add_argument("--from-tdx-dir", default=None, help="从 Tdx 归一化目录取最近 N 天全部标的作为 universe")
    ap.add_argument("--tdx-lookback", type=int, default=5, help="--from-tdx-dir 时取最近几个文件(天)")
    ap.add_argument("--start-date", default=None, help="起始日 YYYY-MM-DD；缺省=end-date 前推 count 交易日")
    ap.add_argument("--end-date", default=date.today().strftime("%Y-%m-%d"), help="结束日，默认今天")
    ap.add_argument("--count", type=int, default=320, help="向接口请求的 K 线根数(覆盖窗口)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="并发拉取线程数（默认 16；设 1 回退单线程原行为）")
    ap.add_argument("--out-dir", default="external_data/daily-market-data-westock/ths_exports/normalized",
                    help="逐日归一化 CSV 输出目录")
    args = ap.parse_args()

    symbols = resolve_universe(args)
    start = args.start_date or (
        datetime.strptime(args.end_date, "%Y-%m-%d").date() - timedelta(days=args.count)
    ).strftime("%Y-%m-%d")
    # pandas Timedelta 仅用于近似；精确交易日由接口返回控制

    print(f"[ingest] universe={len(symbols)} symbols, window {start}..{args.end_date}, workers={args.workers}")
    cov = ingest(symbols, start, args.end_date, args.count, Path(args.out_dir), workers=args.workers)
    print(json.dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
