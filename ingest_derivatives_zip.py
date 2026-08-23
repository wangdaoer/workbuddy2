"""衍生品（期货 + 期权）月 zip → 连续合约面板，接入 high_risk_quant_model3。

源数据（网盘 `FfpWJxmRgdcy` 下 `202201`..`202607` 月 zip）：
  - 每个 zip 内含逐交易日 CSV（`20240801_1.csv` …），GBK 编码。
  - 列（GBK）：合约代码, 今开盘, 最高价, 最低价, 成交量, 成交金额, 持仓量,
            持仓变化, 今收盘, 今结算, 前结算, 涨跌1, 涨跌2, Delta

标的构成：
  - 指数期货：IF/IH/IC/IM（沪深300/上证50/中证500/中证1000）
  - 国债期货：T/TF/TS/TL（10Y/5Y/2Y/30Y）
  - 指数期权：HO/IO/MO（含 -C-/ -P-，带行权价/到期/Delta）

设计要点：
  - 期货前缀无 '-'，期权含 '-'（如 HO2408-C-2100）。
  - 连续合约默认取「近月」：对每个交易日，选到期月 >= 该月且最小的合约；
    支持 near/second/third。需要无未来函数——只用当日及之前的到期/持仓信息。
  - 合约乘数（元/点）用于估算名义成交：IF/IH=300, IC/IM=200,
    T/TF/TL=10000, TS=20000。

用法：
  python3 ingest_derivatives_zip.py --zips 202407_t.zip 202408_t.zip \
      --out external_data/derivatives/futures_near.csv [--contract near]
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# 期货标的与合约乘数（元/指数点）
FUTURES = {"IF", "IH", "IC", "IM", "T", "TF", "TS", "TL"}
MULT = {"IF": 300, "IH": 300, "IC": 200, "IM": 200,
        "T": 10000, "TF": 10000, "TS": 20000, "TL": 10000}
OPTIONS = {"HO", "IO", "MO"}

RAW_RENAME = {
    "合约代码": "code", "今开盘": "open", "最高价": "high", "最低价": "low",
    "今收盘": "close", "成交金额": "amount", "持仓量": "oi", "持仓变化": "oi_chg",
    "成交量": "volume", "今结算": "settlement", "前结算": "prev_settle",
    "涨跌1": "change1", "涨跌2": "change2", "Delta": "delta",
}


def load_month(zf: zipfile.ZipFile) -> pd.DataFrame:
    """读一个衍生品月 zip，返回长表（含 trade_date 列）。"""
    frames = []
    for name in sorted(zf.namelist()):
        if not name.lower().endswith(".csv"):
            continue
        ymd = "".join(ch for ch in name if ch.isdigit())[:8]
        raw = zf.read(name).decode("gbk", errors="replace")
        df = pd.read_csv(io.StringIO(raw))
        df.columns = [c.strip() for c in df.columns]
        df["trade_date"] = ymd
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_assets(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """拆分期货 / 期权。期货=前缀在 FUTURES 且不含 '-'；期权=含 '-'（HO/IO/MO）。"""
    code = long["code"].astype(str)
    prefix = code.str.extract(r"^([A-Za-z]+)")[0]
    is_opt = code.str.contains("-")
    is_fut = prefix.isin(FUTURES) & (~is_opt)
    return long[is_fut].copy(), long[is_opt].copy()


def build_continuous(fut: pd.DataFrame, contract: str = "near") -> pd.DataFrame:
    """构建连续合约面板。

    contract: 'near' | 'second' | 'third' —— 按到期月排序取第 N 近月。
    同时给出 notional_turnover（名义成交，元）。
    """
    fut = fut.copy()
    for c in ["open", "high", "low", "close", "volume", "amount", "oi", "settlement"]:
        fut[c] = pd.to_numeric(fut[c], errors="coerce")
    fut = fut.dropna(subset=["close", "volume"])
    fut["expiry"] = fut["code"].str.extract(r"(\d{4})")[0].astype(int)
    # ym 取 4 位 YYMM 与 expiry 同维度，保证 expiry>=ym 的近月筛选语义正确
    fut["ym"] = fut["trade_date"].str[2:6].astype(int)
    fut["underlying"] = fut["code"].str.extract(r"^([A-Za-z]+)")[0]

    rank_map = {"near": 0, "second": 1, "third": 2}
    rank = rank_map[contract]

    rows = []
    for (und, td), g in fut.groupby(["underlying", "trade_date"]):
        ym = int(td[2:6])
        cand = g[g["expiry"] >= ym].sort_values("expiry")
        if cand.empty:
            cand = g.sort_values("expiry")
        if len(cand) <= rank:
            near = cand.iloc[-1]
        else:
            near = cand.iloc[rank]
        rows.append(near)
    cont = pd.DataFrame(rows).sort_values(["underlying", "trade_date"]).reset_index(drop=True)
    cont["notional_turnover"] = cont.apply(
        lambda r: r["volume"] * r["close"] * MULT.get(r["underlying"], 300), axis=1)
    cont["contract_rank"] = rank
    return cont


def ingest(zips: list[str], out: Path, contract: str = "near") -> dict:
    long = pd.concat([load_month(zipfile.ZipFile(z)) for z in zips], ignore_index=True)
    # 源文件 code 列带尾随空白，先按 GBK 原名剥离，再按 ascii 名兜底
    for col in ("合约代码", "code"):
        if col in long.columns:
            long[col] = long[col].astype(str).str.strip()
    long = long.rename(columns={k: v for k, v in RAW_RENAME.items() if k in long.columns})
    fut, opt = split_assets(long)
    cont = build_continuous(fut, contract=contract)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cont.to_csv(out, index=False)

    # 期权单独落盘（供波动率曲面等后续使用）
    opt_out = out.parent / f"options_long_{out.stem.split('_')[0]}.csv"
    if not opt.empty:
        opt.to_csv(opt_out, index=False)
    else:
        opt_out = None

    return {
        "months": sorted(cont["trade_date"].str[:6].unique().tolist()),
        "date_start": cont["trade_date"].min(),
        "date_end": cont["trade_date"].max(),
        "n_days": int(cont["trade_date"].nunique()),
        "underlyings": sorted(cont["underlying"].unique().tolist()),
        "contract_rank": contract,
        "futures_rows": int(len(cont)),
        "options_rows": int(len(opt)),
        "avg_notional_turnover_by_underlying_cny": {
            u: int(cont[cont.underlying == u]["notional_turnover"].mean())
            for u in sorted(cont["underlying"].unique())
        },
        "futures_panel": str(out),
        "options_panel": str(opt_out) if opt_out else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Derivatives monthly zip -> continuous contract panel")
    ap.add_argument("--zips", nargs="+", required=True, help="月 zip 路径列表")
    ap.add_argument("--out", required=True, help="输出连续期货面板 CSV")
    ap.add_argument("--contract", default="near", choices=["near", "second", "third"])
    args = ap.parse_args()
    cov = ingest(args.zips, Path(args.out), contract=args.contract)
    print(__import__("json").dumps(cov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
