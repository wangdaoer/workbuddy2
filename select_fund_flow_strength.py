# -*- coding: utf-8 -*-
"""
优中选优 · 强中选强 —— 主力资金流选股（基于 ths_hs_a_share 单源）

设计要点（按用户要求）：
1) 主数据源统一用 D:/codex/daily-market-data/ths_exports/normalized 下的
   ths_hs_a_share_*.xls / *.csv。其中 .xls 含「大单净额」「特大单净额」「总市值」
   「现价」「昨收」「主力净量」等资金流字段；.csv 为英文列基础行情（无资金流），
   仅用于补全价格/市值。脚本自动识别两种 schema。
2) 每日增量主表（master_a_share.csv）：只解析“尚未入库”的交易日文件，幂等追加，
   不重复解析历史，满足“每日新增、避免重复工作”。
3) 资金净额 = 大单净额 + 特大单净额（单位：元）。
4) 核心维度 = 资金净额 / 总市值（占比越高越好）。
5) 预筛：剔除 ST/*ST/PT；剔除涨停近似（现价/昨收-1 ≥ 9.5%）；
   市值 ∈ [30亿, 5000亿]；股价 ≤ 200 元；绝对净额 ≥ 5000万（大钱真流入门槛，含涨停板）。
6) 四维度（强中选强，跨日/当日结合）：
   A 当日净额/市值占比 0.30（核心，越高越好）
   B 强度=绝对净额规模 0.20（大钱真流入）
   C 持续净流入 0.25（近N日净额为正的交易日占比，跨日）
   D 价格趋势 0.25（近N日价格区间涨幅，跨日）
   跨日维度来自 master 的历史交易日快照（非单日），实现“既有当日、
   又有持续、有强度、有趋势”。

用法：
  python select_fund_flow_strength.py                 # 选最新交易日 top30
  python select_fund_flow_strength.py --date 2026-08-19 --top-n 30
  python select_fund_flow_strength.py --rebuild       # 清空主表重建（慎用）
  python select_fund_flow_strength.py --no-master     # 跳过主表增量（直接用源文件）

只读取源（只读），只写入 outputs/fund_flow_selection/。
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# ── 路径配置 ────────────────────────────────────────────────
SRC_DIR = "D:/codex/daily-market-data/ths_exports/normalized"
OUT_DIR = "outputs/fund_flow_selection"
MASTER_PATH = os.path.join(OUT_DIR, "master_a_share.csv")

# ── 可调参数 ────────────────────────────────────────────────
CONFIG = {
    "min_cap": 3e9,          # 市值下限：30 亿
    "max_cap": 5e11,         # 市值上限：5000 亿
    "max_price": 200.0,      # 股价上限：200 元
    "limit_up_pct": 0.095,   # 涨停近似阈值（覆盖主板10%/双创20%）
    "exclude_st": True,      # 剔除 ST/*ST/PT
    "exclude_limit_up": False,  # 是否剔除涨停板；False=保留（含涨停）
    "min_abs_net": 5e7,      # 绝对净额下限：≥5000 万（大钱真流入门槛）
    "top_n": 50,
    "window": 5,             # 持续性/趋势回看交易日数（不含当日）
    "weights": {             # 四维度权重（合计 1.0）
        "ratio": 0.30,      # A 当日净额/市值占比（核心）
        "absnet": 0.20,     # B 强度：绝对净额规模
        "persist": 0.25,    # C 持续净流入（近N日正净流入占比）
        "trend": 0.25,      # D 价格趋势（近N日涨幅）
    },
}

_pct = lambda s: s.rank(pct=True) if len(s) else s


# ── 鲁棒读取 ths_hs_a_share（xls=GBK/TSV，csv=UTF8/逗号）──────
def _read_ths(path):
    for sep, enc in (("\t", "gb18030"), ("\t", "utf-8-sig"),
                     (",", "utf-8-sig"), (",", "gb18030"), ("\t", "utf-8")):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
            if len(df.columns) > 3:
                return df
        except Exception:
            pass
    return None


def _code6(s):
    s = str(s)
    d = re.sub(r"\D", "", s)
    return d[-6:].zfill(6) if d else ""


def _clean_name(s):
    s = str(s)
    # 去除 THS 给中文名加的字间距（如 “红 宝 丽”）
    return re.sub(r"\s+", "", s).strip()


def _parse_file(path, date, source):
    """解析单个 ths_hs_a_share 文件 → 主表行。"""
    raw = _read_ths(path)
    if raw is None:
        return None
    raw.columns = [c.strip() for c in raw.columns]

    rec = {
        "date": date, "source": source,
        "code6": "", "name": "", "price": np.nan, "prev_close": np.nan,
        "change_pct": np.nan, "large_net": np.nan, "super_net": np.nan,
        "net_amount": np.nan, "main_net_volume": np.nan, "turnover": np.nan,
        "market_cap": np.nan, "industry_sub": "", "industry": "",
    }

    if "代码" in raw.columns:  # xls（中文列，含资金流）
        rec["code6"] = raw["代码"].map(_code6)
        rec["name"] = raw["名称"].map(_clean_name)
        rec["price"] = pd.to_numeric(raw.get("现价"), errors="coerce")
        rec["prev_close"] = pd.to_numeric(raw.get("昨收"), errors="coerce")
        cp = pd.to_numeric(raw.get("涨幅").astype(str).str.replace("%", "", regex=False)
                           .str.replace(",", "", regex=False), errors="coerce")
        if cp.notna().any():
            rec["change_pct"] = cp
        rec["large_net"] = pd.to_numeric(raw.get("大单净额"), errors="coerce")
        rec["super_net"] = pd.to_numeric(raw.get("特大单净额"), errors="coerce")
        rec["main_net_volume"] = pd.to_numeric(raw.get("主力净量"), errors="coerce")
        rec["turnover"] = pd.to_numeric(raw.get("换手").astype(str).str.replace("%", "", regex=False)
                                       .str.replace(",", "", regex=False), errors="coerce")
        rec["market_cap"] = pd.to_numeric(raw.get("总市值"), errors="coerce")
        rec["industry_sub"] = raw.get("细分行业", pd.Series([""] * len(raw)))
        rec["industry"] = raw.get("所属行业", pd.Series([""] * len(raw)))
        ln = pd.to_numeric(raw.get("大单净额"), errors="coerce")
        sm = pd.to_numeric(raw.get("特大单净额"), errors="coerce")
        rec["net_amount"] = (ln + sm)
    elif "security_code" in raw.columns:  # csv（英文列，无资金流）
        rec["code6"] = raw["security_code"].map(_code6)
        rec["name"] = raw.get("security_name", pd.Series([""] * len(raw))).map(_clean_name)
        rec["price"] = pd.to_numeric(raw.get("close_price"), errors="coerce")
        rec["prev_close"] = pd.to_numeric(raw.get("prev_close"), errors="coerce")
        rec["change_pct"] = pd.to_numeric(raw.get("change_ratio"), errors="coerce")
        rec["market_cap"] = pd.to_numeric(raw.get("market_cap"), errors="coerce")
        rec["turnover"] = pd.to_numeric(raw.get("turnover_rate"), errors="coerce")
        # csv 无大单/特大单 → net_amount 保持 NaN
    else:
        return None

    df = pd.DataFrame(rec)
    df = df[df["code6"] != ""]
    return df


def _candidate_files(src_dir):
    """列出 ths_hs_a_share_YYYY-MM-DD.(xls|csv)，排除 THS 页面状态快照
    (*.prev-*.xls / *.moneyflow-page-prev-*.xls / *.self200-prev-*.xls)。
    按交易日分组，优先取 .xls（含资金流）。"""
    pat = re.compile(r"ths_hs_a_share_(\d{4}-\d{2}-\d{2})\.(xls|csv)$")
    by_date = {}
    for p in glob.glob(os.path.join(src_dir, "ths_hs_a_share_*")):
        m = pat.match(os.path.basename(p))
        if not m:
            continue
        date, ext = m.group(1), m.group(2)
        cur = by_date.get(date)
        # 优先 xls
        if cur is None or (ext == "xls" and cur[1] != "xls"):
            by_date[date] = (p, ext)
    return by_date


def build_master(src_dir=SRC_DIR, rebuild=False):
    """增量构建主表。返回 (master_df, added_dates_list)。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    existing = pd.DataFrame()
    existing_dates = set()
    if not rebuild and os.path.exists(MASTER_PATH):
        existing = pd.read_csv(MASTER_PATH, dtype={"code6": str})
        if "date" in existing.columns:
            existing_dates = set(existing["date"].astype(str))

    cands = _candidate_files(src_dir)
    new_frames = []
    added = []
    for date in sorted(cands.keys()):
        if date in existing_dates:
            continue
        path, ext = cands[date]
        df = _parse_file(path, date, ext)
        if df is not None and len(df):
            new_frames.append(df)
            added.append(date)

    if new_frames:
        combined = pd.concat([existing] + new_frames, ignore_index=True)
        # 去重保险（同 date+code6 仅保留一行，后写优先）
        combined = combined.drop_duplicates(subset=["date", "code6"], keep="last")
        combined.to_csv(MASTER_PATH, index=False)
        print(f"[主表] 新增 {len(added)} 个交易日：{added[0]} … {added[-1]}"
              f"（累计 {combined['date'].nunique()} 日 / {len(combined)} 行）")
    else:
        combined = existing
        print(f"[主表] 无新增交易日（已含 {len(existing_dates)} 日）。")
    return combined, added


def _prior_aggregates(master, date, window):
    """跨日计算：对 date 之前最近 window 个交易日的每只标的，
    返回 (pos_frac=近window日净额为正的交易日占比,
          trend_ret=近window日价格区间涨幅)。
    单次 groupby，避免逐行查询。"""
    prior = master[(master["date"] < date) & (master["net_amount"].notna())].copy()
    if prior.empty:
        return pd.DataFrame(columns=["pos_frac", "trend_ret"])
    prior = prior.sort_values(["code6", "date"])
    prior["rn"] = prior.groupby("code6").cumcount(ascending=False)  # 0=最近一个 prior 日
    tail = prior[prior["rn"] < window]
    if tail.empty:
        return pd.DataFrame(columns=["pos_frac", "trend_ret"])
    g = tail.groupby("code6")
    agg = g.agg(pos_frac=("net_amount", lambda s: float((s > 0).mean())),
                first_price=("price", "first"),
                last_price=("price", "last"))
    agg["trend_ret"] = agg["last_price"] / agg["first_price"] - 1
    return agg[["pos_frac", "trend_ret"]]


def select(master, date, top_n=30, cfg=CONFIG):
    """在指定交易日做优中选优·强中选强选股，返回含维度的结果 df。"""
    w = cfg["weights"]
    win = cfg["window"]

    df = master[master["date"] == date].copy()
    if df.empty:
        raise SystemExit(f"[错误] 主表中无 {date} 数据，请先 build_master。")
    # 仅保留含净额（xls 来源）的标的
    df = df[df["net_amount"].notna()].copy()
    if df.empty:
        raise SystemExit(f"[错误] {date} 当日无资金流数据（可能仅有 csv 源）。")

    n0 = len(df)

    # ── 第一层「优」预筛 ──
    cap = df["market_cap"]
    price = df["price"]
    # 涨停近似：现价/昨收 - 1（昨收缺失则回退涨幅）
    pct = df["change_pct"]
    derived = df["price"] / df["prev_close"] - 1
    pct = pct.where(pct.notna(), derived)
    df["change_pct"] = pct  # 兜底填充，供展示与涨停判断
    df["_pct"] = pct

    mask = pd.Series(True, index=df.index)
    if cfg["exclude_st"]:
        mask &= ~df["name"].str.contains("ST", case=False, na=False)
    mask &= cap.between(cfg["min_cap"], cfg["max_cap"])
    mask &= price <= cfg["max_price"]
    if cfg.get("exclude_limit_up"):
        mask &= ~(df["_pct"] >= cfg["limit_up_pct"])  # 剔除涨停近似（默认关闭→保留）
    df = df[mask].copy()
    n1 = len(df)

    # 绝对净额下限（≥5000万）：大钱真流入门槛
    df = df[df["net_amount"] >= cfg["min_abs_net"]].copy()
    n2 = len(df)

    # ── 第二层「强中选强」复合打分（跨日 + 当日）──
    # A 当日净额/市值占比（核心，越高越好）
    df["A_ratio"] = df["net_amount"] / df["market_cap"]
    # B 强度：绝对净额规模（大钱）
    df["B_absnet"] = df["net_amount"]
    # C 持续净流入 + D 价格趋势（跨日，来自 master 历史快照）
    agg = _prior_aggregates(master, date, cfg["window"])
    df = df.merge(agg, left_on="code6", right_index=True, how="left")
    df["C_persist"] = df["pos_frac"].fillna(df["pos_frac"].median())
    df["D_trend"] = df["trend_ret"].fillna(df["trend_ret"].median())

    df["pr_A"] = _pct(df["A_ratio"])
    df["pr_B"] = _pct(df["B_absnet"])
    df["pr_C"] = _pct(df["C_persist"])
    df["pr_D"] = _pct(df["D_trend"])

    df["composite"] = (w["ratio"] * df["pr_A"] + w["absnet"] * df["pr_B"]
                       + w["persist"] * df["pr_C"] + w["trend"] * df["pr_D"]) * 100

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    _net_yi = cfg["min_abs_net"] / 1e8
    limit_txt = f"≥{_net_yi:.0f}亿" if _net_yi >= 1 else f"≥{int(_net_yi*10)}000万"
    meta = {
        "date": date, "universe": n0, "qualified": n2,
        "detail": (f"全市场含净额 {n0} 只 → 预筛合格 {n1} 只 "
                   f"(剔除 ST/市值<30亿或>5000亿/股价>200元"
                   f"{'' if cfg.get('exclude_limit_up') else '/含涨停板'}) "
                   f"→ 绝对净额{limit_txt} {n2} 只"),
    }
    return df, meta


def _write_outputs(df, meta, top_n, cfg=CONFIG):
    date = meta["date"]
    win = cfg["window"]
    top = df.head(top_n)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"strong_fund_flow_{date}.csv")
    md_path = os.path.join(OUT_DIR, f"strong_fund_flow_{date}.md")

    out_cols = ["rank", "code6", "name", "price", "change_pct",
                "large_net", "super_net", "net_amount", "A_ratio",
                "C_persist", "D_trend", "composite",
                "market_cap", "industry"]
    out = top[out_cols].copy()
    out = out.rename(columns={
        "rank": "排名", "code6": "代码", "name": "名称", "price": "现价", "change_pct": "涨幅%",
        "large_net": "大单净额(元)", "super_net": "特大单净额(元)",
        "net_amount": "资金净额(元)", "A_ratio": "净额/市值",
        "C_persist": f"近{win}日正净流入占比", "D_trend": f"近{win}日涨幅%",
        "composite": "综合分", "market_cap": "总市值(元)", "industry": "行业",
    })
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    w = cfg["weights"]
    lines = [f"# 优中选优·强中选强 主力资金流选股 — {date}", ""]
    lines.append(f"> {meta['detail']}")
    lines.append("")
    lines.append("**维度与权重**（资金净额 = 大单净额 + 特大单净额，单位元）：")
    lines.append(f"- A 当日净额/总市值占比（核心，越高越好）：权重 {w['ratio']:.0%}")
    lines.append(f"- B 强度=绝对净额规模（大钱真流入）：权重 {w['absnet']:.0%}")
    lines.append(f"- C 持续净流入（近{win}日净额为正的交易日占比，跨日）：权重 {w['persist']:.0%}")
    lines.append(f"- D 价格趋势（近{win}日价格区间涨幅，跨日）：权重 {w['trend']:.0%}")
    lines.append("")
    lines.append(f"## Top {top_n}")
    lines.append("")
    hdr = f"|排名|代码|名称|现价|涨幅%|资金净额(元)|净额/市值|近{win}日正净流入占比|近{win}日涨幅%|综合分|行业|"
    lines.append(hdr)
    lines.append("|" + "---|" * (hdr.count("|") - 1) + "---|")
    for _, r in out.iterrows():
        lines.append(
            f"|{int(r['排名'])}|{r['代码']}|{r['名称']}|{r['现价']:.2f}|"
            f"{r['涨幅%']:.2f}|{r['资金净额(元)']:.0f}|{r['净额/市值']*100:.3f}%|"
            f"{r[f'近{win}日正净流入占比']*100:.1f}%|{r[f'近{win}日涨幅%']*100:.1f}%|"
            f"{r['综合分']:.1f}|{r['行业']}|")
    lines.append("")
    lines.append("> 说明：数据源自 ths_hs_a_share 单源（xls 含资金流）；主表每日增量、幂等不重跑。")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return csv_path, md_path


def main():
    ap = argparse.ArgumentParser(description="优中选优·强中选强 主力资金流选股")
    ap.add_argument("--date", help="选股交易日 YYYY-MM-DD（默认主表最新一日）")
    ap.add_argument("--top-n", type=int, default=CONFIG["top_n"])
    ap.add_argument("--src", default=SRC_DIR, help="ths_hs_a_share 源目录")
    ap.add_argument("--rebuild", action="store_true", help="清空主表重建")
    ap.add_argument("--no-master", action="store_true",
                    help="跳过主表增量（直接用源目录现有文件）")
    args = ap.parse_args()

    if args.no_master and os.path.exists(MASTER_PATH):
        master = pd.read_csv(MASTER_PATH, dtype={"code6": str})
        print(f"[主表] 跳过增量，直接读取 {MASTER_PATH}（{master['date'].nunique()} 日）")
    else:
        master, _ = build_master(args.src, rebuild=args.rebuild)

    if args.date:
        date = args.date
    else:
        date = str(master["date"].max())
        print(f"[默认] 选股交易日 = 主表最新一日 {date}")

    df, meta = select(master, date, top_n=args.top_n)
    csv_path, md_path = _write_outputs(df, meta, args.top_n)

    print(f"[预筛] {meta['detail']}")
    print(f"[输出] {csv_path}")
    print(f"[输出] {md_path}")
    print(f"\n=== Top {args.top_n} 预览 ===")
    prev = df.head(args.top_n)[
        ["rank", "code6", "name", "price", "net_amount", "A_ratio",
         "C_persist", "D_trend", "composite"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(prev.to_string(index=False))


if __name__ == "__main__":
    main()
