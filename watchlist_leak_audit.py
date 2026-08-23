"""watchlist_leak_audit.py — 量化用户"每日自选股名单"的前视污染度。

输入：D:/codex/outputs/stock-analysis-dashboard/input/ths_money_flow_YYYY-MM-DD.xls
      （同花顺资金流导出，TSV/GBK，列含 代码 名称 ...）
输出：
  1) 每日名单的"内部前视"：某日名单里有多少票，在面板里当天根本不在市/停牌/不流动
     （=用户自己把未来或已退市/未上市的票放进了当天名单）。
  2) 静态复用污染：若把"末日名单"或"全集"当固定宇宙回测历史，detect_static_leak 给出污染度。
  3) 诚实 PIT 自选宇宙：票在日 t 合格当且仅当出现在 ≤t 的名单里（无前视用法）。

需先有 Tdx 面板 external_data/daily-market-data-tdx/data_panel.csv（构建 wide close/amount）。
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np
import pandas as pd

WATCH_DIR = r"D:\codex\outputs\stock-analysis-dashboard\input"
PANEL = r"external_data/daily-market-data/data_panel.csv"
OUT_DIR = r"outputs/watchlist_audit"
# 自选名单合理性护栏：真实名单 129~354 只/日；超过该阈值 = 全市场误导出
# （如 2026-07-02 的 5204 行垃圾文件）。此类文件会经 PIT 累积逻辑把宇宙
# 永久污染成近全市场，必须跳过。
MAX_WATCHLIST_SYMBOLS = 1500

CODE_RE = re.compile(r'^([A-Z]{2})(\d{6})$')  # SZ300196

def norm_code(c: str) -> str | None:
    c = (c or "").strip().upper()
    m = CODE_RE.match(c)
    return m.group(2) if m else None

def find_col(header, key):
    for i, h in enumerate(header):
        if key in h.replace(' ', ''):
            return i
    return -1

def load_daily_files():
    """返回 (files: list[(date_str, path)], per_day: dict[date]->set(codes), names: dict[code]->name)

    注意：目录里存在 'xxx.prev-154115.xls' 备份文件，列布局与正式文件不同，
    且同一日期可能有两个文件。按日期去重，优先保留非 .prev 的正式文件。
    """
    pat = os.path.join(WATCH_DIR, "ths_money_flow_*.xls")
    # 收集每个日期对应的候选文件，去重
    cand = {}  # date -> (is_prev, path)
    for fp in glob.glob(pat):
        base = os.path.basename(fp)
        m = re.search(r'(\d{4}-\d{2}-\d{2})', base)
        if not m:
            continue
        d = m.group(1)
        is_prev = '.prev' in base
        if d not in cand or (not cand[d][0] and is_prev):
            # 优先非 prev
            if d not in cand:
                cand[d] = (is_prev, fp)
            elif is_prev is False and cand[d][0] is True:
                cand[d] = (is_prev, fp)
    files = []
    per_day = {}
    names = {}
    for d in sorted(cand.keys()):
        fp = cand[d][1]
        codes = set()
        with open(fp, encoding='gbk', errors='replace') as fh:
            header = [h.replace(' ', '') for h in fh.readline().rstrip('\n').split('\t')]
            ci = find_col(header, '代码')
            ni = find_col(header, '名称')
            if ci < 0:
                ci = 0
            if ni < 0:
                ni = 1
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) <= ci:
                    continue
                code = norm_code(parts[ci])
                if code:
                    codes.add(code)
                    if len(parts) > ni and ni != ci:
                        names[code] = parts[ni]
        if len(codes) > MAX_WATCHLIST_SYMBOLS:
            print(f"[warn] 跳过疑似全市场误导出: {os.path.basename(fp)} "
                  f"({len(codes)} 只 > {MAX_WATCHLIST_SYMBOLS})，防止污染 PIT 宇宙")
            continue
        if codes:
            files.append((d, fp))
            per_day[d] = codes
    return files, per_day, names

def load_panel_wide():
    use = ['date', 'symbol', 'close', 'amount']
    df = pd.read_csv(PANEL, usecols=use)
    # 面板 symbol 被 pandas 解析为 int（000001->1）；统一转回 6 位零填充字符串
    df['symbol'] = df['symbol'].astype(int).astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['close'])
    wide_close = df.pivot(index='date', columns='symbol', values='close')
    wide_amount = df.pivot(index='date', columns='symbol', values='amount')
    return wide_close, wide_amount

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files, per_day, names = load_daily_files()
    print(f"[parse] {len(files)} 个每日名单, 区间 {files[0][0]} ~ {files[-1][0]}")
    all_codes = set().union(*per_day.values())
    print(f"[parse] 每日平均 {np.mean([len(v) for v in per_day.values()]):.0f} 只, 全集 {len(all_codes)} 只")

    close, amount = load_panel_wide()
    panel_dates = close.index
    panel_syms = set(close.columns)
    print(f"[panel] {len(panel_dates)} 交易日, {len(panel_syms)} 只标的")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pit_universe as pit

    # PIT 资格必须在【全历史】上算（rolling 流动性需要长窗口），再按日切片
    elig_full = pit.pit_eligible(close, amount, None)

    # ---- 1) 每日名单"内部前视"检查 ----
    # 把文件日期映射到面板里 <=该日 的最近交易日（名单反映当日收盘）
    rows = []
    unknown_total = 0
    for d, fp in files:
        dt = pd.to_datetime(d)
        # 最近交易日 <= dt
        mask = panel_dates <= dt
        if not mask.any():
            tdate = panel_dates[0]
        else:
            tdate = panel_dates[mask][-1]
        codes = per_day[d]
        unknown = [c for c in codes if c not in panel_syms]
        unknown_total += len(unknown)
        present = [c for c in codes if c in panel_syms]
        # 当日是否在面板里（有收盘价）——不在= Phantom/未来票（硬前视）
        present_on_day = close.loc[tdate, present].notna() if present else pd.Series(dtype=bool)
        n_absent = int((~present_on_day).sum())
        # 在面板里但 PIT 不合格（停牌/不流动/上市不足）
        elig_day = elig_full.reindex(columns=present).loc[tdate] if present else pd.Series(dtype=bool)
        n_present_but_ill = int((present_on_day & ~elig_day).sum())
        n = len(codes)
        rows.append({
            'file_date': d, 'trade_date': str(tdate.date()),
            'n': n, 'unknown_not_in_panel': len(unknown),
            'absent_on_day': n_absent,
            'present_but_illiquid': n_present_but_ill,
        })
    daily_df = pd.DataFrame(rows)
    daily_df.to_csv(os.path.join(OUT_DIR, 'daily_internal_leak.csv'), index=False)
    print(f"\n[内部前视] 未知代码(不在面板)合计 {unknown_total} 次出现")
    print(daily_df.to_string(index=False))
    # 汇总
    tot = len(files)
    absent_days = (daily_df['absent_on_day'] > 0).sum()
    ill_days = (daily_df['present_but_illiquid'] > 0).sum()
    print(f"[内部前视汇总] {tot} 天中，{absent_days} 天含'当日不在面板'的票(Phantom/未来票硬前视)，"
          f"{ill_days} 天含'在市但不合格(停牌/不流动)'的票")

    # ---- 2) 静态复用污染 ----
    # 末日名单（常见错误：用今天自选股回测历史）
    last_date = files[-1][0]
    final_list = list(per_day[last_date])
    # 全集
    union_list = list(all_codes)
    for tag, lst in [('final_day', final_list), ('union', union_list)]:
        try:
            r = pit.detect_static_leak(close, amount, None, static_list=lst)
            print(f"\n[静态复用:{tag}] n={r['n_watchlist']} "
                  f"mean_leak={r['mean_leak_ratio']:.3f} max={r['max_leak_ratio']:.3f} min={r['min_leak_ratio']:.3f}")
        except Exception as e:
            print(f"[静态复用:{tag}] 跳过: {e}")

    # ---- 3) 诚实 PIT 自选宇宙 ----
    # 构造 mask: 票 c 在日 t 合格 iff 存在文件日 d<=t 且 c 在 per_day[d]
    # 取与面板对齐的交易日序列
    dates = panel_dates
    # 每个文件日映射到 <= 的最近交易日
    def map_td(dt):
        m = panel_dates <= dt
        return panel_dates[m][-1] if m.any() else None
    # 构建 cumulative membership
    membership = pd.DataFrame(False, index=dates, columns=sorted(all_codes))
    for d, codes in per_day.items():
        tdate = map_td(pd.to_datetime(d))
        if tdate is not None:
            for c in codes:
                if c in membership.columns:
                    membership.loc[tdate:, c] = True
    # 与面板实际可交易取交集（再平衡日 t 仅当面板里也在市）
    pit_wl = (membership & close.notna() & (amount.fillna(0) > 0))
    n_elig_by_date = pit_wl.sum(axis=1)
    print(f"\n[诚实PIT自选宇宙] 平均每日合格 {n_elig_by_date.mean():.0f} 只, "
          f"首/末交易日合格 {int(n_elig_by_date.iloc[0])}/{int(n_elig_by_date.iloc[-1])}")
    # 存为 mask 供模型接入
    try:
        pit_wl.astype('float16').to_parquet(os.path.join(OUT_DIR, 'watchlist_pit_mask.parquet'))
        print(f"[保存] PIT 自选宇宙掩码 -> {OUT_DIR}/watchlist_pit_mask.parquet")
    except Exception as e:
        # 回退：存 npz
        np.savez(os.path.join(OUT_DIR, 'watchlist_pit_mask.npz'),
                 mask=pit_wl.values.astype(float), index=pit_wl.index.astype(str),
                 columns=np.array(pit_wl.columns, dtype=str))
        print(f"[保存] PIT 自选宇宙掩码 -> {OUT_DIR}/watchlist_pit_mask.npz (parquet 不可用: {e})")


def load_watchlist_mask(close: pd.DataFrame, out_dir: str = OUT_DIR) -> pd.DataFrame:
    """载入 PIT 自选宇宙掩码并对齐到面板（close）的 index/columns。

    返回与 close 同形的 bool DataFrame：日 t、票 c 合格当且仅当 c 曾出现在 ≤t 的
    每日名单且当日 PIT 可交易。未在面板中的票/日填 False。
    """
    parquet = os.path.join(out_dir, 'watchlist_pit_mask.parquet')
    npz = os.path.join(out_dir, 'watchlist_pit_mask.npz')
    if os.path.exists(parquet):
        m = pd.read_parquet(parquet)
    elif os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        m = pd.DataFrame(d['mask'], index=d['index'].astype(str),
                         columns=d['columns'].astype(str))
    else:
        raise FileNotFoundError("未找到 watchlist 掩码，请先运行 watchlist_leak_audit.py")
    m.index = pd.to_datetime(m.index)
    m_raw = m
    mask_last = m.index.max()
    m = m.reindex(index=close.index, columns=close.columns)
    # 尾部前向填充：面板比掩码新的日期（用户当天尚未导出名单文件）沿用最后一次
    # 已知的累积名单成员。掩码语义是"≤t 曾出现过"，故沿用 ≤mask_last 的信息，
    # 不引入任何未来信息（PIT 安全）。不做 ffill 会让这些日期全部变 False，
    # 导致 user_watchlist / 配额静默失效且无告警。
    tail = m.index > mask_last
    m = m.fillna(False).astype(bool)
    if tail.any():
        n_tail = int(tail.sum())
        last_row = (m_raw.loc[mask_last].reindex(close.columns)
                    .fillna(False).astype(bool))
        m.loc[tail] = last_row.values
        print(f"[mask] 面板比掩码新 {n_tail} 天（掩码末日 {mask_last.date()}）"
              f"→ 已按 PIT 语义前向沿用最后一次名单成员（{int(last_row.sum())} 只）")
    return m


if __name__ == "__main__":
    main()
