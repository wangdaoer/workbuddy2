"""build_leakfree_nor_feed.py — 无泄漏 per-as-of 重建 NOR soft feed (方案 B P0-3 残留清理).

问题 (P0-3 残留, 见 leakfree_backtest_report.md §四):
  soft_score_feed.csv 由 production_soft_score.py 全量 walk-forward 构建, 混合验证
  validate_blend_leakfree.py 中 NOR 分量未做 per-as-of purge; 且 causal_soft_blend 的
  trailing IC 用 shift(1): IC[t] 需 label[t]=open[t+2]/open[t+1]-1, 而 label[t] 在 t+2 开盘
  才成熟 -> as-of d 的 trailing 用到 IC[d-1](需 open[d+1], d 收盘时未知) = 1 天未来信息泄漏.

修复 (与 build_leakfree_scorer_feed.py 同一纪律):
  1. 严格成熟训练窗口: 对信号日 i, 训练窗口末行 = i-2 (label[i-2]=open[i]/open[i-1]-1 在
     i 日收盘时成熟; label[i-1] 需 open[i+1] 未成熟 -> 排除). 用交易日历精确判定.
  2. trailing IC shift(1)->shift(2): as-of d 只用 IC <= d-2 (label[d-2]=open[d]/open[d-1]-1
     在 d 收盘时成熟), 杜绝 1 天未来信息.
  3. 每个 as-of 日强制重训: 该 as-of 的 soft 分由「仅用 <=as-of 成熟样本训练」的模型产出,
     与 TI leakfree feed 的 per-as-of 语义完全一致.
  4. 产物与 leakfree_feed_fwd60.csv 同构: 第一列 asof, 列=symbol, 值=soft 分;
     供 validate_blend_leakfree.py --nor-feed-leakfree 消费.
  5. 审计 leakfree_nor_audit.json: 每 as-of 记录重训窗口与 future_label_leak(=0 断言).

用法:
  python build_leakfree_nor_feed.py                              # 默认面板/as-of 窗口
  python build_leakfree_nor_feed.py --panel <panel.csv> --end 2026-08-21
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import next_open_return_label
from train_next_open_rank_model import (
    MIN_LIVE_SYMBOLS, build_features, clean_matrix, daily_ic, normalize_weights,
)
from p10c_ensemble import MLP, select_positive, MTH, RETRAIN, TRAIN_DAYS
import pit_universe as pit

# ---- 常量 (与 production_soft_score.py 保持一致, 不再 import 该模块以独立运行) ----
HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
OUT = HERE / "outputs" / "production_soft_score"
MAX_ABS = 0.22
LIQ_LOOKBACK = 252
LIQ_THR = 1e8
THR_HI = 0.03
THR_LO = 0.0
IC_WIN = 60
IC_MIN = 40
# next_open 标签 label[t] = open[t+2]/open[t+1]-1, 在 t 后第 2 个交易日开盘成熟.
LABEL_MATURITY_TD = 2
DEFAULT_START = "2025-06-03"   # 与 build_leakfree_scorer_feed.py 批量 as-of 窗口一致
DEFAULT_END = "2026-08-20"


def load_trading_dates(panel: Path) -> np.ndarray:
    """交易日历 (升序 datetime64[ns]), 用于精确判定标签成熟日. 与 leakfree TI 同口径."""
    df = pd.read_csv(panel, usecols=["date"])
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return np.sort(np.unique(d.to_numpy(dtype="datetime64[ns]").astype("datetime64[ns]")))


def _maturity_dates(ig: pd.Series, tdates: np.ndarray, h: int) -> pd.Series:
    """每条样本的真实标签成熟日 = 交易日历中第 h 个交易日 (对齐 label 口径 shift(-h)).
    面板不足 h 个交易日 -> NaT (保守视为未成熟)."""
    vals = ig.to_numpy(dtype="datetime64[ns]")
    out = pd.Series(pd.NaT, index=ig.index, dtype="datetime64[ns]")
    ok = ~pd.isna(vals)
    if not ok.any():
        return out
    pos = np.searchsorted(tdates, vals[ok], side="left")
    enough = (pos + h) < len(tdates)
    idx_ok = np.where(ok)[0][enough]
    out.iloc[idx_ok] = pd.Series(tdates[pos[enough] + h], index=ig.index[idx_ok])
    return out


def build_panel(panel_csv: Path) -> dict:
    """复制 production_soft_score.build_panel 的清洗/特征/标签逻辑 (用 --panel 面板)."""
    raw = pd.read_csv(panel_csv)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["symbol"] = raw["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    raw = raw.dropna(subset=["symbol", "date"])
    raw["symbol"] = raw["symbol"].astype(int).map(lambda s: f"{s:06d}")

    def _piv(col):
        return (raw.pivot_table(index="date", columns="symbol", values=col)
                    .sort_index().sort_index(axis=1))

    close = clean_matrix(_piv("close"), MAX_ABS)
    open_px = clean_matrix(_piv("open").reindex_like(close), MAX_ABS)
    high = clean_matrix(_piv("high").reindex_like(close), MAX_ABS)
    low = clean_matrix(_piv("low").reindex_like(close), MAX_ABS)
    amount = _piv("amount").reindex_like(close)
    symbols = list(close.columns)
    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    # universe=pit (与 production_soft_score 默认一致): 训练+打分共用诚实 PIT 宇宙掩码
    universe_mask = pit.pit_eligible(close, amount, None, thr=LIQ_THR,
                                     lookback=LIQ_LOOKBACK, min_days=60)
    feat_arrays = {k: fr.values for k, fr in features.items()}
    label_arr = label.values
    return dict(close=close, open_px=open_px, symbols=symbols, features=features, label=label,
                universe_mask=universe_mask, feat_arrays=feat_arrays, label_arr=label_arr)


def main() -> None:
    ap = argparse.ArgumentParser(description="无泄漏 per-as-of 重建 NOR soft feed")
    ap.add_argument("--panel", default=str(PANEL), help="价格面板 (与 validate_blend_leakfree 同源)")
    ap.add_argument("--start", default=DEFAULT_START, help="批量 as-of 窗口起点 (与 TI leakfree 对齐)")
    ap.add_argument("--end", default=DEFAULT_END, help="批量 as-of 窗口截止")
    ap.add_argument("--output-dir", default=str(OUT))
    ap.add_argument("--out-name", default="leakfree_nor_feed.csv")
    ap.add_argument("--audit-name", default="leakfree_nor_audit.json")
    args = ap.parse_args()

    t0 = time.time()
    panel_path = Path(args.panel)
    tdates = load_trading_dates(panel_path)
    P = build_panel(panel_path)
    symbols = P["symbols"]
    features = P["features"]
    label = P["label"]
    feat_arrays = P["feat_arrays"]
    label_arr = P["label_arr"]
    universe_mask = P["universe_mask"]
    fnames = list(features.keys())
    n = len(label.index)
    first = TRAIN_DAYS + MTH  # 253

    # as-of 列表与 TI leakfree 完全一致 (W-MON, 同窗口)
    asof_list = pd.date_range(args.start, args.end, freq="W-MON")
    asof_set = set(asof_list)
    asof_set_td = {pd.Timestamp(d).normalize() for d in asof_list}  # 供交易日匹配

    linear_score = pd.DataFrame(np.nan, index=label.index, columns=symbols)
    mlp_score = pd.DataFrame(np.nan, index=label.index, columns=symbols)
    current_weights = None
    selected = fnames
    mlp_fit = None
    cur_live_bool = np.ones(len(symbols), dtype=bool)
    cur_live_cols = np.arange(len(symbols))
    retrain_count = 0

    audit = []          # per-as-of 审计
    last_retrain = {"i": None, "date": None, "window_start": None, "window_end": None}

    for i in range(first, n):
        date = label.index[i]
        step = i - first
        is_asof = date in asof_set_td
        do_retrain = (step % RETRAIN == 0) or is_asof
        if do_retrain:
            live = universe_mask.loc[date]
            if live.sum() < MIN_LIVE_SYMBOLS:
                live = pd.Series(True, index=symbols)
            cur_live_bool = live.values.astype(bool)
            cur_live_cols = np.where(cur_live_bool)[0]
            # ---- 严格成熟训练窗口 [i-TRAIN_DAYS-MTH, i-MTH) = [i-253, i-2] ----
            # label[t]=open[t+2]/open[t+1]-1: 末行 t=i-2 的 label 需 open[i], i 日收盘时已知
            # (成熟); t=i-1 的 label 需 open[i+1], i 日收盘时未知 -> 被 iloc 排除. 无泄漏.
            start_tr = i - TRAIN_DAYS - MTH
            end_tr = i - MTH
            cols = cur_live_cols
            feat_slice = {f: features[f].iloc[start_tr:end_tr, cols] for f in fnames}
            label_slice = label.iloc[start_tr:end_tr, cols]
            training_ic = daily_ic(feat_slice, label_slice)
            mean_ic = training_ic.mean()
            selected = select_positive(training_ic)
            if len(selected) < 3:
                selected = fnames
            current_weights = normalize_weights(mean_ic[selected])

            Xs, ys = [], []
            for t in range(start_tr, end_tr):
                row = np.column_stack([feat_arrays[f][t, cols] for f in selected])
                yt = label_arr[t, cols]
                mask = ~np.isnan(row).any(axis=1) & ~np.isnan(yt)
                if mask.sum() < 50:
                    continue
                Xs.append(row[mask])
                ys.append(yt[mask])
            mlp_fit = MLP().fit(np.vstack(Xs), np.concatenate(ys)) if Xs else None
            retrain_count += 1
            last_retrain = {"i": i, "date": date, "window_start": start_tr, "window_end": end_tr}

            # ---- 泄漏自检 (as-of 日): 训练窗口末行 label 成熟日 <= as-of ----
            # 窗口 [start_tr, end_tr) 末行索引 = end_tr-1 = i-2, 其 label 成熟日 = 该行日期后
            # 第 LABEL_MATURITY_TD 个交易日 = i 日 (= 当前 as-of). as-of 收盘时已知 -> 无泄漏.
            if is_asof:
                last_date = label.index[end_tr - 1]
                mature = _maturity_dates(pd.Series([last_date]), tdates, LABEL_MATURITY_TD).iloc[0]
                leaked = 0 if (not pd.isna(mature) and mature <= date) else 1
                audit.append({
                    "asof": str(date.date()), "retrain_date": str(date.date()),
                    "window_start": str(label.index[start_tr].date()),
                    "window_end_excl": str(label.index[end_tr - 1].date()),
                    "window_last_row_maturity": str(mature.date()) if not pd.isna(mature) else "NaT",
                    "train_rows": end_tr - start_tr,
                    "live_symbols": int(live.sum()),
                    "future_label_leak": leaked,
                    "skipped": False,
                })
                if leaked > 0:
                    raise SystemExit(f"[leakfree] as-of {date.date()}: 训练窗口泄漏 {leaked} 样本!")

        if current_weights is not None:
            lin = np.zeros(len(symbols))
            for f, w in current_weights.items():
                lin = lin + feat_arrays[f][i] * w
            lin[~cur_live_bool] = np.nan
            linear_score.iloc[i] = lin
            if mlp_fit is not None:
                row = np.column_stack([feat_arrays[f][i, cur_live_cols] for f in selected])
                mask = ~np.isnan(row).any(axis=1)
                pred = np.full(cur_live_cols.size, np.nan)
                if mask.sum() > 0:
                    pred[mask] = mlp_fit.predict(row[mask])
                mlp_score.iloc[i, cur_live_cols] = pred
        # 注: soft 混合分在循环结束后统一计算 (trailing 需完整 IC 序列)；
        #     每 as-of 的分数即该日(强制重训后)的 linear/mlp 组合, 无额外采样。

    # ---- trailing IC (严格因果: shift(2), 修 shift(1) 泄漏) ----
    mlp_ic = daily_ic({"mlp": mlp_score}, label)["mlp"]
    trailing = mlp_ic.shift(2).rolling(IC_WIN, min_periods=IC_MIN).mean()
    adv = ((THR_HI - trailing) / (THR_HI - THR_LO)).clip(0, 1).fillna(1.0)
    alpha_dead = (trailing < THR_LO).fillna(False)
    soft = adv.values[:, None] * linear_score.values + (1 - adv.values[:, None]) * mlp_score.values
    soft = pd.DataFrame(soft, index=label.index, columns=symbols)

    # ---- as-of 采样输出 ----
    feed_rows = {}
    for d in asof_list:
        if d in soft.index:
            feed_rows[d] = soft.loc[d]
        else:
            feed_rows[d] = pd.Series(np.nan, index=symbols)
    feed = pd.DataFrame(feed_rows).T
    feed.index.name = "asof"
    feed = feed.sort_index()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    feed.reset_index().to_csv(out_path, index=False, encoding="utf-8")

    # 审计汇总
    n_lk = len(audit)
    n_skip = len(asof_list) - n_lk
    max_leak = max((a["future_label_leak"] for a in audit), default=0)
    audit_summary = {
        "note": ("leakfree NOR: 训练窗口 [i-253, i-2] 严格成熟 (label[t]=open[t+2]/open[t+1]-1, "
                 "末行 t=i-2 的 label 在信号日 i 收盘时已知); trailing IC 用 shift(2) "
                 "(as-of d 只用 IC<=d-2), 修 production_soft_score shift(1) 的 1 天泄漏."),
        "label_maturity_td": LABEL_MATURITY_TD,
        "asof_count": len(asof_list),
        "asof_with_retrain": n_lk,
        "asof_skipped_non_trading_day": n_skip,
        "max_future_label_leak": max_leak,
        "trailing_ic_shift": 2,
        "asof_range": [str(asof_list[0].date()), str(asof_list[-1].date())],
        "per_asof": audit,
    }
    audit_path = out_dir / args.audit_name
    audit_path.write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")

    n_non_nan = int(feed.notna().any(axis=1).sum())
    print(f"[leakfree-nor] 面板 {panel_path.name} 交易日 {len(tdates)} | as-of {len(asof_list)} "
          f"(重训 {n_lk}, 非交易日跳过 {n_skip}) | retrain 总次数 {retrain_count}")
    print(f"  max_future_label_leak = {max_leak} (应为 0)")
    print(f"  trailing IC shift(2): as-of {feed.index.min().date()} .. {feed.index.max().date()}, "
          f"非空行 {n_non_nan} | feed {feed.shape[0]} x {feed.shape[1]}")
    cov_at_last = feed.iloc[-1].notna().sum() if len(feed) else 0
    print(f"  末日 as-of {feed.index[-1].date()}: 非NaN 符号数 {int(cov_at_last)}")
    print(f"  -> {out_path}\n  -> {audit_path}")
    if max_leak > 0:
        raise SystemExit("[leakfree-nor] future_label_leak > 0, 中止!")
    print(f"  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
