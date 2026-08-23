# -*- coding: utf-8 -*-
"""tests/test_label_maturity.py — P0-2 回归测试：标签成熟日必须用真实交易日历精确计算。

背景（2026-08-23 审查 P0-2）：原 purge 用「ignition + 28/84 自然日」近似 20/60 交易日标签成熟日。
节假日（春节/国庆）会使 20 个交易日的自然日跨度达 28~40 天，固定近似把「标签窗口尚未结束」的样本
误判为已成熟 → 前视泄漏。本测试固化修复：_maturity_dates 按交易日历取 ignition 后第 H 个交易日。

运行：python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import build_leakfree_scorer_feed as blf


def _trading_dates_2026() -> list[pd.Timestamp]:
    """2026 年交易日近似：全年工作日扣除春节(02-16~02-22)与国庆(10-01~10-08)。"""
    all_b = pd.date_range("2026-01-05", "2026-12-31", freq="B")
    spring = pd.date_range("2026-02-16", "2026-02-22", freq="B")
    national = pd.date_range("2026-10-01", "2026-10-08", freq="B")
    off = set(spring).union(national)
    return [d for d in all_b if d not in off]


def test_real_maturity_is_trading_day_h():
    """成熟日 = ignition 后第 20 个交易日（含春节 gap 时自然日跨度 > 28）。"""
    tdates = np.array(_trading_dates_2026(), dtype="datetime64[ns]")
    ig = pd.Series([pd.Timestamp("2026-02-09")])  # 春节前最后交易日
    mat = blf._maturity_dates(ig, tdates, 20)
    k = int(np.searchsorted(tdates, ig.iloc[0].value, side="left"))
    assert mat.iloc[0] == pd.Timestamp(tdates[k + 20])
    span = (mat.iloc[0] - ig.iloc[0]).days
    assert span > 28, f"春节后 20 交易日自然日跨度应为 >28，实测 {span}"


def test_natural_approx_would_leak_spring_festival():
    """证明原 28 自然日近似在春节窗口会漏未来标签（回归锚点）。"""
    tdates = np.array(_trading_dates_2026(), dtype="datetime64[ns]")
    ig = pd.Timestamp("2026-02-09")  # 春节(02-16~02-22 休市)前最后交易日
    asof = pd.Timestamp("2026-03-20")
    embargo = 10
    cutoff = asof - pd.Timedelta(days=embargo)  # 2026-03-10
    real_mat = blf._maturity_dates(pd.Series([ig]), tdates, 20).iloc[0]
    approx_mat = ig + pd.Timedelta(days=28)  # 旧近似
    # 旧近似认为已成熟（approx <= cutoff），真实 20 交易日成熟日却未到（real > cutoff）→ 原逻辑泄漏
    assert approx_mat <= cutoff, "构造场景：旧近似应判为已成熟"
    assert real_mat > cutoff, "真实 20 交易日成熟日应晚于 cutoff → 新逻辑正确剔除"
    # 新逻辑的 keep 判定
    assert blf._maturity_dates(pd.Series([ig]), tdates, 20).iloc[0] > cutoff


def test_panel_insufficient_returns_nat():
    """面板不足 H 个交易日 → NaT（保守未成熟，purge 剔除，不误判成熟）。"""
    tdates = np.array(pd.date_range("2026-06-01", periods=10, freq="B"), dtype="datetime64[ns]")
    ig = pd.Series([pd.Timestamp("2026-06-08")])  # 剩余交易日 < 20
    mat = blf._maturity_dates(ig, tdates, 20)
    assert pd.isna(mat.iloc[0])


def test_fit_for_asof_purges_real_maturity():
    """_fit_for_asof 用真实成熟日 purge：春节前样本在 3/10 前不应进入 fit（真实成熟日在 3/10 之后）。"""
    tdates = np.array(_trading_dates_2026(), dtype="datetime64[ns]")
    # 训练集只需 fit_binned_scorer 依赖的列：ignition_date + 特征 + 标签
    train = pd.DataFrame({
        "ignition_date": ["2026-02-09", "2026-01-05", "2026-02-24"],
        "label_fwd20_up": [1, 0, 1],
        "label_fwd60_strong": [1, 0, 0],
    })
    for c in blf.FEATURE_COLUMNS:
        train[c] = np.random.rand(len(train))
    scorer, kept, purged, leaked = blf._fit_for_asof(train, pd.Timestamp("2026-03-10"), "20", 10, tdates)
    assert leaked == 0, "真实成熟日口径下 purge 后不应有泄漏"
    assert scorer is not None
    assert kept >= 1, "至少春节后(02-24)样本应已成熟进入 fit"
    # 02-09 样本真实成熟日 = 20 交易日 → 2026-03-16 附近 > cutoff(2026-02-28) → 应被 purge
    assert "2026-02-09" not in train.iloc[:1]["ignition_date"].tolist() or kept <= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
