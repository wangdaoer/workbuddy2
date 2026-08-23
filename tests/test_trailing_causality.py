# -*- coding: utf-8 -*-
"""tests/test_trailing_causality.py — P0-3 残留回归测试：trailing IC 必须用 shift(2)。

背景（2026-08-23）：NOR 软打分标签 label[t] = open[t+2]/open[t+1]-1（next_open_return_label,
horizon_days=1），**t 行的标签在 t+2 开盘才成熟**。因此因果 trailing 在 as-of d（d 日收盘后）
只能用 IC <= d-2：
  - 原 `causal_soft_blend` 用 `mlp_ic.shift(1)`：trailing[d] 用到 IC[d-1]（需 label[d-1]=open[d+1]/
    open[d]，open[d+1] 明日开盘在 d 收盘时未知）= 1 天未来信息泄漏。
  - 修复：shift(1)→shift(2)。本测试固化：
      1. shift(2) 的 trailing **免疫**于未来标签（把末日两行标签塞入有效值，trailing 逐位不变）；
      2. 对照 shift(1) 会受影响（证明 shift(1) 确实泄漏）；
      3. causal_soft_blend 产出的 trailing == 手算 shift(2)（防回归改回 shift(1)）。

运行：python -m pytest tests/test_trailing_causality.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import production_soft_score as ps


def _synthetic():
    """构造最小合成面板：200 交易日 × 50 标的，标签口径与生产一致。
    返回 (lin, mlp, label, syms, open_px)。"""
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    syms = [f"{i:06d}" for i in range(50)]
    rng = np.random.default_rng(0)
    open_px = pd.DataFrame(np.abs(rng.standard_normal((200, 50))) + 10.0,
                           index=dates, columns=syms)
    label = open_px.shift(-2) / (open_px.shift(-1) + 1e-12) - 1.0  # 同 execution_rules
    lin = pd.DataFrame(rng.standard_normal((200, 50)), index=dates, columns=syms)
    mlp = pd.DataFrame(rng.standard_normal((200, 50)), index=dates, columns=syms)
    return lin, mlp, label, syms, open_px


def _trailing_shift(mlp_ic: pd.Series, shift: int) -> pd.Series:
    return mlp_ic.shift(shift).rolling(ps.IC_WIN, min_periods=ps.IC_MIN).mean()


def test_shift2_trailing_excludes_last_two_ic():
    """核心因果性：把 IC 末两行从 NaN 改为有效值（模拟未来 IC 被错误塞入），
    shift(2) trailing 应逐位不变（窗口 ic[p-61..p-2] 天然排除末日两行）；
    对照 shift(1) 末日 trailing 必须变（窗口含 ic[n-1] → 证明 shift(1) 泄漏）。"""
    lin, mlp, label, syms, _ = _synthetic()
    mlp_ic = ps.daily_ic({"mlp": mlp}, label)["mlp"]
    t1_orig = _trailing_shift(mlp_ic, 1)
    t2_orig = _trailing_shift(mlp_ic, 2)

    # 末日两行 IC 原本为 NaN（label 末两行需未来 open）；填入有效值模拟泄漏
    ic_tam = mlp_ic.copy()
    ic_tam.iloc[-1] = 0.5
    ic_tam.iloc[-2] = 0.7
    t1_tam = _trailing_shift(ic_tam, 1)
    t2_tam = _trailing_shift(ic_tam, 2)

    # shift(2)：任意 trailing[p] 窗口 = ic[p-61..p-2]，p<=n-1 → 不含 ic[n-2]/ic[n-1]
    assert np.array_equal(t2_orig.values, t2_tam.values, equal_nan=True), \
        "shift(2) trailing 仍受末日 IC 影响（回归！）"
    # 对照：shift(1) 的 trailing[n-1] 窗口 ic[n-60..n-1] 含 ic[n-1] → 必须变化
    assert not np.array_equal(t1_orig.values, t1_tam.values, equal_nan=True), \
        "测试无效：shift(1) trailing 未受影响，篡改未能模拟未来信息"


def test_causal_soft_blend_uses_shift2():
    """行为锚定：causal_soft_blend 返回的 trailing == 手算 shift(2)（≠ shift(1)）。"""
    lin, mlp, label, syms, _ = _synthetic()
    _, trailing, _, _, mlp_ic = ps.causal_soft_blend(lin, mlp, label, syms)
    ref2 = _trailing_shift(mlp_ic, 2)
    ref1 = _trailing_shift(mlp_ic, 1)
    assert np.array_equal(trailing.values, ref2.values, equal_nan=True), \
        "causal_soft_blend 的 trailing 不是 shift(2)（回归为 shift(1)？）"
    # 同一份 mlp_ic 下 shift(1) 与 shift(2) 确有差异（测试有判别力）
    assert not np.array_equal(ref1.values, ref2.values, equal_nan=True), \
        "测试无效：shift(1) 与 shift(2) 结果相同"


def test_last_row_does_not_need_tomorrow():
    """生产语义：末日（今天收盘）的 trailing 不依赖任何未来开盘价。"""
    lin, mlp, label, syms, _ = _synthetic()
    _, trailing, _, _, _ = ps.causal_soft_blend(lin, mlp, label, syms)
    last = trailing.iloc[-1]
    assert not pd.isna(last), "shift(2) 末日 trailing 应可用（窗口 ≥ min_periods）"
    # 末日 trailing 只依赖 <= 末-2 行的 IC：把 label 末日两行改为常量 0，
    # IC 末日两行为 NaN（常量无相关），rolling skipna 下末值仍不变
    lab0 = label.copy()
    lab0.iloc[-1] = 0.0
    lab0.iloc[-2] = 0.0
    _, trailing0, _, _, _ = ps.causal_soft_blend(lin, mlp, lab0, syms)
    assert np.isclose(trailing.iloc[-1], trailing0.iloc[-1], equal_nan=True), \
        "末日 trailing 依赖未来标签（回归！）"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
