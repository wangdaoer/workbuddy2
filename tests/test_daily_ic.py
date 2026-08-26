"""daily_ic 向量化实现的回归测试。

核心保证：向量化 daily_ic 与逐点 _daily_ic_legacy 数值等价（spearman IC 语义），
且边界行为一致（NaN 自动排除、有效样本 <30 置 NaN、常数序列置 NaN）。
"""
import numpy as np
import pandas as pd

import train_next_open_rank_model as tm


def _panel(T=80, N=60, seed=0, nan_frac=0.05):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="B")
    cols = [f"S{i}" for i in range(N)]

    def mk():
        a = rng.normal(size=(T, N))
        a[rng.random((T, N)) < nan_frac] = np.nan
        return pd.DataFrame(a, index=idx, columns=cols)

    return mk, idx, cols


def test_vectorized_matches_legacy():
    mk, idx, cols = _panel()
    f1, f2 = mk(), mk()
    f3 = pd.DataFrame(np.ones((len(idx), len(cols))), index=idx, columns=cols)
    f4 = mk()
    f4.iloc[:55, :] = np.nan  # 前 55 行有效样本 < 30
    label = mk()
    feats = {"a": f1, "b": f2, "const": f3, "sparse": f4}

    new = tm.daily_ic(feats, label)
    leg = tm._daily_ic_legacy(feats, label)

    assert list(new.columns) == list(leg.columns)
    maxerr = 0.0
    nan_mismatch = 0
    for c in new.columns:
        for d in new.index:
            nv, lv = new.loc[d, c], leg.loc[d, c]
            if pd.isna(nv) and pd.isna(lv):
                continue
            if pd.isna(nv) != pd.isna(lv):
                nan_mismatch += 1
                continue
            maxerr = max(maxerr, abs(float(nv) - float(lv)))
    assert nan_mismatch == 0
    assert maxerr < 1e-9


def test_constant_factor_all_nan():
    mk, idx, cols = _panel()
    f = pd.DataFrame(np.ones((len(idx), len(cols))), index=idx, columns=cols)
    label = mk()
    ic = tm.daily_ic({"c": f}, label)
    assert ic["c"].isna().all()


def test_sparse_factor_partially_nan():
    mk, idx, cols = _panel()
    f = mk()
    f.iloc[:55, :] = np.nan  # drop 前 55 行，使有效样本 < 30
    label = mk()
    ic = tm.daily_ic({"s": f}, label)
    assert ic["s"].iloc[:55].isna().all()
    # 之后若有效样本 >= 30 可能出现非 NaN（随机面板下大概率）
