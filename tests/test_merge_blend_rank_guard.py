"""merge_blend_into_overlay 顶入 rank 守卫的回归测试。

背景（2026-09-18）：守卫写成 `any(b >= a for a, b in zip(ranks, ranks[1:]))`，
方向反了。keep_ranks 已 sort_values() 升序，升序列表的相邻元素天然 b > a，
故该判据对「正确」输入恒为 True —— 守卫在 rank 正确时报错、在 rank 错误时放行。
因为 zip 在 len(keep_ranks) <= 1 时为空、any([]) 为 False，该缺陷在 blend 顶入
0/1 席时被长期掩盖，直到首次出现 4 席顶入（ranks 2/5/13/14）才暴露。

本测试锁定正确语义：升序（含空洞）通过；有重复则失败。
"""
from pathlib import Path

import pandas as pd
import pytest

import merge_blend_into_overlay as mb


def _write_overlay(path: Path, n_seats: int = 11, n_rows: int = 60) -> None:
    """构造最小 derisked overlay：前 n_seats 席等权 0.05，其余非选中。"""
    rows = []
    for i in range(n_rows):
        sel = i < n_seats
        rows.append({
            "symbol": f"{i:06d}",
            "stock_name": f"S{i}",
            "final_score": float(n_rows - i),
            "personal_selected": sel,
            "target_weight_after_behavior": 0.05 if sel else 0.0,
            "personal_adjusted_target_weight": 0.05 if sel else 0.0,
            "target_weight": 0.05 if sel else 0.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_blend(path: Path, pairs) -> None:
    """pairs = [(symbol, blended_rank), ...]，按给定顺序写盘。"""
    rows = [
        {
            "asof": "2026-09-18",
            "symbol": s,
            "ti_score": 1.0,
            "nor_score": 1.0,
            "blended_score": 1.0,
            "blended_rank": float(r),
        }
        for s, r in pairs
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _run(tmp_path, monkeypatch, pairs, top_n=20, max_blend_seats=None):
    overlay = tmp_path / "derisked.csv"
    blend = tmp_path / "blended.csv"
    out = tmp_path / "blended_overlay.csv"
    _write_overlay(overlay)
    _write_blend(blend, pairs)
    argv = [
        "merge_blend_into_overlay.py",
        "--overlay", str(overlay),
        "--blend", str(blend),
        "--output", str(out),
        "--top-n", str(top_n),
    ]
    if max_blend_seats is not None:
        argv += ["--max-blend-seats", str(max_blend_seats)]
    monkeypatch.setattr("sys.argv", argv)
    mb.main()
    return pd.read_csv(out, dtype=str)


def test_multiple_blend_seats_with_rank_gaps_pass(tmp_path, monkeypatch):
    """核心回归：顶入 >=2 席且 rank 有空洞（升序）必须通过并正确落盘。

    这是 2026-09-18 缺陷的直接复现条件 —— 多席顶入（真实场景为 ranks 2/5/13/14）。
    overlay 前 11 行(000000..000010)为选中席；blend 取 000020/000030/000040
    （在候选全集内但不在原 selected 内）→ 应全部记为 blend_inserted。
    rank 故意留空洞：2 / 5 / 11 / 14。
    """
    pairs = [
        ("999998", 1),   # 不在 overlay 候选内 → 被过滤
        ("000020", 2),   # 顶入
        ("999997", 3),   # 不在 overlay 候选内 → 被过滤
        ("000030", 5),   # 顶入（rank 空洞）
        ("000040", 11),  # 顶入（rank 空洞）
        ("000050", 14),  # 顶入（rank 空洞）
    ]
    df = _run(tmp_path, monkeypatch, pairs)
    inserted = sorted(df.loc[df["blend_inserted"].astype(str).str.lower() == "true", "symbol"].tolist())
    assert inserted == ["000020", "000030", "000040", "000050"], inserted
    # 暴露档位不变：仍 11 席、总权重 0.55
    sel = df.loc[df["personal_selected"].astype(str).str.lower() == "true"]
    assert len(sel) == 11
    assert abs(float(sel["target_weight_after_behavior"].astype(float).sum()) - 0.55) < 1e-9


def test_single_blend_seat_still_passes(tmp_path, monkeypatch):
    """回归护栏：<=1 席顶入（zip 为空）仍应通过 —— 缺陷曾被此路径掩盖。"""
    df = _run(tmp_path, monkeypatch, [("000020", 1), ("999996", 2), ("999995", 3)])
    sel = df.loc[df["personal_selected"].astype(str).str.lower() == "true"]
    assert len(sel) == 11
    assert abs(float(sel["target_weight_after_behavior"].astype(float).sum()) - 0.55) < 1e-9
    inserted = df.loc[df["blend_inserted"].astype(str).str.lower() == "true", "symbol"].tolist()
    assert inserted == ["000020"]


def test_duplicate_rank_is_rejected(tmp_path, monkeypatch):
    """负向用例：blend 名单出现重复 blended_rank 时必须 fail-closed。"""
    overlay = tmp_path / "derisked.csv"
    blend = tmp_path / "blended.csv"
    out = tmp_path / "blended_overlay.csv"
    _write_overlay(overlay)
    _write_blend(blend, [("000020", 1), ("000030", 1), ("000040", 2)])  # rank 1 重复
    monkeypatch.setattr("sys.argv", [
        "merge_blend_into_overlay.py",
        "--overlay", str(overlay),
        "--blend", str(blend),
        "--output", str(out),
        "--top-n", "20",
    ])
    with pytest.raises(SystemExit) as ei:
        mb.main()
    assert "非严格递增" in str(ei.value)
    assert not out.exists(), "校验失败时不得留下坏产物"
