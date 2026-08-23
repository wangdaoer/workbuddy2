# -*- coding: utf-8 -*-
"""反推原始内容并与当前文件 diff，生成统一补丁（仓库非 git，无 git diff 可用）。"""
from pathlib import Path
import difflib

ROOT = Path(__file__).resolve().parent

# (相对路径, 原始片段, 改进后片段) — 片段均来自本次确切编辑
edits = [
    ("build_dashboard.py",
     r'''# 设为 True 时，dashboard 不把 blended overlay 当主名单来源，仅作隔离标注。
BLENDED_QUARANTINED = True''',
     r'''# 隔离态唯一真源 = .QUARANTINE 标记文件存在性（与 publish_blended_release.py G2 一致）：
# 标记存在 → 隔离生效，dashboard 不把 blended overlay 当主名单来源，仅作隔离标注；
# 解除隔离 = 删除该标记文件（见报告§五前置条件），无需再改本脚本常量。
OVERLAY_Q = ROOT / "outputs" / "watchlist_audit" / "full_overlay_calibrated_blended.csv.QUARANTINE"
BLENDED_QUARANTINED = OVERLAY_Q.exists()'''),
    ("build_premarket_brief.py",
     r'''# ⚠️ 隔离研究快照（2026-08-23 起）：未通过 run 一致性校验，禁止作为实盘主名单消费。
BLENDED_QUARANTINED = True''',
     r'''# ⚠️ 隔离研究快照（2026-08-23 起）：未通过 run 一致性校验，禁止作为实盘主名单消费。
# 隔离态唯一真源 = .QUARANTINE 标记文件存在性（与 publish_blended_release.py G2 一致）：
# 标记存在 → 隔离生效，主名单回退降仓版→原始 overlay；解除隔离 = 删除该标记文件。
OVERLAY_Q = OVERLAY_DIR / "full_overlay_calibrated_blended.csv.QUARANTINE"
BLENDED_QUARANTINED = OVERLAY_Q.exists()'''),
    ("publish_blended_release.py",
     r'''def read_quarantine_flag(src: Path) -> bool:
    """从脚本源码读 BLENDED_QUARANTINED 字面量（避免 import 重副作用）。"""
    m = re.search(r"BLENDED_QUARANTINED\s*=\s*(True|False)", src.read_text(encoding="utf-8"))
    return m.group(1) == "True" if m else False''',
     r'''def read_quarantine_flag(src: Path) -> bool:
    """从脚本源码读隔离态（隔离态唯一真源 = .QUARANTINE 标记文件存在性，避免 import 重副作用）。

    新版脚本（build_dashboard.py / build_premarket_brief.py）以
    `BLENDED_QUARANTINED = OVERLAY_Q.exists()` 派生隔离态；此处识别该形式并实时返回标记状态，
    与脚本运行时行为保持一致。旧版字面量 `= True/False` 仍兼容。
    """
    txt = src.read_text(encoding="utf-8")
    if re.search(r"BLENDED_QUARANTINED\s*=\s*OVERLAY_Q\.exists\(\)", txt):
        return OVERLAY_Q.exists()
    m = re.search(r"BLENDED_QUARANTINED\s*=\s*(True|False)", txt)
    return m.group(1) == "True" if m else False'''),
    ("outputs/high_return_v2/trend_ignition_daily_pool/blend_manifest.json",
     r'''  "note": "权重 w_ti/w_nor/w_horizon 仍沿用泄漏回测调优结果，待 per-as-of 无泄漏重做后替换。"''',
     r'''  "note": "权重 w_ti/w_nor/w_horizon 沿用泄漏回测调优结果；经 step5 无泄漏验证（holdout 净20=-10.49% n=3、全轴准OOS 净20=-8.06% n=14）方案 B 纯 OOS 确凿为负，已隔离、不晋级实盘，权重保持冻结不再替换。解除隔离须满足报告§五前置条件（删 .QUARANTINE 标记 + 统计可信正 OOS 样本≥10 且均值>0 + 全部门控通过）。"'''),
]

def _to_eol(s: str, eol: str) -> str:
    """归一化换行为目标文件的 EOL（CRLF/LF），避免源脚本自身换行差异影响匹配。"""
    return s.replace("\r\n", "\n").replace("\n", eol)

patch_parts = []
for rel, old, new in edits:
    p = ROOT / rel
    cur = p.read_text(encoding="utf-8", newline="")  # 原样读取，保留磁盘真实换行
    eol = "\r\n" if "\r\n" in cur else "\n"           # 按文件检测 EOL
    old_e, new_e = _to_eol(old, eol), _to_eol(new, eol)
    assert new_e in cur, f"改进片段未在 {rel} 中找到，跳过"
    orig = cur.replace(new_e, old_e)  # 反推原始
    a = orig.splitlines(keepends=True)
    b = cur.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile="a/" + rel, tofile="b/" + rel)
    patch_parts.append("".join(diff))

out = ROOT / "improved_changes_2026-08-23.patch"
# 保留各文件原始换行（CRLF/LF），仅去除行尾空格与制表符（保留 \r 与 \n），
# 避免 plain `patch`/`git apply` 因 CRLF 与长 note 行尾随空白敏感而失败。
content = "".join(patch_parts)
content = "\n".join(l.rstrip(" \t") for l in content.split("\n"))
if not content.endswith("\n"):
    content += "\n"
out.write_text(content, encoding="utf-8", newline="\n")
print("written:", out)
print("bytes:", out.stat().st_size)
for rel, *_ in edits:
    print("  covered:", rel)
