# -*- coding: utf-8 -*-
"""精确字符串替换方式应用本批改进（仓库非 git，且源文件为混合换行 CRLF/LF，
`patch`/`git apply` 对 CRLF 与无末行换行敏感，故用内容替换而非 diff 打补丁）。

用法：
  python apply_changes.py            # 实际写入
  python apply_changes.py --dry-run # 只报告将要改什么，不写盘

特性：
  - 按文件检测真实换行（CRLF/LF），old/new 片段归一化到同换行后再替换，避免换行差异导致不匹配；
  - 已应用过（找到 new）则跳过并提示；old 与 new 都找不到则报错；
  - 不改动文件其余内容的换行格式。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

# (相对路径, 原始片段, 改进后片段) — 片段均来自本次确切编辑，与 improved_changes_2026-08-23.patch 一致
EDITS = [
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
    return s.replace("\r\n", "\n").replace("\n", eol)


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    changed = 0
    skipped = 0
    for rel, old, new in EDITS:
        p = ROOT / rel
        if not p.exists():
            print(f"[缺失] {rel} 不存在，跳过")
            continue
        raw = p.read_bytes()
        cur = raw.decode("utf-8")
        eol = "\r\n" if "\r\n" in cur else "\n"
        old_e, new_e = _to_eol(old, eol), _to_eol(new, eol)
        if old_e in cur:
            if dry:
                print(f"[将改] {rel}")
            else:
                cur2 = cur.replace(old_e, new_e)
                p.write_bytes(cur2.encode("utf-8"))
                print(f"[已改] {rel}")
            changed += 1
        elif new_e in cur:
            print(f"[已应用] {rel} 已是改进态，跳过")
            skipped += 1
        else:
            print(f"[错误] {rel} 未找到原始片段，无法定位（请确认文件版本）")
            return 2
    print(f"\n完成：将改/已改 {changed} 处，已应用跳过 {skipped} 处" + ("（dry-run）" if dry else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
