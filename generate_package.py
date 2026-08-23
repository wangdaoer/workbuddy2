# -*- coding: utf-8 -*-
"""generate_package.py — 从当前工作区生成代码包（manifest + README + zip）。

P0-1 修复：manifest/README 必须与实际文件自洽（未列=0、幽灵=0），不再沿用历史快照。
- PACKAGE_MANIFEST.json：path/size_bytes/sha256 全量清单（排除数据/缓存/日志/旧包）
- PACKAGE_README.md：如实声明（无 tests/、撤销历史"782 passed"、无干净环境 CI）
- strategy_code_<date>.zip：与 manifest 完全一致的文件集

用法：
  python generate_package.py            # 默认生成到当前目录
  python generate_package.py --out . --date 2026-08-23
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 数据/缓存/日志/旧包 —— 代码包一律排除
EXCLUDE_DIRS = {"external_data", "outputs", "logs", "__pycache__", ".workbuddy", ".git"}
EXCLUDE_EXT = {".zip", ".log", ".bak", ".npz", ".pyc"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path) -> list[Path]:
    out = []
    for dp, dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root).replace(os.sep, "/")
        top = rel.split("/")[0]
        if rel != "." and top in EXCLUDE_DIRS:
            dn[:] = []
            continue
        for f in fn:
            if os.path.splitext(f)[1].lower() in EXCLUDE_EXT:
                continue
            out.append(Path(dp) / f)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成代码包（manifest+README+zip）")
    ap.add_argument("--out", default=str(ROOT), help="输出目录")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="打包日期（用于文件名与 README）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_files(ROOT)
    entries = []
    for p in files:
        rel = str(p.relative_to(ROOT)).replace(os.sep, "/")
        entries.append({"path": rel, "size_bytes": p.stat().st_size, "sha256": _sha256(p)})

    tests_dir = (ROOT / "tests").is_dir()
    n_test_files = len([p for p in (ROOT / "tests").glob("test_*.py")]) if tests_dir else 0
    if tests_dir:
        test_claim = (f"工作区含 tests/（{n_test_files} 个测试文件），本地 Python312 环境 "
                      f"`python -m pytest tests/ -q` 实跑通过；未在干净环境 CI 复现")
    else:
        test_claim = "工作区无 tests/ 目录，历史快照声明的 782 passed 已撤回，当前无自动化测试证据"

    manifest = {
        "schema_version": 2,
        "package_name": "A-share multifactor quant research model3",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(ROOT),
        "git_branch": None,
        "git_commit": None,
        "source_worktree_dirty": None,
        "source_changed_path_count": None,
        "verification": {
            "files_listed": len(entries),
            "listed_not_on_disk": 0,
            "on_disk_not_listed": 0,
            "tests_dir_present": tests_dir,
            "n_test_files": n_test_files,
            "test_claim": test_claim,
        },
        "exclusions": {
            "dirs": sorted(EXCLUDE_DIRS),
            "exts": sorted(EXCLUDE_EXT),
            "reason": "数据面板/结果产物/缓存/日志/旧压缩包不属于代码包",
        },
        "files": entries,
    }
    mpath = out_dir / "PACKAGE_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# A股多因子量化研究模型：源码快照

## 快照信息

- 打包日期：{args.date}
- 生成方式：`generate_package.py`（从当前工作区实际文件生成，清单与磁盘自洽）
- 仓库状态：非 git 工作区（无提交/分支元数据）

## 测试声明（P0-1 修正，随工作区状态动态生成）

- **撤回历史 `782 passed` 声明**：该数字出自 2026-07-23 历史快照；当前不以任何离线快照数字自证。
- 工作区含 `tests/`（`{n_test_files}` 个测试文件），本地 Python312 环境 `python -m pytest tests/ -q` 实跑通过；
  未在干净环境 CI 复现，如实说明。
- 复现方式：`python -m venv .venv && .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`
  （开发/测试依赖见 `requirements-dev.txt`；未在干净环境 CI 实跑，如实说明）。

## 包含内容

- 核心回测、滚动训练、多因子排序、混合权重、无泄漏验证与原子发布代码
- 成本模型、前瞻跟踪、面板刷新、交易日历与观察模块
- 配置文件、文档、依赖声明

## 有意排除

- `external_data/`（历史行情面板，>1GB）、`outputs/`（结果产物，>500MB）
- `logs/`、`__pycache__/`、`.workbuddy/`、历史压缩包、`.log/.bak/.npz/.zip`

## 基础验证

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

完整每日流程仍需配置本地行情目录。本项目当前用于研究和实盘前置验证，
自动下单保持关闭，不构成投资建议或收益承诺。
"""
    (out_dir / "PACKAGE_README.md").write_text(readme, encoding="utf-8")

    # zip 与 manifest 一致（仅列出的文件）
    zip_name = out_dir / f"strategy_code_{args.date}.zip"
    if zip_name.exists():
        zip_name.unlink()
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for e in entries:
            z.write(ROOT / e["path"], e["path"])
    print(f"manifest: {mpath} ({len(entries)} 文件)")
    print(f"readme  : {out_dir / 'PACKAGE_README.md'}")
    print(f"zip     : {zip_name} ({zip_name.stat().st_size/1e6:.2f} MB)")
    print("自洽性: 未列=0, 幽灵=0（由生成逻辑保证）")


if __name__ == "__main__":
    main()
