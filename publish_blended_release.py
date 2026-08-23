"""publish_blended_release.py — 方案 B 原子发布 + 一致性 fail-closed 门控 (step5).

目标：用"原子目录"替代"覆盖旧文件"的脆弱发布方式。
  解除 blended overlay 隔离 / 晋级实盘前，必须通过全部一致性门控：
    G1  run_id 绑定存在
    G2  隔离已解除（.QUARANTINE 标记删除 且 两脚本 BLENDED_QUARANTINED=False）
    G3  三者全交集覆盖率 ≥ min_blend_coverage
    G4  纯 OOS 门控：holdout 窗口 net(next_open+成本) 均值 > oos_min（默认 0）
    G5  报告↔产物一致：报告 holdout 20日 数字 必须等于 leakfree_backtest.csv 实测
    G6  manifest 输入 sha 与当前文件重算 sha 一致（无静默换源）
    G7  pool as-of 与 manifest.asof 一致

判定：
  全部通过 → 在 outputs/published/blend/<run_id>/ 落原子发布目录（pin 所有 hash），
             并原子写 LATEST 指针。
  任一失败 → 写 outputs/published/blend/PUBLISH_BLOCKED.json（含逐条原因），
             绝不晋级、绝不触碰隔离标记与源产物（只读门控）。

本脚本只读源产物 + 只写 outputs/published/blend/，不修改任何生产/隔离状态。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOL_DIR = ROOT / "outputs" / "high_return_v2" / "trend_ignition_daily_pool"
AUDIT_DIR = ROOT / "outputs" / "watchlist_audit"
PUBLISH_DIR = ROOT / "outputs" / "published" / "blend"

MANIFEST = POOL_DIR / "blend_manifest.json"
OVERLAY = AUDIT_DIR / "full_overlay_calibrated_blended.csv"
OVERLAY_Q = AUDIT_DIR / "full_overlay_calibrated_blended.csv.QUARANTINE"
POOL_LATEST = POOL_DIR / "blended_pool_latest.csv"
POOL_CSV = POOL_DIR / "blended_pool.csv"
LEAK_CSV = POOL_DIR / "leakfree_backtest.csv"
LEAK_REPORT = POOL_DIR / "leakfree_backtest_report.md"
SWEEP_CSV = POOL_DIR / "blend_weight_sweep.csv"
WFV_CSV = POOL_DIR / "blend_wfv_oos.csv"
DASHBOARD = ROOT / "build_dashboard.py"
PREMARKET = ROOT / "build_premarket_brief.py"


def sha256(path: Path, head: int | None = None) -> str:
    h = hashlib.sha256()
    data = path.read_bytes()[:head] if head else path.read_bytes()
    h.update(data)
    return h.hexdigest()[:16]


def read_quarantine_flag(src: Path) -> bool:
    """从脚本源码读隔离态（隔离态唯一真源 = .QUARANTINE 标记文件存在性，避免 import 重副作用）。

    新版脚本（build_dashboard.py / build_premarket_brief.py）以
    `BLENDED_QUARANTINED = OVERLAY_Q.exists()` 派生隔离态；此处识别该形式并实时返回标记状态，
    与脚本运行时行为保持一致。旧版字面量 `= True/False` 仍兼容。
    """
    txt = src.read_text(encoding="utf-8")
    if re.search(r"BLENDED_QUARANTINED\s*=\s*OVERLAY_Q\.exists\(\)", txt):
        return OVERLAY_Q.exists()
    m = re.search(r"BLENDED_QUARANTINED\s*=\s*(True|False)", txt)
    return m.group(1) == "True" if m else False


def gate_report_matches_sweep(report: Path, sweep: Path, w_ti: float, w_horizon: float) -> tuple[bool, str]:
    """G5: 报告'当前生产权重 w_ti/w_horizon holdout 净20'必须 == blend_weight_sweep.csv 同网格点实测。"""
    import pandas as pd
    if not sweep.exists():
        return False, "缺少 blend_weight_sweep.csv"
    txt = report.read_text(encoding="utf-8")
    m = re.search(rf"当前生产权重 {w_ti:g}/{w_horizon:g}：holdout 净20 = \*\*([-\d.]+)%\*\*", txt)
    if not m:
        return False, f"报告未找到 '当前生产权重 {w_ti:g}/{w_horizon:g} holdout 净20' 数字"
    report_val = float(m.group(1)) / 100.0
    df = pd.read_csv(sweep)
    row = df[(df.w_ti == w_ti) & (df.w_horizon == w_horizon)]
    if len(row) == 0:
        return False, f"sweep 缺 {w_ti:g}/{w_horizon:g} 行"
    csv_val = float(row.iloc[0]["hold_n20"])
    diff = abs(report_val - csv_val)
    if diff > 0.005:
        return False, (f"报告 {report_val:+.2%} 与 sweep {w_ti:g}/{w_horizon:g} holdout_n20={csv_val:+.2%} "
                       f"差异 {diff:+.2%}，超出 0.5% 容差")
    return True, f"报告 {report_val:+.2%} == sweep {csv_val:+.2%}"


def main() -> None:
    ap = argparse.ArgumentParser(description="方案 B 原子发布 + 一致性门控")
    ap.add_argument("--min-blend-coverage", type=int, default=30)
    ap.add_argument("--oos-min", type=float, default=0.0,
                    help="纯 OOS net 均值门槛（默认 0：必须正收益才晋级）")
    ap.add_argument("--oos-min-samples", type=int, default=3,
                    help="holdout 最少 as-of 数（面板限制下物理上限≈3），低于则 OOS 证据不足不晋级")
    ap.add_argument("--dry-run", action="store_true", help="只评估门控，不写任何发布目录")
    args = ap.parse_args()

    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []
    ok = True

    def gate(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        gates.append({"gate": name, "pass": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # ---- G1 run 绑定 ----
    if not MANIFEST.exists():
        gate("G1_run_binding", False, "缺少 blend_manifest.json")
        manifest = {}
    else:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rid = manifest.get("run_id")
        gate("G1_run_binding", bool(rid), f"run_id={rid}")

    # ---- G2 隔离解除 ----
    q_marker = OVERLAY_Q.exists()
    q_dash = read_quarantine_flag(DASHBOARD) if DASHBOARD.exists() else False
    q_pre = read_quarantine_flag(PREMARKET) if PREMARKET.exists() else False
    cleared = (not q_marker) and (not q_dash) and (not q_pre)
    detail = ("隔离标记已删除且两脚本 BLENDED_QUARANTINED=False" if cleared
              else f"隔离仍生效 (marker={q_marker}, dashboard={q_dash}, premarket={q_pre})")
    gate("G2_quarantine_cleared", cleared, detail)

    # ---- G3 覆盖率 ----
    cov_all = manifest.get("coverage", {}).get("all_three")
    if cov_all is None:
        gate("G3_coverage", False, "manifest 缺 coverage.all_three")
    else:
        gate("G3_coverage", cov_all >= args.min_blend_coverage,
             f"三者全交集={cov_all} ≥ 门槛 {args.min_blend_coverage}")

    # ---- G4 纯 OOS 门控（多样本，自然日口径，基于 blend_weight_sweep.csv）----
    # P0-3 修复：G4 只评估「待发布 manifest 的精确参数」一次，禁止在 holdout 网格上选优（窥探）。
    # 参数必须 == manifest.params（w_ti/w_horizon）；sweep 缺该行或样本不足 → fail-closed。
    if not SWEEP_CSV.exists():
        gate("G4_oos_gate", False, "缺少 blend_weight_sweep.csv（请先跑 tune_blend_weights_leakfree.py）")
    elif not manifest:
        gate("G4_oos_gate", False, "缺少 blend_manifest.json，无法绑定待发布参数")
    else:
        import pandas as pd
        params = manifest.get("params", {})
        w_ti, w_horizon = params.get("w_ti"), params.get("w_horizon")
        if w_ti is None or w_horizon is None:
            gate("G4_oos_gate", False, "manifest 缺 params.w_ti/w_horizon，无法绑定参数")
        else:
            df = pd.read_csv(SWEEP_CSV)
            row = df[(df["w_ti"] == w_ti) & (df["w_horizon"] == w_horizon)]
            if len(row) == 0:
                gate("G4_oos_gate", False,
                     f"sweep 缺待发布参数行 w_ti={w_ti:g}/w_horizon={w_horizon:g}"
                     f"（fail-closed，禁止用其他参数代替）")
            else:
                r = row.iloc[0]
                n_ok = int(r["hold_n"]) if pd.notna(r["hold_n"]) else 0
                hold_n20 = float(r["hold_n20"]) if pd.notna(r["hold_n20"]) else float("nan")
                if n_ok < args.oos_min_samples:
                    gate("G4_oos_gate", False,
                         f"待发布参数 w_ti={w_ti:g}/w_horizon={w_horizon:g} holdout 样本 {n_ok} < "
                         f"{args.oos_min_samples}，OOS 证据不足")
                else:
                    gate("G4_oos_gate", hold_n20 > args.oos_min,
                         f"待发布参数 w_ti={w_ti:g}/w_horizon={w_horizon:g} holdout 净20={hold_n20:+.2%} "
                         f"(n={n_ok})；要求 > {args.oos_min:+.0%}")
        # 全轴准 OOS 复核（统计可信优先依据）
        if WFV_CSV.exists():
            wfv = pd.read_csv(WFV_CSV)
            wfv_n = wfv["n20"].dropna()
            if len(wfv_n) >= 10:
                wfv_mean = float(wfv_n.mean())
                gate("G4b_oos_wfv", wfv_mean > args.oos_min,
                     f"全轴准OOS 净20={wfv_mean:+.2%} (n={len(wfv_n)})；要求 > {args.oos_min:+.0%}")
            else:
                gate("G4b_oos_wfv", False, f"准OOS 样本 {len(wfv_n)} < 10，证据不足")
        else:
            gate("G4b_oos_wfv", False, "缺少 blend_wfv_oos.csv（请带 --wfv 重跑扫描）")

    # ---- G5 报告↔产物一致 ----
    if LEAK_REPORT.exists() and SWEEP_CSV.exists():
        p_ = manifest.get("params", {}) if manifest else {}
        g5_w_ti = p_.get("w_ti", 0.9)
        g5_w_horizon = p_.get("w_horizon", 0.5)
        p, d = gate_report_matches_sweep(LEAK_REPORT, SWEEP_CSV, g5_w_ti, g5_w_horizon)
        gate("G5_report_artifact_consistency", p, d)
    else:
        gate("G5_report_artifact_consistency", False, "报告或 blend_weight_sweep.csv 缺失")

    # ---- G6 manifest 输入 sha 校验 ----
    if manifest:
        inputs = manifest.get("inputs", {})
        mism = []
        for key, val in inputs.items():
            if key.endswith("_sha"):
                fkey = key[:-4]
                fpath = Path(inputs[fkey])
                if not fpath.exists():
                    mism.append(f"{fkey}: 文件缺失")
                elif sha256(fpath) != val:
                    mism.append(f"{fkey}: sha 不符(当前 {sha256(fpath)} ≠ {val})")
        gate("G6_manifest_sha", len(mism) == 0,
             "输入 sha 全部一致" if not mism else "；".join(mism))
    else:
        gate("G6_manifest_sha", False, "无 manifest 可校验")

    # ---- G7 pool as-of 一致 ----
    if POOL_LATEST.exists() and manifest.get("asof"):
        import pandas as pd
        try:
            a = pd.read_csv(POOL_LATEST)["asof"].iloc[0]
            gate("G7_pool_asof", str(a) == str(manifest["asof"]),
                 f"pool as-of={a} == manifest as-of={manifest['asof']}")
        except Exception as e:  # noqa
            gate("G7_pool_asof", False, f"读 pool as-of 失败: {e}")
    else:
        gate("G7_pool_asof", False, "pool 或 manifest.asof 缺失")

    # ---- 判定 ----
    result = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": manifest.get("run_id"),
        "all_gates_pass": ok,
        "gates": gates,
    }

    if args.dry_run:
        print(f"\n[dry-run] 不写发布目录。综合判定: {'晋级' if ok else '阻断'}。")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0 if ok else 1)

    if ok:
        rid = manifest["run_id"]
        rel_dir = PUBLISH_DIR / rid
        # P1-3 修复：先写临时目录，全部拷贝+校验完成后原子替换；中途失败不触碰旧发布目录。
        tmp_dir = PUBLISH_DIR / f".{rid}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        try:
            # 拷贝产物（pin 快照）
            for f in [POOL_CSV, POOL_LATEST, OVERLAY, MANIFEST, LEAK_CSV, LEAK_REPORT,
                      SWEEP_CSV, WFV_CSV,
                      ROOT / "cost_model.py", ROOT / "validate_blend_leakfree.py",
                      ROOT / "tune_blend_weights_leakfree.py"]:
                if f.exists():
                    shutil.copy2(f, tmp_dir / f.name)
            publish_manifest = {
                "published_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": rid,
                "source_manifest": json.loads(MANIFEST.read_text(encoding="utf-8")),
                "gates": gates,
                "artifacts": {
                    f.name: sha256(f) for f in
                    [POOL_CSV, POOL_LATEST, OVERLAY, MANIFEST, LEAK_CSV, LEAK_REPORT]
                    if f.exists()
                },
            }
            (tmp_dir / "publish_manifest.json").write_text(
                json.dumps(publish_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            # 校验临时目录产物可读（防半成品晋级）
            (tmp_dir / "publish_manifest.json").read_text(encoding="utf-8")
        except Exception as e:  # noqa
            shutil.rmtree(tmp_dir, ignore_errors=True)
            sys.exit(f"\n❌ 发布目录准备失败，未触碰旧发布：{e}")
        # 临时目录完整 → 原子替换旧发布目录（保留旧目录直至新目录全部就绪）
        if rel_dir.exists():
            shutil.rmtree(rel_dir)
        tmp_dir.rename(rel_dir)
        # 原子写 LATEST 指针
        tmp = PUBLISH_DIR / "LATEST.tmp"
        tmp.write_text(rid, encoding="utf-8")
        tmp.replace(PUBLISH_DIR / "LATEST")
        print(f"\n✅ 晋级：原子发布目录 {rel_dir} （LATEST -> {rid}）")
    else:
        blocked = PUBLISH_DIR / "PUBLISH_BLOCKED.json"
        blocked.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n⛔ 阻断：未晋级。原因见 {blocked}")
        print("   隔离标记与源产物未被修改。解除条件：G2 隔离解除 + G4 OOS 转正 + G5 报告与产物对齐。")


if __name__ == "__main__":
    main()
