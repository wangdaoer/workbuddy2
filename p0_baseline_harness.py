"""P0 baseline harness for high_risk_quant_model3.

目的（仅管线验证，非真实基线）:
- 在沙盒内重建代码期望的数据目录布局:
  external_data/daily-market-data/ths_exports/normalized/ths_hs_a_share_YYYY-MM-DD.csv
- 生成**合成**小样本（point-in-time 随机游走，无未来函数），证明
  build_data_panel.py -> train_next_open_rank_model.py 端到端可跑通。
- 输出 SHA-256 全链路指纹 + 运行卡（复用现有治理理念：fail-closed / 可复现）。

真实数据接入后：只需把真实 ths_hs_a_share_YYYY-MM-DD.csv 放进同一 normalized 目录，
（必要时按真实列名调整 build_data_panel 的 schema 分支），再跑一遍 harness 即可。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NORMALIZED_DIR = HERE / "external_data" / "daily-market-data" / "ths_exports" / "normalized"
OUTPUT_DIR = HERE / "outputs" / "p0"
PANEL_CSV = OUTPUT_DIR / "data_panel.csv"
MODEL_OUT = OUTPUT_DIR / "next_open_rank_model"

N_DAYS = 320          # 足够驱动 walk-forward（train_days=252, retrain=20）
N_SYMBOLS = 80
SEED = 20260723
DATA_START = date(2024, 1, 2)


def _trading_days(n: int, start: date) -> list[date]:
    days: list[date] = []
    cur = start
    while len(days) < n:
        if cur.weekday() < 5:  # 周一~周五
            days.append(cur)
        cur += timedelta(days=1)
    return days


def generate_fixture(force: bool = False) -> list[Path]:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(NORMALIZED_DIR.glob("ths_hs_a_share_*.csv"))
    if existing and not force:
        print(f"[fixture] 已存在 {len(existing)} 个合成文件，跳过生成（force=True 可重建）。")
        return existing

    # 清理旧合成文件
    for p in existing:
        p.unlink()

    rng = np.random.default_rng(SEED)
    days = _trading_days(N_DAYS, DATA_START)
    # 构造类 A 股代码（主板/创业板前缀），6 位字符串
    prefixes = ("000", "001", "002", "003", "300", "600", "601", "603")
    symbols = [f"{rng.choice(prefixes)}{rng.integers(0, 1000):03d}" for _ in range(N_SYMBOLS)]
    # 每个标的独立随机游走起点与波动率（point-in-time：第 t 天只用 <=t 的信息）
    base = rng.uniform(5.0, 50.0, size=N_SYMBOLS)
    vol = rng.uniform(0.01, 0.04, size=N_SYMBOLS)

    written: list[Path] = []
    prev_close = base.copy()
    for d in days:
        ret = rng.normal(0.0003, 1.0, size=N_SYMBOLS) * vol  # 当日收益率冲击
        close = prev_close * (1.0 + ret)
        close = np.maximum(close, 0.5)
        open_px = prev_close * (1.0 + rng.normal(0, 0.3, size=N_SYMBOLS) * vol)
        high = np.maximum(open_px, close) * (1.0 + np.abs(rng.normal(0, 0.5, size=N_SYMBOLS) * vol))
        low = np.minimum(open_px, close) * (1.0 - np.abs(rng.normal(0, 0.5, size=N_SYMBOLS) * vol))
        low = np.maximum(low, 0.3)
        volume = rng.integers(1_000, 5_000_000, size=N_SYMBOLS).astype(float)
        # 真实 THS 归一化导出通常含更丰富了段；这里用 build_data_panel 的通用分支可接受的
        # 最小列集（date/symbol/open/high/low/close/volume）。amount 由流水线推导。
        frame = pd.DataFrame(
            {
                "date": d.isoformat(),
                "symbol": symbols,
                "open": np.round(open_px, 4),
                "high": np.round(high, 4),
                "low": np.round(low, 4),
                "close": np.round(close, 4),
                "volume": volume,
            }
        )
        path = NORMALIZED_DIR / f"ths_hs_a_share_{d.isoformat()}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        written.append(path)
        prev_close = close

    print(f"[fixture] 生成 {len(written)} 个合成文件 -> {NORMALIZED_DIR}")
    return written


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)

    # 1) build_data_panel.py
    print("\n=== [1/2] build_data_panel.py ===")
    p1 = subprocess.run(
        [sys.executable, "build_data_panel.py", "--output", str(PANEL_CSV)],
        cwd=HERE, env=env, capture_output=True, text=True,
    )
    print(p1.stdout.strip())
    if p1.returncode != 0:
        print(p1.stderr.strip())
        raise RuntimeError("build_data_panel.py 失败")

    # 2) train_next_open_rank_model.py
    print("\n=== [2/2] train_next_open_rank_model.py ===")
    p2 = subprocess.run(
        [
            sys.executable, "train_next_open_rank_model.py",
            "--data", str(PANEL_CSV),
            "--output-dir", str(MODEL_OUT),
            "--train-days", "252",
            "--retrain-frequency", "20",
            "--top-n", "20",
        ],
        cwd=HERE, env=env, capture_output=True, text=True,
    )
    if p2.returncode != 0:
        print(p2.stderr.strip()[:4000])
        raise RuntimeError("train_next_open_rank_model.py 失败")
    print(p2.stdout.strip())

    # 3) 指纹 + 运行卡
    panel_digest = _sha256(PANEL_CSV)
    model_files = sorted(MODEL_OUT.glob("*.csv")) + sorted(MODEL_OUT.glob("*.json"))
    model_digests = {p.name: _sha256(p) for p in model_files}
    metrics = json.loads((MODEL_OUT / "metrics.json").read_text(encoding="utf-8"))

    run_card = {
        "harness": "p0_baseline_harness",
        "purpose": "PIPELINE_VALIDATION_ONLY__NOT_A_REAL_BASELINE",
        "data_mode": "SYNTHETIC_FIXTURE",
        "synthetic_seed": SEED,
        "n_days": N_DAYS,
        "n_symbols": N_SYMBOLS,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "data_panel_csv": str(PANEL_CSV),
        "data_panel_sha256": panel_digest,
        "model_output_dir": str(MODEL_OUT),
        "model_output_sha256": model_digests,
        "metrics_summary": {
            k: metrics.get(k)
            for k in (
                "annualized_return", "sharpe_like", "max_drawdown",
                "total_return", "n_trading_days", "final_equity",
            )
            if k in metrics
        },
        "fail_closed": True,
    }
    (OUTPUT_DIR / "run_card.json").write_text(
        json.dumps(run_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "run_card.md").write_text(_run_card_md(run_card, metrics), encoding="utf-8")
    print(f"\n[run-card] 已写入 {OUTPUT_DIR / 'run_card.json'}")
    return run_card


def _run_card_md(card: dict, metrics: dict) -> str:
    lines = [
        "# P0 基线运行卡（管线验证）",
        "",
        f"> **重要**：本卡基于**合成样本**生成，仅证明 `build_data_panel.py` → "
        f"`train_next_open_rank_model.py` 端到端可跑通。**不是真实策略基线**，IC/收益无意义。",
        "",
        "## 环境",
        f"- Python {card['environment']['python']} / {card['environment']['platform']}",
        f"- numpy {card['environment']['numpy']} · pandas {card['environment']['pandas']} · scipy {card['environment']['scipy']}",
        "",
        "## 数据指纹（SHA-256）",
        f"- 面板 `data_panel.csv`: `{card['data_panel_sha256'][:16]}…`",
        f"- 合成种子: {card['synthetic_seed']} · 天数: {card['n_days']} · 标的数: {card['n_symbols']}",
        "",
        "## 模型输出指纹",
    ]
    for name, digest in card["model_output_sha256"].items():
        lines.append(f"- `{name}`: `{digest[:16]}…`")
    lines += [
        "",
        "## 指标摘要（合成，仅验证用）",
        "```json",
        json.dumps(card["metrics_summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 真实数据接入后的一键复跑",
        "```bash",
        "cd high_risk_quant_model3",
        "# 1) 把真实 ths_hs_a_share_YYYY-MM-DD.csv 放进 external_data/daily-market-data/ths_exports/normalized/",
        "python3 p0_baseline_harness.py --real   # 用真实数据重跑（去掉合成样本）",
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="真实数据模式：不生成合成样本，直接用 normalized 目录内已有文件。")
    ap.add_argument("--force-fixtures", action="store_true", help="强制重建合成样本。")
    args = ap.parse_args()

    if not args.real:
        generate_fixture(force=args.force_fixtures)
    else:
        files = sorted(NORMALIZED_DIR.glob("ths_hs_a_share_*.csv"))
        if not files:
            raise SystemExit(f"[real] 在 {NORMALIZED_DIR} 未找到任何真实样本，请先上传。")
        print(f"[real] 使用 {len(files)} 个真实样本。")

    run_pipeline()


if __name__ == "__main__":
    main()
