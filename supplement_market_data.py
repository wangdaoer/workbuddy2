"""补充历史行情数据生成器（P0 数据自补，用户授权）。

背景：high_risk_quant_model3 训练需要 ≥~300 交易日的逐日全市场行情，
但真实数据仅在网盘/上传中各 3 天且不可达。按用户指示"找不到自己补充"，
这里生成 **确定性、point-in-time、无未来函数** 的合成历史面板。

关键设计——植入可验证信号（P1 版含可观测非线性交互）：
  次日开盘收益 next_open_ret[t] = open[t+1]/open[t] - 1
  由 latent 因子 f[t]（主动量）、g[t]（交互/regime）驱动：
    next_open_ret[t] = a*f[t] + b*tanh(2f[t]) + c*f[t]*g[t] + noise
  - 收盘价随机游走同时由 f[t] 与 g[t] 驱动，使 momentum_20（≈f）与
    volatility_20/distance_ma20（≈g）成为 f、g 的**可观测代理（带符号）**。
  - 线性项 a*f + b*tanh(2f) 单调 → incumbent 的 IC 加权线性模型可捕获；
    双线性交互项 c*f*g 投影到 (f,g) 线性 span 为 0 → 线性模型**结构性抓不到**，
    而 MLP 等非线性模型可捕获。这样补充数据可用于**验证 P1 提升框架**：
    当数据真含可观测非线性时，非线性挑战者应胜过线性 incumbent。

输出：normalized/ths_hs_a_share_YYYY-MM-DD.csv（date,symbol,open,high,low,close,volume,amount）
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NORMALIZED_DIR = HERE / "external_data" / "daily-market-data" / "ths_exports" / "normalized"
OUTPUT_DIR = HERE / "outputs" / "p0_supplemented"
PANEL_CSV = OUTPUT_DIR / "data_panel.csv"
MODEL_OUT = OUTPUT_DIR / "next_open_rank_model"

N_DAYS = 370
N_PER_PREFIX = 120
SEED = 20260723
DATA_START = date(2025, 1, 2)

PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605")
A_LIN = 0.006      # 线性动量分量
B_TANH = 0.004     # 非线性 tanh 分量（单调，线性模型可捕获）
C_INT = 0.006      # 双线性交互项 f*g（可观测，线性模型结构性抓不到）


def _trading_days(n: int, start: date) -> list[date]:
    days: list[date] = []
    cur = start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def generate(force: bool = True) -> list[Path]:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    for p in NORMALIZED_DIR.glob("ths_hs_a_share_*.csv"):
        p.unlink()

    rng = np.random.default_rng(SEED)
    days = _trading_days(N_DAYS, DATA_START)
    symbols = [f"{pre}{i:03d}" for pre in PREFIXES for i in range(N_PER_PREFIX)]
    n_sym = len(symbols)

    # latent 因子：AR(1) 标准化（f 主信号，g 交互信号）
    shock_f = rng.normal(0, 1, (n_sym, N_DAYS))
    shock_g = rng.normal(0, 1, (n_sym, N_DAYS))
    f = np.zeros((n_sym, N_DAYS))
    g = np.zeros((n_sym, N_DAYS))
    for t in range(1, N_DAYS):
        f[:, t] = 0.90 * f[:, t - 1] + shock_f[:, t]
        g[:, t] = 0.88 * g[:, t - 1] + shock_g[:, t]
    fz = (f - f.mean(0)) / (f.std(0) + 1e-9)
    gz = (g - g.mean(0)) / (g.std(0) + 1e-9)

    # 收盘价随机游走由 f 与 g 共同驱动（使 momentum_20≈f、volatility_20/distance_ma20≈g
    # 成为 f、g 的可观测代理；交互项 f*g 因此可由可见因子重建，无未来函数）
    base = rng.uniform(5.0, 60.0, n_sym)
    close = np.zeros((n_sym, N_DAYS))
    close[:, 0] = base
    for t in range(1, N_DAYS):
        daily_ret = 0.0002 + 0.012 * fz[:, t] + 0.008 * gz[:, t] + rng.normal(0, 0.013, n_sym)
        close[:, t] = close[:, t - 1] * (1 + daily_ret)

    # 开盘价：植入 next_open 信号（用 f[t],g[t]，当日收盘已知，无前视）
    # 含双线性交互项 c*f*g（可观测，线性模型结构性抓不到）
    open_px = np.zeros((n_sym, N_DAYS))
    open_px[:, 0] = close[:, 0] * (1 - 0.002)
    for t in range(0, N_DAYS - 1):
        sig = (
            A_LIN * fz[:, t]
            + B_TANH * np.tanh(2.0 * fz[:, t])
            + C_INT * fz[:, t] * gz[:, t]
        )
        open_px[:, t + 1] = open_px[:, t] * (1 + sig + rng.normal(0, 0.005, n_sym))

    high = np.maximum(open_px, close) * (1 + np.abs(rng.normal(0, 0.004, (n_sym, N_DAYS))))
    low = np.minimum(open_px, close) * (1 - np.abs(rng.normal(0, 0.004, (n_sym, N_DAYS))))
    low = np.maximum(low, 0.3)
    volume = np.exp(rng.normal(14.0, 0.8, (n_sym, N_DAYS))) * (1 + np.abs(fz))
    volume = np.round(volume, 0)
    amount = volume * close

    written: list[Path] = []
    for ti, d in enumerate(days):
        frame = pd.DataFrame(
            {
                "date": d.isoformat(),
                "symbol": symbols,
                "open": np.round(open_px[:, ti], 4),
                "high": np.round(high[:, ti], 4),
                "low": np.round(low[:, ti], 4),
                "close": np.round(close[:, ti], 4),
                "volume": volume[:, ti],
                "amount": np.round(amount[:, ti], 2),
            }
        )
        path = NORMALIZED_DIR / f"ths_hs_a_share_{d.isoformat()}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        written.append(path)
    print(f"[supplement] 生成 {len(written)} 天 × {n_sym} 标的（含植入信号）-> {NORMALIZED_DIR}")
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

    print("\n=== [1/2] build_data_panel.py ===")
    p1 = subprocess.run(
        [sys.executable, "build_data_panel.py", "--output", str(PANEL_CSV)],
        cwd=HERE, env=env, capture_output=True, text=True,
    )
    print(p1.stdout.strip())
    if p1.returncode != 0:
        print(p1.stderr.strip()); raise RuntimeError("build_data_panel 失败")

    print("\n=== [2/2] train_next_open_rank_model.py ===")
    p2 = subprocess.run(
        [sys.executable, "train_next_open_rank_model.py", "--data", str(PANEL_CSV),
         "--output-dir", str(MODEL_OUT), "--train-days", "252", "--retrain-frequency", "20",
         "--top-n", "20"],
        cwd=HERE, env=env, capture_output=True, text=True,
    )
    if p2.returncode != 0:
        print(p2.stderr.strip()[:4000]); raise RuntimeError("train 失败")
    print(p2.stdout.strip())

    panel_digest = _sha256(PANEL_CSV)
    model_files = sorted(MODEL_OUT.glob("*.csv")) + sorted(MODEL_OUT.glob("*.json"))
    model_digests = {p.name: _sha256(p) for p in model_files}
    metrics = json.loads((MODEL_OUT / "metrics.json").read_text(encoding="utf-8"))

    # 验证植入信号是否被捕获：读 daily_feature_ic 看 momentum_20 的 IC
    ic_path = MODEL_OUT / "daily_feature_ic.csv"
    top_ic = ""
    if ic_path.exists():
        ic = pd.read_csv(ic_path)
        if "feature" in ic.columns:
            mean_ic = ic.groupby("feature")[["ic"]].mean().sort_values("ic", ascending=False)
            top_ic = mean_ic.head(8).to_string()

    run_card = {
        "harness": "supplement_market_data",
        "purpose": "SUPPLEMENTED_SYNTHETIC_WITH_PLANTED_SIGNAL__NOT_A_REAL_BASELINE",
        "data_mode": "SYNTHETIC_SUPPLEMENT",
        "note": "数据为用户授权自补；含可验证非线性信号（单调项 a·f+b·tanh(2f) 可由线性模型捕获，"
                "双线性交互项 c·f·g 可观测但线性模型结构性抓不到，用于校验 P1 提升框架）。非真实行情。",
        "synthetic_seed": SEED,
        "n_days": N_DAYS,
        "n_symbols": N_PER_PREFIX * len(PREFIXES),
        "signal_coeffs": {"linear": A_LIN, "tanh": B_TANH, "interaction": C_INT},
        "data_panel_sha256": panel_digest,
        "model_output_sha256": model_digests,
        "metrics_summary": {k: metrics.get(k) for k in
                            ("annualized_return", "sharpe_like", "max_drawdown",
                             "total_return", "final_equity") if k in metrics},
        "fail_closed": True,
    }
    (OUTPUT_DIR / "run_card.json").write_text(json.dumps(run_card, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "top_feature_ic.txt").write_text(top_ic, encoding="utf-8")
    print(f"\n[run-card] {OUTPUT_DIR / 'run_card.json'}")
    print("\n=== 因子 IC 排名（前 8，验证信号是否被捕获）===\n" + top_ic)
    return run_card


def main() -> None:
    generate(force=True)
    run_pipeline()


if __name__ == "__main__":
    main()
