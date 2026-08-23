"""每日市场宽度信号（宽度闸门）——与 next_open_rank 模型共用同一套暴露逻辑。

本脚本把 `train_next_open_rank_model.build_breadth_exposure` 的默认参数与算法
**原样复用**，作为生产链路的独立信号层：每天收盘后计算"宽度驱动的应暴露水平"，
供每日 overlay 降仓、日报/盘前简报/仪表盘展示弱势市预警。

设计原则（遵守源文件只读红线）：
- 只读 data_panel.csv，不写任何输入源；产物一律写到 outputs/market_regime/。
- 原始 full_overlay_calibrated.csv 不被改动；降仓由下游 apply_market_exposure.py 生成
  衍生文件，可随时回退。

暴露规则（与模型一致）：
  宽度 = 站上 MA60 的个股占比
  宽度 < 0.45  -> 暴露 0.55（谨慎）
  宽度 < 0.32  -> 暴露 0.20（崩溃防御）
  否则         -> 暴露 1.00（常态满仓）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_prices, pivot_prices

HERE = Path(__file__).resolve().parent
DEFAULT_PANEL = HERE / "external_data" / "daily-market-data" / "data_panel.csv"
OUT_DIR = HERE / "outputs" / "market_regime"

# ---- 与 train_next_open_rank_model.build_breadth_exposure 默认参数保持一致 ----
MA_WINDOW = 60
THRESHOLD = 0.45
BELOW_EXPOSURE = 0.55
CRASH_THRESHOLD = 0.32
CRASH_EXPOSURE = 0.20
RECOMMENDED_SEATS_FULL = 20  # 与 run_daily_overlay --reselect-top-n 20 对应


def build_breadth_exposure(
    close: pd.DataFrame,
    ma_window: int = MA_WINDOW,
    threshold: float = THRESHOLD,
    below_exposure: float = BELOW_EXPOSURE,
    crash_threshold: float = CRASH_THRESHOLD,
    crash_exposure: float = CRASH_EXPOSURE,
) -> pd.Series:
    """原样复用 train_next_open_rank_model.build_breadth_exposure。"""
    ma = close.rolling(ma_window).mean()
    breadth = close.gt(ma).mean(axis=1)
    exposure = pd.Series(1.0, index=close.index)
    exposure = exposure.where(~breadth.lt(threshold), below_exposure)
    exposure = exposure.where(~breadth.lt(crash_threshold), crash_exposure)
    return exposure.fillna(1.0).clip(lower=0.0, upper=1.0)


def regime_label(exposure: float, breadth: float) -> str:
    if exposure <= CRASH_EXPOSURE + 1e-9:
        return "crash_defense"  # 崩溃防御
    if exposure <= BELOW_EXPOSURE + 1e-9:
        return "caution"  # 谨慎
    return "normal"  # 常态


def main() -> None:
    parser = argparse.ArgumentParser(description="每日市场宽度信号（宽度闸门）")
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--asof", default=None, help="目标日 YYYY-MM-DD（默认取面板末日）")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    raw = load_prices(Path(args.panel), None, None)
    close = pivot_prices(raw, "close").sort_index()

    exposure_series = build_breadth_exposure(close)
    ma = close.rolling(MA_WINDOW).mean()
    breadth_series = close.gt(ma).mean(axis=1)

    asof = args.asof or str(close.index[-1])[:10]
    # 取 asof 当日（若存在），否则取最近一个交易日
    if asof in exposure_series.index:
        exp = float(exposure_series.loc[asof])
        brd = float(breadth_series.loc[asof])
    else:
        recent = exposure_series[exposure_series.index <= asof]
        if recent.empty:
            raise SystemExit(f"面板中无 <= {asof} 的交易日")
        asof = str(recent.index[-1])[:10]
        exp = float(exposure_series.loc[asof])
        brd = float(breadth_series.loc[asof])

    label = regime_label(exp, brd)
    recommended_seats = max(0, round(RECOMMENDED_SEATS_FULL * exp))

    signal = {
        "date": asof,
        "breadth_above_ma60": round(brd, 4),
        "exposure_target": round(exp, 4),
        "regime": label,
        "recommended_seats": recommended_seats,
        "params": {
            "ma_window": MA_WINDOW,
            "threshold": THRESHOLD,
            "below_exposure": BELOW_EXPOSURE,
            "crash_threshold": CRASH_THRESHOLD,
            "crash_exposure": CRASH_EXPOSURE,
        },
        "source": "market_breadth_signal.py (mirrors train_next_open_rank_model.build_breadth_exposure)",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"breadth_signal_{asof}.json"
    json_path.write_text(json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8")
    # 固定名（供下游稳定读取）
    (out_dir / "breadth_signal.json").write_text(
        json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    label_cn = {"normal": "常态满仓", "caution": "谨慎（宽度预警）", "crash_defense": "崩溃防御"}[label]
    md = (
        f"# 市场宽度信号 {asof}\n\n"
        f"- **宽度(站上MA60个股占比)**：{brd:.1%}\n"
        f"- **建议暴露水平**：{exp:.2f}（{label_cn}）\n"
        f"- **建议席位**：{recommended_seats} / {RECOMMENDED_SEATS_FULL}\n"
        f"- 规则：宽度<{THRESHOLD:.0%}→0.55；宽度<{CRASH_THRESHOLD:.0%}→0.20\n"
    )
    (out_dir / f"breadth_signal_{asof}.md").write_text(md, encoding="utf-8")

    print(f"[breadth] {asof} 宽度={brd:.1%} 暴露={exp:.2f} 状态={label_cn} 建议席位={recommended_seats}")
    print(f"[breadth] 写出 {json_path}")


if __name__ == "__main__":
    main()
