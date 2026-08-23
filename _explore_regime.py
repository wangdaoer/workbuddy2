"""P10e 前置探查：哪些 regime 信号能识别 W4（最新逆风窗口）为 adverse，且不在 W2-W3 误杀。

候选信号：
  A) market_exposure（run_walk_forward 自带风险闸门，load_market_exposure）
  B) breadth bear：close>MA120 的标的比例 < 0.5
  C) trend bear：等权指数 < MA120
探查各信号在 W1..W4 的 adverse 日占比，判断其识别力。
"""
from pathlib import Path
import numpy as np, pandas as pd
from execution_rules import next_open_return_label
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix, load_market_exposure, build_features

HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
raw = load_prices(PANEL, None, None)
close = clean_matrix(pivot_prices(raw, "close"), 0.22)
open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), 0.22)
high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), 0.22)
low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), 0.22)
amount = pivot_prices(raw, "amount").reindex_like(close)
features = build_features(close, open_px, high, low, amount)
label = next_open_return_label(open_px, max_abs_daily_return=0.22)
market_exposure = load_market_exposure(None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08, below_ma_exposure=0.60, crash_exposure=0.0)

# 信号 B：breadth
ma120 = close.rolling(120, min_periods=120).mean()
breadth = (close > ma120).mean(axis=1)
# 信号 C：等权指数 trend
ew_idx = close.mean(axis=1)
ew_ma120 = ew_idx.rolling(120, min_periods=120).mean()
trend_bear = ew_idx < ew_ma120

n = len(close.index)
edges = [0, n // 4, n // 2, 3 * n // 4, n]
sig = {
    "A_market_exposure<1": (market_exposure < 1.0).astype(float),
    "B_breadth<0.5": (breadth < 0.5).astype(float),
    "C_trend_bear": trend_bear.astype(float).reindex(close.index).fillna(0.0),
}
print(f"{'signal':22s}" + "".join(f"{'W'+str(k+1):>14}" for k in range(4)) + "   overall")
for name, s in sig.items():
    row = []
    for k in range(4):
        seg = s.iloc[edges[k]:edges[k + 1]]
        row.append(seg.mean())
    print(f"{name:22s}" + "".join(f"{v:>14.2f}" for v in row) + f"   {s.mean():.2f}")

# W4 内逐月 adverse 占比（看信号 B/C 是否持续触发）
print("\nW4 窗口（最新~133日）逐段 adverse 占比：")
w4 = slice(edges[3], edges[4])
for name, s in sig.items():
    seg = s.iloc[w4]
    print(f"  {name:22s} W4 均值={seg.mean():.2f}  min={seg.min():.0f} max={seg.max():.0f}")
print("\nmarket_exposure 取值分布：", market_exposure.value_counts().to_dict())
