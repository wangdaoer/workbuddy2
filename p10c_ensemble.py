"""P10c：rank 线性模型 + P4 MLP 非线性集成（同 curated 16 因子，成本感知）。

动机（P10b 第 5 节方向 1）：因子扩张被证伪（34→12 真实维度仍打不过原始 16），
天花板在「决策函数」而非「因子数量」。本脚本在同一套 curated 16 因子上，把：
  - incumbent 线性综合分（select_positive + mean-IC 权重）
  - P4 MLP 非线性分（单一 meta-feature，select_positive 子集上训练）
在**每个 retrain 点复用同一 live 标的**分别打分，再集成：
  ensemble_ew  = rank(linear) + rank(mlp)            （等权 rank 集成）
  ensemble_icw = w_lin·rank(linear) + w_mlp·rank(mlp)（IC 加权，w∝|IC|）

两集成分均作为单一 meta-feature 注入 run_walk_forward（执行/约束/成本与 incumbent
完全一致），只替换 alpha 决策函数。对比四方（均基于真实面板 + P8b 动态流动性语义）：
  A) incumbent  : 16 因子 + select_positive 线性（= P8b 冠军，脚本内精确复刻）
  B) mlp_only   : P4 MLP 单一 meta-feature
  C) ensemble_ew: 等权 rank 集成
  D) ensemble_icw: IC 加权集成

精确复刻关键：incumbent 的线性综合分在 build_linear_mlp_scores 内逐 retrain 用
「retrain 当日 live 集」对成熟窗口分片计算 IC（与 run_walk_forward 内部 ic_r 完全一致），
并按 block 将非 live 标的置 NaN，作为单一 meta-feature 喂入（liquid_mask=None，回测内部
不再重算 IC）。该综合分与 P8b 的选股/权重逐日一致。扫 AUM ∈ {1e6,1e8,5e8,1e9}，并与 P8b
保存指标交叉校验复刻正确性。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import next_open_return_label
from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import (
    MIN_LIVE_SYMBOLS,
    build_features,
    calculate_walk_forward_metrics,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    mature_ic_window,
    normalize_weights,
    run_walk_forward,
)


class MLP:
    """轻量 2 层 tanh 网络（numpy 实现，无外部依赖）。配置较 P1 更紧凑以控制 44×retrain 耗时。"""

    def __init__(self, hid: int = 16, lr: float = 0.05, iters: int = 150, cap: int = 8000):
        self.hid = hid
        self.lr = lr
        self.iters = iters
        self.cap = cap

    @staticmethod
    def _phi(z):
        return np.tanh(z)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        if X.shape[0] > self.cap:
            idx = np.random.default_rng(0).choice(X.shape[0], self.cap, replace=False)
            X, y = X[idx], y[idx]
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        ys = (y - self.y_mean) / self.y_std
        n, d = Xs.shape
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((d, self.hid)) * 0.1
        self.b1 = np.zeros(self.hid)
        self.W2 = rng.standard_normal((self.hid, 1)) * 0.1
        self.b2 = 0.0
        for _ in range(self.iters):
            H = self._phi(Xs @ self.W1 + self.b1)
            out = H @ self.W2 + self.b2
            err = out.ravel() - ys
            dW2 = (H.T @ err).reshape(-1, 1) / n
            db2 = err.mean()
            dH = (err.reshape(-1, 1) @ self.W2.T) * (1 - H ** 2)
            dW1 = Xs.T @ dH / n
            db1 = dH.mean(0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, float) - self.mu) / self.sd
        return (self._phi(Xs @ self.W1 + self.b1) @ self.W2 + self.b2).ravel()


HERE = Path(__file__).resolve().parent
PANEL = HERE / "external_data" / "daily-market-data-tdx" / "data_panel.csv"
P8B = HERE / "outputs" / "p8b_dynamic_liquidity" / "metrics.json"
OUT = HERE / "outputs" / "p10c_ensemble"
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"

MAX_ABS = 0.22
TRAIN_DAYS = 252
MTH = 1
LIQ_LOOKBACK = 252
THR = 1e8
RETRAIN = 20
TOP_N = 20
REBALANCE = 1
COMMISSION_BPS = 1.0
IMPACT_BPS = 0.7
IMPACT_REF = 0.01
SPLIT = 440
AUMS = [1_000_000.0, 100_000_000.0, 500_000_000.0, 1_000_000_000.0]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def select_positive(training_ic: pd.DataFrame) -> list[str]:
    m = training_ic.mean()
    return [f for f in m.index if pd.notna(m[f]) and m[f] > 0]


def half_return(eq_path: Path, split: int = SPLIT) -> float | None:
    eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if split >= len(eq):
        return None
    return float(eq["equity"].iloc[-1] / eq["equity"].iloc[split - 1] - 1)


def decompose_cost(eq: pd.DataFrame) -> dict:
    turnover = eq["turnover"].sum()
    total_cost = eq["cost"].sum()
    commission = turnover * COMMISSION_BPS / 1e4
    impact = total_cost - commission
    return {
        "total_cost": float(total_cost), "commission_cost": float(commission),
        "impact_cost": float(impact),
        "impact_share": float(impact / total_cost) if total_cost > 0 else 0.0,
    }


def build_linear_mlp_scores(features, label, symbols, feat_arrays, label_arr, liquid_mask_aligned,
                            refresh_live_daily: bool = False):
    """统一 walk-forward 循环：每个 retrain 点复用 retrain 当日 live 集，产出线性综合分与 MLP 分。

    在每个 retrain 点：用 retrain 当日 live 集对**成熟窗口分片**计算 IC（与 run_walk_forward
    内部 ic_r 完全一致），得到线性权重；并在同一 live 子集上训练 MLP。随后在整段 retrain 块
    内用该 live 集作为候选域（非 live 标的置 NaN），逐日产出 linear_score / mlp_score。
    返回 linear_score, mlp_score（dates×symbols，非 block-live 标的 = NaN）。
    """
    n = len(label.index)
    first = TRAIN_DAYS + MTH
    fnames = list(features.keys())
    linear_score = pd.DataFrame(np.nan, index=label.index, columns=symbols)
    mlp_score = pd.DataFrame(np.nan, index=label.index, columns=symbols)
    current_weights = None
    selected = fnames
    mlp_fit = None
    cur_live_bool = np.ones(len(symbols), dtype=bool)  # 当前 block 的 live 布尔（默认全市场）
    retrain_count = 0
    t0 = time.time()

    for i in range(first, n):
        # 修复(2026-08-05)：原为 range(first, n-2)，硬截断最后两天 → 生产"今天收盘"拿不到
        # 今天的分数（08-04/08-05 全 NaN）。打分只用当日特征(features[i])，无前视；
        # 末两日 label 为 NaN 只影响回测评估，不影响打分。扩到 n 使末日也有分。
        step = i - first
        date = label.index[i]
        if step % RETRAIN == 0:
            live = liquid_mask_aligned.loc[date]
            if live.sum() < MIN_LIVE_SYMBOLS:
                live = pd.Series(True, index=symbols)
            cur_live_bool = live.values.astype(bool)
            cols = np.where(cur_live_bool)[0]
            # retrain 当日 live 集对成熟窗口分片计算 IC（与 P8b 内部 ic_r 一致）。
            # 注意：分片 daily_ic 仅覆盖成熟窗口（252 行），不可再套 mature_ic_window
            # 的位置切片（会错位）。ic_r 已即成熟窗口，直接取均值即可。
            start_tr = i - TRAIN_DAYS - MTH
            end_tr = i - MTH
            feat_slice = {f: features[f].iloc[start_tr:end_tr, cols] for f in fnames}
            label_slice = label.iloc[start_tr:end_tr, cols]
            training_ic = daily_ic(feat_slice, label_slice)  # 已即成熟窗口
            mean_ic = training_ic.mean()
            selected = select_positive(training_ic)
            if len(selected) < 3:
                selected = fnames
            current_weights = normalize_weights(mean_ic[selected])

            # MLP 训练（retrain 当日 live 集、select_positive 子集、成熟窗口）
            Xs, ys = [], []
            for t in range(start_tr, end_tr):
                row = np.column_stack([feat_arrays[f][t, cols] for f in selected])
                yt = label_arr[t, cols]
                mask = ~np.isnan(row).any(axis=1) & ~np.isnan(yt)
                if mask.sum() < 50:
                    continue
                Xs.append(row[mask])
                ys.append(yt[mask])
            mlp_fit = MLP().fit(np.vstack(Xs), np.concatenate(ys)) if Xs else None
            retrain_count += 1
            if retrain_count % 10 == 0:
                log(f"  retrain {retrain_count} @ {date.date()} ({time.time()-t0:.0f}s 累计)")

        if refresh_live_daily and step % REBALANCE == 0 and step % RETRAIN != 0:
            # 实验变体(2026-08-05)：非 retrain 日按"当日流动性"刷新 live 集——今天流动才给分，
            # 解决"live 集冻结在 retrain 日(如 05-19)导致 73.5% 新流动票无分"的覆盖错位。
            live = liquid_mask_aligned.loc[date]
            if live.sum() < MIN_LIVE_SYMBOLS:
                live = pd.Series(True, index=symbols)
            cur_live_bool = live.values.astype(bool)
            cols = np.where(cur_live_bool)[0]

        if step % REBALANCE == 0 and current_weights is not None:
            # 线性综合分（全符号行，非 block-live 置 NaN）
            lin = np.zeros(len(symbols))
            for f, w in current_weights.items():
                lin = lin + feat_arrays[f][i] * w
            lin[~cur_live_bool] = np.nan
            linear_score.iloc[i] = lin
            if mlp_fit is not None:
                row = np.column_stack([feat_arrays[f][i, cols] for f in selected])
                mask = ~np.isnan(row).any(axis=1)
                pred = np.full(cols.size, np.nan)
                if mask.sum() > 0:
                    pred[mask] = mlp_fit.predict(row[mask])
                mlp_score.iloc[i, cols] = pred

    log(f"  线性/MLP 分构建完成；retrain 次数={retrain_count}（耗时 {time.time()-t0:.0f}s）")
    return linear_score, mlp_score


def rank_pct(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def main() -> None:
    RUN_LOG.write_text("", encoding="utf-8")
    log("=== P10c 启动 ===")
    raw = load_prices(PANEL, None, None)
    close = clean_matrix(pivot_prices(raw, "close"), MAX_ABS)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), MAX_ABS)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), MAX_ABS)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), MAX_ABS)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    symbols = list(close.columns)

    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=MAX_ABS)
    market_exposure = load_market_exposure(
        None, close.index, ma_window=120, risk_off_drawdown_20d=-0.08,
        below_ma_exposure=0.60, crash_exposure=0.0,
    )

    # 动态流动性 mask（与 P8b 一致）
    roll_med = amount.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK).median()
    liquid_mask = (roll_med >= THR).fillna(False)
    liquid_mask_aligned = liquid_mask.reindex(index=close.index, columns=close.columns).fillna(False)
    log(f"动态流动性 mask: THR={THR:.0e}, 平均每日达标标的数={liquid_mask.sum(axis=1).mean():.0f}")

    # 预计算 numpy 矩阵
    feat_arrays = {k: fr.values for k, fr in features.items()}
    label_arr = label.values

    # 1) 统一循环产出线性 + MLP 分（逐 retrain 用 retrain 当日 live 集对成熟窗口分片算 IC）
    log("=== 构建线性综合分 + MLP 非线性分（统一 retrain 当日 live 集）===")
    linear_score, mlp_score = build_linear_mlp_scores(
        features, label, symbols, feat_arrays, label_arr, liquid_mask_aligned
    )

    # 各 meta-feature 全样本 IC（单次 daily_ic 各自，用于报告与 IC 加权集成）
    log("计算各 meta-feature IC（单次 daily_ic）...")
    ic_lin = daily_ic({"linear": linear_score}, label)["linear"].mean()
    ic_mlp = daily_ic({"mlp": mlp_score}, label)["mlp"].mean()
    log(f"线性综合分 IC={ic_lin:.4f}  MLP 分 IC={ic_mlp:.4f}")

    # 2) 集成分
    lin_rank = rank_pct(linear_score)
    mlp_rank = rank_pct(mlp_score)
    union = linear_score.notna() & mlp_score.notna()
    rl = lin_rank.where(union)
    rm = mlp_rank.where(union)
    ensemble_ew = rl + rm
    w_lin = abs(ic_lin) / (abs(ic_lin) + abs(ic_mlp)) if (abs(ic_lin) + abs(ic_mlp)) > 0 else 0.5
    w_mlp = 1.0 - w_lin
    ensemble_icw = rl * w_lin + rm * w_mlp
    log(f"IC 加权集成权重: w_lin={w_lin:.3f} w_mlp={w_mlp:.3f}")
    ic_ew = daily_ic({"ensemble_ew": ensemble_ew}, label)["ensemble_ew"].mean()
    ic_icw = daily_ic({"ensemble_icw": ensemble_icw}, label)["ensemble_icw"].mean()
    log(f"ensemble_ew IC={ic_ew:.4f}  ensemble_icw IC={ic_icw:.4f}")

    # 3) 回测四方 × 4 AUM（均 liquid_mask=None，incumbent 用预烘焙的精确线性综合分）
    p8b = json.loads(Path(P8B).read_text(encoding="utf-8")) if Path(P8B).exists() else {}
    p8b_sweep = {r["aum"]: r for r in p8b.get("aum_sweep", [])}

    report = {
        "data_source": "hsjday.zip (real A-share, clean panel)",
        "method": "rank linear + lightweight MLP nonlinear ENSEMBLE on curated 16 factors; incumbent replicated exactly via per-retrain live-set composite score (P8b dynamic-liquidity semantics)",
        "mlp_config": {"hid": 16, "lr": 0.05, "iters": 150, "cap": 8000},
        "ic": {
            "linear_composite": float(ic_lin), "mlp": float(ic_mlp),
            "ensemble_ew": float(ic_ew), "ensemble_icw": float(ic_icw),
            "ic_weight_lin": float(w_lin), "ic_weight_mlp": float(w_mlp),
        },
        "aum_sweep": [],
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for aum in AUMS:
        log(f"\n=== AUM={aum:,.0f} ===")
        row = {"aum": aum, "variants": []}
        # A) incumbent（精确复刻 P8b 的线性综合分作为单一 meta-feature）
        eq, _, _ = run_walk_forward(
            close=close, open_px=open_px, features={"incumbent": linear_score}, label=label,
            ic=daily_ic({"incumbent": linear_score}, label),
            train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
            rebalance_frequency=REBALANCE, max_position_weight=0.04, leverage=1.0,
            commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
            impact_model="sqrt", impact_ref_participation=IMPACT_REF,
            max_buy_open_gap=0.06, limit_buffer=0.995,
            market_exposure=market_exposure, initial_capital=aum,
            max_training_horizon=MTH, feature_directions=None, amount=amount,
            feature_selection=None, max_daily_amount_participation=None,
            liquid_mask=None,
        )
        m = calculate_walk_forward_metrics(eq, aum)
        eq.to_csv(OUT / f"equity_aum_{int(aum)}_incumbent.csv", index=False, encoding="utf-8")
        dec = decompose_cost(eq)
        row["variants"].append(_variant("incumbent", m, dec, OUT / f"equity_aum_{int(aum)}_incumbent.csv"))
        p8r = p8b_sweep.get(aum)
        vs = f" vs_P8b={m.get('total_return',0)-p8r.get('total_return',0):+.4f}" if p8r else ""
        log(f"  {'incumbent':12s} total={m.get('total_return'):+.4f} sharpe={m.get('sharpe_like'):.3f} "
            f"dd={m.get('max_drawdown'):.3f} impact_share={dec['impact_share']:.2f}{vs}")

        for key, score_df in [
            ("mlp_only", mlp_score),
            ("ensemble_ew", ensemble_ew),
            ("ensemble_icw", ensemble_icw),
        ]:
            ic_run = daily_ic({key: score_df}, label)  # ic 键须与 features 键一致
            eq, _, _ = run_walk_forward(
                close=close, open_px=open_px, features={key: score_df}, label=label, ic=ic_run,
                train_days=TRAIN_DAYS, retrain_frequency=RETRAIN, top_n=TOP_N,
                rebalance_frequency=REBALANCE, max_position_weight=0.04, leverage=1.0,
                commission_bps=COMMISSION_BPS, impact_bps=IMPACT_BPS,
                impact_model="sqrt", impact_ref_participation=IMPACT_REF,
                max_buy_open_gap=0.06, limit_buffer=0.995,
                market_exposure=market_exposure, initial_capital=aum,
                max_training_horizon=MTH, feature_directions=None, amount=amount,
                feature_selection=None, max_daily_amount_participation=None,
                liquid_mask=None,
            )
            m = calculate_walk_forward_metrics(eq, aum)
            eq.to_csv(OUT / f"equity_aum_{int(aum)}_{key}.csv", index=False, encoding="utf-8")
            dec = decompose_cost(eq)
            row["variants"].append(_variant(key, m, dec, OUT / f"equity_aum_{int(aum)}_{key}.csv"))
            log(f"  {key:12s} total={m.get('total_return'):+.4f} sharpe={m.get('sharpe_like'):.3f} "
                f"dd={m.get('max_drawdown'):.3f} impact_share={dec['impact_share']:.2f}")
        report["aum_sweep"].append(row)
        (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 复刻一致性校验
    if p8b_sweep:
        log("\n=== 复刻一致性（incumbent vs P8b 保存）===")
        for aum in AUMS:
            mine = next(v for r in report["aum_sweep"] if r["aum"] == aum for v in r["variants"] if v["variant"] == "incumbent")
            ref = p8b_sweep.get(aum)
            if ref:
                log(f"  AUM={aum:,.0f}: 复刻 total={mine['total_return']:+.4f} | P8b={ref['total_return']:+.4f} | "
                    f"Δ={mine['total_return']-ref['total_return']:+.4f} | "
                    f"sharpe 复刻={mine['sharpe_like']:.3f} vs P8b={ref.get('sharpe_like'):.3f}")

    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=== P10c 完成 ===")


def _variant(name, m, dec, eq_path):
    return {
        "variant": name, "total_return": m.get("total_return"),
        "annualized_return": m.get("annualized_return"), "max_drawdown": m.get("max_drawdown"),
        "sharpe_like": m.get("sharpe_like"),
        "second_half_return": half_return(eq_path),
        "avg_gross_exposure": m.get("avg_gross_exposure"), "avg_turnover": m.get("avg_turnover"),
        "avg_positions_count": m.get("avg_positions_count"), "final_equity": m.get("final_equity"),
        "impact_cost": dec["impact_cost"], "impact_share": dec["impact_share"],
    }


if __name__ == "__main__":
    main()
