"""Compare the base next-open rank model with a personal behavior overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apply_personal_trade_overlay import apply_overlay, load_symbol_history, _load_rules
from execution_rules import apply_open_constraints, next_open_return_label
from run_backtest import load_prices, max_drawdown, pivot_prices, sharpe_like
from train_next_open_rank_model import (
    build_breadth_exposure,
    build_features,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    normalize_weights,
)


def build_candidate_table(
    score: pd.Series,
    metrics: pd.DataFrame,
    top_n: int,
    candidate_pool_n: int,
    raw_weight: float,
) -> pd.DataFrame:
    pool_n = max(int(candidate_pool_n), int(top_n))
    ranked = score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(pool_n)
    out = pd.DataFrame(
        {
            "symbol": [str(symbol).zfill(6) for symbol in ranked.index],
            "rank": range(1, len(ranked) + 1),
            "score": ranked.to_numpy(dtype=float),
        },
        index=ranked.index,
    )
    metric_snapshot = metrics.reindex(ranked.index).reset_index(drop=True)
    out = pd.concat([out.reset_index(drop=True), metric_snapshot], axis=1)
    out["selected"] = out["rank"].le(int(top_n))
    out["target_weight"] = np.where(out["selected"], float(raw_weight), 0.0)
    return out


def targets_from_overlay_table(
    overlay: pd.DataFrame,
    all_symbols: pd.Index,
    exposure: float,
) -> pd.Series:
    target = pd.Series(0.0, index=all_symbols, dtype=float)
    if overlay.empty or "personal_selected" not in overlay:
        return target
    selected = overlay[overlay["personal_selected"].astype(bool)]
    if selected.empty:
        return target
    weights = pd.to_numeric(
        selected["personal_adjusted_target_weight"],
        errors="coerce",
    ).fillna(0.0)
    symbols = selected["symbol"].astype(str).str.zfill(6)
    aligned = pd.Series(weights.to_numpy(dtype=float), index=symbols)
    common = target.index.intersection(aligned.index)
    target.loc[common] = aligned.loc[common] * float(exposure)
    return target


def latest_metric_snapshot(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    row_index: int,
) -> pd.DataFrame:
    c = close.iloc[row_index]
    o = open_px.iloc[row_index]
    h = high.iloc[row_index]
    l = low.iloc[row_index]
    ma20 = close.rolling(20).mean().iloc[row_index]
    ma60 = close.rolling(60).mean().iloc[row_index]
    ma120 = close.rolling(120).mean().iloc[row_index]
    out = pd.DataFrame(index=close.columns)
    out["close"] = c
    out["return_5d"] = close.pct_change(5, fill_method=None).iloc[row_index]
    out["return_20d"] = close.pct_change(20, fill_method=None).iloc[row_index]
    out["return_60d"] = close.pct_change(60, fill_method=None).iloc[row_index]
    out["return_120d"] = close.pct_change(120, fill_method=None).iloc[row_index]
    out["distance_ma20"] = c / (ma20 + 1e-12) - 1.0
    out["distance_ma60"] = c / (ma60 + 1e-12) - 1.0
    out["distance_ma120"] = c / (ma120 + 1e-12) - 1.0
    out["intraday_return"] = c / (o + 1e-12) - 1.0
    out["close_position"] = (c - l) / ((h - l).replace(0, np.nan) + 1e-12)
    out["avg_amount_20d"] = amount.replace(0, np.nan).rolling(20).median().iloc[row_index]
    return out


def build_metric_frames(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    close_range = (high - low).replace(0, np.nan)
    return {
        "close": close,
        "return_5d": close.pct_change(5, fill_method=None),
        "return_20d": close.pct_change(20, fill_method=None),
        "return_60d": close.pct_change(60, fill_method=None),
        "return_120d": close.pct_change(120, fill_method=None),
        "distance_ma20": close / (ma20 + 1e-12) - 1.0,
        "distance_ma60": close / (ma60 + 1e-12) - 1.0,
        "distance_ma120": close / (ma120 + 1e-12) - 1.0,
        "intraday_return": close / (open_px + 1e-12) - 1.0,
        "close_position": (close - low) / (close_range + 1e-12),
        "avg_amount_20d": amount.replace(0, np.nan).rolling(20).median(),
    }


def metric_snapshot_from_frames(metric_frames: dict[str, pd.DataFrame], row_index: int) -> pd.DataFrame:
    return pd.DataFrame({name: frame.iloc[row_index] for name, frame in metric_frames.items()})


def score_from_weights(
    features: dict[str, pd.DataFrame],
    weights: pd.Series,
    row_index: int,
) -> pd.Series:
    score = None
    for name, weight in weights.items():
        if name not in features:
            continue
        part = features[name].iloc[row_index] * float(weight)
        score = part if score is None else score + part
    if score is None:
        return pd.Series(dtype=float)
    return score


def weights_for_signal_date(weights: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
    frame = weights.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    usable = frame[frame["date"].le(signal_date)]
    if usable.empty:
        usable = frame.head(1)
    row = usable.iloc[-1].drop(labels=["date"])
    return pd.to_numeric(row, errors="coerce").fillna(0.0)


def base_targets_from_candidates(
    candidates: pd.DataFrame,
    all_symbols: pd.Index,
    exposure: float,
) -> pd.Series:
    target = pd.Series(0.0, index=all_symbols, dtype=float)
    selected = candidates[candidates["selected"].astype(bool)]
    if selected.empty:
        return target
    weights = pd.to_numeric(selected["target_weight"], errors="coerce").fillna(0.0)
    symbols = selected["symbol"].astype(str).str.zfill(6)
    aligned = pd.Series(weights.to_numpy(dtype=float), index=symbols)
    common = target.index.intersection(aligned.index)
    target.loc[common] = aligned.loc[common] * float(exposure)
    return target


def strategy_metrics(nav: pd.Series, initial_capital: float) -> dict[str, float]:
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    total_return = float(nav.iloc[-1] / initial_capital - 1.0)
    if total_return <= -1.0:
        annualized = -1.0
    else:
        annualized = float((1.0 + total_return) ** (252 / max(len(nav), 1)) - 1.0)
    return {
        "final_equity": float(nav.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
    }


def run_comparison_walk_forward(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    symbol_history: pd.DataFrame,
    rules: dict[str, Any],
    train_days: int,
    retrain_frequency: int,
    top_n: int,
    candidate_pool_n: int,
    rebalance_frequency: int,
    max_position_weight: float,
    leverage: float,
    commission_bps: float,
    impact_bps: float,
    max_buy_open_gap: float,
    limit_buffer: float,
    market_exposure: pd.Series,
    initial_capital: float,
    trained_weights: pd.DataFrame | None = None,
    max_abs_daily_return: float = 0.22,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_features(close, open_px, high, low, amount)
    metric_frames = build_metric_frames(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=max_abs_daily_return)
    ic = pd.DataFrame() if trained_weights is not None else daily_ic(features, label)

    base_equity = initial_capital
    overlay_equity = initial_capital
    base_positions = pd.Series(0.0, index=close.columns)
    overlay_positions = pd.Series(0.0, index=close.columns)
    current_weights = pd.Series(1.0 / len(features), index=list(features))
    nav_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for i in range(train_days, len(close.index) - 2):
        signal_date = close.index[i]
        if trained_weights is not None:
            current_weights = weights_for_signal_date(trained_weights, signal_date)
            if (i - train_days) % retrain_frequency == 0:
                weight_rows.append({"date": signal_date.strftime("%Y-%m-%d"), **current_weights.to_dict()})
        elif (i - train_days) % retrain_frequency == 0:
            current_weights = normalize_weights(ic.iloc[i - train_days : i].mean())
            weight_rows.append({"date": signal_date.strftime("%Y-%m-%d"), **current_weights.to_dict()})

        exposure = float(market_exposure.reindex([signal_date]).iloc[0])
        if (i - train_days) % rebalance_frequency == 0:
            score = score_from_weights(features, current_weights, i)
            raw_weight = min(max_position_weight, leverage / max(top_n, 1))
            metrics = metric_snapshot_from_frames(metric_frames, i)
            candidates = build_candidate_table(
                score,
                metrics,
                top_n=top_n,
                candidate_pool_n=candidate_pool_n,
                raw_weight=raw_weight,
            )
            base_target = base_targets_from_candidates(candidates, close.columns, exposure)
            overlay_table = apply_overlay(
                candidates,
                symbol_history,
                rules,
                reselect_top_n=top_n,
                base_target_weight=raw_weight,
            )
            overlay_target = targets_from_overlay_table(overlay_table, close.columns, exposure)

            changed = overlay_table[
                overlay_table["selected"].astype(bool)
                | overlay_table["personal_selected"].astype(bool)
                | overlay_table["personal_action"].ne("allow")
            ].copy()
            changed.insert(0, "signal_date", signal_date.strftime("%Y-%m-%d"))
            audit_rows.extend(changed.to_dict("records"))
        else:
            base_target = base_positions.copy()
            overlay_target = overlay_positions.copy()

        base_target = apply_open_constraints(
            base_positions,
            base_target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=max_buy_open_gap,
            limit_buffer=limit_buffer,
        )
        overlay_target = apply_open_constraints(
            overlay_positions,
            overlay_target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=max_buy_open_gap,
            limit_buffer=limit_buffer,
        )

        realized = label.iloc[i].reindex(close.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        base_turnover = float((base_target - base_positions).abs().sum())
        overlay_turnover = float((overlay_target - overlay_positions).abs().sum())
        cost_rate = (commission_bps + impact_bps) / 1e4
        base_cost = base_turnover * cost_rate
        overlay_cost = overlay_turnover * cost_rate
        base_gross_return = float((base_target * realized).sum())
        overlay_gross_return = float((overlay_target * realized).sum())
        base_equity *= 1.0 + base_gross_return - base_cost
        overlay_equity *= 1.0 + overlay_gross_return - overlay_cost
        base_positions = base_target
        overlay_positions = overlay_target

        nav_rows.append(
            {
                "date": close.index[i + 2].strftime("%Y-%m-%d"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "base_equity": base_equity,
                "overlay_equity": overlay_equity,
                "base_gross_return": base_gross_return,
                "overlay_gross_return": overlay_gross_return,
                "base_cost": base_cost,
                "overlay_cost": overlay_cost,
                "base_turnover": base_turnover,
                "overlay_turnover": overlay_turnover,
                "base_gross_exposure": float(base_positions.abs().sum()),
                "overlay_gross_exposure": float(overlay_positions.abs().sum()),
                "market_exposure": exposure,
                "base_positions_count": int(base_positions.ne(0).sum()),
                "overlay_positions_count": int(overlay_positions.ne(0).sum()),
            }
        )

    return (
        pd.DataFrame(nav_rows),
        pd.DataFrame(audit_rows),
        pd.DataFrame(weight_rows),
        ic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a personal behavior overlay against the base rank model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/personal_behavior_overlay")
    parser.add_argument("--symbol-history", default=None)
    parser.add_argument("--rules", default="configs/personal_trade_habit_overlay.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--retrain-frequency", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--candidate-pool-n", type=int, default=120)
    parser.add_argument("--rebalance-frequency", type=int, default=5)
    parser.add_argument("--max-position-weight", type=float, default=0.029)
    parser.add_argument("--leverage", type=float, default=0.93)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--impact-bps", type=float, default=0.7)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--max-buy-open-gap", type=float, default=0.05)
    parser.add_argument("--limit-buffer", type=float, default=0.995)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--market-ma-window", type=int, default=120)
    parser.add_argument("--market-risk-off-drawdown-20d", type=float, default=-0.08)
    parser.add_argument("--market-below-ma-exposure", type=float, default=0.65)
    parser.add_argument("--market-crash-exposure", type=float, default=0.0)
    parser.add_argument("--breadth-filter", action="store_true")
    parser.add_argument("--breadth-ma-window", type=int, default=60)
    parser.add_argument("--breadth-threshold", type=float, default=0.45)
    parser.add_argument("--breadth-below-exposure", type=float, default=0.55)
    parser.add_argument("--breadth-crash-threshold", type=float, default=0.32)
    parser.add_argument("--breadth-crash-exposure", type=float, default=0.20)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    rules = _load_rules(Path(args.rules) if args.rules else None)
    symbol_history = load_symbol_history(Path(args.symbol_history) if args.symbol_history else None)
    trained_weights = pd.read_csv(args.weights) if args.weights else None

    market_exposure = load_market_exposure(
        args.benchmark,
        close.index,
        ma_window=args.market_ma_window,
        risk_off_drawdown_20d=args.market_risk_off_drawdown_20d,
        below_ma_exposure=args.market_below_ma_exposure,
        crash_exposure=args.market_crash_exposure,
    )
    if args.breadth_filter:
        breadth_exposure = build_breadth_exposure(
            close,
            ma_window=args.breadth_ma_window,
            threshold=args.breadth_threshold,
            below_exposure=args.breadth_below_exposure,
            crash_threshold=args.breadth_crash_threshold,
            crash_exposure=args.breadth_crash_exposure,
        )
        market_exposure = pd.concat([market_exposure, breadth_exposure], axis=1).min(axis=1)

    equity, audit, weights, ic = run_comparison_walk_forward(
        close=close,
        open_px=open_px,
        high=high,
        low=low,
        amount=amount,
        symbol_history=symbol_history,
        rules=rules,
        train_days=args.train_days,
        retrain_frequency=args.retrain_frequency,
        top_n=args.top_n,
        candidate_pool_n=args.candidate_pool_n,
        rebalance_frequency=args.rebalance_frequency,
        max_position_weight=args.max_position_weight,
        leverage=args.leverage,
        commission_bps=args.commission_bps,
        impact_bps=args.impact_bps,
        max_buy_open_gap=args.max_buy_open_gap,
        limit_buffer=args.limit_buffer,
        market_exposure=market_exposure,
        initial_capital=args.initial_capital,
        trained_weights=trained_weights,
        max_abs_daily_return=args.max_abs_daily_return,
    )
    if equity.empty:
        raise ValueError("No backtest rows generated. Check train-days and input history length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_dir / "behavior_selection_audit.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(output_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(output_dir / "daily_feature_ic.csv", encoding="utf-8-sig")

    base_nav = pd.Series(equity["base_equity"].to_numpy(), index=pd.to_datetime(equity["date"]))
    overlay_nav = pd.Series(equity["overlay_equity"].to_numpy(), index=pd.to_datetime(equity["date"]))
    base_metrics = strategy_metrics(base_nav, args.initial_capital)
    overlay_metrics = strategy_metrics(overlay_nav, args.initial_capital)
    action_counts = audit["personal_action"].value_counts().to_dict() if "personal_action" in audit else {}
    selected_action_counts = (
        audit[audit["personal_selected"].astype(bool)]["personal_action"].value_counts().to_dict()
        if "personal_selected" in audit and "personal_action" in audit
        else {}
    )
    metrics = {
        "strategy": "personal_behavior_overlay_comparison",
        "initial_capital": args.initial_capital,
        "base": base_metrics,
        "personal_overlay": overlay_metrics,
        "incremental": {
            "final_equity_diff": overlay_metrics["final_equity"] - base_metrics["final_equity"],
            "total_return_diff": overlay_metrics["total_return"] - base_metrics["total_return"],
            "max_drawdown_diff": overlay_metrics["max_drawdown"] - base_metrics["max_drawdown"],
            "sharpe_like_diff": overlay_metrics["sharpe_like"] - base_metrics["sharpe_like"],
        },
        "avg_base_turnover": float(equity["base_turnover"].mean()),
        "avg_overlay_turnover": float(equity["overlay_turnover"].mean()),
        "avg_base_gross_exposure": float(equity["base_gross_exposure"].mean()),
        "avg_overlay_gross_exposure": float(equity["overlay_gross_exposure"].mean()),
        "avg_market_exposure": float(equity["market_exposure"].mean()),
        "action_counts": action_counts,
        "selected_action_counts": selected_action_counts,
        "symbol_history_used": bool(args.symbol_history),
        "parameters": {
            "train_days": args.train_days,
            "retrain_frequency": args.retrain_frequency,
            "top_n": args.top_n,
            "candidate_pool_n": args.candidate_pool_n,
            "rebalance_frequency": args.rebalance_frequency,
            "max_position_weight": args.max_position_weight,
            "leverage": args.leverage,
            "selection_mode": rules.get("selection_mode"),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Personal behavior overlay comparison saved to: {output_dir}")


if __name__ == "__main__":
    main()
