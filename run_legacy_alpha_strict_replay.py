"""Strict replay study for the legacy concentrated alpha.

This script is research-only. It does not touch daily production state and
never promotes a strategy automatically.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from execution_rules import (
    LIMIT_DOWN_PRICE_COLUMNS,
    LIMIT_RATE_COLUMNS,
    LIMIT_UP_PRICE_COLUMNS,
    apply_open_constraints_with_diagnostics,
)
from run_backtest import annualized_return, load_prices, max_drawdown, pivot_prices, sharpe_like
from train_next_open_rank_model import (
    build_breadth_exposure,
    build_features,
    clean_matrix,
    load_market_exposure,
    rank_pct,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "high_return_v2" / "legacy_alpha_strict_replay"

INITIAL_CAPITAL = 1_000_000.0
TOP_N = 10
MAX_POSITION_WEIGHT = 0.10
LEVERAGE = 1.0
COMMISSION_BPS = 3.0
IMPACT_BPS = 7.0
MAX_BUY_OPEN_GAP = 0.03
LIMIT_BUFFER = 0.995
MAX_ABS_DAILY_RETURN = 0.22
FACTOR_REBALANCE_FREQUENCY = 8

LEGACY_ALPHA_WEIGHTS: dict[str, float] = {
    "momentum60": 0.20,
    "trend120": 0.20,
    "reversal5": 0.25,
    "low_vol20": 0.25,
    "liquidity20": 0.10,
}
STRONG_FOCUS_BOOST = 0.20

SEGMENTS: tuple[tuple[str, str | None, str | None], ...] = (
    ("full_history", None, None),
    ("2025H2", "2025-07-01", "2025-12-31"),
    ("2026-01-01_to_2026-06-15", "2026-01-01", "2026-06-15"),
)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    use_market: bool
    use_breadth: bool
    is_control: bool = False


@dataclass
class VariantRun:
    variant: VariantSpec
    curve: pd.DataFrame
    diagnostics: pd.DataFrame
    metrics: dict[str, Any]


PREREGISTERED_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("Control", use_market=True, use_breadth=True, is_control=True),
    VariantSpec("OldTop10-Strict", use_market=False, use_breadth=False),
    VariantSpec("OldTop10-Strict+Market", use_market=True, use_breadth=False),
    VariantSpec("OldTop10-Strict+Market+Breadth", use_market=True, use_breadth=True),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the preregistered legacy concentrated alpha under strict next-open execution."
    )
    parser.add_argument("--data", required=True, help="Historical OHLCV panel (CSV or Parquet).")
    parser.add_argument("--benchmark", required=True, help="Benchmark CSV with date and close columns.")
    parser.add_argument(
        "--control-equity",
        required=True,
        help="External current champion equity curve CSV used for the preregistered control row.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--asof-date", default="2026-07-21")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--max-abs-daily-return", type=float, default=MAX_ABS_DAILY_RETURN)
    parser.add_argument("--market-ma-window", type=int, default=120)
    parser.add_argument("--market-risk-off-drawdown-20d", type=float, default=-0.08)
    parser.add_argument("--market-below-ma-exposure", type=float, default=0.60)
    parser.add_argument("--market-crash-exposure", type=float, default=0.0)
    parser.add_argument("--breadth-ma-window", type=int, default=60)
    parser.add_argument("--breadth-threshold", type=float, default=0.45)
    parser.add_argument("--breadth-below-exposure", type=float, default=0.55)
    parser.add_argument("--breadth-crash-threshold", type=float, default=0.32)
    parser.add_argument("--breadth-crash-exposure", type=float, default=0.20)
    return parser.parse_args(list(argv) if argv is not None else None)


def _optional_panel(
    raw: pd.DataFrame,
    close: pd.DataFrame,
    candidate_columns: tuple[str, ...],
) -> pd.DataFrame | None:
    column = next((name for name in candidate_columns if name in raw.columns), None)
    if column is None:
        return None
    return pivot_prices(raw, column).reindex_like(close)


def load_control_equity_curve(path: Path, *, asof_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"date", "equity"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Control equity curve is missing columns {missing}: {path}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError(f"Control equity curve contains invalid dates: {path}")
    if out["date"].duplicated().any():
        raise ValueError(f"Control equity curve contains duplicate dates: {path}")
    if out["equity"].isna().any() or out["equity"].le(0.0).any():
        raise ValueError(f"Control equity curve contains non-positive equity: {path}")

    asof = pd.Timestamp(asof_date).normalize()
    future = out.loc[out["date"].gt(asof)]
    if not future.empty:
        latest = pd.Timestamp(future["date"].max()).date().isoformat()
        raise ValueError(
            f"Control equity curve contains dates after {asof.date().isoformat()}: {path}; latest={latest}"
        )

    for column in (
        "turnover",
        "cost",
        "gross_exposure",
        "market_exposure",
        "positions_count",
        "blocked_limit_up_buys",
        "blocked_limit_down_sells",
        "blocked_open_gap_buys",
        "blocked_orders_total",
    ):
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.sort_values("date").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError(f"Control equity curve requires at least two rows: {path}")

    out["gross_return"] = out["equity"].pct_change(fill_method=None).fillna(0.0)
    out["variant_id"] = "Control"
    return out[
        [
            "variant_id",
            "date",
            "equity",
            "gross_return",
            "cost",
            "turnover",
            "gross_exposure",
            "market_exposure",
            "positions_count",
            "blocked_limit_up_buys",
            "blocked_limit_down_sells",
            "blocked_open_gap_buys",
            "blocked_orders_total",
        ]
    ].copy()


def build_legacy_alpha_score(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    base = build_features(close, open_px, high, low, amount)
    returns = close.pct_change(fill_method=None)
    trend120 = rank_pct(close / (close.rolling(120, min_periods=120).mean() + 1e-12) - 1.0)
    low_vol20 = rank_pct(-returns.rolling(20, min_periods=20).std())
    strong_focus_boost = returns.ge(0.07).astype(float) * STRONG_FOCUS_BOOST

    required_components = {
        "momentum60": base["momentum_60"],
        "trend120": trend120,
        "reversal5": base["reversal_5"],
        "low_vol20": low_vol20,
        "liquidity20": base["liquidity_20"],
    }
    eligibility_mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    for frame in required_components.values():
        eligibility_mask &= frame.notna()

    score = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    for name, weight in LEGACY_ALPHA_WEIGHTS.items():
        score = score + required_components[name] * weight
    score = (score + strong_focus_boost).where(eligibility_mask)

    definitions = {
        "momentum60": "rank_pct of trailing 60-session close return",
        "trend120": "rank_pct of close / MA120 - 1",
        "reversal5": "rank_pct of trailing 5-session reversal (-return_5)",
        "low_vol20": "rank_pct of negative 20-session close-return volatility",
        "liquidity20": "rank_pct of log1p trailing 20-session median amount",
        "strong_focus_boost": "add 0.20 when close-to-close daily return is at least 7%",
    }
    return score, definitions


def build_variant_exposure(
    spec: VariantSpec,
    close: pd.DataFrame,
    benchmark_path: Path,
    args: argparse.Namespace,
) -> pd.Series:
    exposure = pd.Series(1.0, index=close.index, dtype=float)
    if spec.use_market:
        market = load_market_exposure(
            str(benchmark_path),
            close.index,
            ma_window=args.market_ma_window,
            risk_off_drawdown_20d=args.market_risk_off_drawdown_20d,
            below_ma_exposure=args.market_below_ma_exposure,
            crash_exposure=args.market_crash_exposure,
        )
        exposure = pd.concat([exposure, market], axis=1).min(axis=1)
    if spec.use_breadth:
        breadth = build_breadth_exposure(
            close,
            ma_window=args.breadth_ma_window,
            threshold=args.breadth_threshold,
            below_exposure=args.breadth_below_exposure,
            crash_threshold=args.breadth_crash_threshold,
            crash_exposure=args.breadth_crash_exposure,
        )
        exposure = pd.concat([exposure, breadth], axis=1).min(axis=1)
    return exposure.fillna(1.0).clip(lower=0.0, upper=1.0)


def _target_from_score(score_row: pd.Series, exposure_scale: float) -> tuple[pd.Series, int, int]:
    candidates = score_row.dropna().sort_values(ascending=False)
    symbols = score_row.index
    target = pd.Series(0.0, index=symbols, dtype=float)
    if candidates.empty or exposure_scale <= 0.0:
        return target, int(len(candidates)), 0

    selected = candidates.index[:TOP_N]
    raw_weight = min(MAX_POSITION_WEIGHT, LEVERAGE / max(len(selected), 1))
    target.loc[selected] = raw_weight
    gross = float(target.abs().sum())
    if gross > LEVERAGE:
        target = target / gross * LEVERAGE
    target *= float(exposure_scale)
    return target, int(len(candidates)), int(len(selected))


def summarize_curve(curve: pd.DataFrame) -> dict[str, Any]:
    nav = pd.Series(curve["equity"].to_numpy(dtype=float), index=pd.to_datetime(curve["date"]), name="equity")
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    metrics: dict[str, Any] = {
        "final_equity": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "annualized_return": float(annualized_return(nav)),
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
        "trade_days": int(len(nav)),
        "avg_turnover": float(curve["turnover"].mean()) if curve["turnover"].notna().any() else None,
        "avg_gross_exposure": (
            float(curve["gross_exposure"].mean()) if curve["gross_exposure"].notna().any() else None
        ),
        "avg_market_exposure": (
            float(curve["market_exposure"].mean()) if curve["market_exposure"].notna().any() else None
        ),
        "avg_positions_count": (
            float(curve["positions_count"].mean()) if curve["positions_count"].notna().any() else None
        ),
    }
    for column in (
        "blocked_limit_up_buys",
        "blocked_limit_down_sells",
        "blocked_open_gap_buys",
        "blocked_orders_total",
    ):
        values = pd.to_numeric(curve[column], errors="coerce") if column in curve else pd.Series(dtype=float)
        metrics[column] = int(values.fillna(0.0).sum()) if values.notna().any() else None
    return metrics


def run_strict_replay_variant(
    spec: VariantSpec,
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    score: pd.DataFrame,
    market_exposure: pd.Series,
    *,
    initial_capital: float,
    limit_rate: pd.DataFrame | None = None,
    limit_up_price: pd.DataFrame | None = None,
    limit_down_price: pd.DataFrame | None = None,
) -> VariantRun:
    positions = pd.Series(0.0, index=close.columns, dtype=float)
    equity = float(initial_capital)
    curve_rows = [
        {
            "variant_id": spec.variant_id,
            "date": close.index[0].strftime("%Y-%m-%d"),
            "equity": equity,
            "gross_return": 0.0,
            "cost": 0.0,
            "turnover": 0.0,
            "gross_exposure": 0.0,
            "market_exposure": float(market_exposure.reindex([close.index[0]]).iloc[0]),
            "positions_count": 0,
            "blocked_limit_up_buys": 0,
            "blocked_limit_down_sells": 0,
            "blocked_open_gap_buys": 0,
            "blocked_orders_total": 0,
        }
    ]
    diagnostics_rows: list[dict[str, Any]] = []
    desired_target = pd.Series(0.0, index=close.columns, dtype=float)

    for i in range(len(close.index) - 2):
        signal_date = close.index[i]
        execute_date = close.index[i + 1]
        realize_date = close.index[i + 2]
        if i % FACTOR_REBALANCE_FREQUENCY == 0:
            desired_target, candidate_count, selected_count = _target_from_score(
                score.iloc[i], float(market_exposure.reindex([signal_date]).iloc[0])
            )
        else:
            candidate_count = int(score.iloc[i].notna().sum())
            selected_count = int(desired_target.ne(0.0).sum())
        target = desired_target.copy()
        adjusted, counts = apply_open_constraints_with_diagnostics(
            current=positions,
            target=target,
            open_row=open_px.iloc[i + 1],
            prev_close_row=close.iloc[i],
            max_buy_open_gap=MAX_BUY_OPEN_GAP,
            limit_buffer=LIMIT_BUFFER,
            block_limit_up_buys=True,
            block_limit_down_sells=True,
            limit_rate_row=limit_rate.loc[execute_date] if limit_rate is not None else None,
            limit_up_price_row=limit_up_price.loc[execute_date] if limit_up_price is not None else None,
            limit_down_price_row=limit_down_price.loc[execute_date] if limit_down_price is not None else None,
        )

        turnover = float((adjusted - positions).abs().sum())
        cost = turnover * (COMMISSION_BPS + IMPACT_BPS) / 1e4
        realized = open_px.iloc[i + 2].div(open_px.iloc[i + 1] + 1e-12).sub(1.0).reindex(close.columns).fillna(0.0)
        gross_return = float((adjusted * realized).sum())
        equity *= 1.0 + gross_return - cost
        positions = adjusted

        curve_rows.append(
            {
                "variant_id": spec.variant_id,
                "date": realize_date.strftime("%Y-%m-%d"),
                "equity": equity,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "gross_exposure": float(positions.abs().sum()),
                "market_exposure": float(market_exposure.reindex([signal_date]).iloc[0]),
                "positions_count": int(positions.ne(0.0).sum()),
                "blocked_limit_up_buys": int(counts["blocked_limit_up_buys"]),
                "blocked_limit_down_sells": int(counts["blocked_limit_down_sells"]),
                "blocked_open_gap_buys": int(counts["blocked_open_gap_buys"]),
                "blocked_orders_total": int(counts["blocked_orders_total"]),
            }
        )
        diagnostics_rows.append(
            {
                "variant_id": spec.variant_id,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "execute_date": execute_date.strftime("%Y-%m-%d"),
                "realize_date": realize_date.strftime("%Y-%m-%d"),
                "candidate_count": candidate_count,
                "selected_count": selected_count,
                "executed_positions_count": int(positions.ne(0.0).sum()),
                "turnover": turnover,
                "cost": cost,
                "gross_return": gross_return,
                "target_market_exposure": float(market_exposure.reindex([signal_date]).iloc[0]),
                **{name: int(value) for name, value in counts.items()},
            }
        )

    curve = pd.DataFrame(curve_rows)
    if len(curve) < 2:
        raise ValueError(f"Not enough observations for {spec.variant_id}")
    diagnostics = pd.DataFrame(diagnostics_rows)
    return VariantRun(spec, curve, diagnostics, summarize_curve(curve))


def _mean_or_none(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else None


def _sum_or_none(series: pd.Series) -> int | None:
    numeric = pd.to_numeric(series, errors="coerce")
    return int(numeric.fillna(0.0).sum()) if numeric.notna().any() else None


def compute_segment_metrics(
    curve: pd.DataFrame,
    *,
    variant_id: str,
    segment_id: str,
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    working = curve.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    if start is not None:
        working = working.loc[working["date"].ge(pd.Timestamp(start))]
    if end is not None:
        working = working.loc[working["date"].le(pd.Timestamp(end))]
    working = working.sort_values("date").reset_index(drop=True)

    row: dict[str, Any] = {
        "segment_id": segment_id,
        "segment_start": start,
        "segment_end": end,
        "variant_id": variant_id,
        "segment_available": False,
        "rows": int(len(working)),
        "first_date": None,
        "last_date": None,
        "start_equity": None,
        "end_equity": None,
        "total_return": None,
        "annualized_return": None,
        "max_drawdown": None,
        "sharpe_like": None,
        "avg_turnover": None,
        "avg_gross_exposure": None,
        "avg_market_exposure": None,
        "avg_positions_count": None,
        "blocked_limit_up_buys": None,
        "blocked_limit_down_sells": None,
        "blocked_open_gap_buys": None,
        "blocked_orders_total": None,
    }
    if len(working) < 2:
        return row

    nav = pd.Series(working["equity"].to_numpy(dtype=float), index=working["date"], name="equity")
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    row.update(
        {
            "segment_available": True,
            "rows": int(len(working)),
            "first_date": working["date"].iloc[0].strftime("%Y-%m-%d"),
            "last_date": working["date"].iloc[-1].strftime("%Y-%m-%d"),
            "start_equity": float(nav.iloc[0]),
            "end_equity": float(nav.iloc[-1]),
            "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
            "annualized_return": float(annualized_return(nav)),
            "max_drawdown": float(max_drawdown(nav)),
            "sharpe_like": float(sharpe_like(returns)),
            "avg_turnover": _mean_or_none(working["turnover"]),
            "avg_gross_exposure": _mean_or_none(working["gross_exposure"]),
            "avg_market_exposure": _mean_or_none(working["market_exposure"]),
            "avg_positions_count": _mean_or_none(working["positions_count"]),
            "blocked_limit_up_buys": _sum_or_none(working["blocked_limit_up_buys"]),
            "blocked_limit_down_sells": _sum_or_none(working["blocked_limit_down_sells"]),
            "blocked_open_gap_buys": _sum_or_none(working["blocked_open_gap_buys"]),
            "blocked_orders_total": _sum_or_none(working["blocked_orders_total"]),
        }
    )
    return row


def add_control_deltas(segment_frame: pd.DataFrame) -> pd.DataFrame:
    out = segment_frame.copy()
    control = out.loc[out["variant_id"].eq("Control"), ["segment_id", "total_return", "annualized_return", "max_drawdown", "sharpe_like"]]
    control = control.rename(
        columns={
            "total_return": "control_total_return",
            "annualized_return": "control_annualized_return",
            "max_drawdown": "control_max_drawdown",
            "sharpe_like": "control_sharpe_like",
        }
    )
    out = out.merge(control, on="segment_id", how="left")
    out["delta_total_return_vs_control"] = out["total_return"] - out["control_total_return"]
    out["delta_annualized_return_vs_control"] = out["annualized_return"] - out["control_annualized_return"]
    out["delta_max_drawdown_vs_control"] = out["max_drawdown"] - out["control_max_drawdown"]
    out["delta_sharpe_like_vs_control"] = out["sharpe_like"] - out["control_sharpe_like"]
    return out


def _fmt_pct(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "n/a" if pd.isna(numeric) else f"{float(numeric):.2%}"


def _fmt_num(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "n/a" if pd.isna(numeric) else f"{float(numeric):.3f}"


def _fmt_int(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "n/a" if pd.isna(numeric) else str(int(numeric))


def build_segment_markdown(segment_frame: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines = [
        "# Legacy Alpha Strict Replay",
        "",
        "research_only: true  ",
        "promotion_allowed: false  ",
        "trade_instruction: false",
        "",
        "Fixed replay conditions:",
        "",
        f"- Old alpha weights: `{json.dumps(LEGACY_ALPHA_WEIGHTS, ensure_ascii=True, separators=(',', ':'))}`",
        f"- Strong focus boost: `+{STRONG_FOCUS_BOOST:.2f}` when daily return is at least `7%`.",
        f"- Portfolio: `Top{TOP_N}`, single-name cap `{MAX_POSITION_WEIGHT:.0%}`, leverage `{LEVERAGE:.1f}`.",
        f"- Costs: commission `{COMMISSION_BPS:.1f}` bps + impact `{IMPACT_BPS:.1f}` bps.",
        f"- Execution: next-open, `max_buy_open_gap={MAX_BUY_OPEN_GAP:.0%}`, limit-up buys blocked, limit-down sells blocked.",
        f"- Control source: `{manifest['control_equity_path']}`",
        "",
    ]
    for segment_id, _, _ in SEGMENTS:
        subset = segment_frame.loc[segment_frame["segment_id"].eq(segment_id)].copy()
        lines.extend(
            [
                f"## {segment_id}",
                "",
                "| Variant | Return | Ann. | Max DD | Sharpe | Avg Turnover | Avg Exposure | Blocked Orders | Delta vs Control |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subset.to_dict("records"):
            lines.append(
                "| {variant} | {ret} | {ann} | {dd} | {sharpe} | {turnover} | {exposure} | {blocked} | {delta} |".format(
                    variant=row["variant_id"],
                    ret=_fmt_pct(row["total_return"]),
                    ann=_fmt_pct(row["annualized_return"]),
                    dd=_fmt_pct(row["max_drawdown"]),
                    sharpe=_fmt_num(row["sharpe_like"]),
                    turnover=_fmt_pct(row["avg_turnover"]),
                    exposure=_fmt_pct(row["avg_gross_exposure"]),
                    blocked=_fmt_int(row["blocked_orders_total"]),
                    delta=_fmt_pct(row["delta_total_return_vs_control"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `trend120` is fixed as `close / MA120 - 1`.",
            "- `low_vol20` is fixed as the cross-sectional rank of negative 20-session close-return volatility.",
            "- Control reads only the external champion equity curve; position overlap is not computed in this replay artifact.",
            "- This study writes no daily-production state and cannot auto-promote any variant.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    combined_curves: pd.DataFrame,
    diagnostics: pd.DataFrame,
    segment_frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_curves.to_csv(output_dir / "full_history_equity_curves.csv", index=False)
    diagnostics.to_csv(output_dir / "execution_diagnostics.csv", index=False)
    segment_frame.to_csv(output_dir / "segment_comparison.csv", index=False)

    (output_dir / "segment_comparison.json").write_text(
        json.dumps(
            {
                "research_only": True,
                "promotion_allowed": False,
                "segments": segment_frame.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "segment_comparison.md").write_text(
        build_segment_markdown(segment_frame, manifest),
        encoding="utf-8",
    )
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run_replay_study(
    *,
    data_path: Path,
    benchmark_path: Path,
    control_equity_path: Path,
    output_dir: Path,
    asof_date: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    max_abs_daily_return: float = MAX_ABS_DAILY_RETURN,
    market_ma_window: int = 120,
    market_risk_off_drawdown_20d: float = -0.08,
    market_below_ma_exposure: float = 0.60,
    market_crash_exposure: float = 0.0,
    breadth_ma_window: int = 60,
    breadth_threshold: float = 0.45,
    breadth_below_exposure: float = 0.55,
    breadth_crash_threshold: float = 0.32,
    breadth_crash_exposure: float = 0.20,
) -> dict[str, Any]:
    args = argparse.Namespace(
        market_ma_window=market_ma_window,
        market_risk_off_drawdown_20d=market_risk_off_drawdown_20d,
        market_below_ma_exposure=market_below_ma_exposure,
        market_crash_exposure=market_crash_exposure,
        breadth_ma_window=breadth_ma_window,
        breadth_threshold=breadth_threshold,
        breadth_below_exposure=breadth_below_exposure,
        breadth_crash_threshold=breadth_crash_threshold,
        breadth_crash_exposure=breadth_crash_exposure,
    )

    raw = load_prices(data_path, start_date, end_date)
    asof = pd.Timestamp(asof_date).normalize()
    latest_input_date = pd.to_datetime(raw["date"], errors="coerce").max()
    if pd.isna(latest_input_date):
        raise ValueError(f"Input panel contains no valid dates: {data_path}")
    if pd.Timestamp(latest_input_date).normalize() > asof:
        raise ValueError(
            f"Input panel contains dates after asof_date={asof.date().isoformat()}: "
            f"latest={pd.Timestamp(latest_input_date).date().isoformat()}"
        )
    close = clean_matrix(pivot_prices(raw, "close"), max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    limit_rate = _optional_panel(raw, close, LIMIT_RATE_COLUMNS)
    limit_up_price = _optional_panel(raw, close, LIMIT_UP_PRICE_COLUMNS)
    limit_down_price = _optional_panel(raw, close, LIMIT_DOWN_PRICE_COLUMNS)
    score, factor_definitions = build_legacy_alpha_score(close, open_px, high, low, amount)

    curve_frames = [load_control_equity_curve(control_equity_path, asof_date=asof_date)]
    diagnostics_frames: list[pd.DataFrame] = []
    variant_summaries: dict[str, dict[str, Any]] = {
        "Control": summarize_curve(curve_frames[0])
    }

    for spec in PREREGISTERED_VARIANTS:
        if spec.is_control:
            continue
        exposure = build_variant_exposure(spec, close, benchmark_path, args)
        run = run_strict_replay_variant(
            spec,
            close,
            open_px,
            score,
            exposure,
            initial_capital=initial_capital,
            limit_rate=limit_rate,
            limit_up_price=limit_up_price,
            limit_down_price=limit_down_price,
        )
        curve_frames.append(run.curve)
        diagnostics_frames.append(run.diagnostics)
        variant_summaries[spec.variant_id] = run.metrics

    combined_curves = pd.concat(curve_frames, ignore_index=True).sort_values(["variant_id", "date"]).reset_index(drop=True)
    diagnostics = (
        pd.concat(diagnostics_frames, ignore_index=True).sort_values(["variant_id", "signal_date"]).reset_index(drop=True)
        if diagnostics_frames
        else pd.DataFrame()
    )

    common_start = max(pd.to_datetime(frame["date"]).min() for frame in curve_frames)
    common_end = min(pd.to_datetime(frame["date"]).max() for frame in curve_frames)
    segment_rows: list[dict[str, Any]] = []
    for frame in curve_frames:
        variant_id = str(frame["variant_id"].iloc[0])
        for segment_id, start, end in SEGMENTS:
            if segment_id == "full_history":
                start = common_start.strftime("%Y-%m-%d")
                end = common_end.strftime("%Y-%m-%d")
            segment_rows.append(
                compute_segment_metrics(frame, variant_id=variant_id, segment_id=segment_id, start=start, end=end)
            )
    segment_frame = add_control_deltas(pd.DataFrame(segment_rows))
    validation = segment_frame[segment_frame["segment_id"].isin(
        ["2025H2", "2026-01-01_to_2026-06-15"]
    ) & ~segment_frame["variant_id"].eq("Control")]
    failed_gates: dict[str, list[str]] = {}
    for variant_id, rows in validation.groupby("variant_id", sort=False):
        failures: list[str] = []
        if len(rows) != 2 or not rows["segment_available"].astype(bool).all():
            failures.append("missing_preregistered_segment")
        if pd.to_numeric(rows["total_return"], errors="coerce").le(0.0).any():
            failures.append("non_positive_segment_return")
        if pd.to_numeric(rows["delta_total_return_vs_control"], errors="coerce").le(0.0).any():
            failures.append("non_positive_excess_vs_control")
        if pd.to_numeric(rows["max_drawdown"], errors="coerce").lt(-0.40).any():
            failures.append("segment_drawdown_below_floor")
        failures.append("control_position_overlap_unavailable")
        failed_gates[str(variant_id)] = failures

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "promotion_allowed": False,
        "trade_instruction": False,
        "data_path": str(data_path.resolve()),
        "benchmark_path": str(benchmark_path.resolve()),
        "control_equity_path": str(control_equity_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "asof_date": asof_date,
        "segments": [
            {
                "segment_id": segment_id,
                "start": common_start.strftime("%Y-%m-%d") if segment_id == "full_history" else start,
                "end": common_end.strftime("%Y-%m-%d") if segment_id == "full_history" else end,
            }
            for segment_id, start, end in SEGMENTS
        ],
        "preregistered_variants": [spec.variant_id for spec in PREREGISTERED_VARIANTS],
        "legacy_alpha_weights": LEGACY_ALPHA_WEIGHTS,
        "factor_definitions": factor_definitions,
        "strong_focus_boost": STRONG_FOCUS_BOOST,
        "execution": {
            "top_n": TOP_N,
            "max_position_weight": MAX_POSITION_WEIGHT,
            "leverage": LEVERAGE,
            "commission_bps": COMMISSION_BPS,
            "impact_bps": IMPACT_BPS,
            "max_buy_open_gap": MAX_BUY_OPEN_GAP,
            "limit_buffer": LIMIT_BUFFER,
            "block_limit_up_buys": True,
            "block_limit_down_sells": True,
            "max_abs_daily_return": max_abs_daily_return,
            "factor_rebalance_frequency": FACTOR_REBALANCE_FREQUENCY,
        },
        "failed_gates": failed_gates,
        "research_decision": {
            variant_id: "rejected" if failures else "historical_replay_passed_not_unseen"
            for variant_id, failures in failed_gates.items()
        },
        "variant_summaries": variant_summaries,
        "outputs": [
            "full_history_equity_curves.csv",
            "execution_diagnostics.csv",
            "segment_comparison.csv",
            "segment_comparison.json",
            "segment_comparison.md",
            "replay_manifest.json",
        ],
        "control_position_overlap_supported": False,
        "production_integration": "disabled_research_only",
    }
    write_outputs(output_dir, combined_curves, diagnostics, segment_frame, manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    result = run_replay_study(
        data_path=Path(args.data),
        benchmark_path=Path(args.benchmark),
        control_equity_path=Path(args.control_equity),
        output_dir=Path(args.output_dir),
        asof_date=args.asof_date,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        max_abs_daily_return=args.max_abs_daily_return,
        market_ma_window=args.market_ma_window,
        market_risk_off_drawdown_20d=args.market_risk_off_drawdown_20d,
        market_below_ma_exposure=args.market_below_ma_exposure,
        market_crash_exposure=args.market_crash_exposure,
        breadth_ma_window=args.breadth_ma_window,
        breadth_threshold=args.breadth_threshold,
        breadth_below_exposure=args.breadth_below_exposure,
        breadth_crash_threshold=args.breadth_crash_threshold,
        breadth_crash_exposure=args.breadth_crash_exposure,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
