"""Shared production benchmark-risk exposure rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def benchmark_market_exposure(
    close: pd.Series,
    *,
    ma_window: int,
    risk_off_drawdown_20d: float,
    below_ma_exposure: float,
    crash_exposure: float,
) -> pd.DataFrame:
    """Return point-in-time benchmark state and the approved exposure multiplier."""

    if ma_window < 1:
        raise ValueError("ma_window must be positive")
    series = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    series = series[series.gt(0.0)]
    if series.index.has_duplicates:
        raise ValueError("benchmark close index must be unique")

    ma = series.rolling(ma_window).mean()
    return_20d = series.pct_change(20, fill_method=None)
    exposure = pd.Series(1.0, index=series.index, dtype=float)
    exposure = exposure.where(~series.lt(ma), float(below_ma_exposure))
    exposure = exposure.where(
        ~return_20d.le(float(risk_off_drawdown_20d)),
        float(crash_exposure),
    )
    exposure = exposure.fillna(1.0).clip(lower=0.0, upper=1.0)

    reason = pd.Series("risk_on", index=series.index, dtype=object)
    reason = reason.where(~series.lt(ma), "below_ma")
    reason = reason.where(
        ~return_20d.le(float(risk_off_drawdown_20d)),
        "drawdown_20d_crash",
    )
    return pd.DataFrame(
        {
            "close": series,
            "moving_average": ma,
            "return_20d": return_20d,
            "risk_exposure": exposure,
            "risk_reason": reason,
        }
    )


def load_benchmark_market_exposure(
    benchmark_path: str | Path | None,
    trade_dates: pd.Index,
    *,
    ma_window: int,
    risk_off_drawdown_20d: float,
    below_ma_exposure: float,
    crash_exposure: float,
) -> pd.Series:
    """Load benchmark history and align the production exposure to trade dates."""

    dates = pd.DatetimeIndex(pd.to_datetime(trade_dates, errors="raise"))
    if not benchmark_path:
        return pd.Series(1.0, index=dates)
    benchmark = pd.read_csv(benchmark_path, parse_dates=["date"])
    if not {"date", "close"}.issubset(benchmark.columns):
        raise ValueError("benchmark must include date and close columns")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "close"]).sort_values("date")
    if benchmark["date"].duplicated().any():
        raise ValueError("benchmark dates must be unique")
    aligned = benchmark.set_index("date")["close"].reindex(dates).ffill()
    state = benchmark_market_exposure(
        aligned,
        ma_window=ma_window,
        risk_off_drawdown_20d=risk_off_drawdown_20d,
        below_ma_exposure=below_ma_exposure,
        crash_exposure=crash_exposure,
    )
    return state["risk_exposure"].reindex(dates).fillna(1.0)


def latest_benchmark_risk_state(
    benchmark_path: str | Path,
    asof_date: str,
    *,
    ma_window: int,
    risk_off_drawdown_20d: float,
    below_ma_exposure: float,
    crash_exposure: float,
) -> dict[str, object]:
    """Return the exact-date production risk state for a pre-live decision."""

    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(path)
    benchmark = pd.read_csv(path, parse_dates=["date"])
    if not {"date", "close"}.issubset(benchmark.columns):
        raise ValueError("benchmark must include date and close columns")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "close"]).sort_values("date")
    if benchmark["date"].duplicated().any():
        raise ValueError("benchmark dates must be unique")

    target = pd.Timestamp(asof_date).normalize()
    history = benchmark[benchmark["date"].dt.normalize().le(target)].copy()
    if history.empty or history.iloc[-1]["date"].normalize() != target:
        latest = None if history.empty else history.iloc[-1]["date"].strftime("%Y-%m-%d")
        raise ValueError(
            f"benchmark is not current to {target.strftime('%Y-%m-%d')}; latest={latest}"
        )
    minimum_history = max(int(ma_window), 21)
    if len(history) < minimum_history:
        raise ValueError(
            f"benchmark history is insufficient: {len(history)} < {minimum_history}"
        )

    state = benchmark_market_exposure(
        history.set_index("date")["close"],
        ma_window=ma_window,
        risk_off_drawdown_20d=risk_off_drawdown_20d,
        below_ma_exposure=below_ma_exposure,
        crash_exposure=crash_exposure,
    ).iloc[-1]
    return {
        "asof_date": target.strftime("%Y-%m-%d"),
        "benchmark_path": str(path.resolve()),
        "benchmark_close": float(state["close"]),
        "benchmark_ma": float(state["moving_average"]),
        "benchmark_return_20d": float(state["return_20d"]),
        "risk_exposure": float(state["risk_exposure"]),
        "risk_reason": str(state["risk_reason"]),
        "ma_window": int(ma_window),
        "risk_off_drawdown_20d": float(risk_off_drawdown_20d),
        "below_ma_exposure": float(below_ma_exposure),
        "crash_exposure": float(crash_exposure),
    }
