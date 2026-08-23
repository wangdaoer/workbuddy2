"""Forward-only observer for weak-breadth and high-exposure mismatch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "market_breadth_mismatch_observer_20260723.json"

LEDGER_FIELDS = (
    "registration_id",
    "date",
    "recorded_at",
    "source_latest_date",
    "observation_day_index",
    "breadth_above_ma60",
    "cross_sectional_median_return20",
    "available_symbols",
    "breadth_eligible_symbols",
    "return_eligible_symbols",
    "actual_gross_exposure",
    "benchmark_market_exposure_target",
    "weak_breadth",
    "high_benchmark_exposure",
    "weak_breadth_high_exposure_mismatch",
    "next_observed_date",
    "next_incumbent_return",
)

CONFIG_KEYS = {
    "schema_version",
    "registration_id",
    "status",
    "frozen_at",
    "observation_start_date",
    "target_matured_trade_days",
    "breadth_ma_window",
    "cross_sectional_return_window",
    "weak_breadth_threshold",
    "high_benchmark_exposure_threshold",
    "research_only",
    "promotion_allowed",
    "automatic_model_change",
    "selection_effect",
    "position_effect",
    "notes",
}


@dataclass(frozen=True)
class ObserverConfig:
    registration_id: str
    frozen_at: str
    observation_start_date: str
    target_matured_trade_days: int
    breadth_ma_window: int
    cross_sectional_return_window: int
    weak_breadth_threshold: float
    high_benchmark_exposure_threshold: float
    research_only: bool
    promotion_allowed: bool
    automatic_model_change: bool
    selection_effect: bool
    position_effect: bool
    config_sha256: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_observer_config(path: Path) -> ObserverConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        missing = sorted(CONFIG_KEYS.difference(raw if isinstance(raw, dict) else {}))
        extra = sorted(set(raw if isinstance(raw, dict) else {}).difference(CONFIG_KEYS))
        raise ValueError(f"observer config keys mismatch: missing={missing} extra={extra}")
    if raw["schema_version"] != 1:
        raise ValueError("unsupported observer config schema_version")
    if raw["status"] != "preregistered_research_only":
        raise ValueError("observer status must remain preregistered_research_only")
    if int(raw["breadth_ma_window"]) != 60:
        raise ValueError("observer breadth_ma_window must remain 60")
    if int(raw["cross_sectional_return_window"]) != 20:
        raise ValueError("observer cross_sectional_return_window must remain 20")
    if int(raw["target_matured_trade_days"]) < 1:
        raise ValueError("target_matured_trade_days must be positive")
    weak = float(raw["weak_breadth_threshold"])
    high = float(raw["high_benchmark_exposure_threshold"])
    if not 0.0 < weak < 1.0 or not 0.0 <= high <= 1.0:
        raise ValueError("observer thresholds are outside valid ranges")
    if not bool(raw["research_only"]):
        raise ValueError("observer must remain research_only")
    for key in (
        "promotion_allowed",
        "automatic_model_change",
        "selection_effect",
        "position_effect",
    ):
        if bool(raw[key]):
            raise ValueError(f"observer {key} must remain false")
    return ObserverConfig(
        registration_id=str(raw["registration_id"]),
        frozen_at=str(raw["frozen_at"]),
        observation_start_date=str(raw["observation_start_date"]),
        target_matured_trade_days=int(raw["target_matured_trade_days"]),
        breadth_ma_window=int(raw["breadth_ma_window"]),
        cross_sectional_return_window=int(raw["cross_sectional_return_window"]),
        weak_breadth_threshold=weak,
        high_benchmark_exposure_threshold=high,
        research_only=True,
        promotion_allowed=False,
        automatic_model_change=False,
        selection_effect=False,
        position_effect=False,
        config_sha256=_sha256_file(path),
    )


def _load_unique_dates(path: Path, required: set[str], *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} is empty")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError(f"{label} contains duplicate dates")
    return frame.sort_values("date").reset_index(drop=True)


def load_market_diagnostics(path: Path) -> pd.DataFrame:
    required = {
        "date",
        "breadth_above_ma60",
        "cross_sectional_median_return20",
        "available_symbols",
        "breadth_eligible_symbols",
        "return_eligible_symbols",
        "model_market_exposure_target",
    }
    frame = _load_unique_dates(path, required, label="market diagnostics")
    for column in required.difference({"date"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_incumbent_equity(path: Path) -> pd.DataFrame:
    required = {
        "date",
        "gross_return",
        "cost",
        "gross_exposure",
        "market_exposure",
    }
    frame = _load_unique_dates(path, required, label="incumbent equity")
    for column in required.difference({"date"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required.difference({"date"}))].isna().any(axis=None):
        raise ValueError("incumbent equity contains non-numeric required values")
    frame["net_return"] = frame["gross_return"] - frame["cost"]
    if frame["net_return"].le(-1.0).any():
        raise ValueError("incumbent net returns must be greater than -100%")
    return frame


def load_trade_audit(path: Path) -> pd.DataFrame:
    required = {
        "signal_date",
        "realize_date",
        "gross_return",
        "cost",
    }
    if not path.exists():
        raise FileNotFoundError(f"incumbent trade audit not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"incumbent trade audit missing columns: {missing}")
    frame = frame.copy()
    frame["signal_date"] = pd.to_datetime(
        frame["signal_date"], errors="raise"
    ).dt.normalize()
    frame["realize_date"] = pd.to_datetime(
        frame["realize_date"], errors="raise"
    ).dt.normalize()
    if frame["signal_date"].duplicated().any():
        raise ValueError("incumbent trade audit contains duplicate signal dates")
    if frame["realize_date"].duplicated().any():
        raise ValueError("incumbent trade audit contains duplicate realize dates")
    for column in ("gross_return", "cost"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["gross_return", "cost"]].isna().any(axis=None):
        raise ValueError("incumbent trade audit contains non-numeric returns")
    if frame["realize_date"].le(frame["signal_date"]).any():
        raise ValueError("incumbent trade audit realize dates must follow signal dates")
    frame["net_return"] = frame["gross_return"] - frame["cost"]
    if frame["net_return"].le(-1.0).any():
        raise ValueError("incumbent trade audit net returns must be greater than -100%")
    return frame.sort_values("signal_date").reset_index(drop=True)


def _iso_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _serialize(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def build_observation_rows(
    diagnostics: pd.DataFrame,
    equity: pd.DataFrame,
    config: ObserverConfig,
    *,
    source_latest_date: str,
    observation_day_index: int = 1,
) -> list[dict[str, object]]:
    diagnostic_latest = pd.Timestamp(diagnostics["date"].max()).normalize()
    equity_latest = pd.Timestamp(equity["date"].max()).normalize()
    if diagnostic_latest != equity_latest:
        raise ValueError("market diagnostics and incumbent equity latest dates must match")
    start = pd.Timestamp(config.observation_start_date).normalize()
    date = diagnostic_latest
    if date < start:
        return []
    recorded_at = datetime.now().isoformat(timespec="seconds")
    market = diagnostics.set_index("date").loc[date]
    portfolio = equity.set_index("date").loc[equity_latest]
    values = market[
        [
            "breadth_above_ma60",
            "cross_sectional_median_return20",
            "available_symbols",
            "breadth_eligible_symbols",
            "return_eligible_symbols",
            "model_market_exposure_target",
        ]
    ]
    if values.isna().any():
        raise ValueError(f"market diagnostics incomplete on {_iso_date(date)}")
    target = float(market["model_market_exposure_target"])
    breadth = float(market["breadth_above_ma60"])
    weak_breadth = breadth < config.weak_breadth_threshold
    high_exposure = target >= config.high_benchmark_exposure_threshold
    return [
        {
            "registration_id": config.registration_id,
            "date": _iso_date(date),
            "recorded_at": recorded_at,
            "source_latest_date": source_latest_date,
            "observation_day_index": observation_day_index,
            "breadth_above_ma60": breadth,
            "cross_sectional_median_return20": float(
                market["cross_sectional_median_return20"]
            ),
            "available_symbols": int(market["available_symbols"]),
            "breadth_eligible_symbols": int(market["breadth_eligible_symbols"]),
            "return_eligible_symbols": int(market["return_eligible_symbols"]),
            "actual_gross_exposure": float(portfolio["gross_exposure"]),
            "benchmark_market_exposure_target": target,
            "weak_breadth": weak_breadth,
            "high_benchmark_exposure": high_exposure,
            "weak_breadth_high_exposure_mismatch": weak_breadth and high_exposure,
            "next_observed_date": None,
            "next_incumbent_return": None,
        }
    ]


def _read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEDGER_FIELDS:
            raise ValueError("market breadth mismatch ledger schema does not match")
        return list(reader)


def merge_ledger_rows(
    existing: list[dict[str, str]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    serialized = [{field: str(row.get(field, "")) for field in LEDGER_FIELDS} for row in existing]
    keys = {
        (str(row.get("registration_id")), str(row.get("date"))) for row in serialized
    }
    for row in new_rows:
        key = (str(row["registration_id"]), str(row["date"]))
        if key in keys:
            continue
        serialized.append(
            {field: _serialize(row.get(field)) for field in LEDGER_FIELDS}
        )
        keys.add(key)
    return sorted(
        serialized,
        key=lambda row: (str(row.get("registration_id")), str(row.get("date"))),
    )


def mature_ledger_rows(
    ledger_rows: list[dict[str, str]],
    trade_audit: pd.DataFrame,
    equity: pd.DataFrame,
    config: ObserverConfig,
) -> list[dict[str, str]]:
    trade_by_signal = trade_audit.set_index("signal_date")
    equity_by_date = equity.set_index("date")
    matured: list[dict[str, str]] = []
    for original in ledger_rows:
        row = original.copy()
        if row.get("registration_id") != config.registration_id:
            matured.append(row)
            continue
        signal_date = pd.Timestamp(row["date"]).normalize()
        if signal_date not in trade_by_signal.index:
            matured.append(row)
            continue
        trade = trade_by_signal.loc[signal_date]
        realize_date = pd.Timestamp(trade["realize_date"]).normalize()
        if realize_date not in equity_by_date.index:
            raise ValueError(
                f"incumbent equity missing trade realize date {_iso_date(realize_date)}"
            )
        trade_return = float(trade["net_return"])
        equity_return = float(equity_by_date.loc[realize_date, "net_return"])
        if not math.isclose(trade_return, equity_return, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"trade audit and equity return mismatch on {_iso_date(realize_date)}"
            )
        existing_date = str(row.get("next_observed_date") or "").strip()
        existing_return = _parse_float(row.get("next_incumbent_return"))
        if existing_date and existing_date != _iso_date(realize_date):
            raise ValueError(f"matured outcome date changed for {_iso_date(signal_date)}")
        if existing_return is not None and not math.isclose(
            existing_return, trade_return, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"matured outcome changed for {_iso_date(signal_date)}")
        row["next_observed_date"] = _iso_date(realize_date)
        row["next_incumbent_return"] = _serialize(trade_return)
        matured.append(row)
    return matured


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _latest_snapshot(
    diagnostics: pd.DataFrame,
    equity: pd.DataFrame,
    config: ObserverConfig,
) -> dict[str, object]:
    diagnostic_by_date = diagnostics.set_index("date")
    equity_by_date = equity.set_index("date")
    common = diagnostic_by_date.index.intersection(equity_by_date.index)
    if common.empty:
        raise ValueError("market diagnostics and incumbent equity have no shared dates")
    date = common.max()
    market = diagnostic_by_date.loc[date]
    portfolio = equity_by_date.loc[date]
    breadth = float(market["breadth_above_ma60"])
    target = float(market["model_market_exposure_target"])
    if not math.isfinite(breadth):
        raise ValueError("latest market breadth is unavailable")
    return {
        "date": _iso_date(date),
        "breadth_above_ma60": breadth,
        "cross_sectional_median_return20": float(
            market["cross_sectional_median_return20"]
        ),
        "available_symbols": int(market["available_symbols"]),
        "breadth_eligible_symbols": int(market["breadth_eligible_symbols"]),
        "return_eligible_symbols": int(market["return_eligible_symbols"]),
        "actual_gross_exposure": float(portfolio["gross_exposure"]),
        "benchmark_market_exposure_target": target,
        "weak_breadth": breadth < config.weak_breadth_threshold,
        "high_benchmark_exposure": target
        >= config.high_benchmark_exposure_threshold,
        "weak_breadth_high_exposure_mismatch": (
            breadth < config.weak_breadth_threshold
            and target >= config.high_benchmark_exposure_threshold
        ),
    }


def summarize_observations(
    ledger_rows: list[dict[str, str]],
    diagnostics: pd.DataFrame,
    equity: pd.DataFrame,
    config: ObserverConfig,
    *,
    source_latest_date: str,
) -> dict[str, object]:
    observed = sorted(
        [row for row in ledger_rows if row.get("registration_id") == config.registration_id],
        key=lambda row: str(row["date"]),
    )
    matured = [row for row in observed if _parse_float(row.get("next_incumbent_return")) is not None]
    mismatches = [
        row for row in observed if _parse_bool(row.get("weak_breadth_high_exposure_mismatch"))
    ]
    mismatch_matured = [
        float(_parse_float(row["next_incumbent_return"]))
        for row in matured
        if _parse_bool(row.get("weak_breadth_high_exposure_mismatch"))
    ]
    other_matured = [
        float(_parse_float(row["next_incumbent_return"]))
        for row in matured
        if not _parse_bool(row.get("weak_breadth_high_exposure_mismatch"))
    ]
    mismatch_mean = _mean(mismatch_matured)
    other_mean = _mean(other_matured)
    matured_count = len(matured)
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "registration_id": config.registration_id,
        "config_sha256": config.config_sha256,
        "status": (
            "diagnostic_review_ready"
            if matured_count >= config.target_matured_trade_days
            else "collecting"
        ),
        "frozen_at": config.frozen_at,
        "observation_start_date": config.observation_start_date,
        "source_latest_date": source_latest_date,
        "target_matured_trade_days": config.target_matured_trade_days,
        "valid_observation_count": len(observed),
        "matured_outcome_count": matured_count,
        "remaining_matured_outcomes": max(
            config.target_matured_trade_days - matured_count, 0
        ),
        "mismatch_observation_count": len(mismatches),
        "mismatch_observation_ratio": (
            len(mismatches) / len(observed) if observed else None
        ),
        "mismatch_matured_outcome_count": len(mismatch_matured),
        "other_matured_outcome_count": len(other_matured),
        "mismatch_mean_next_return": mismatch_mean,
        "other_mean_next_return": other_mean,
        "mismatch_next_return_spread": (
            mismatch_mean - other_mean
            if mismatch_mean is not None and other_mean is not None
            else None
        ),
        "latest_snapshot": _latest_snapshot(diagnostics, equity, config),
        "diagnostic_thresholds": {
            "weak_breadth_below": config.weak_breadth_threshold,
            "high_benchmark_exposure_at_least": config.high_benchmark_exposure_threshold,
        },
        "research_only": config.research_only,
        "promotion_allowed": config.promotion_allowed,
        "automatic_model_change": config.automatic_model_change,
        "selection_effect": config.selection_effect,
        "position_effect": config.position_effect,
        "trade_instruction": False,
    }


def render_report(summary: Mapping[str, object]) -> str:
    latest = summary.get("latest_snapshot") or {}
    return "\n".join(
        [
            "# Market Breadth Mismatch Forward Observer",
            "",
            f"- Registration: `{summary['registration_id']}`",
            f"- Source latest date: `{summary['source_latest_date']}`",
            f"- Status: `{summary['status']}`",
            f"- Matured outcomes: `{summary['matured_outcome_count']}/{summary['target_matured_trade_days']}`",
            f"- Research only: `{summary['research_only']}`",
            f"- Selection effect: `{summary['selection_effect']}`",
            f"- Position effect: `{summary['position_effect']}`",
            "",
            "## Latest Market State",
            "",
            f"- Date: `{latest.get('date')}`",
            f"- Breadth above MA60: `{latest.get('breadth_above_ma60')}`",
            f"- Cross-sectional median return20: `{latest.get('cross_sectional_median_return20')}`",
            f"- Actual gross exposure: `{latest.get('actual_gross_exposure')}`",
            f"- Benchmark market exposure target: `{latest.get('benchmark_market_exposure_target')}`",
            f"- Weak-breadth/high-exposure mismatch: `{latest.get('weak_breadth_high_exposure_mismatch')}`",
            "",
            "## Forward Outcomes",
            "",
            f"- Valid observations: `{summary['valid_observation_count']}`",
            f"- Mismatch observations: `{summary['mismatch_observation_count']}`",
            f"- Mismatch ratio: `{summary['mismatch_observation_ratio']}`",
            f"- Mismatch mean next return: `{summary['mismatch_mean_next_return']}`",
            f"- Other mean next return: `{summary['other_mean_next_return']}`",
            f"- Mismatch next-return spread: `{summary['mismatch_next_return_spread']}`",
            "",
            "The observer cannot change rankings, positions, orders, or production configuration.",
            "",
        ]
    )


def _csv_text(rows: list[dict[str, str]]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _stage_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_publish_text(files: Mapping[Path, str]) -> None:
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    existed: dict[Path, bool] = {}
    try:
        for path, content in files.items():
            staged[path] = _stage_text(path, content)
        for path in files:
            existed[path] = path.exists()
            if path.exists():
                backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.bak")
                path.replace(backup)
                backups[path] = backup
            staged[path].replace(path)
    except Exception:
        for path, backup in backups.items():
            path.unlink(missing_ok=True)
            backup.replace(path)
        for path, did_exist in existed.items():
            if not did_exist:
                path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def run_observer(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(getattr(args, "config", DEFAULT_CONFIG)).resolve()
    config = load_observer_config(config_path)
    diagnostics = load_market_diagnostics(Path(args.market_diagnostics).resolve())
    equity = load_incumbent_equity(Path(args.incumbent_equity).resolve())
    trade_audit = load_trade_audit(Path(args.incumbent_trade_audit).resolve())
    source_latest_date = _iso_date(diagnostics["date"].max())
    ledger_path = Path(args.ledger).resolve()
    existing = _read_ledger(ledger_path)
    existing_registration_count = sum(
        row.get("registration_id") == config.registration_id for row in existing
    )
    new_rows = build_observation_rows(
        diagnostics,
        equity,
        config,
        source_latest_date=source_latest_date,
        observation_day_index=existing_registration_count + 1,
    )
    summary_path = Path(args.summary).resolve()
    report_path = Path(args.report).resolve()
    ledger = mature_ledger_rows(
        merge_ledger_rows(existing, new_rows),
        trade_audit,
        equity,
        config,
    )
    summary = summarize_observations(
        ledger,
        diagnostics,
        equity,
        config,
        source_latest_date=source_latest_date,
    )
    _atomic_publish_text(
        {
            ledger_path: _csv_text(ledger),
            summary_path: json.dumps(
                summary, ensure_ascii=False, allow_nan=False, indent=2
            )
            + "\n",
            report_path: render_report(summary),
        }
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track weak-breadth/high-benchmark-exposure mismatch."
    )
    parser.add_argument("--market-diagnostics", required=True)
    parser.add_argument("--incumbent-equity", required=True)
    parser.add_argument("--incumbent-trade-audit", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_observer(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
