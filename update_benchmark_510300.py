"""Refresh the canonical 510300 benchmark with cross-source verification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time as time_module
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class BenchmarkRow:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: int
    observed_time: time | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "date": self.date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
        }


def _validated_row(row: BenchmarkRow, source: str) -> BenchmarkRow:
    prices = (row.open, row.high, row.low, row.close)
    if not all(math.isfinite(value) and value > 0.0 for value in prices):
        raise ValueError(f"{source} contains invalid prices for {row.date}")
    if row.low > min(row.open, row.close) or row.high < max(row.open, row.close):
        raise ValueError(f"{source} contains inconsistent OHLC for {row.date}")
    if row.volume <= 0 or row.amount < 0:
        raise ValueError(f"{source} contains invalid volume or amount for {row.date}")
    return row


def parse_sina_quote(text: str, symbol: str) -> BenchmarkRow:
    parts = text.split('"')
    if len(parts) < 3 or not parts[1].strip():
        raise ValueError(f"Sina returned an empty quote for {symbol}")
    fields = next(csv.reader([parts[1]]))
    if len(fields) < 32:
        raise ValueError(f"Sina returned an incomplete quote for {symbol}")
    quote_time = datetime.strptime(fields[31], "%H:%M:%S").time()
    row = BenchmarkRow(
        date=datetime.strptime(fields[30], "%Y-%m-%d").date(),
        open=float(fields[1]),
        high=float(fields[4]),
        low=float(fields[5]),
        close=float(fields[3]),
        volume=int(float(fields[8])),
        amount=int(round(float(fields[9]))),
        observed_time=quote_time,
    )
    return _validated_row(row, "Sina")


def parse_sohu_history(text: str) -> dict[date, BenchmarkRow]:
    prefix = "historySearchHandler("
    if not text.startswith(prefix) or not text.rstrip().endswith(")"):
        raise ValueError("Sohu returned malformed history data")
    payload = json.loads(text[len(prefix): text.rfind(")")])
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError("Sohu returned malformed history data")
    result: dict[date, BenchmarkRow] = {}
    for values in payload[0].get("hq") or []:
        if not isinstance(values, list) or len(values) < 9:
            raise ValueError("Sohu returned an incomplete history row")
        row = BenchmarkRow(
            date=datetime.strptime(values[0], "%Y-%m-%d").date(),
            open=float(values[1]),
            high=float(values[6]),
            low=float(values[5]),
            close=float(values[2]),
            volume=int(round(float(values[7]) * 100.0)),
            amount=int(round(float(values[8]) * 10_000.0)),
        )
        result[row.date] = _validated_row(row, "Sohu")
    if not result:
        raise ValueError("Sohu returned no history rows")
    return result


def parse_yahoo_chart(payload: Mapping[str, object]) -> dict[date, BenchmarkRow]:
    try:
        chart = payload["chart"]
        if not isinstance(chart, Mapping) or chart.get("error") is not None:
            raise ValueError("Yahoo returned a chart error")
        results = chart["result"]
        if not isinstance(results, list) or not results:
            raise ValueError("Yahoo returned no chart result")
        result = results[0]
        if not isinstance(result, Mapping):
            raise ValueError("Yahoo returned malformed chart data")
        timestamps = result["timestamp"]
        indicators = result["indicators"]
        if not isinstance(indicators, Mapping):
            raise ValueError("Yahoo returned malformed indicators")
        quote_list = indicators["quote"]
        if not isinstance(quote_list, list) or not quote_list:
            raise ValueError("Yahoo returned no quote rows")
        quote = quote_list[0]
        if not isinstance(quote, Mapping) or not isinstance(timestamps, list):
            raise ValueError("Yahoo returned malformed quote rows")
    except (KeyError, TypeError) as exc:
        raise ValueError("Yahoo returned malformed chart data") from exc

    zone = ZoneInfo("Asia/Shanghai")
    rows: dict[date, BenchmarkRow] = {}
    for index, timestamp in enumerate(timestamps):
        try:
            values = {
                key: quote[key][index]
                for key in ("open", "high", "low", "close", "volume")
            }
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Yahoo returned an incomplete quote row") from exc
        if any(value is None for value in values.values()):
            continue
        row = BenchmarkRow(
            date=datetime.fromtimestamp(int(timestamp), zone).date(),
            open=round(float(values["open"]), 3),
            high=round(float(values["high"]), 3),
            low=round(float(values["low"]), 3),
            close=round(float(values["close"]), 3),
            volume=int(round(float(values["volume"]))),
            amount=0,
        )
        try:
            rows[row.date] = _validated_row(row, "Yahoo")
        except ValueError:
            # 停牌/无数据行（如 volume<=0）：按"缺席"跳过，而非让整段历史拉取失败
            continue
    if not rows:
        raise ValueError("Yahoo returned no usable quote rows")
    return rows


def _request_text(
    url: str,
    *,
    params: Mapping[str, object],
    headers: Mapping[str, str],
    encoding: str = "utf-8",
) -> str:
    query = urlencode(params)
    request = Request(f"{url}?{query}" if query else url, headers=dict(headers))
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} from {url}")
                return response.read().decode(encoding, errors="strict")
        except (OSError, TimeoutError, URLError) as exc:
            last_error = exc
            if attempt < 2:
                time_module.sleep(attempt + 1)
    raise RuntimeError(f"Request failed after 3 attempts: {url}") from last_error


def fetch_sina_quote(symbol: str) -> BenchmarkRow:
    text = _request_text(
        "https://hq.sinajs.cn/list=" + f"sh{symbol}",
        params={},
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
        encoding="gbk",
    )
    return parse_sina_quote(text, symbol)


def fetch_sohu_history(symbol: str, start: date, end: date) -> dict[date, BenchmarkRow]:
    text = _request_text(
        "https://q.stock.sohu.com/hisHq",
        params={
            "code": f"cn_{symbol}",
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "stat": 1,
            "order": "D",
            "period": "d",
            "callback": "historySearchHandler",
            "rt": "jsonp",
        },
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://q.stock.sohu.com/"},
        encoding="gbk",
    )
    return parse_sohu_history(text)


def fetch_yahoo_history(symbol: str, start: date, end: date) -> dict[date, BenchmarkRow]:
    period1 = int(datetime.combine(start, time.min, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=2), time.min, tzinfo=timezone.utc).timestamp())
    text = _request_text(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.SS",
        params={"period1": period1, "period2": period2, "interval": "1d", "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"},
    )
    return parse_yahoo_chart(json.loads(text))


def _validate_agreement(left: BenchmarkRow, right: BenchmarkRow, labels: str) -> None:
    if left.date != right.date:
        raise ValueError(f"{labels} source disagreement: dates do not match")
    for field in ("open", "high", "low", "close"):
        if abs(getattr(left, field) - getattr(right, field)) > 0.0015:
            raise ValueError(f"{labels} source disagreement for {left.date}: {field}")
    if abs(left.volume - right.volume) > max(100, int(left.volume * 0.000001)):
        raise ValueError(f"{labels} source disagreement for {left.date}: volume")
    if left.amount > 0 and right.amount > 0:
        if abs(left.amount - right.amount) > max(10_000, int(left.amount * 0.00001)):
            raise ValueError(f"{labels} source disagreement for {left.date}: amount")


SOURCE_ORDER = ("Sohu", "Yahoo", "Sina")
SOURCE_PAIRS = (("Sohu", "Yahoo"), ("Yahoo", "Sina"), ("Sohu", "Sina"))
SOURCE_FAILURES = (KeyError, OSError, RuntimeError, TimeoutError, TypeError, URLError, ValueError)


def _select_source_quorum(
    rows: Mapping[str, BenchmarkRow],
    trading_date: date,
) -> tuple[list[str], list[str]]:
    """Return a mutually agreeing 2-of-3 source quorum and mismatch details."""

    matching_pairs: list[tuple[str, str]] = []
    mismatches: list[str] = []
    for left_label, right_label in SOURCE_PAIRS:
        if left_label not in rows or right_label not in rows:
            continue
        try:
            _validate_agreement(
                rows[left_label],
                rows[right_label],
                f"{left_label}/{right_label}",
            )
        except ValueError as exc:
            mismatches.append(str(exc))
        else:
            matching_pairs.append((left_label, right_label))

    if len(rows) == 3 and len(matching_pairs) == 3:
        return list(SOURCE_ORDER), mismatches
    if matching_pairs:
        return list(matching_pairs[0]), mismatches

    available = [label for label in SOURCE_ORDER if label in rows]
    details = f"; details: {'; '.join(mismatches)}" if mismatches else ""
    raise ValueError(
        f"Benchmark requires two agreeing sources for {trading_date}; "
        f"available={available}{details}"
    )


def _compose_quorum_row(
    rows: Mapping[str, BenchmarkRow],
    quorum_sources: list[str],
) -> BenchmarkRow:
    if "Sina" in quorum_sources:
        return rows["Sina"]
    if {"Sohu", "Yahoo"}.issubset(quorum_sources):
        sohu_row = rows["Sohu"]
        yahoo_row = rows["Yahoo"]
        return BenchmarkRow(
            date=yahoo_row.date,
            open=yahoo_row.open,
            high=yahoo_row.high,
            low=yahoo_row.low,
            close=yahoo_row.close,
            volume=yahoo_row.volume,
            amount=sohu_row.amount,
        )
    raise ValueError(f"Unsupported benchmark quorum: {quorum_sources}")


def _load_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"Benchmark schema must be exactly: {', '.join(COLUMNS)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame.empty or frame["date"].isna().any():
        raise ValueError("Benchmark contains no rows or invalid dates")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("Benchmark dates must be unique and increasing")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(COLUMNS[1:])].isna().any().any():
        raise ValueError("Benchmark contains invalid numeric values")
    for record in frame.to_dict("records"):
        _validated_row(
            BenchmarkRow(
                date=record["date"].date(),
                open=float(record["open"]),
                high=float(record["high"]),
                low=float(record["low"]),
                close=float(record["close"]),
                volume=int(record["volume"]),
                amount=int(record["amount"]),
            ),
            "local benchmark",
        )
    return frame


def _stage_csv(path: Path, frame: pd.DataFrame) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", date_format="%Y-%m-%d")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stage_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = _stage_csv(path, frame)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = _stage_json(path, payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv_and_status(
    benchmark_path: Path,
    frame: pd.DataFrame,
    status_path: Path,
    payload: Mapping[str, object],
) -> None:
    benchmark_temporary = _stage_csv(benchmark_path, frame)
    try:
        status_temporary = _stage_json(status_path, payload)
    except Exception:
        benchmark_temporary.unlink(missing_ok=True)
        raise

    benchmark_backup = benchmark_path.with_name(
        f".{benchmark_path.name}.{uuid.uuid4().hex}.bak"
    )
    status_backup = status_path.with_name(f".{status_path.name}.{uuid.uuid4().hex}.bak")
    status_existed = status_path.exists()
    try:
        if not benchmark_path.is_file():
            raise OSError(f"Benchmark path is not a file: {benchmark_path}")
        if status_existed and not status_path.is_file():
            raise OSError(f"Status path is not a file: {status_path}")
        shutil.copy2(benchmark_path, benchmark_backup)
        if status_existed:
            shutil.copy2(status_path, status_backup)
        os.replace(benchmark_temporary, benchmark_path)
        os.replace(status_temporary, status_path)
    except Exception:
        if benchmark_backup.exists():
            os.replace(benchmark_backup, benchmark_path)
        if status_backup.exists():
            os.replace(status_backup, status_path)
        elif not status_existed and status_path.is_file():
            status_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in (
            benchmark_temporary,
            status_temporary,
            benchmark_backup,
            status_backup,
        ):
            temporary.unlink(missing_ok=True)


def refresh_benchmark(
    benchmark_path: Path,
    asof_date: date,
    *,
    symbol: str = "510300",
    status_path: Path | None = None,
    fetch_sina: Callable[[str], BenchmarkRow] = fetch_sina_quote,
    fetch_sohu: Callable[[str, date, date], dict[date, BenchmarkRow]] = fetch_sohu_history,
    fetch_yahoo: Callable[[str, date, date], dict[date, BenchmarkRow]] = fetch_yahoo_history,
) -> dict[str, object]:
    benchmark_path = Path(benchmark_path)
    frame = _load_benchmark(benchmark_path)
    latest = frame.iloc[-1]["date"].date()
    if latest > asof_date:
        raise ValueError(
            f"Benchmark last date {latest} is after requested asof-date {asof_date}"
        )

    query_start = min(latest, asof_date)
    source_warnings: list[str] = []
    unavailable_sources: list[str] = []

    try:
        sina: BenchmarkRow | None = fetch_sina(symbol)
    except SOURCE_FAILURES as exc:
        sina = None
        unavailable_sources.append("Sina")
        source_warnings.append(f"Sina unavailable: {type(exc).__name__}: {exc}")
    if (
        sina is not None
        and sina.date == asof_date
        and (sina.observed_time is None or sina.observed_time < time(15, 0))
    ):
        rendered_time = sina.observed_time.isoformat() if sina.observed_time else "missing"
        sina = None
        unavailable_sources.append("Sina")
        source_warnings.append(
            f"Sina quote before market close and excluded from quorum: {rendered_time}"
        )

    try:
        sohu = fetch_sohu(symbol, query_start, asof_date)
    except SOURCE_FAILURES as exc:
        sohu = {}
        unavailable_sources.append("Sohu")
        source_warnings.append(f"Sohu unavailable: {type(exc).__name__}: {exc}")
    try:
        yahoo = fetch_yahoo(symbol, query_start, asof_date)
    except SOURCE_FAILURES as exc:
        yahoo = {}
        unavailable_sources.append("Yahoo")
        source_warnings.append(f"Yahoo unavailable: {type(exc).__name__}: {exc}")

    relevant_sohu = {key: value for key, value in sohu.items() if latest < key <= asof_date}
    relevant_yahoo = {key: value for key, value in yahoo.items() if latest < key <= asof_date}

    rows_to_add: list[BenchmarkRow] = []
    updated: pd.DataFrame | None = None
    target_rows: dict[str, BenchmarkRow] = {}
    if asof_date in sohu:
        target_rows["Sohu"] = sohu[asof_date]
    elif "Sohu" not in unavailable_sources:
        unavailable_sources.append("Sohu")
        source_warnings.append("Sohu missing requested asof-date")
    if asof_date in yahoo:
        target_rows["Yahoo"] = yahoo[asof_date]
    elif "Yahoo" not in unavailable_sources:
        unavailable_sources.append("Yahoo")
        source_warnings.append("Yahoo missing requested asof-date")
    if sina is not None and sina.date == asof_date:
        target_rows["Sina"] = sina
    elif sina is not None and sina.date < asof_date:
        unavailable_sources.append("Sina")
        source_warnings.append(
            f"Sina quote stale for requested asof-date: {sina.date.isoformat()}"
        )

    result_sources, mismatch_warnings = _select_source_quorum(target_rows, asof_date)
    source_warnings.extend(mismatch_warnings)
    excluded_available = [
        label for label in SOURCE_ORDER if label in target_rows and label not in result_sources
    ]
    for label in excluded_available:
        if label not in unavailable_sources:
            unavailable_sources.append(label)
        source_warnings.append(
            f"{label} excluded because the other two sources formed the agreeing quorum"
        )
    target_row = _compose_quorum_row(target_rows, result_sources)

    if latest == asof_date:
        existing = BenchmarkRow(
            date=latest,
            open=float(frame.iloc[-1]["open"]),
            high=float(frame.iloc[-1]["high"]),
            low=float(frame.iloc[-1]["low"]),
            close=float(frame.iloc[-1]["close"]),
            volume=int(frame.iloc[-1]["volume"]),
            amount=int(frame.iloc[-1]["amount"]),
        )
        _validate_agreement(existing, target_row, "local/quorum")
        status = "already_fresh_degraded" if unavailable_sources else "already_fresh"
    else:
        prior_dates = sorted(
            (set(relevant_sohu) | set(relevant_yahoo)) - {asof_date}
        )
        for trading_date in prior_dates:
            if trading_date not in relevant_sohu or trading_date not in relevant_yahoo:
                raise ValueError(
                    "Historical catch-up before asof-date requires both Sohu and Yahoo; "
                    f"missing verification for {trading_date}"
                )
            _validate_agreement(
                relevant_sohu[trading_date],
                relevant_yahoo[trading_date],
                "Sohu/Yahoo",
            )
            rows_to_add.append(
                _compose_quorum_row(
                    {
                        "Sohu": relevant_sohu[trading_date],
                        "Yahoo": relevant_yahoo[trading_date],
                    },
                    ["Sohu", "Yahoo"],
                )
            )
        rows_to_add.append(target_row)
        additions = pd.DataFrame([row.as_record() for row in rows_to_add])
        updated = pd.concat([frame, additions], ignore_index=True)
        updated["date"] = pd.to_datetime(updated["date"])
        status = "updated_degraded" if unavailable_sources else "updated"

    stale_sources = [label for label in SOURCE_ORDER if label in unavailable_sources]

    result: dict[str, object] = {
        "status": status,
        "symbol": symbol,
        "asof_date": asof_date.isoformat(),
        "previous_last_date": latest.isoformat(),
        "latest_date": asof_date.isoformat(),
        "rows_added": len(rows_to_add),
        "source_agreement": True,
        "source_mode": "degraded_two_source" if stale_sources else "full_history_agreement",
        "sources": result_sources,
        "stale_sources": stale_sources,
        "quorum_required": 2,
        "warnings": source_warnings,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_path": str(benchmark_path.resolve()),
    }
    if updated is not None:
        if status_path is not None:
            _atomic_write_csv_and_status(
                benchmark_path,
                updated,
                Path(status_path),
                result,
            )
        else:
            _atomic_write_csv(benchmark_path, updated)
    elif status_path is not None:
        _atomic_write_json(Path(status_path), result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh and verify the 510300 benchmark.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--symbol", default="510300")
    parser.add_argument("--status-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asof_date = datetime.strptime(args.asof_date, "%Y-%m-%d").date()
        result = refresh_benchmark(
            Path(args.benchmark),
            asof_date,
            symbol=args.symbol,
            status_path=Path(args.status_output) if args.status_output else None,
        )
    except Exception as exc:
        print(f"Benchmark refresh failed: {exc}", file=sys.stderr)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
