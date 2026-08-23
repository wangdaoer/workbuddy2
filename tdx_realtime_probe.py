"""Collect and audit TongDaXin realtime quote snapshots.

This module is research-only. It reads quote snapshots from the local
`oficcejo/tdx-api` probe and writes auditable files for deciding whether the
source is stable enough to feed intraday dashboards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, time as day_time
from pathlib import Path
from typing import Any

import yaml

from workspace_paths import resolve_workspace_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("outputs/realtime_tdx")
DEFAULT_TDX_REPO = Path(os.environ.get("QUANT_TDX_API_REPO", r"D:\codex\_repo_cache\tdx-api"))
DEFAULT_GO_EXE = Path(os.environ.get("QUANT_GO_EXE", r"D:\codex\tools\go-go1.26.5\go\bin\go.exe"))
DEFAULT_EXTRA_CODES = ["510300"]
PRIORITY_RE = re.compile(r"merged_priority_watchlist_(\d{8})(?:_cn)?\.csv$")
SYMBOL_COLUMNS = ("股票代码", "symbol", "code", "证券代码", "代码")


@dataclass(frozen=True)
class ProbePaths:
    quote_csv: Path
    status_csv: Path
    summary_json: Path
    report_md: Path
    latest_json: Path


def normalize_symbol(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = re.sub(r"^(sh|sz|bj)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(sh|sz|bj)$", "", text, flags=re.IGNORECASE)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return digits[-6:].zfill(6)


def unique_symbols(symbols: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for symbol in symbols:
        if symbol is None or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def date_token(asof_date: str) -> str:
    return re.sub(r"\D", "", asof_date)[:8]


def iso_date_from_token(token: str) -> str:
    return f"{token[:4]}-{token[4:6]}-{token[6:8]}"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def config_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    return value if value is not None else config.get(name, default)


def latest_priority_watchlist(output_root: Path, asof_date: str | None = None) -> Path | None:
    candidates = []
    for path in output_root.glob("merged_priority_watchlist_*.csv"):
        match = PRIORITY_RE.match(path.name)
        if not match:
            continue
        token = match.group(1)
        if asof_date and token != date_token(asof_date):
            continue
        candidates.append((token, path.name.endswith("_cn.csv"), path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def infer_asof_date(output_root: Path, explicit: str | None) -> str:
    if explicit:
        token = date_token(explicit)
        if len(token) != 8:
            raise ValueError(f"Invalid --asof-date: {explicit}")
        return iso_date_from_token(token)
    watchlist = latest_priority_watchlist(output_root)
    if watchlist is None:
        return datetime.now().strftime("%Y-%m-%d")
    match = PRIORITY_RE.match(watchlist.name)
    if match is None:
        return datetime.now().strftime("%Y-%m-%d")
    return iso_date_from_token(match.group(1))


def detect_symbol_column(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("Watchlist CSV has no header row")
    for column in SYMBOL_COLUMNS:
        if column in fieldnames:
            return column
    raise ValueError(f"Watchlist CSV does not contain a symbol column: {fieldnames}")


def load_watchlist_symbols(path: Path, top_n: int) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        column = detect_symbol_column(reader.fieldnames)
        symbols = [normalize_symbol(row.get(column)) for row in reader]
    return unique_symbols(symbols)[:top_n]


def resolve_probe_codes(
    *,
    explicit_codes: str | None,
    watchlist: Path | None,
    top_n: int,
    extra_codes: list[str],
) -> tuple[list[str], Path | None]:
    if explicit_codes:
        base = [normalize_symbol(item) for item in explicit_codes.split(",")]
        return unique_symbols(base + [normalize_symbol(item) for item in extra_codes]), None
    if watchlist is None:
        base = ["000001", "600519", "002472", "605389"]
        return unique_symbols(base + [normalize_symbol(item) for item in extra_codes]), None
    base = load_watchlist_symbols(watchlist, top_n)
    return unique_symbols(base + [normalize_symbol(item) for item in extra_codes]), watchlist


def ensure_probe_database(tdx_repo: Path, codes_db: Path | None = None) -> Path:
    if codes_db is not None:
        return codes_db
    source = tdx_repo / "web" / "data" / "database" / "codes.db"
    target = tdx_repo / "_probe" / "codes_probe.db"
    if not source.exists():
        raise FileNotFoundError(f"TDX code database not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
        shutil.copy2(source, target)
    return target


def extract_probe_json(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        text = raw[match.start() :].strip()
        try:
            parsed, _end = decoder.raw_decode(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "ok" in parsed and "quotes" in parsed:
            return parsed
    raise ValueError(f"No TDX probe JSON object found in output: {raw[-500:]}")


def go_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GOPATH", r"D:\codex\go")
    env.setdefault("GOMODCACHE", r"D:\codex\go\pkg\mod")
    env.setdefault("GOCACHE", r"D:\codex\go\cache")
    return env


def run_go_probe(
    *,
    go_exe: Path,
    tdx_repo: Path,
    codes_db: Path,
    codes: list[str],
    timeout_seconds: int,
    host: str | None = None,
) -> dict[str, Any]:
    if not go_exe.exists():
        raise FileNotFoundError(f"Go executable not found: {go_exe}")
    probe_dir = tdx_repo / "_probe"
    if not probe_dir.exists():
        raise FileNotFoundError(f"TDX probe directory not found: {probe_dir}")
    command = [
        str(go_exe),
        "run",
        ".\\_probe",
        "--codes",
        ",".join(codes),
        "--codes-db",
        str(codes_db),
    ]
    if host:
        command.extend(["--host", host])
    completed = subprocess.run(
        command,
        cwd=tdx_repo,
        env=go_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    raw = f"{completed.stdout}\n{completed.stderr}"
    parsed = extract_probe_json(raw)
    parsed["process_returncode"] = completed.returncode
    parsed["raw_tail"] = raw[-1200:]
    return parsed


def flatten_quote_rows(sample_id: int, collected_at: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for quote in result.get("quotes") or []:
        buy1 = quote.get("buy1") or {}
        sell1 = quote.get("sell1") or {}
        rows.append(
            {
                "sample_id": sample_id,
                "collected_at": collected_at,
                "probe_generated_at": result.get("generated_at"),
                "elapsed_ms": result.get("elapsed_ms"),
                "exchange": quote.get("exchange"),
                "code": normalize_symbol(quote.get("code")),
                "latest": quote.get("latest"),
                "change_pct": quote.get("change_pct"),
                "last_close": quote.get("last_close"),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "server_time": quote.get("server_time"),
                "total_hand": quote.get("total_hand"),
                "current_hand": quote.get("current_hand"),
                "amount": quote.get("amount"),
                "buy1_price": buy1.get("price"),
                "buy1_number": buy1.get("number"),
                "sell1_price": sell1.get("price"),
                "sell1_number": sell1.get("number"),
                "raw_rate": quote.get("raw_rate"),
            }
        )
    return rows


def flatten_status_row(sample_id: int, collected_at: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "collected_at": collected_at,
        "ok": bool(result.get("ok")),
        "elapsed_ms": result.get("elapsed_ms"),
        "quote_count": len(result.get("quotes") or []),
        "error": result.get("error", ""),
        "process_returncode": result.get("process_returncode"),
    }


def is_china_intraday_window(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    current = now.time()
    return day_time(9, 25) <= current <= day_time(11, 35) or day_time(12, 55) <= current <= day_time(15, 5)


def summarize_probe(
    *,
    asof_date: str,
    codes: list[str],
    sample_rows: list[dict[str, Any]],
    quote_rows: list[dict[str, Any]],
    watchlist: Path | None,
    generated_at: str,
) -> dict[str, Any]:
    ok_rows = [row for row in sample_rows if row.get("ok")]
    elapsed = [float(row["elapsed_ms"]) for row in ok_rows if row.get("elapsed_ms") is not None]
    requested = set(codes)
    returned = {str(row["code"]) for row in quote_rows if row.get("code")}
    changed_codes = []
    for code in sorted(returned):
        code_rows = [row for row in quote_rows if row.get("code") == code]
        states = {
            (
                str(row.get("latest")),
                str(row.get("server_time")),
                str(row.get("buy1_price")),
                str(row.get("sell1_price")),
            )
            for row in code_rows
        }
        if len(states) > 1:
            changed_codes.append(code)
    success_rate = len(ok_rows) / len(sample_rows) if sample_rows else 0.0
    intraday_window = is_china_intraday_window()
    if not ok_rows:
        assessment = "failed"
    elif intraday_window and len(sample_rows) >= 2 and changed_codes:
        assessment = "intraday_live_ok"
    elif intraday_window and len(sample_rows) >= 2:
        assessment = "intraday_connected_no_tick_change"
    else:
        assessment = "snapshot_ok_intraday_pending"
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "research_only": True,
        "trade_instruction": False,
        "source": "oficcejo/tdx-api via TongDaXin quote protocol",
        "asof_date": asof_date,
        "watchlist": str(watchlist) if watchlist else None,
        "requested_codes": codes,
        "requested_code_count": len(codes),
        "samples_requested": len(sample_rows),
        "samples_succeeded": len(ok_rows),
        "success_rate": success_rate,
        "quotes_returned": len(quote_rows),
        "unique_codes_returned": len(returned),
        "missing_codes": sorted(requested - returned),
        "avg_elapsed_ms": sum(elapsed) / len(elapsed) if elapsed else None,
        "max_elapsed_ms": max(elapsed) if elapsed else None,
        "min_elapsed_ms": min(elapsed) if elapsed else None,
        "changed_code_count": len(changed_codes),
        "changed_codes": changed_codes,
        "intraday_window": intraday_window,
        "assessment": assessment,
        "limitations": [
            "TDX quote snapshots are used for observation and dashboard freshness only.",
            "They do not replace end-of-day normalized data in the research/backtest pipeline.",
            "Weekend or after-hours runs can prove connectivity but cannot prove intraday tick updates.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_paths(output_dir: Path, asof_date: str, run_token: str) -> ProbePaths:
    token = date_token(asof_date)
    return ProbePaths(
        quote_csv=output_dir / f"tdx_realtime_quotes_{token}_{run_token}.csv",
        status_csv=output_dir / f"tdx_realtime_status_{token}_{run_token}.csv",
        summary_json=output_dir / f"tdx_realtime_summary_{token}_{run_token}.json",
        report_md=output_dir / f"tdx_realtime_report_{token}_{run_token}.md",
        latest_json=output_dir / "tdx_realtime_latest.json",
    )


def render_report(summary: dict[str, Any], paths: ProbePaths) -> str:
    status = {
        "intraday_live_ok": "盘中连续验证通过",
        "intraday_connected_no_tick_change": "盘中已连通，但本轮未观察到跳动",
        "snapshot_ok_intraday_pending": "快照连通通过，等待盘中验证",
        "failed": "未通过",
    }.get(str(summary["assessment"]), str(summary["assessment"]))
    missing = ", ".join(summary["missing_codes"]) if summary["missing_codes"] else "无"
    changed = ", ".join(summary["changed_codes"]) if summary["changed_codes"] else "无"
    avg_elapsed = summary["avg_elapsed_ms"]
    avg_elapsed_text = "无" if avg_elapsed is None else f"{avg_elapsed:.0f} ms"
    return "\n".join(
        [
            f"# TDX 实时行情源验证报告 {summary['asof_date']}",
            "",
            f"- 状态：{status}",
            f"- 样本：{summary['samples_succeeded']} / {summary['samples_requested']} 成功",
            f"- 覆盖：{summary['unique_codes_returned']} / {summary['requested_code_count']} 只",
            f"- 平均耗时：{avg_elapsed_text}",
            f"- 本轮价格/盘口发生变化的代码：{changed}",
            f"- 缺失代码：{missing}",
            f"- 是否盘中窗口：{summary['intraday_window']}",
            "",
            "## 使用边界",
            "",
            "- 本模块只做行情可用性和盘中观察验证，不产生买卖指令。",
            "- 正式模型训练、回测和日更仍以统一每日数据目录为主。",
            "- 若要证明盘中实时性，需要在交易时间用多样本采样观察价格或盘口变化。",
            "",
            "## 输出文件",
            "",
            f"- 行情明细：`{paths.quote_csv}`",
            f"- 采样状态：`{paths.status_csv}`",
            f"- 汇总 JSON：`{paths.summary_json}`",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe TDX realtime quote snapshots for research-only intraday validation.")
    parser.add_argument("--config", default="configs/tdx_realtime_probe.yaml")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--codes", default=None, help="Comma separated code list. Overrides watchlist loading.")
    parser.add_argument("--watchlist", default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--extra-code", action="append", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tdx-repo", default=None)
    parser.add_argument("--go-exe", default=None)
    parser.add_argument("--codes-db", default=None)
    parser.add_argument("--host", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_workspace_path(PROJECT_ROOT, args.config) if args.config else None
    config = load_config(config_path)
    high_return_root = resolve_workspace_path(
        PROJECT_ROOT,
        Path(config.get("priority_output_root", "outputs/high_return_v2")),
    )
    asof_date = infer_asof_date(high_return_root, args.asof_date)
    output_dir = resolve_workspace_path(PROJECT_ROOT, Path(config_value(args, config, "output_dir", DEFAULT_OUTPUT_ROOT)))
    tdx_repo = Path(config_value(args, config, "tdx_repo", DEFAULT_TDX_REPO))
    go_exe = Path(config_value(args, config, "go_exe", DEFAULT_GO_EXE))
    top_n = int(config_value(args, config, "top_n", 20))
    samples = max(1, int(config_value(args, config, "samples", 1)))
    interval_seconds = max(0.0, float(config_value(args, config, "interval_seconds", 30.0)))
    timeout_seconds = max(5, int(config_value(args, config, "timeout_seconds", 60)))
    extra_codes = list(config.get("extra_codes", DEFAULT_EXTRA_CODES))
    if args.extra_code:
        extra_codes.extend(args.extra_code)

    watchlist = (
        resolve_workspace_path(PROJECT_ROOT, args.watchlist)
        if args.watchlist
        else latest_priority_watchlist(high_return_root, asof_date)
    )
    codes, watchlist_used = resolve_probe_codes(
        explicit_codes=args.codes,
        watchlist=watchlist,
        top_n=top_n,
        extra_codes=extra_codes,
    )
    codes_db = ensure_probe_database(tdx_repo, Path(args.codes_db) if args.codes_db else None)
    run_token = datetime.now().strftime("%H%M%S")
    paths = build_paths(output_dir, asof_date, run_token)

    sample_rows: list[dict[str, Any]] = []
    quote_rows: list[dict[str, Any]] = []
    for sample_id in range(1, samples + 1):
        collected_at = datetime.now().isoformat(timespec="seconds")
        try:
            result = run_go_probe(
                go_exe=go_exe,
                tdx_repo=tdx_repo,
                codes_db=codes_db,
                codes=codes,
                timeout_seconds=timeout_seconds,
                host=args.host,
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result = {"ok": False, "quotes": [], "error": str(exc), "elapsed_ms": None, "process_returncode": None}
        sample_rows.append(flatten_status_row(sample_id, collected_at, result))
        quote_rows.extend(flatten_quote_rows(sample_id, collected_at, result))
        if sample_id < samples:
            time.sleep(interval_seconds)

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary = summarize_probe(
        asof_date=asof_date,
        codes=codes,
        sample_rows=sample_rows,
        quote_rows=quote_rows,
        watchlist=watchlist_used,
        generated_at=generated_at,
    )
    summary["artifacts"] = {
        "quotes": str(paths.quote_csv),
        "status": str(paths.status_csv),
        "summary": str(paths.summary_json),
        "report": str(paths.report_md),
    }
    write_csv(paths.quote_csv, quote_rows)
    write_csv(paths.status_csv, sample_rows)
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.report_md.write_text(render_report(summary, paths), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
