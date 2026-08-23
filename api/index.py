"""Vercel entry point for the research workbench and its health API."""

from copy import deepcopy
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


RESPONSE = {
    "status": "ok",
    "project": "lianghua",
    "service": "deployment-health",
    "message": "The quantitative research repository is available.",
    "research_only": True,
    "trade_instruction": False,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_ROOT / "index.html"
HTML_PATHS = {"/", "/index.html"}
API_PATHS = {"/api", "/api/", "/health", "/health/"}
RESEARCH_PATHS = {"/api/research/latest", "/api/research/latest/"}
MAX_RESEARCH_SNAPSHOT_BYTES = 1024 * 1024
RESEARCH_SCHEMA_VERSION = 2
SYMBOL_PATTERN = re.compile(r"^[0-9]{6}$")
STOCK_NAME_PATTERN = re.compile(r"^[\u3400-\u9fffA-Za-z0-9*·（）()＋+&.\-]{1,30}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
STRATEGY_FAMILY_LABELS = {
    "trend_momentum": "趋势动量",
    "strong_pullback": "强势回调二波",
    "hidden_accumulation": "隐性吸筹",
}
ALLOWED_STRATEGY_FAMILIES = set(STRATEGY_FAMILY_LABELS)
ALLOWED_PRIORITY_BUCKETS = {
    "model_focus",
    "action_focus",
    "risk_watch",
    "pattern_watch",
    "review_later",
}
ALLOWED_WARNING_CODES = {
    "artifact_missing:stability_report",
    "data_stale",
    "run_failed",
}
FORBIDDEN_RESEARCH_KEYS = {
    "project_root",
    "commands",
    "argv",
    "artifacts",
    "personal_target_weight",
    "personal_action",
    "shadow_account_notes",
    "target_leverage",
}
ALLOWED_RESEARCH_KEYS = {
    "schema_version",
    "project",
    "research_only",
    "trade_instruction",
    "asof_date",
    "generated_at",
    "published_at",
    "run_status",
    "freshness",
    "summary",
    "coverage",
    "quality",
    "watchlist",
    "source_integrity",
}
ALLOWED_RESEARCH_SECTION_KEYS = {
    "freshness": {"status", "age_days", "stale_after_days"},
    "summary": {
        "priority_rows",
        "selected_rows",
        "change_rows",
        "model_decision_rows",
        "early_pattern_rows",
    },
    "coverage": {
        "database_latest_date",
        "database_asof_rows",
        "database_daily_rows",
        "database_observation_rows",
        "benchmark_latest_date",
        "benchmark_source_agreement",
    },
    "quality": {
        "missing_stock_names",
        "warnings",
    },
    "watchlist": {"bucket_counts", "strategy_family_counts", "top10"},
    "source_integrity": {"run_card_sha256", "watchlist_sha256"},
}
ALLOWED_TOP10_KEYS = {
    "symbol",
    "stock_name",
    "strategy_family",
    "strategy_family_cn",
    "priority_bucket",
    "priority_score",
}
FORBIDDEN_RESEARCH_TEXT_PATTERN = re.compile(
    r"(?:"
    r"(?:^|[\s\"'=])[a-z]:"
    r"|[a-z][a-z0-9+.-]*://"
    r"|(?:^|[^a-z0-9])[^\s/\\]+\.(?:py|js|ps1|sh|bat|cmd|sqlite3?|db|csv)(?:[^a-z0-9]|$)"
    r"|target[\s_.-]*(?:weight|leverage)"
    r"|personal[\s_-]*(?:action|position|trade)"
    r"|(?:^|[^a-z0-9])(?:buy|sell|execute|broker|purchase|allocation|position)(?:[^a-z0-9]|$)"
    r"|(?:place[\s_-]*order|order[\s_-]*(?:now|id|quantity|side))"
    r"|(?:^|[^a-z0-9])(?:py|python3?|node|deno|rscript|git|powershell|pwsh|cmd(?:\.exe)?|bash|zsh|curl|wget|sqlite3)(?=\s|$)"
    r"|--[a-z0-9_-]+"
    r"|(?:api[\s_-]*key|access[\s_-]*token|client[\s_-]*secret|password|bearer)"
    r"|(?:^|[^a-z0-9])(?:sk[-_](?:proj[-_]|live[-_])?|gh[pousr]_|vercel_blob_rw_|xox[baprs]-|AKIA[0-9A-Z]{8,})"
    r"|(?:买入|卖出|下单|仓位|持仓|加仓|减仓|清仓|建仓|止损|平仓|目标权重|交易指令|执行交易)"
    r")",
    re.IGNORECASE,
)


class ResearchSnapshotMissing(Exception):
    """The configured private snapshot does not exist."""


class ResearchSnapshotUnavailable(Exception):
    """Private research storage has not been configured."""


def _fetch_research_snapshot() -> dict[str, Any]:
    url = os.environ.get("RESEARCH_BLOB_URL", "").strip()
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()
    if not url or not token:
        raise ResearchSnapshotUnavailable
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".private.blob.vercel-storage.com")
    ):
        raise ValueError("RESEARCH_BLOB_URL must reference a private Vercel Blob")

    request = Request(
        _consistent_blob_url(url),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = response.read(MAX_RESEARCH_SNAPSHOT_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 404:
            raise ResearchSnapshotMissing from exc
        raise RuntimeError("Private research storage returned an HTTP error") from exc
    except URLError as exc:
        raise RuntimeError("Private research storage could not be reached") from exc
    if len(payload) > MAX_RESEARCH_SNAPSHOT_BYTES:
        raise ValueError("Research snapshot exceeds the maximum allowed size")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Research snapshot must be a JSON object")
    return value


def _consistent_blob_url(url: str) -> str:
    """Bypass private Blob cache so an overwritten daily snapshot is current."""
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["cache"] = "0"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def _validate_research_snapshot(
    snapshot: dict[str, Any], *, today: date | None = None
) -> dict[str, Any]:
    value = deepcopy(snapshot)
    if value.get("schema_version") != RESEARCH_SCHEMA_VERSION:
        raise ValueError("Unsupported research snapshot schema")
    if value.get("project") != "lianghua":
        raise ValueError("Research snapshot project does not match")
    if value.get("research_only") is not True:
        raise ValueError("Research snapshot must remain research-only")
    if value.get("trade_instruction") is not False:
        raise ValueError("Research snapshot must not contain trade instructions")
    _validate_research_schema(value)
    if _contains_forbidden_research_key(value):
        raise ValueError("Research snapshot contains a forbidden field")
    if _contains_forbidden_research_text(value):
        raise ValueError("Research snapshot contains unsafe text")

    asof_value = value.get("asof_date")
    try:
        asof_date = date.fromisoformat(str(asof_value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Research snapshot has an invalid asof_date") from exc
    current_date = today or datetime.now(timezone.utc).date()
    age_days = (current_date - asof_date).days
    if age_days < 0:
        raise ValueError("Research snapshot asof_date is in the future")

    freshness = value.get("freshness")
    if not isinstance(freshness, dict):
        raise ValueError("Research snapshot is missing freshness metadata")
    stale_after_days = freshness.get("stale_after_days")
    if not isinstance(stale_after_days, int) or stale_after_days < 0:
        raise ValueError("Research snapshot has invalid freshness metadata")
    freshness["age_days"] = age_days
    freshness["status"] = "fresh" if age_days <= stale_after_days else "stale"
    if value.get("run_status") != "success":
        freshness["status"] = "failed"
    return value


def _contains_forbidden_research_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RESEARCH_KEYS:
                return True
            if _contains_forbidden_research_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_research_key(item) for item in value)
    return False


def _validate_research_schema(value: dict[str, Any]) -> None:
    if set(value) - ALLOWED_RESEARCH_KEYS:
        raise ValueError("Research snapshot contains an unknown top-level field")
    for section, allowed_keys in ALLOWED_RESEARCH_SECTION_KEYS.items():
        item = value.get(section)
        if not isinstance(item, dict):
            raise ValueError(f"Research snapshot has an invalid {section} section")
        if set(item) - allowed_keys:
            raise ValueError(f"Research snapshot contains an unknown {section} field")
    top10 = value["watchlist"].get("top10")
    if not isinstance(top10, list):
        raise ValueError("Research snapshot top10 must be a list")
    for row in top10:
        if not isinstance(row, dict) or set(row) - ALLOWED_TOP10_KEYS:
            raise ValueError("Research snapshot contains an invalid top10 row")
    _validate_research_values(value)


def _validate_research_values(value: dict[str, Any]) -> None:
    if value.get("run_status") not in {"success", "failed"}:
        raise ValueError("Research snapshot has an invalid run status")
    for field in ("generated_at", "published_at"):
        timestamp = value.get(field)
        if timestamp is None and field == "generated_at":
            continue
        try:
            datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Research snapshot has an invalid timestamp") from exc

    _validate_nonnegative_numbers(value["summary"].values())
    coverage = value["coverage"]
    _validate_nonnegative_numbers(
        coverage.get(field)
        for field in (
            "database_asof_rows",
            "database_daily_rows",
            "database_observation_rows",
        )
    )
    for field in ("database_latest_date", "benchmark_latest_date"):
        date_value = coverage.get(field)
        if date_value is not None:
            try:
                date.fromisoformat(str(date_value))
            except ValueError as exc:
                raise ValueError("Research snapshot has an invalid coverage date") from exc
    agreement = coverage.get("benchmark_source_agreement")
    if agreement is not None and not isinstance(agreement, bool):
        raise ValueError("Research snapshot has invalid benchmark agreement")

    quality = value["quality"]
    _validate_nonnegative_numbers((quality.get("missing_stock_names"),))
    warnings = quality.get("warnings")
    if not isinstance(warnings, list) or any(
        warning not in ALLOWED_WARNING_CODES for warning in warnings
    ):
        raise ValueError("Research snapshot contains an unapproved warning")

    watchlist = value["watchlist"]
    _validate_count_map(watchlist.get("bucket_counts"), ALLOWED_PRIORITY_BUCKETS)
    _validate_count_map(
        watchlist.get("strategy_family_counts"), ALLOWED_STRATEGY_FAMILIES
    )
    for row in watchlist["top10"]:
        symbol = row.get("symbol")
        stock_name = row.get("stock_name")
        strategy_family = row.get("strategy_family")
        if not isinstance(symbol, str) or not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError("Research snapshot contains an invalid symbol")
        if not isinstance(stock_name, str) or not STOCK_NAME_PATTERN.fullmatch(stock_name):
            raise ValueError("Research snapshot contains an invalid stock name")
        if strategy_family not in ALLOWED_STRATEGY_FAMILIES:
            raise ValueError("Research snapshot contains an invalid strategy family")
        if row.get("strategy_family_cn") != STRATEGY_FAMILY_LABELS[strategy_family]:
            raise ValueError("Research snapshot contains an invalid strategy label")
        if row.get("priority_bucket") not in ALLOWED_PRIORITY_BUCKETS:
            raise ValueError("Research snapshot contains an invalid priority bucket")
        score = row.get("priority_score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise ValueError("Research snapshot contains an invalid priority score")

    integrity = value["source_integrity"]
    if any(
        not isinstance(integrity.get(field), str)
        or not SHA256_PATTERN.fullmatch(integrity[field])
        for field in ("run_card_sha256", "watchlist_sha256")
    ):
        raise ValueError("Research snapshot has invalid source integrity")


def _validate_nonnegative_numbers(values: Any) -> None:
    for item in values:
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item < 0
        ):
            raise ValueError("Research snapshot contains an invalid count")


def _validate_count_map(value: Any, allowed_keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) - allowed_keys:
        raise ValueError("Research snapshot contains an invalid count map")
    _validate_nonnegative_numbers(value.values())


def _contains_forbidden_research_text(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_forbidden_research_text(str(key))
            or _contains_forbidden_research_text(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_research_text(item) for item in value)
    if not isinstance(value, str):
        return False
    if any(character in value for character in ("/", "\\", "\n", "\r")):
        return True
    if any(ord(character) < 32 for character in value):
        return True
    return FORBIDDEN_RESEARCH_TEXT_PATTERN.search(value) is not None


def _research_response(
    fetcher: Callable[[], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    fetcher = fetcher or _fetch_research_snapshot
    base = {
        "project": "lianghua",
        "research_only": True,
        "trade_instruction": False,
    }
    try:
        snapshot = _validate_research_snapshot(fetcher())
    except ResearchSnapshotMissing:
        return 404, {
            **base,
            "status": "missing",
            "message": "Research data is not published.",
        }
    except ResearchSnapshotUnavailable:
        return 503, {
            **base,
            "status": "unavailable",
            "message": "Research storage is not configured.",
        }
    except Exception:
        return 502, {
            **base,
            "status": "error",
            "message": "Research data could not be verified.",
        }

    freshness_status = snapshot["freshness"]["status"]
    if freshness_status == "stale":
        status = "stale"
    elif freshness_status == "failed":
        status = "failed"
    else:
        status = "ok"
    return 200, {**base, "status": status, "data": snapshot}


class handler(BaseHTTPRequestHandler):
    """Serve the homepage and dependency-free health responses on Vercel."""

    def _response_for_path(self) -> tuple[int, str, str, bytes]:
        parsed_path = urlsplit(self.path)
        path = parsed_path.path
        query = dict(parse_qsl(parsed_path.query, keep_blank_values=True))

        if path in HTML_PATHS:
            try:
                return (
                    200,
                    "text/html; charset=utf-8",
                    "public, max-age=0, must-revalidate",
                    INDEX_FILE.read_bytes(),
                )
            except OSError:
                body = json.dumps(
                    {"status": "error", "message": "Homepage is unavailable."}
                ).encode("utf-8")
                return 500, "application/json; charset=utf-8", "no-store", body

        is_rewrite_target = path in {"/api", "/api/"} and query.get("route") == "research-latest"
        if path in RESEARCH_PATHS or is_rewrite_target:
            status, payload = _research_response()
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return status, "application/json; charset=utf-8", "private, no-store", body

        if path in API_PATHS:
            body = json.dumps(RESPONSE, ensure_ascii=False).encode("utf-8")
            return (
                200,
                "application/json; charset=utf-8",
                "public, max-age=0, must-revalidate",
                body,
            )

        body = json.dumps(
            {"status": "not_found", "message": "Route not found."}
        ).encode("utf-8")
        return 404, "application/json; charset=utf-8", "no-store", body

    def _send_response(self, *, include_body: bool) -> None:
        status, content_type, cache_control, body = self._response_for_path()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_response(include_body=True)

    def do_HEAD(self) -> None:
        self._send_response(include_body=False)

    def log_message(self, format: str, *args: object) -> None:
        return
