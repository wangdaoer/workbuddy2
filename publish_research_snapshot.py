"""Publish a sanitized research snapshot to a private Vercel Blob store."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from export_research_snapshot import (
    DEFAULT_STALE_AFTER_DAYS,
    build_research_snapshot,
    discover_latest_sources,
    _parse_datetime,
)


DEFAULT_BLOB_PATHNAME = "research/latest.json"


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    return (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def publish_research_snapshot(
    run_card_path: Path,
    watchlist_path: Path,
    *,
    pathname: str = DEFAULT_BLOB_PATHNAME,
    published_at: datetime | None = None,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
    token: str | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Upload and read back one snapshot before reporting publication success."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN") if token is None else token
    if not token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN is not configured")
    if not pathname or pathname.startswith("/") or ".." in Path(pathname).parts:
        raise ValueError("pathname must be a relative Blob pathname")

    if client_factory is None:
        try:
            from vercel.blob import BlobClient
        except ImportError as exc:
            raise RuntimeError(
                "Install publishing dependencies with: "
                "python -m pip install -r requirements-publish.txt"
            ) from exc
        client_factory = BlobClient

    snapshot = build_research_snapshot(
        run_card_path,
        watchlist_path,
        published_at=published_at,
        stale_after_days=stale_after_days,
    )
    payload = snapshot_bytes(snapshot)
    client = client_factory(token=token)
    uploaded = client.put(
        pathname,
        payload,
        access="private",
        content_type="application/json",
        overwrite=True,
        cache_control_max_age=60,
    )
    _validate_private_blob_url(uploaded.url)

    downloaded = client.get(
        uploaded.url,
        access="private",
        use_cache=False,
        token=token,
    )
    if downloaded.status_code != 200:
        raise RuntimeError(
            f"Blob verification returned HTTP {downloaded.status_code}"
        )
    if downloaded.content != payload:
        raise RuntimeError("Blob verification content does not match the upload")

    return {
        "status": "published",
        "pathname": uploaded.pathname,
        "url": uploaded.url,
        "asof_date": snapshot["asof_date"],
        "run_status": snapshot["run_status"],
        "freshness": snapshot["freshness"]["status"],
        "priority_rows": snapshot["summary"]["priority_rows"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "verified": True,
    }


def load_local_environment(path: Path) -> None:
    """Load ignored local credentials without logging or returning their values."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Install publishing dependencies with: "
            "python -m pip install -r requirements-publish.txt"
        ) from exc
    load_dotenv(path, override=False)


def _validate_private_blob_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".private.blob.vercel-storage.com")
    ):
        raise RuntimeError("Upload did not return a private Vercel Blob URL")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export, upload, and verify the latest sanitized research snapshot."
    )
    parser.add_argument("--source-dir", default=os.environ.get("QUANT_RESEARCH_OUTPUT_DIR"))
    parser.add_argument("--run-card")
    parser.add_argument("--watchlist")
    parser.add_argument("--pathname", default=DEFAULT_BLOB_PATHNAME)
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--published-at")
    parser.add_argument("--stale-after-days", type=int, default=DEFAULT_STALE_AFTER_DAYS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_local_environment(Path(args.env_file))
    if bool(args.run_card) != bool(args.watchlist):
        raise SystemExit("--run-card and --watchlist must be provided together")
    if args.run_card:
        run_card_path = Path(args.run_card)
        watchlist_path = Path(args.watchlist)
    elif args.source_dir:
        run_card_path, watchlist_path = discover_latest_sources(Path(args.source_dir))
    else:
        raise SystemExit(
            "Provide --source-dir, or set QUANT_RESEARCH_OUTPUT_DIR, or provide both "
            "--run-card and --watchlist"
        )

    result = publish_research_snapshot(
        run_card_path,
        watchlist_path,
        pathname=args.pathname,
        published_at=_parse_datetime(args.published_at) if args.published_at else None,
        stale_after_days=args.stale_after_days,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
