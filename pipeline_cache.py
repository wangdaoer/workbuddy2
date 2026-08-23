"""Strict content-addressed cache for deterministic daily pipeline steps."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4


CACHEABLE_STEPS = {
    "update_panel",
    "trend_state",
    "regime_shadow_compare",
    "train_rank_model",
    "train_breadth_guard_challenger",
    "rank_candidates",
    "early_pattern_watchlist",
}
EXCLUDED_TREE_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "outputs",
    "tests",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_fingerprint(path: Path) -> dict[str, object]:
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        total_size = 0
        for item in files:
            relative = item.relative_to(path).as_posix()
            size = item.stat().st_size
            total_size += size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha256(item).encode("ascii"))
            digest.update(b"\0")
        return {
            "path": str(path.resolve()),
            "kind": "directory",
            "file_count": len(files),
            "size_bytes": total_size,
            "sha256": digest.hexdigest(),
        }
    return {"path": str(path.resolve()), "kind": "missing"}


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _included_project_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_root.rglob("*.py"):
        if any(part in EXCLUDED_TREE_PARTS for part in path.relative_to(project_root).parts):
            continue
        files.append(path)
    config_root = project_root / "configs"
    if config_root.exists():
        for pattern in ("*.yaml", "*.yml", "*.json"):
            files.extend(config_root.rglob(pattern))
    return sorted(set(files))


def _daily_input_files(daily_dir: Path, daily_start: str, asof_date: str) -> list[Path]:
    start = datetime.strptime(daily_start, "%Y-%m-%d").date()
    end = datetime.strptime(asof_date, "%Y-%m-%d").date()
    files: list[Path] = []
    for path in daily_dir.glob("ths_hs_a_share_*"):
        match = re.fullmatch(
            r"ths_hs_a_share_(\d{4}-\d{2}-\d{2})\.(?:csv|xls|xlsx)",
            path.name,
        )
        if not match:
            continue
        trade_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        if start <= trade_date <= end:
            files.append(path)
    return sorted(files)


def build_global_fingerprint(
    *,
    project_root: Path,
    base_panel: Path,
    daily_dir: Path,
    daily_start: str,
    asof_date: str,
    benchmark: Path,
) -> tuple[str, dict[str, object]]:
    code_files = _included_project_files(project_root)
    input_files = [
        base_panel,
        benchmark,
        *_daily_input_files(daily_dir, daily_start, asof_date),
    ]
    payload = {
        "schema_version": 1,
        "asof_date": asof_date,
        "daily_start": daily_start,
        "code_and_config": [path_fingerprint(path) for path in code_files],
        "market_inputs": [path_fingerprint(path) for path in input_files],
    }
    return _stable_hash(payload), payload


def _argument_value(command: Sequence[str], flag: str) -> Path | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return Path(command[index + 1])


def cache_outputs(step_name: str, command: Sequence[str]) -> list[Path]:
    if step_name not in CACHEABLE_STEPS:
        return []
    flag = "--output-dir" if step_name in {
        "trend_state",
        "regime_shadow_compare",
        "train_rank_model",
        "train_breadth_guard_challenger",
    } else "--output"
    path = _argument_value(command, flag)
    return [path] if path is not None else []


class PipelineStepCache:
    def __init__(
        self,
        manifest_path: Path,
        global_fingerprint: str,
        fingerprint_inputs: Mapping[str, object],
    ) -> None:
        self.manifest_path = manifest_path
        self.global_fingerprint = global_fingerprint
        self.fingerprint_inputs = dict(fingerprint_inputs)
        self.entries: dict[str, dict[str, object]] = {}
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if (
                payload.get("schema_version") == 1
                and payload.get("global_fingerprint") == global_fingerprint
                and isinstance(payload.get("entries"), dict)
            ):
                self.entries = dict(payload["entries"])

    def command_fingerprint(self, step_name: str, command: Sequence[str]) -> str:
        return _stable_hash({"step": step_name, "command": list(command)})

    def restore(
        self,
        step_name: str,
        command: Sequence[str],
        dependency_cache_hits: Iterable[bool],
    ) -> bool:
        if step_name not in CACHEABLE_STEPS or not all(dependency_cache_hits):
            return False
        entry = self.entries.get(step_name)
        if not isinstance(entry, dict):
            return False
        if entry.get("command_fingerprint") != self.command_fingerprint(step_name, command):
            return False
        recorded = entry.get("outputs")
        outputs = cache_outputs(step_name, command)
        if not outputs or not isinstance(recorded, list) or len(recorded) != len(outputs):
            return False
        return all(
            path_fingerprint(path) == expected
            for path, expected in zip(outputs, recorded, strict=True)
        )

    def record_success(self, step_name: str, command: Sequence[str]) -> None:
        outputs = cache_outputs(step_name, command)
        if not outputs or any(not path.exists() for path in outputs):
            self.entries.pop(step_name, None)
            return
        self.entries[step_name] = {
            "command_fingerprint": self.command_fingerprint(step_name, command),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": [path_fingerprint(path) for path in outputs],
        }

    def save(self) -> None:
        payload = {
            "schema_version": 1,
            "global_fingerprint": self.global_fingerprint,
            "fingerprint_inputs": self.fingerprint_inputs,
            "entries": self.entries,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
