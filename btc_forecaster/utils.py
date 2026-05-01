"""Shared helpers for JSONL persistence and time handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path and return it as a Path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    """Normalize a timestamp-like value to a timezone-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON-serializable record to a JSONL file."""
    file_path = ensure_parent(path)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Overwrite a JSONL file with records."""
    file_path = ensure_parent(path)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records. Missing files are treated as empty history."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write JSONL via a temporary file, then replace the target."""
    file_path = ensure_parent(path)
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    write_jsonl(temp_path, records)
    temp_path.replace(file_path)
