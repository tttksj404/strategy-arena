from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_entry(path: Path, schema_version: str) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "schema_version": schema_version,
        "generated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "bytes": stat.st_size,
    }


def write_manifest(output_path: Path, entries: list[dict[str, str | int]]) -> None:
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "entries": entries,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
