"""Atomic strategy override writer with baseline snapshots and audit trail."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base (deep copy, override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, path: tuple[str, ...], value: Any) -> None:
    """Set a value in a nested dict using a path tuple."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _get_nested(d: dict, path: tuple[str, ...], default: Any = None) -> Any:
    """Get a value from a nested dict using a path tuple."""
    for key in path:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is default:
            return default
    return d


class OverrideWriter:
    def __init__(self, *, override_path: Path, autotuner_dir: Path) -> None:
        self.override_path = override_path
        self.autotuner_dir = autotuner_dir
        self.baseline_path = autotuner_dir / "baseline_snapshot.json"
        self.audit_path = autotuner_dir / "audit.jsonl"
        self.autotuner_dir.mkdir(parents=True, exist_ok=True)

    def read_current(self) -> dict[str, Any]:
        if not self.override_path.exists():
            return {}
        try:
            return json.loads(self.override_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save_baseline(self, current: dict[str, Any]) -> None:
        self.baseline_path.write_text(
            json.dumps(current, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_baseline(self) -> dict[str, Any] | None:
        if not self.baseline_path.exists():
            return None
        try:
            return json.loads(self.baseline_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def apply_deltas(
        self,
        deltas: list[dict[str, Any]],
        *,
        change_id: str,
        dry_run: bool = False,
    ) -> bool:
        """Apply parameter deltas to override file. Returns True if applied."""
        if not deltas:
            return False

        current = self.read_current()
        self.save_baseline(current)

        # Build override patch
        patch: dict[str, Any] = {}
        for delta in deltas:
            path = tuple(delta["path"])
            _set_nested(patch, path, delta["proposed_value"])

        merged = _deep_merge(current, patch)

        # Audit log
        audit_entry = {
            "change_id": change_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "action": "dry_run" if dry_run else "apply",
            "deltas": deltas,
        }
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry, sort_keys=True) + "\n")

        if dry_run:
            return False

        # Atomic write
        tmp_path = self.override_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(merged, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(self.override_path))
        return True

    def revert_to_baseline(self, *, reason: str) -> bool:
        """Restore baseline snapshot. Returns True if reverted."""
        baseline = self.load_baseline()
        if baseline is None:
            return False

        audit_entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "action": "revert",
            "reason": reason,
        }
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry, sort_keys=True) + "\n")

        tmp_path = self.override_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(baseline, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(self.override_path))
        return True
