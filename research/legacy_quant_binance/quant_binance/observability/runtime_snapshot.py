from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_binance.observability.runtime_state import read_runtime_state


def latest_runtime_artifact_path(
    base_dir: str | Path,
    *,
    filename: str,
    mode: str = "paper-live-shell",
) -> Path | None:
    mode_root = Path(base_dir) / "output" / mode
    latest_path = mode_root / "latest" / filename
    if latest_path.exists():
        return latest_path
    candidates = sorted(
        mode_root.rglob(filename),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_latest_runtime_payloads(
    base_dir: str | Path,
    *,
    mode: str = "paper-live-shell",
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = latest_runtime_artifact_path(base_dir, filename="summary.state.json", mode=mode)
    summary_path = latest_runtime_artifact_path(base_dir, filename="summary.json", mode=mode)
    state_payload = read_runtime_state(state_path) if state_path is not None else {}
    summary_payload = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path is not None
        else {}
    )
    return state_payload, summary_payload
