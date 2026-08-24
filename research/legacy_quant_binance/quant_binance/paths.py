from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    summary_path: Path
    state_path: Path


def prepare_run_paths(
    *,
    base_dir: str | Path,
    mode: str,
    run_id: str | None = None,
) -> RunPaths:
    timestamp = run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(base_dir) / mode / timestamp
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    state_path = root / "summary.state.json"
    return RunPaths(root=root, summary_path=summary_path, state_path=state_path)
