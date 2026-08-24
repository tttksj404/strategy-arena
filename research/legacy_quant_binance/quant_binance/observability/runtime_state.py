from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import shutil


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_runtime_state(path: str | Path, payload: dict[str, Any]) -> None:
    data = _json_ready(dict(payload))
    data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    latest_root = output_path.parent.parent / "latest"
    latest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, latest_root / "summary.state.json")


def read_runtime_state(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
