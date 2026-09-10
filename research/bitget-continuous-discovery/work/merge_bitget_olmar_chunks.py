#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

# ─── How to run ───
# uv run --script merge_bitget_olmar_chunks.py --interval 5m --chunks bitget_5m_olmar_chunk_000_004.json,bitget_5m_olmar_chunk_004_008.json --report bitget_5m_olmar_search_report.json

"""CLI boundary for contract-checked OLMAR chunk merging."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from bitget_archive_panel import ArchivePanelError
from bitget_olmar_chunk_contract import ChunkContractError
from bitget_olmar_chunk_merge import DEFAULT_ARCHIVE, DEFAULT_INTERVAL, MergeError, run
from selection_bias_stats import SelectionBiasError

DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_olmar_search_report.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    chunk_paths = tuple(Path(part) for part in arguments.chunks.split(",") if part)
    try:
        report = run(arguments.archive_root, chunk_paths=chunk_paths, interval=arguments.interval)
    except (ArchivePanelError, ChunkContractError, MergeError, SelectionBiasError) as error:
        print(f"MERGE_ERROR={error}")
        return 1
    arguments.report.write_text(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MERGE_DONE report={arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
