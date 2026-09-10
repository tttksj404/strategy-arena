#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx2[http2,brotli,zstd]", "pydantic>=2"]
# ///

"""Restart a resumable public Bitget candle collection after transient request failures."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import fetch_bitget_history as collector
from bitget_history_manifest import ArchiveManifest, CollectionConfig

DEFAULT_RESTART_ATTEMPTS: Final = 24
RESTART_BASE_SECONDS: Final = 15.0
RESTART_MAX_SECONDS: Final = 300.0


@dataclass(frozen=True, slots=True)
class SupervisorOptions:
    config: CollectionConfig
    restart_attempts: int


def restart_delay_seconds(attempt: int) -> float:
    """Return a bounded delay after a numbered failed collection attempt."""
    return min(RESTART_BASE_SECONDS * 2 ** min(attempt, 4), RESTART_MAX_SECONDS)


def collect_with_restarts(config: CollectionConfig, *, restart_attempts: int) -> ArchiveManifest:
    """Recreate the collector after bounded transient request failures.

    History-data validation errors intentionally propagate immediately.  Each
    restarted collection reuses the manifest and verifies an existing partition
    before it skips it, so a retry cannot silently accept a partial file.
    """
    for attempt in range(restart_attempts + 1):
        try:
            return collector.collect(config)
        except collector.BitgetHistoryRequestError as error:
            if attempt == restart_attempts:
                raise
            delay = restart_delay_seconds(attempt)
            print(
                f"COLLECTION_RESTART attempt={attempt + 1}/{restart_attempts} delay_seconds={delay:g} error={error}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("bounded supervisor must either return or re-raise")


def parse_options(argv: Sequence[str] | None = None) -> SupervisorOptions:
    """Parse supervisor-only options and forward all collector options unchanged."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restart-attempts", type=int, default=DEFAULT_RESTART_ATTEMPTS)
    arguments, collector_argv = parser.parse_known_args(argv)
    if arguments.restart_attempts < 0:
        parser.error("--restart-attempts must be non-negative")
    return SupervisorOptions(collector.parse_config(collector_argv), arguments.restart_attempts)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the existing collector under the transient-failure supervisor."""
    try:
        options = parse_options(argv)
        manifest = collect_with_restarts(options.config, restart_attempts=options.restart_attempts)
    except (collector.BitgetHistoryRequestError, collector.history.HistoryDataError) as error:
        print(f"COLLECTION_ERROR={error}")
        return 1
    manifest_path = Path(options.config.root) / "manifest.json"
    print(f"MANIFEST={manifest_path}")
    print(f"COMPLETE={str(manifest.complete).lower()} PARTITIONS={len(manifest.partitions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
