from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from datetime import UTC, datetime

from django.core.management.base import BaseCommand, CommandError

from pulse.services.bitget_market import BitgetError, BitgetPublicClient
from pulse.services.trend_continuation_forward_collection import ForwardEvidenceError
from pulse.services.trend_continuation_forward_funding import CurrentFundingError
from pulse.services.trend_continuation_funding import FundingCollectionError
from pulse.services.trend_continuation_history import HistoricalCollectionError
from pulse.services.trend_continuation_manifest import ManifestLifecycleError
from pulse.services.trend_continuation_settlement import (
    ForwardCollectionError,
    SettlementIncompleteError,
)
from pulse.services.trend_continuation_watcher import run_watcher_cycle


class Command(BaseCommand):
    help = "Collect and settle research-only trend-continuation forward evidence."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--cycles", type=int, default=1)
        parser.add_argument("--sleep-seconds", type=int, default=0)

    def handle(self, *args: str, **options: str | int | None) -> None:
        del args
        cycles = _integer_option(options.get("cycles"), "--cycles")
        sleep_seconds = _integer_option(options.get("sleep_seconds"), "--sleep-seconds")
        if cycles < 1:
            raise CommandError("--cycles must be >= 1")
        if sleep_seconds < 0:
            raise CommandError("--sleep-seconds must be >= 0")
        fetch_text = BitgetPublicClient().get_text
        reports = []
        failure_codes: list[str] = []
        try:
            for index in range(cycles):
                outcome = run_watcher_cycle(fetch_text, collected_at=datetime.now(UTC))
                reports.append(outcome.payload())
                failure_codes.extend(outcome.failure_codes)
                if failure_codes:
                    break
                if sleep_seconds and index < cycles - 1:
                    time.sleep(sleep_seconds)
        except (
            BitgetError,
            CurrentFundingError,
            ForwardCollectionError,
            ForwardEvidenceError,
            FundingCollectionError,
            HistoricalCollectionError,
            ManifestLifecycleError,
            SettlementIncompleteError,
        ) as error:
            raise CommandError(str(error)) from error
        payload = {
            "mode": "research_only",
            "execution": {"enabled": False},
            "cycles_requested": cycles,
            "cycles_completed": len(reports),
            "reports": reports,
            "failure_codes": sorted(frozenset(failure_codes)),
        }
        self.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        if failure_codes:
            raise CommandError(failure_codes[0])


def _integer_option(value: str | int | None, name: str) -> int:
    if not isinstance(value, int):
        raise CommandError(f"{name} must be an integer")
    return value
