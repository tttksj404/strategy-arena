from __future__ import annotations

import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from pulse.services.bitget_market import BitgetError
from pulse.services.trend_continuation_fixture import (
    HistoricalResearchError,
    HistoricalResearchRequest,
)
from pulse.services.trend_continuation_funding import FundingCollectionError
from pulse.services.trend_continuation_history import HistoricalCollectionError
from pulse.services.trend_continuation_provenance import ProvenanceBlockedSource
from pulse.services.trend_continuation_research import evaluate_historical_research


class Command(BaseCommand):
    help = "Evaluate bounded Bitget trend-continuation history for research only."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--symbol", required=True)
        parser.add_argument(
            "--profile",
            required=True,
            choices=("standard_15m_5m", "triple_leveraged_15m_5m_1m"),
        )
        parser.add_argument("--start", required=True)
        parser.add_argument("--end", required=True)
        parser.add_argument("--max-pages", type=int, default=720)
        parser.add_argument("--input-fixture")

    def handle(self, *args: str, **options: str | int | None) -> None:
        del args
        request = HistoricalResearchRequest(
            symbol=_text_option(options.get("symbol"), "--symbol"),
            profile=_text_option(options.get("profile"), "--profile"),
            starts_at=_timestamp_option(options.get("start"), "--start"),
            ends_at=_timestamp_option(options.get("end"), "--end"),
            max_pages=_integer_option(options.get("max_pages"), "--max-pages"),
            input_fixture=_fixture_option(options.get("input_fixture")),
        )
        try:
            outcome = evaluate_historical_research(
                request,
                collected_at=datetime.now(UTC),
            )
        except (
            BitgetError,
            FundingCollectionError,
            HistoricalCollectionError,
            HistoricalResearchError,
            ProvenanceBlockedSource,
        ) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(
            json.dumps(outcome.payload(), ensure_ascii=False, sort_keys=True, indent=2)
        )
        if not outcome.successful:
            raise CommandError(outcome.failure_code or "historical_research_failed")


def _text_option(value: str | int | None, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CommandError(f"{name} is required")
    return value


def _integer_option(value: str | int | None, name: str) -> int:
    if not isinstance(value, int):
        raise CommandError(f"{name} must be an integer")
    return value


def _timestamp_option(value: str | int | None, name: str) -> datetime:
    raw_value = _text_option(value, name)
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError as error:
        raise CommandError(f"{name} must be an ISO-8601 timestamp") from error


def _fixture_option(value: str | int | None) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None
