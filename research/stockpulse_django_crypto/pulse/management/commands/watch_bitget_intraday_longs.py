from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta

from django.core.management.base import BaseCommand, CommandError

from pulse.services.bitget_market import BitgetError, BitgetPublicClient
from pulse.services.intraday_signal_ledger import record_observations
from pulse.services.intraday_signal_settlement import settle_pending_signals
from pulse.services.live_long_research import collect_research_observations
from pulse.services.live_long_signal_feed import screen_bitget_live_long_alerts


CLOSED_CANDLE_PUBLICATION_BUFFER = timedelta(seconds=2)


def _next_closed_candle_cycle_at(reference_time: datetime) -> datetime:
    """Align a collection pass to a completed one-minute candle plus a short publication buffer."""

    minute_open = reference_time.replace(second=0, microsecond=0)
    current_minute_target = minute_open + CLOSED_CANDLE_PUBLICATION_BUFFER
    if reference_time < current_minute_target:
        return current_minute_target
    return minute_open + timedelta(minutes=1) + CLOSED_CANDLE_PUBLICATION_BUFFER


class Command(BaseCommand):
    help = "Persist and settle public Bitget intraday long observations; never place or modify orders."

    def add_arguments(self, parser):
        parser.add_argument("--interval-seconds", type=int, default=60)
        parser.add_argument("--cycles", type=int, default=1, help="0 runs continuously until interrupted.")

    def handle(self, *args, **options):
        interval_seconds = options["interval_seconds"]
        cycles = options["cycles"]
        if interval_seconds < 60:
            raise CommandError("interval-seconds must be at least 60 so each completed 1m bar is evaluated once.")
        if cycles < 0:
            raise CommandError("cycles must be zero (continuous) or a positive integer.")
        client = BitgetPublicClient()
        completed_cycles = 0
        next_cycle_at = _next_closed_candle_cycle_at(datetime.now(UTC))
        while cycles == 0 or completed_cycles < cycles:
            wait_seconds = max((next_cycle_at - datetime.now(UTC)).total_seconds(), 0)
            if wait_seconds:
                time.sleep(wait_seconds)
            try:
                settled = settle_pending_signals(client.get_text)
                feed = screen_bitget_live_long_alerts()
                collection = collect_research_observations(feed.base_report, client.get_text, datetime.now(UTC))
                recorded = record_observations(collection.observations)
            except BitgetError as error:
                raise CommandError(str(error)) from error
            completed_cycles += 1
            self.stdout.write(
                json.dumps(
                    {
                        "cycle": completed_cycles,
                        "as_of": feed.base_report.as_of.isoformat(),
                        "settled": settled,
                        "recorded": recorded,
                        "research_candidates": len(feed.base_report.candidates),
                        "research_recording_rejections": [
                            {"reason": reason, "count": count} for reason, count in collection.rejection_counts
                        ],
                        "enter_now_signals": [signal.alert_payload() for signal in feed.enter_now_signals],
                        "five_x_eligible_candidates": sum(
                            signal.candidate.five_x_eligible for signal in feed.signals
                        ),
                        "execution": "disabled",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            next_cycle_at += timedelta(seconds=interval_seconds)
            while next_cycle_at <= datetime.now(UTC):
                next_cycle_at += timedelta(seconds=interval_seconds)
