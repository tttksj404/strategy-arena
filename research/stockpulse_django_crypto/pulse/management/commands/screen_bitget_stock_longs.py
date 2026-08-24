from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from pulse.services.bitget_market import BitgetError
from pulse.services.live_long_signal_feed import screen_bitget_live_long_alerts


class Command(BaseCommand):
    help = "Emit public-quote manual entry/exit instructions from closed 15m/5m/1m long filters."

    def handle(self, *args, **options):
        try:
            payload = screen_bitget_live_long_alerts().payload()
        except BitgetError as error:
            raise CommandError(str(error)) from error
        self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
