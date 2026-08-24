from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import assert_never

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from pulse.services.bitget_market import BitgetContract
from pulse.services.trend_continuation_contract import CandidateIdentity, Direction
from pulse.services.trend_continuation_manifest import (
    JsonValue,
    ManifestLifecycleError,
    append_holdout_result,
)
from pulse.services.trend_continuation_research import (
    ForwardObservationDraft,
    capture_forward_observation,
    freeze_selection_idempotently,
)
from pulse.services.trend_continuation_settlement import (
    WatcherLifecyclePhase,
    advance_watcher_lifecycle,
    watcher_lifecycle_phase,
)
from pulse.test_trend_continuation_manifest import BASE, _identity, _observation, _settle, _snapshot
from pulse.trend_continuation_models import TrendContinuationObservation


FIXTURE = Path("tests/fixtures/trend_continuation_complete.json")


def _assert_no_legacy_fields(payload: JsonValue) -> None:
    match payload:
        case dict():
            forbidden = {"action", "entry", "target", "leverage", "take_profit", "stop_loss"}
            if forbidden.intersection(payload):
                raise AssertionError(f"legacy field leaked: {forbidden.intersection(payload)}")
            for value in payload.values():
                _assert_no_legacy_fields(value)
        case list():
            for value in payload:
                _assert_no_legacy_fields(value)
        case str() | int() | float() | bool() | None:
            pass
        case unreachable:
            assert_never(unreachable)


class HistoricalResearchCommandTests(TestCase):
    def test_fixture_command_is_exploratory_disabled_and_never_promotes(self):
        output = StringIO()
        call_command(
            "evaluate_bitget_trend_continuation",
            symbol="TSLAUSDT",
            profile="standard_15m_5m",
            start="2026-01-01T00:00:00+00:00",
            end="2026-01-02T00:00:00+00:00",
            max_pages=2,
            input_fixture=str(FIXTURE),
            stdout=output,
        )
        payload = json.loads(output.getvalue())

        self.assertTrue(payload["exploratory_only"])
        self.assertFalse(payload["execution"]["enabled"])
        self.assertEqual(payload["decision"], "NO_RELIABLE_WINNER")
        self.assertEqual(payload["history"]["required_intervals"], ["15m", "5m"])
        self.assertTrue(payload["data_contracts"]["candles_complete"])
        self.assertTrue(payload["data_contracts"]["funding_complete"])
        self.assertTrue(payload["data_contracts"]["fee_policy_complete"])
        _assert_no_legacy_fields(payload)

    def test_cli_rejects_invalid_ranges_page_bounds_profile_and_missing_fixture(self):
        cases = (
            {"max_pages": 0},
            {"max_pages": 721},
            {"start": "2026-01-01T00:00:00", "end": "2026-01-02T00:00:00+00:00"},
            {"start": "2026-01-02T00:00:00+00:00", "end": "2026-01-01T00:00:00+00:00"},
            {"profile": "unsupported_profile"},
            {"input_fixture": "tests/fixtures/does-not-exist.json"},
        )
        defaults = {
            "symbol": "TSLAUSDT",
            "profile": "standard_15m_5m",
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-02T00:00:00+00:00",
            "max_pages": 2,
            "input_fixture": str(FIXTURE),
        }
        for override in cases:
            with self.subTest(override=override):
                with self.assertRaises(CommandError):
                    call_command("evaluate_bitget_trend_continuation", **(defaults | override))

    def test_incomplete_or_extra_fixture_surface_fails_closed(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        fixture["candles"].pop("15m")
        fixture["unexpected"] = True
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "incomplete.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            with self.assertRaises(CommandError):
                call_command(
                    "evaluate_bitget_trend_continuation",
                    symbol="TSLAUSDT",
                    profile="standard_15m_5m",
                    start="2026-01-01T00:00:00+00:00",
                    end="2026-01-02T00:00:00+00:00",
                    max_pages=2,
                    input_fixture=str(path),
                )

    def test_manifest_freeze_is_strategy_snapshot_idempotent(self):
        snapshot = _snapshot(_identity("TSLAUSDT"), nonce="task5-idempotent")
        first = freeze_selection_idempotently(snapshot, frozen_at=BASE)
        second = freeze_selection_idempotently(snapshot, frozen_at=BASE + timedelta(seconds=1))
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(first.digest, second.digest)


class ForwardWatcherTests(TestCase):
    def _draft(self, evidence_time: datetime) -> ForwardObservationDraft:
        identity = CandidateIdentity(
            contract="TSLAUSDT",
            profile="standard_15m_5m",
            trend_direction=Direction.UP,
            rule_name="technical_score_v1",
            candidate_horizon_bars=1,
            strategy_version="trend-v1",
            feature_schema_version="features-v1",
        )
        return ForwardObservationDraft(
            contract=BitgetContract("TSLAUSDT", "TSLA", 20, 8, 0.0006),
            identity=identity,
            evidence_time=evidence_time,
            reference_price=100.0,
            invalidation_price=99.0,
            trigger_interval_seconds=300,
            evidence_raw_sha256="c" * 64,
            feature_snapshot={"schema": "features-v1", "complete": True},
            funding_snapshot={"complete": True, "coverage_end": evidence_time.isoformat(), "events": []},
            fee_policy_snapshot={"version": "fee-v1", "raw_sha256": "a" * 64},
        )

    def test_quote_is_watcher_only_before_lookup_raw_hashed_and_idempotent(self):
        evidence = BASE + timedelta(days=1)
        observed = evidence + timedelta(seconds=10)
        quote_body = json.dumps(
            {
                "code": "00000",
                "data": [{"symbol": "TSLAUSDT", "lastPr": "100", "bidPr": "99.9", "askPr": "100.1", "ts": str(int(observed.timestamp() * 1000))}],
            }
        )
        order: list[str] = []

        def fetch_text(url: str) -> str:
            order.append("quote")
            self.assertIn("/api/v2/mix/market/ticker", url)
            return quote_body

        def settlement_lookup() -> int:
            order.append("lookup")
            return 0

        first = capture_forward_observation(
            self._draft(evidence),
            fetch_text,
            watcher_collected_at=observed,
            settlement_lookup_started_at=observed + timedelta(seconds=1),
            settlement_lookup=settlement_lookup,
        )
        second = capture_forward_observation(
            self._draft(evidence),
            lambda url: (_ for _ in ()).throw(AssertionError("duplicate must not refetch quote")),
            watcher_collected_at=observed,
            settlement_lookup_started_at=observed + timedelta(seconds=1),
            settlement_lookup=lambda: (_ for _ in ()).throw(AssertionError("duplicate must not lookup")),
        )

        self.assertEqual(order, ["quote", "lookup"])
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.observation.pk, second.observation.pk)
        self.assertEqual(first.observation.quote_snapshot["raw_sha256"], second.observation.quote_snapshot["raw_sha256"])
        self.assertEqual(TrendContinuationObservation.objects.count(), 1)

    def test_exact_80_pause_then_later_20_and_repeat_holdout_is_terminal(self):
        identity = _identity("MSFTUSDT")
        manifest = freeze_selection_idempotently(_snapshot(identity, nonce="task5-lifecycle"), frozen_at=BASE)
        validation_records = []
        for index in range(80):
            record = _observation(identity, BASE + timedelta(days=1, minutes=index * 10))
            _settle(record)
            validation_records.append(record)
        self.assertEqual(watcher_lifecycle_phase(manifest), WatcherLifecyclePhase.FINALIZE_VALIDATION)
        self.assertEqual(advance_watcher_lifecycle(manifest), WatcherLifecyclePhase.COLLECT_HOLDOUT)

        holdout_records = []
        for index in range(20):
            record = _observation(identity, BASE + timedelta(days=2, minutes=index * 10))
            _settle(record)
            holdout_records.append(record)
        self.assertEqual(watcher_lifecycle_phase(manifest), WatcherLifecyclePhase.FINALIZE_HOLDOUT)
        self.assertEqual(advance_watcher_lifecycle(manifest), WatcherLifecyclePhase.COMPLETE)
        with self.assertRaises(ManifestLifecycleError):
            append_holdout_result(manifest, tuple(holdout_records))
        self.assertEqual(watcher_lifecycle_phase(manifest), WatcherLifecyclePhase.TERMINAL_REJECTED)
