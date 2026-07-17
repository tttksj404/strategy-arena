#!/usr/bin/env python3
import contextlib
import json
import datetime as dt
import os
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
import datastore
from admob_ssv import VerifiedReward


def closing_sqlite(path):
    return contextlib.closing(sqlite3.connect(path))


class AppDataLayerTestCase(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self._db_tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._db_tmp.name, "strategy.sqlite")
        self._env_patch = patch.dict(
            os.environ,
            {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "RACELENS_REWARDED_ADS_ENABLED": "1",
            },
            clear=False,
        )
        self._env_patch.start()
        self.client = app_module.app.test_client()
        app_module._BASE_PREDICTION_CACHE.clear()
        app_module._NEGATIVE_BASE_PREDICTION_CACHE.clear()
        app_module._RECENT_CACHE.clear()
        app_module._RECENT_FETCHING.clear()

    def tearDown(self):
        self._env_patch.stop()
        self._db_tmp.cleanup()

    def test_health_alias_and_http_errors_keep_operational_status_codes(self):
        with patch.object(app_module.os.path, "exists", return_value=True):
            health = self.client.get("/health")
        missing = self.client.get("/missing-release-route")

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.get_json()["ok"])
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.get_json()["status"], 404)

    def test_root_serves_the_current_mobile_web_shell(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/_expo/static/js/web/", response.get_data(as_text=True))

    def test_legal_pages_are_live_and_store_safe(self):
        pages = [
            "/legal/privacy",
            "/legal/terms",
            "/legal/account-deletion",
            "/legal/support",
        ]
        with patch.dict(os.environ, {"RACELENS_SUPPORT_EMAIL": "support@racelens.test"}, clear=False):
            for path in pages:
                response = self.client.get(path)
                body = response.get_data(as_text=True)

                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["Content-Type"])
                self.assertIn("RaceLens", body)
                self.assertIn("support@racelens.test", body)
                self.assertNotIn("�", body)
                self.assertNotIn("구매하기", body)
                self.assertNotIn("지금 베팅", body)

    def test_legal_pages_use_real_default_contact_and_required_subscription_terms(self):
        terms = self.client.get("/legal/terms").get_data(as_text=True)
        privacy = self.client.get("/legal/privacy").get_data(as_text=True)

        self.assertIn("tttksj@gmail.com", terms)
        self.assertIn("자동 갱신", terms)
        self.assertIn("구독 관리", terms)
        self.assertIn("환불", terms)
        self.assertIn("만 19세 이상", terms)
        self.assertIn("시행합니다", terms)
        self.assertIn("tttksj@gmail.com", privacy)
        self.assertIn("1년", privacy)
        self.assertIn("Google Firebase Analytics", privacy)
        self.assertIn("Crashlytics", privacy)
        self.assertIn("앱 인스턴스 식별자", privacy)
        self.assertIn("Google에 전송", privacy)
        self.assertIn("Google 보상형 광고", terms)
        self.assertIn("Google Mobile Ads SDK", privacy)
        self.assertNotIn("support@example.invalid", terms + privacy)

    def test_recent_keirin_falls_back_to_today_schedule_without_api_key(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": ""}, clear=False), \
             patch.object(app_module, "_today_kst", return_value=dt.date(2026, 7, 3)):
            response = self.client.get("/recent?sport=keirin&meet=광명")
            payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["sport"], "keirin")
        self.assertEqual(payload["meet"], "광명")
        self.assertEqual(payload["days"][0], "2026-07-03")
        self.assertIn("2026-07-04", payload["days"])
        self.assertIn("2026-07-05", payload["days"])
        self.assertEqual(payload["race_count"], 16)
        self.assertIn("default_race_no", payload)

    def test_recent_horse_falls_back_by_venue_race_days(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": ""}, clear=False), \
             patch.object(app_module, "_today_kst", return_value=dt.date(2026, 7, 3)):
            seoul = self.client.get("/recent?sport=horse&meet=서울").get_json()
            busan = self.client.get("/recent?sport=horse&meet=부경").get_json()

        self.assertEqual(seoul["days"][0], "2026-07-04")
        self.assertNotIn("2026-07-03", seoul["days"])
        self.assertEqual(busan["days"][0], "2026-07-03")
        self.assertEqual(seoul["race_count"], 11)
        self.assertEqual(busan["race_count"], 11)

    def test_recent_accepts_mobile_sport_aliases_without_falling_back_to_keirin(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": ""}, clear=False), \
             patch.object(app_module, "_today_kst", return_value=dt.date(2026, 7, 3)):
            horse = self.client.get("/recent?sport=kra&meet=서울").get_json()
            keirin = self.client.get("/recent?sport=kcycle&meet=광명").get_json()

        self.assertEqual(horse["sport"], "horse")
        self.assertEqual(horse["meet"], "서울")
        self.assertEqual(horse["race_count"], 11)
        self.assertEqual(horse["days"][0], "2026-07-04")
        self.assertEqual(keirin["sport"], "keirin")
        self.assertEqual(keirin["meet"], "광명")
        self.assertEqual(keirin["race_count"], 16)

    def test_live_decision_accepts_mobile_sport_alias_before_prediction(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "pre_race",
            "message": "stored",
            "updated_at": "2026-07-02T12:00:00",
            "odds_age_sec": None,
            "market_odds": [],
            "top": base["rows"][0],
            "rows": base["rows"],
            "decision": "hold",
            "market_used": False,
            "snapshot_phase": "pre_race",
        }

        with patch.object(app_module, "_compute_base_prediction", return_value=base) as base_mock, \
             patch.object(app_module.engine, "compute_live_decision", return_value=decision) as live_mock:
            response = self.client.get(
                "/api/live-decision?sport=kcycle&date=2026-07-03&meet=광명&race_no=5",
                headers={
                    "X-RaceLens-Device-Id": "device-alias-kcycle",
                    "X-RaceLens-Platform": "ios",
                },
            )

        self.assertEqual(response.status_code, 200)
        base_mock.assert_called_once_with("keirin", "2026-07-03", "광명", "5", None)
        live_mock.assert_called_once_with("keirin", "2026-07-03", "광명", "5", base_model_out=base)

    def test_recent_default_race_no_tracks_current_keirin_start_time(self):
        race_no = app_module._default_race_no(
            "keirin",
            "광명",
            ["2026-07-05"],
            now=dt.datetime(2026, 7, 5, 17, 38, tzinfo=dt.timezone(dt.timedelta(hours=9))),
        )

        self.assertEqual(race_no, 13)

    def test_sqlite_data_layer_initializes_logical_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False):
                status = datastore.app_data_layer_status()

        self.assertTrue(status["ready"])
        self.assertEqual(status["storage"], "sqlite")
        self.assertEqual(
            [item["name"] for item in status["schemas"]],
            ["race_data", "prediction", "user_account", "billing", "analytics"],
        )
        self.assertTrue(any("market_odds_snapshots" in item["tables"] for item in status["schemas"]))
        self.assertTrue(any("analysis_usage" in item["tables"] for item in status["schemas"]))
        self.assertTrue(any("subscriptions" in item["tables"] for item in status["schemas"]))

    def test_kra_market_snapshots_preserve_trajectory_and_deduplicate_one_minute(self):
        captured = "2026-07-11T15:50:00+09:00"
        board = [
            {"chulNo": "1", "winOdds": "2.0", "plcOdds": "1.3"},
            {"chulNo": "2", "winOdds": "4.0", "plcOdds": "1.8"},
        ]
        first = datastore.record_market_odds_snapshot(
            "horse", "20260711", "서울", "7", board, captured
        )
        duplicate = datastore.record_market_odds_snapshot(
            "horse", "20260711", "서울", "7", board, "2026-07-11T15:50:30+09:00"
        )
        moved = datastore.record_market_odds_snapshot(
            "horse",
            "20260711",
            "서울",
            "7",
            [{**board[0], "winOdds": "1.9"}, board[1]],
            "2026-07-11T15:50:40+09:00",
        )

        config = datastore.database_config()
        with closing_sqlite(config.sqlite_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM race_data__market_odds_snapshots WHERE sport='horse'"
            ).fetchone()[0]
        self.assertTrue(first)
        self.assertFalse(duplicate)
        self.assertTrue(moved)
        self.assertEqual(2, count)

    def test_app_data_layer_route_requires_admin_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "RACELENS_ADMIN_TOKEN": "secret-admin",
            }, clear=False):
                missing = self.client.get("/api/app-data-layer")
                allowed = self.client.get("/api/app-data-layer", headers={"RACELENS_ADMIN_TOKEN": "secret-admin"})

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(allowed.status_code, 200)
        self.assertTrue(allowed.get_json()["ready"])

    def test_iap_verify_returns_not_configured_without_granting_pro(self):
        response = self.client.post("/api/iap/verify", json={"platform": "ios", "receipt": "receipt-data"})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "not_configured")

    def test_iap_verify_upserts_subscription_after_store_verification(self):
        expires_at = (dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(timespec="seconds")
        verification = app_module.iap.IapVerification(
            ok=True,
            reason="verified",
            product_id="racelens_pro_monthly",
            status="active",
            expires_at=expires_at,
        )
        with patch.dict(os.environ, {"RACELENS_APPLE_SHARED_SECRET": "shared-secret"}, clear=False), \
             patch.object(app_module.iap, "verify_receipt_with_store", return_value=verification):
            response = self.client.post(
                "/api/iap/verify",
                json={"platform": "ios", "receipt": "receipt-data"},
                headers={
                    "X-RaceLens-Device-Id": "device-iap-1",
                    "X-RaceLens-Platform": "ios",
                },
            )
            payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["app_session"]["entitlement"], "pro")
        with closing_sqlite(os.environ["DATABASE_URL"].replace("sqlite:///", "", 1)) as conn:
            row = conn.execute("SELECT platform, product_id, status FROM billing__subscriptions").fetchone()
        self.assertEqual(row, ("ios", "racelens_pro_monthly", "active"))

    def test_live_decision_records_device_prediction_and_view_event(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "odds_live",
            "message": "stored",
            "updated_at": "2026-07-02T12:00:00",
            "odds_age_sec": 0,
            "market_odds": [{"code": "WIN", "label": "단승", "selection": "4", "odds": 1.9}],
            "top": base["rows"][0],
            "rows": base["rows"],
            "decision": "hold",
            "market_used": True,
            "snapshot_phase": "live_odds",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision):
                response = self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                    headers={
                        "X-RaceLens-Device-Id": "device-test-1",
                        "X-RaceLens-Platform": "ios",
                    },
                )
                payload = response.get_json()
                status = datastore.app_data_layer_status()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["data_layer"]["ready"])
        self.assertEqual(payload["app_session"]["device_id"], "device-test-1")
        self.assertEqual(payload["app_session"]["entitlement"], "free")
        self.assertEqual(payload["app_session"]["free_analysis_limit"], 3)
        self.assertEqual(payload["app_session"]["free_analysis_used"], 1)
        self.assertEqual(payload["app_session"]["free_analysis_remaining"], 2)
        counts = {item["name"]: item["row_count"] for item in status["schemas"]}
        self.assertEqual(counts["user_account"], 3)
        self.assertEqual(counts["prediction"], 1)
        self.assertEqual(counts["analytics"], 1)
        self.assertEqual(counts["race_data"], 1)

    def test_no_card_hold_refunds_free_live_decision_quota(self):
        base = {
            "error": "no_race",
            "error_kind": "no_race",
            "message": "출주표 미공개 — 경주일 아님/카드 미발표",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base):
                response = self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-07-07&meet=광명&race_no=5",
                    headers={
                        "X-RaceLens-Device-Id": "device-no-card-refund",
                        "X-RaceLens-Platform": "ios",
                    },
                )
                with closing_sqlite(db_path) as conn:
                    used = conn.execute("SELECT count FROM user_account__analysis_usage").fetchone()[0]

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "hold")
        self.assertEqual(payload["error_kind"], "no_race")
        self.assertEqual(payload["app_session"]["free_analysis_used"], 0)
        self.assertEqual(payload["app_session"]["free_analysis_remaining"], 3)
        self.assertEqual(used, 0)

    def test_free_live_decision_quota_is_enforced_before_prediction(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision) as live_mock:
                responses = [
                    self.client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers={
                            "X-RaceLens-Device-Id": "device-quota-1",
                            "X-RaceLens-Platform": "ios",
                        },
                    )
                    for _ in range(4)
                ]
                with closing_sqlite(db_path) as conn:
                    prediction_count = conn.execute("SELECT COUNT(*) FROM prediction__predictions").fetchone()[0]

        last_payload = responses[-1].get_json()
        self.assertEqual([response.status_code for response in responses], [200, 200, 200, 200])
        self.assertEqual(live_mock.call_count, 3)
        self.assertEqual(prediction_count, 3)
        self.assertEqual(last_payload["status"], "blocked")
        self.assertEqual(last_payload["snapshot_phase"], "quota_exhausted")
        self.assertEqual(last_payload["app_session"]["free_analysis_used"], 3)
        self.assertEqual(last_payload["app_session"]["free_analysis_remaining"], 0)

    def test_rewarded_ad_claim_is_disabled_without_server_verification_flag(self):
        with patch.dict(os.environ, {"RACELENS_REWARDED_ADS_ENABLED": ""}, clear=False):
            response = self.client.post(
                "/api/rewarded-ad/claim",
                headers={
                    "X-RaceLens-Device-Id": "device-rewarded-disabled",
                    "X-RaceLens-Platform": "android",
                },
                json={"placement": "quota_gate"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "rewarded_ads_disabled")

    def test_direct_rewarded_ad_claim_never_mints_credit_when_enabled(self):
        response = self.client.post(
            "/api/rewarded-ad/claim",
            headers={
                "X-RaceLens-Device-Id": "device-direct-claim",
                "X-RaceLens-Platform": "android",
            },
            json={"placement": "quota_gate"},
        )

        self.assertEqual(response.status_code, 410)
        self.assertEqual(response.get_json()["error"], "ssv_required")

    def test_verified_ssv_credit_is_idempotent_and_extends_quota(self):
        reward = VerifiedReward(
            transaction_id="txn-verified-00000001",
            device_id="device-verified-00000001",
            ad_unit="ca-app-pub-1234567890123456/1234567890",
            reward_amount=1,
            reward_item="analysis_credit",
            timestamp_ms=int(dt.datetime.now(dt.UTC).timestamp() * 1000),
            custom_data="quota_gate",
        )
        with patch.dict(
            os.environ,
            {"RACELENS_ADMOB_REWARDED_AD_UNIT_ID": reward.ad_unit},
            clear=False,
        ), patch.object(app_module.admob_ssv, "verify_callback", return_value=reward):
            first = self.client.get("/api/rewarded-ad/ssv?first=1")
            duplicate = self.client.get("/api/rewarded-ad/ssv?first=1")
            session = self.client.get(
                "/api/app-session",
                headers={
                    "X-RaceLens-Device-Id": reward.device_id,
                    "X-RaceLens-Platform": "android",
                },
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.get_data(as_text=True), "")
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(session.get_json()["app_session"]["rewarded_analysis_credits"], 1)

    def test_admob_ui_verification_callback_returns_ok_without_minting_credit(self):
        reward = VerifiedReward(
            transaction_id="txn-ui-verification-0001",
            device_id="racelens_ssv_test_001",
            ad_unit="1234567890",
            reward_amount=1,
            reward_item="analysis_credit",
            timestamp_ms=int(dt.datetime.now(dt.UTC).timestamp() * 1000),
            custom_data="",
        )
        with patch.dict(
            os.environ,
            {"RACELENS_ADMOB_REWARDED_AD_UNIT_ID": "ca-app-pub-1234567890123456/6412636739"},
            clear=False,
        ), patch.object(
            app_module.admob_ssv,
            "verify_callback",
            return_value=reward,
        ), patch.object(
            app_module.datastore,
            "claim_verified_rewarded_ad_credit_safely",
        ) as claim_credit:
            response = self.client.get("/api/rewarded-ad/ssv?verification=1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "")
        claim_credit.assert_not_called()

    def test_rewarded_ad_credit_extends_free_analysis_after_daily_limit(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            headers = {
                "X-RaceLens-Device-Id": "device-rewarded-ad",
                "X-RaceLens-Platform": "ios",
            }
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision) as live_mock:
                first_responses = [
                    self.client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers=headers,
                    )
                    for _ in range(3)
                ]
                blocked = self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                    headers=headers,
                )
                reward_session, reward_granted, reward_duplicate = datastore.claim_verified_rewarded_ad_credit(
                    headers["X-RaceLens-Device-Id"],
                    headers["X-RaceLens-Platform"],
                    "txn-rewarded-extension-0001",
                    "ca-app-pub-1234567890123456/1234567890",
                    1,
                    "analysis_credit",
                    dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                )
                rewarded = self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                    headers=headers,
                )
                blocked_again = self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                    headers=headers,
                )
                with closing_sqlite(db_path) as conn:
                    row = conn.execute("SELECT count, rewarded_credits FROM user_account__analysis_usage").fetchone()

        self.assertTrue(all(response.get_json()["status"] == "ready" for response in first_responses))
        self.assertEqual(blocked.get_json()["snapshot_phase"], "quota_exhausted")
        self.assertTrue(reward_granted)
        self.assertFalse(reward_duplicate)
        self.assertEqual(reward_session["rewarded_analysis_credits"], 1)
        self.assertEqual(rewarded.get_json()["status"], "ready")
        self.assertEqual(rewarded.get_json()["app_session"]["rewarded_analysis_credits"], 0)
        self.assertEqual(blocked_again.get_json()["snapshot_phase"], "quota_exhausted")
        self.assertEqual(row[0], 3)
        self.assertEqual(row[1], 0)
        self.assertEqual(live_mock.call_count, 4)

    def test_rewarded_ad_credit_stops_at_daily_reward_cap(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            headers = {
                "X-RaceLens-Device-Id": "device-rewarded-ad-daily-cap",
                "X-RaceLens-Platform": "ios",
            }
            with patch.dict(
                os.environ,
                {
                    "DATABASE_URL": f"sqlite:///{db_path}",
                    "RACELENS_REWARDED_AD_DAILY_CAP": "1",
                },
                clear=False,
            ), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision):
                for _ in range(3):
                    self.client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers=headers,
                    )
                first_session, first_granted, first_duplicate = datastore.claim_verified_rewarded_ad_credit(
                    headers["X-RaceLens-Device-Id"],
                    headers["X-RaceLens-Platform"],
                    "txn-daily-cap-00000001",
                    "ca-app-pub-1234567890123456/1234567890",
                    1,
                    "analysis_credit",
                    dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                )
                self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                    headers=headers,
                )
                second_session, second_granted, second_duplicate = datastore.claim_verified_rewarded_ad_credit(
                    headers["X-RaceLens-Device-Id"],
                    headers["X-RaceLens-Platform"],
                    "txn-daily-cap-00000002",
                    "ca-app-pub-1234567890123456/1234567890",
                    1,
                    "analysis_credit",
                    dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                )
                with closing_sqlite(db_path) as conn:
                    row = conn.execute("SELECT rewarded_credits, rewarded_claims FROM user_account__analysis_usage").fetchone()

        self.assertTrue(first_granted)
        self.assertFalse(first_duplicate)
        self.assertFalse(second_granted)
        self.assertFalse(second_duplicate)
        self.assertEqual(first_session["rewarded_analysis_credits"], 1)
        self.assertEqual(second_session["rewarded_analysis_credits"], 0)
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], 1)

    def test_direct_rewarded_ad_claim_stays_blocked_for_repeated_clients(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(
                os.environ,
                {
                    "DATABASE_URL": f"sqlite:///{db_path}",
                    "RACELENS_REWARDED_AD_IP_PER_MIN_CAP": "1",
                },
                clear=False,
            ):
                first_reward = self.client.post(
                    "/api/rewarded-ad/claim",
                    headers={
                        "X-Forwarded-For": "203.0.113.9",
                        "X-RaceLens-Device-Id": "device-rewarded-ip-a",
                        "X-RaceLens-Platform": "ios",
                    },
                    json={"placement": "quota_gate"},
                )
                second_reward = self.client.post(
                    "/api/rewarded-ad/claim",
                    headers={
                        "X-Forwarded-For": "203.0.113.9",
                        "X-RaceLens-Device-Id": "device-rewarded-ip-b",
                        "X-RaceLens-Platform": "ios",
                    },
                    json={"placement": "quota_gate"},
                )

        self.assertEqual(first_reward.status_code, 410)
        self.assertEqual(second_reward.status_code, 410)
        self.assertEqual(first_reward.get_json()["error"], "ssv_required")
        self.assertEqual(second_reward.get_json()["error"], "ssv_required")

    def test_free_quota_limit_env_override_raises_daily_wall(self):
        """RACELENS_FREE_DAILY_ANALYSIS_LIMIT env로 한도 상향 (IAP 전 무제한 운영용). 메커니즘은 유지."""
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "RACELENS_FREE_DAILY_ANALYSIS_LIMIT": "5",
            }, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision) as live_mock:
                responses = [
                    self.client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers={
                            "X-RaceLens-Device-Id": "device-quota-env",
                            "X-RaceLens-Platform": "ios",
                        },
                    )
                    for _ in range(6)
                ]

        last_payload = responses[-1].get_json()
        self.assertEqual(live_mock.call_count, 5)
        self.assertEqual(responses[0].get_json()["app_session"]["free_analysis_limit"], 5)
        self.assertEqual([r.get_json().get("status") for r in responses[:5]].count("blocked"), 0)
        self.assertEqual(last_payload["status"], "blocked")
        self.assertEqual(last_payload["app_session"]["free_analysis_remaining"], 0)

    def test_concurrent_free_live_decisions_cannot_exceed_daily_limit(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")

            def request_once(_index):
                with app_module.app.test_client() as client:
                    return client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers={
                            "X-RaceLens-Device-Id": "device-concurrent-quota",
                            "X-RaceLens-Platform": "ios",
                        },
                    ).get_json()

            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision):
                datastore.app_data_layer_status()
                with ThreadPoolExecutor(max_workers=6) as pool:
                    payloads = list(pool.map(request_once, range(6)))
                with closing_sqlite(db_path) as conn:
                    used = conn.execute("SELECT count FROM user_account__analysis_usage").fetchone()[0]
                    prediction_count = conn.execute("SELECT COUNT(*) FROM prediction__predictions").fetchone()[0]

        self.assertEqual(used, 3)
        self.assertEqual(prediction_count, 3)
        self.assertEqual(sum(1 for payload in payloads if payload["status"] == "blocked"), 3)

    def test_app_session_reads_quota_without_consuming_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False):
                responses = [
                    self.client.get(
                        "/api/app-session",
                        headers={
                            "X-RaceLens-Device-Id": "device-session-only",
                            "X-RaceLens-Platform": "ios",
                        },
                    )
                    for _ in range(3)
                ]
                with closing_sqlite(db_path) as conn:
                    usage_count = conn.execute("SELECT COUNT(*) FROM user_account__analysis_usage").fetchone()[0]

        payloads = [response.get_json() for response in responses]
        self.assertEqual([response.status_code for response in responses], [200, 200, 200])
        self.assertEqual(usage_count, 0)
        self.assertTrue(all(payload["app_session"]["free_analysis_used"] == 0 for payload in payloads))
        self.assertTrue(all(payload["app_session"]["free_analysis_remaining"] == 3 for payload in payloads))

    def test_active_subscription_bypasses_free_quota(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision) as live_mock:
                session = datastore.ensure_app_session("device-pro-1", "ios")
                with closing_sqlite(db_path) as conn:
                    conn.execute(
                        "INSERT INTO billing__subscriptions (user_id, platform, product_id, status, expires_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (session["user_id"], "ios", "racelens_pro_monthly", "active", None, "2026-07-03T00:00:00+00:00"),
                    )
                    conn.commit()
                responses = [
                    self.client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers={
                            "X-RaceLens-Device-Id": "device-pro-1",
                            "X-RaceLens-Platform": "ios",
                        },
                    )
                    for _ in range(4)
                ]

        payloads = [response.get_json() for response in responses]
        self.assertEqual(live_mock.call_count, 4)
        self.assertTrue(all(payload["app_session"]["entitlement"] == "pro" for payload in payloads))
        self.assertTrue(all(payload["status"] != "blocked" for payload in payloads))

    def test_force_pro_env_bypasses_free_quota_without_subscription_row(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}", "RACELENS_FORCE_PRO": "1"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision) as live_mock:
                app_session = self.client.get(
                    "/api/app-session",
                    headers={
                        "X-RaceLens-Device-Id": "device-force-pro",
                        "X-RaceLens-Platform": "ios",
                    },
                ).get_json()
                responses = [
                    self.client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers={
                            "X-RaceLens-Device-Id": "device-force-pro",
                            "X-RaceLens-Platform": "ios",
                        },
                    )
                    for _ in range(4)
                ]
                with closing_sqlite(db_path) as conn:
                    subscription_count = conn.execute("SELECT COUNT(*) FROM billing__subscriptions").fetchone()[0]
                    usage_count = conn.execute("SELECT COUNT(*) FROM user_account__analysis_usage").fetchone()[0]

        payloads = [response.get_json() for response in responses]
        self.assertEqual(app_session["app_session"]["entitlement"], "pro")
        self.assertEqual(subscription_count, 0)
        self.assertEqual(usage_count, 0)
        self.assertEqual(live_mock.call_count, 4)
        self.assertTrue(all(payload["app_session"]["entitlement"] == "pro" for payload in payloads))
        self.assertTrue(all(payload["status"] != "blocked" for payload in payloads))

    def test_production_can_disable_public_pro_and_keep_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "RACELENS_ENV": "production",
                "RACELENS_PUBLIC_PRO": "0",
                "RACELENS_FORCE_PRO": "1",
                "RACELENS_PRO_DEVICE_IDS": "device-allowlisted",
            }, clear=False):
                forced = datastore.ensure_app_session("device-forced-only", "ios")
                allowed = datastore.ensure_app_session("device-allowlisted", "ios")

        self.assertEqual(forced["entitlement"], "free")
        self.assertEqual(allowed["entitlement"], "pro")

    def test_production_public_pro_applies_to_every_client_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "RACELENS_ENV": "production",
                "RACELENS_PUBLIC_PRO": "1",
            }, clear=False):
                web = datastore.ensure_app_session("web-device", "web")
                native = datastore.ensure_app_session("native-device", "ios")

        self.assertEqual(web["entitlement"], "pro")
        self.assertEqual(native["entitlement"], "pro")

    def test_healthz_exposes_entitlement_mode(self):
        with patch.object(app_module.os.path, "exists", return_value=True), \
             patch.dict(os.environ, {"RACELENS_ENV": "production"}, clear=False):
            response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["entitlement_mode"], "production")

    def test_ip_new_user_cap_reuses_recent_user_for_new_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "RACELENS_IP_NEW_USER_CAP": "2",
            }, clear=False):
                first = datastore.ensure_app_session("ip-device-1", "ios", ip_address="203.0.113.10")
                second = datastore.ensure_app_session("ip-device-2", "ios", ip_address="203.0.113.10")
                third = datastore.ensure_app_session("ip-device-3", "ios", ip_address="203.0.113.10")

        self.assertNotEqual(first["user_id"], second["user_id"])
        self.assertEqual(third["user_id"], second["user_id"])
        self.assertEqual(third["device_id"], "ip-device-3")

    def test_live_decision_ip_minute_rate_limit_returns_429_before_prediction(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "stored",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "RACELENS_LIVE_DECISION_IP_PER_MIN_CAP": "2",
            }, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", return_value=decision) as live_mock:
                responses = [
                    self.client.get(
                        "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                        headers={
                            "X-RaceLens-Device-Id": f"rate-device-{idx}",
                            "X-Forwarded-For": "198.51.100.42",
                        },
                    )
                    for idx in range(3)
                ]

        self.assertEqual([response.status_code for response in responses], [200, 200, 429])
        self.assertEqual(live_mock.call_count, 2)
        self.assertEqual(responses[-1].get_json()["status"], "rate_limited")

    def test_live_decision_keeps_serving_when_database_write_fails(self):
        base = {"kind": "ok", "rows": [{"bno": 4, "name": "A", "pwin": 0.6, "pplc": 0.8}]}
        decision = {
            "ok": True,
            "status": "ready",
            "message": "served without db",
            "market_odds": [],
            "market_used": False,
            "poll_delay_ms": 15000,
            "snapshot_phase": "pre_race",
        }

        with patch.object(app_module, "_compute_base_prediction", return_value=base), \
             patch.object(app_module.engine, "compute_live_decision", return_value=decision), \
             patch.object(datastore, "record_live_decision", side_effect=sqlite3.OperationalError("disk full")):
            response = self.client.get(
                "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                headers={"X-RaceLens-Device-Id": "device-fail-open"},
            )
            payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["data_layer"]["ready"])
        self.assertIn("OperationalError", payload["data_layer"]["error"])
        self.assertEqual(payload["app_session"]["device_id"], "device-fail-open")

    def test_api_routes_allow_mobile_web_client_headers(self):
        with patch.dict(os.environ, {"RACELENS_ALLOWED_ORIGINS": ""}, clear=False):
            response = self.client.options("/api/live-decision")
            recent = self.client.get("/recent?sport=keirin&meet=광명", headers={"Origin": "http://127.0.0.1:19006"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertIn("POST", response.headers["Access-Control-Allow-Methods"])
        self.assertIn("X-RaceLens-Device-Id", response.headers["Access-Control-Allow-Headers"])
        self.assertIn("X-RaceLens-Platform", response.headers["Access-Control-Allow-Headers"])
        self.assertIn("X-RaceLens-Analytics", response.headers["Access-Control-Allow-Headers"])
        self.assertEqual(recent.headers["Access-Control-Allow-Origin"], "*")

    def test_api_cors_can_be_restricted_to_release_origins(self):
        with patch.dict(os.environ, {"RACELENS_ALLOWED_ORIGINS": "https://app.racelens.example"}, clear=False):
            allowed = self.client.options("/api/live-decision", headers={"Origin": "https://app.racelens.example"})
            recent_allowed = self.client.get("/recent?sport=keirin&meet=광명", headers={"Origin": "https://app.racelens.example"})
            blocked = self.client.options("/api/live-decision", headers={"Origin": "https://evil.example"})
            recent_blocked = self.client.get("/recent?sport=keirin&meet=광명", headers={"Origin": "https://evil.example"})

        self.assertEqual(allowed.headers["Access-Control-Allow-Origin"], "https://app.racelens.example")
        self.assertEqual(allowed.headers["Vary"], "Origin")
        self.assertEqual(recent_allowed.headers["Access-Control-Allow-Origin"], "https://app.racelens.example")
        self.assertEqual(recent_allowed.headers["Vary"], "Origin")
        self.assertNotIn("Access-Control-Allow-Origin", blocked.headers)
        self.assertNotIn("Access-Control-Allow-Origin", recent_blocked.headers)

    def test_production_errors_do_not_expose_tracebacks(self):
        app_module.app.config["TESTING"] = False
        try:
            with patch.dict(os.environ, {"RACELENS_DEBUG_ERRORS": "0"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction", side_effect=RuntimeError("secret backend detail")):
                response = self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5",
                    headers={"X-RaceLens-Device-Id": "device-error-safety"},
                )

            with app_module.app.test_request_context("/api/__qa_error"):
                error_response, error_status = app_module.handle_all_errors(RuntimeError("secret handler detail"))
        finally:
            app_module.app.config["TESTING"] = True

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["snapshot_phase"], "pending")
        self.assertNotIn("secret backend detail", response.get_data(as_text=True))
        handler_payload = error_response.get_json()
        self.assertEqual(error_status, 500)
        self.assertEqual(handler_payload["error"], "internal_server_error")
        self.assertNotIn("trace", handler_payload)
        self.assertNotIn("detail", handler_payload)
        self.assertNotIn("secret handler detail", error_response.get_data(as_text=True))

    def test_ux_events_records_privacy_safe_event(self):
        event = {
            "app": "racelens",
            "version": "0.1.0",
            "name": "analysis_result",
            "sessionId": "sess_test",
            "anonymousId": "anon_test",
            "platform": "web",
            "timestamp": "2026-07-02T12:00:00Z",
            "payload": {
                "tab": "analyze",
                "sport": "keirin",
                "raceNo": 5,
                "marketUsed": True,
                "top1Pct": 61.8,
                "trifectaPct": 22.1,
                "latencyMs": 238,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False):
                response = self.client.post("/api/ux-events", json=event)
                status = datastore.app_data_layer_status()
                with closing_sqlite(db_path) as conn:
                    row = conn.execute(
                        "SELECT event_name, payload_json FROM analytics__user_view_events"
                    ).fetchone()

        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.get_json()["stored"])
        counts = {item["name"]: item["row_count"] for item in status["schemas"]}
        self.assertEqual(counts["user_account"], 2)
        self.assertEqual(counts["analytics"], 1)
        self.assertEqual(row[0], "analysis_result")
        stored = json.loads(row[1])
        self.assertEqual(stored["sessionId"], "sess_test")
        self.assertEqual(stored["raceNo"], 5)
        self.assertNotIn("meet", stored)
        self.assertNotIn("selection", stored)
        self.assertNotIn("deviceId", stored)
        self.assertNotIn("userId", stored)

    def test_ux_events_drops_removed_lab_tab_enum(self):
        event = {
            "app": "racelens",
            "version": "0.1.0",
            "name": "tab_select",
            "sessionId": "sess_test",
            "anonymousId": "anon_lab",
            "platform": "web",
            "timestamp": "2026-07-02T12:00:00Z",
            "payload": {"tab": "lab", "previousTab": "home"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False):
                response = self.client.post("/api/ux-events", json=event)
                with closing_sqlite(db_path) as conn:
                    stored = json.loads(conn.execute(
                        "SELECT payload_json FROM analytics__user_view_events"
                    ).fetchone()[0])

        self.assertEqual(response.status_code, 202)
        self.assertNotIn("tab", stored)
        self.assertEqual(stored["previousTab"], "home")

    def test_ux_events_rejects_forbidden_payload_fields_before_storage(self):
        event = {
            "app": "racelens",
            "version": "0.1.0",
            "name": "analysis_result",
            "sessionId": "sess_test",
            "anonymousId": "anon_test",
            "platform": "web",
            "timestamp": "2026-07-02T12:00:00Z",
            "payload": {"tab": "analyze", "sport": "keirin", "raceNo": 5, "selection": "1-2-3"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "strategy.sqlite")
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_path}"}, clear=False):
                response = self.client.post("/api/ux-events", json=event)
                status = datastore.app_data_layer_status()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "forbidden_payload_field")
        counts = {item["name"]: item["row_count"] for item in status["schemas"]}
        self.assertEqual(counts["user_account"], 0)
        self.assertEqual(counts["analytics"], 0)

    def test_ux_events_preflight_allows_mobile_analytics_header(self):
        with patch.dict(os.environ, {"RACELENS_ALLOWED_ORIGINS": ""}, clear=False):
            response = self.client.options("/api/ux-events")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertIn("POST", response.headers["Access-Control-Allow-Methods"])
        self.assertIn("X-RaceLens-Analytics", response.headers["Access-Control-Allow-Headers"])


if __name__ == "__main__":
    unittest.main()
