#!/usr/bin/env python3
import base64
import os
import sys
import time
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

import admob_ssv


def _callback_query(private_key, **overrides) -> bytes:
    values = {
        "ad_network": "5450213213286189855",
        "ad_unit": "ca-app-pub-1234567890123456/1234567890",
        "custom_data": "quota_gate",
        "reward_amount": "1",
        "reward_item": "analysis_credit",
        "timestamp": str(int(time.time() * 1000)),
        "transaction_id": "txn-ssv-00000001",
        "user_id": "device-ssv-00000001",
    }
    values.update(overrides)
    signed = urlencode(values).encode("ascii")
    signature = private_key.sign(signed, ec.ECDSA(hashes.SHA256()))
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return signed + f"&signature={encoded_signature}&key_id=321".encode("ascii")


class AdMobSsvVerifierTestCase(unittest.TestCase):
    def setUp(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_pem = self.private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

    def test_verifies_raw_google_callback_and_returns_typed_reward(self):
        raw_query = _callback_query(self.private_key)

        with patch.object(admob_ssv, "_public_key_pem", return_value=self.public_pem):
            reward = admob_ssv.verify_callback(
                raw_query,
                expected_ad_unit="ca-app-pub-1234567890123456/1234567890",
            )

        self.assertEqual(reward.transaction_id, "txn-ssv-00000001")
        self.assertEqual(reward.device_id, "device-ssv-00000001")
        self.assertEqual(reward.reward_amount, 1)
        self.assertEqual(reward.reward_item, "analysis_credit")

    def test_rejects_tampered_callback_before_crediting(self):
        raw_query = _callback_query(self.private_key).replace(b"reward_amount=1", b"reward_amount=9")

        with patch.object(admob_ssv, "_public_key_pem", return_value=self.public_pem):
            with self.assertRaises(admob_ssv.SsvVerificationError):
                admob_ssv.verify_callback(
                    raw_query,
                    expected_ad_unit="ca-app-pub-1234567890123456/1234567890",
                )

    def test_accepts_google_numeric_ad_unit_for_configured_full_id(self):
        raw_query = _callback_query(self.private_key, ad_unit="1234567890")

        with patch.object(admob_ssv, "_public_key_pem", return_value=self.public_pem):
            reward = admob_ssv.verify_callback(
                raw_query,
                expected_ad_unit="ca-app-pub-1234567890123456/1234567890",
            )

        self.assertEqual(reward.ad_unit, "1234567890")

    def test_accepts_google_ui_verification_ad_unit_without_matching_suffix(self):
        raw_query = _callback_query(self.private_key, ad_unit="1234567890")

        with patch.object(admob_ssv, "_public_key_pem", return_value=self.public_pem):
            reward = admob_ssv.verify_callback(
                raw_query,
                expected_ad_unit="ca-app-pub-9999999999999999/9999999999",
            )

        self.assertEqual(reward.ad_unit, admob_ssv.ADMOB_UI_VERIFICATION_AD_UNIT)

    def test_rejects_wrong_google_numeric_ad_unit(self):
        raw_query = _callback_query(self.private_key, ad_unit="9999999999")

        with patch.object(admob_ssv, "_public_key_pem", return_value=self.public_pem):
            with self.assertRaisesRegex(admob_ssv.SsvVerificationError, "unexpected ad unit"):
                admob_ssv.verify_callback(
                    raw_query,
                    expected_ad_unit="ca-app-pub-1234567890123456/1234567890",
                )

    def test_rejects_wrong_ad_unit_and_stale_callback(self):
        stale = _callback_query(self.private_key, timestamp=str(int((time.time() - 90000) * 1000)))

        with patch.object(admob_ssv, "_public_key_pem", return_value=self.public_pem):
            with self.assertRaises(admob_ssv.SsvVerificationError):
                admob_ssv.verify_callback(stale, expected_ad_unit="ca-app-pub-9999999999999999/9999999999")

    def test_accepts_string_key_ids_from_google_key_response(self):
        response = Mock()
        response.json.return_value = {"keys": [{"keyId": "321", "pem": self.public_pem}]}

        with patch.object(admob_ssv.requests, "get", return_value=response):
            keys = admob_ssv._fetch_public_keys()

        response.raise_for_status.assert_called_once_with()
        self.assertEqual(keys, {321: self.public_pem})


if __name__ == "__main__":
    unittest.main()
