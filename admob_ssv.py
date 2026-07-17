"""Google AdMob rewarded-ad server-side verification (SSV)."""

from __future__ import annotations

import base64
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Final
from urllib.parse import parse_qsl, unquote_to_bytes

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


VERIFIER_KEYS_URL: Final = "https://www.gstatic.com/admob/reward/verifier-keys.json"
DEFAULT_KEY_CACHE_SECONDS: Final = 23 * 60 * 60
DEFAULT_CALLBACK_MAX_AGE_SECONDS: Final = 24 * 60 * 60
ADMOB_UI_VERIFICATION_AD_UNIT: Final = "1234567890"
_DEVICE_ID = re.compile(r"^[A-Za-z0-9._:-]{8,96}$")
_TRANSACTION_ID = re.compile(r"^[A-Za-z0-9._:~-]{8,128}$")
_FULL_AD_UNIT_ID = re.compile(r"^ca-app-pub-\d{16}/(\d{10})$")
_cache_lock = threading.Lock()
_key_cache: tuple[float, dict[int, str]] | None = None


class SsvVerificationError(ValueError):
    """Raised when an AdMob callback cannot be trusted or accepted."""


@dataclass(frozen=True, slots=True)
class VerifiedReward:
    transaction_id: str
    device_id: str
    ad_unit: str
    reward_amount: int
    reward_item: str
    timestamp_ms: int
    custom_data: str


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or str(default)).strip()))
    except ValueError:
        return default


def _fetch_public_keys() -> dict[int, str]:
    response = requests.get(VERIFIER_KEYS_URL, timeout=5)
    response.raise_for_status()
    payload = response.json()
    keys: dict[int, str] = {}
    for item in payload.get("keys", []):
        key_id = item.get("keyId", item.get("key_id"))
        pem = item.get("pem")
        if isinstance(key_id, str) and key_id.isdigit():
            key_id = int(key_id)
        if isinstance(key_id, int) and isinstance(pem, str) and pem.startswith("-----BEGIN PUBLIC KEY-----"):
            keys[key_id] = pem
    if not keys:
        raise SsvVerificationError("Google verifier key response was empty")
    return keys


def _public_key_pem(key_id: int) -> str:
    global _key_cache
    now = time.monotonic()
    ttl = min(24 * 60 * 60, _positive_int_env("RACELENS_ADMOB_KEY_CACHE_SECONDS", DEFAULT_KEY_CACHE_SECONDS))
    with _cache_lock:
        if _key_cache is None or now - _key_cache[0] >= ttl:
            _key_cache = (now, _fetch_public_keys())
        pem = _key_cache[1].get(key_id)
        if pem is None:
            _key_cache = (now, _fetch_public_keys())
            pem = _key_cache[1].get(key_id)
    if pem is None:
        raise SsvVerificationError("Unknown AdMob verifier key")
    return pem


def _decode_signature(value: bytes) -> bytes:
    try:
        decoded = unquote_to_bytes(value.decode("ascii"))
        return base64.urlsafe_b64decode(decoded + b"=" * (-len(decoded) % 4))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SsvVerificationError("Invalid AdMob signature encoding") from exc


def _signed_parts(raw_query: bytes) -> tuple[bytes, bytes, int]:
    marker = b"&signature="
    if raw_query.count(marker) != 1:
        raise SsvVerificationError("Missing or duplicate AdMob signature")
    signed_query, signature_tail = raw_query.split(marker, 1)
    signature_value, separator, key_value = signature_tail.partition(b"&key_id=")
    if not separator or not signature_value or not key_value or b"&" in key_value:
        raise SsvVerificationError("Invalid AdMob signature fields")
    try:
        key_id = int(key_value.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SsvVerificationError("Invalid AdMob key id") from exc
    return signed_query, _decode_signature(signature_value), key_id


def _callback_values(signed_query: bytes) -> dict[str, str]:
    try:
        pairs = parse_qsl(signed_query.decode("ascii"), keep_blank_values=True, strict_parsing=True)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SsvVerificationError("Invalid AdMob callback query") from exc
    values: dict[str, str] = {}
    for key, value in pairs:
        if key in values:
            raise SsvVerificationError(f"Duplicate AdMob callback field: {key}")
        values[key] = value
    return values


def _matches_expected_ad_unit(ad_unit: str, expected_ad_unit: str) -> bool:
    if ad_unit == ADMOB_UI_VERIFICATION_AD_UNIT:
        return True
    if ad_unit == expected_ad_unit:
        return True
    match = _FULL_AD_UNIT_ID.fullmatch(expected_ad_unit)
    return bool(match and ad_unit == match.group(1))


def verify_callback(raw_query: bytes, *, expected_ad_unit: str) -> VerifiedReward:
    """Verify Google's raw callback query and return the accepted reward claims."""
    if not expected_ad_unit:
        raise SsvVerificationError("Expected AdMob rewarded ad unit is not configured")
    signed_query, signature, key_id = _signed_parts(raw_query)
    try:
        public_key = serialization.load_pem_public_key(_public_key_pem(key_id).encode("ascii"))
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            raise SsvVerificationError("AdMob verifier key is not an EC public key")
        public_key.verify(signature, signed_query, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError, TypeError) as exc:
        if isinstance(exc, SsvVerificationError):
            raise
        raise SsvVerificationError("AdMob callback signature verification failed") from exc

    values = _callback_values(signed_query)
    ad_unit = values.get("ad_unit", "")
    transaction_id = values.get("transaction_id", "")
    device_id = values.get("user_id", "")
    reward_item = values.get("reward_item", "")
    custom_data = values.get("custom_data", "")
    if not _matches_expected_ad_unit(ad_unit, expected_ad_unit):
        raise SsvVerificationError(f"AdMob callback used an unexpected ad unit: {ad_unit!r}")
    if not _TRANSACTION_ID.fullmatch(transaction_id):
        raise SsvVerificationError("Invalid AdMob transaction id")
    if not _DEVICE_ID.fullmatch(device_id):
        raise SsvVerificationError("Invalid AdMob user id")
    expected_reward_item = (os.environ.get("RACELENS_ADMOB_REWARD_ITEM") or "analysis_credit").strip()
    if reward_item != expected_reward_item:
        raise SsvVerificationError("Unexpected AdMob reward item")
    try:
        reward_amount = int(values.get("reward_amount", ""))
        timestamp_ms = int(values.get("timestamp", ""))
    except ValueError as exc:
        raise SsvVerificationError("Invalid AdMob reward amount or timestamp") from exc
    if reward_amount != 1:
        raise SsvVerificationError("Unexpected AdMob reward amount")
    age_seconds = time.time() - (timestamp_ms / 1000)
    max_age = _positive_int_env("RACELENS_ADMOB_CALLBACK_MAX_AGE_SECONDS", DEFAULT_CALLBACK_MAX_AGE_SECONDS)
    if age_seconds < -300 or age_seconds > max_age:
        raise SsvVerificationError("Expired AdMob callback")
    if len(custom_data) > 128:
        raise SsvVerificationError("AdMob custom data is too long")
    return VerifiedReward(
        transaction_id=transaction_id,
        device_id=device_id,
        ad_unit=ad_unit,
        reward_amount=reward_amount,
        reward_item=reward_item,
        timestamp_ms=timestamp_ms,
        custom_data=custom_data,
    )
