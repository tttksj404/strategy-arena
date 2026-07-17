import datetime as dt
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IapVerification:
    ok: bool
    reason: str
    product_id: str = ""
    status: str = "inactive"
    expires_at: str | None = None


def verifier_configured(platform: str) -> bool:
    if platform == "ios":
        return bool((os.environ.get("RACELENS_APPLE_SHARED_SECRET") or "").strip())
    return False


def verify_receipt_with_store(platform: str, receipt: str) -> IapVerification:
    if platform == "ios":
        return _verify_apple_receipt(receipt)
    if platform == "android":
        return IapVerification(ok=False, reason="not_configured")
    return IapVerification(ok=False, reason="unsupported_platform")


def _verify_apple_receipt(receipt: str) -> IapVerification:
    secret = (os.environ.get("RACELENS_APPLE_SHARED_SECRET") or "").strip()
    if not secret:
        return IapVerification(ok=False, reason="not_configured")
    payload = json.dumps({
        "receipt-data": receipt,
        "password": secret,
        "exclude-old-transactions": True,
    }).encode("utf-8")
    for url in (
        "https://buy.itunes.apple.com/verifyReceipt",
        "https://sandbox.itunes.apple.com/verifyReceipt",
    ):
        verification = _post_apple_verify(url, payload)
        if verification.reason == "use_sandbox":
            continue
        return verification
    return IapVerification(ok=False, reason="store_rejected")


def _post_apple_verify(url: str, payload: bytes) -> IapVerification:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            raw = response.read(131072)
    except (OSError, urllib.error.URLError) as exc:
        return IapVerification(ok=False, reason=f"store_error:{type(exc).__name__}")
    data = json.loads(raw.decode("utf-8"))
    status = int(data.get("status") or 0)
    if status == 21007:
        return IapVerification(ok=False, reason="use_sandbox")
    if status != 0:
        return IapVerification(ok=False, reason="store_rejected")
    latest = data.get("latest_receipt_info") if isinstance(data.get("latest_receipt_info"), list) else []
    latest_item = latest[-1] if latest and isinstance(latest[-1], dict) else {}
    product_id = str(latest_item.get("product_id") or "racelens_pro_monthly")
    expires_ms = str(latest_item.get("expires_date_ms") or "").strip()
    expires_at = _expires_from_ms(expires_ms)
    active = expires_at is None or expires_at > dt.datetime.now(dt.UTC)
    return IapVerification(
        ok=active,
        reason="verified" if active else "expired",
        product_id=product_id,
        status="active" if active else "expired",
        expires_at=expires_at.isoformat(timespec="seconds") if expires_at else None,
    )


def _expires_from_ms(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromtimestamp(int(value) / 1000, tz=dt.UTC)
    except ValueError:
        return None
