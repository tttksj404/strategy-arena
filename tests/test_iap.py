import base64
import datetime as dt
import json
import os
from unittest.mock import patch

import iap


def test_android_receipt_stays_disabled_without_google_play_api_verification():
    forged_receipt = base64.b64encode(
        json.dumps(
            {
                "productId": "racelens_pro_monthly",
                "expiryTimeMillis": str(int((dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).timestamp() * 1000)),
            }
        ).encode("utf-8")
    ).decode("ascii")

    with patch.dict(os.environ, {"RACELENS_GOOGLE_SA_JSON": "{}"}, clear=False):
        configured = iap.verifier_configured("android")
        verification = iap.verify_receipt_with_store("android", forged_receipt)

    assert configured is False
    assert verification.ok is False
    assert verification.reason == "not_configured"
