#!/usr/bin/env bash
set -eu

base_url="${1:-${ORACLE_BASE_URL:-}}"
if [ -z "$base_url" ]; then
  echo "Usage: $0 https://your-domain-or-ip" >&2
  exit 2
fi

python3 - "$base_url" <<'PY'
from datetime import date, timedelta
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


base = sys.argv[1].rstrip("/")
expected_entitlement = os.environ.get("SMOKE_EXPECT_ENTITLEMENT", "free")
if expected_entitlement not in {"free", "pro"}:
    raise SystemExit("SMOKE_EXPECT_ENTITLEMENT must be free or pro")

smoke_device_id = f"oracle-smoke-{uuid.uuid4()}"
forbidden_names = ["김태훈", "방민재", "이원석"]


def build_url(path, params=None):
    url = f"{base}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return url


def fetch(path, params=None, timeout=30):
    url = build_url(path, params)
    request = urllib.request.Request(url, headers={"X-RaceLens-Device-Id": smoke_device_id})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        raise SystemExit(f"{path} returned HTTP {error.code}: {body[:240]}")
    except urllib.error.URLError as error:
        raise SystemExit(f"{path} request failed: {error.reason}")
    elapsed = time.perf_counter() - started
    if status != 200:
        raise SystemExit(f"{path} returned HTTP {status}")
    return {"url": url, "status": status, "body": body, "elapsed": elapsed}


def parse_json(result, label):
    try:
        parsed = json.loads(result["body"])
    except json.JSONDecodeError as error:
        raise SystemExit(f"{label} returned invalid JSON: {error}")
    if not isinstance(parsed, dict):
        raise SystemExit(f"{label} JSON must be an object")
    return parsed


def assert_clean_text(result, label, required_text):
    body = result["body"]
    if required_text not in body or "RaceLens" not in body:
        raise SystemExit(f"{label} missing required store-review text")
    if "�" in body:
        raise SystemExit(f"{label} contains broken replacement characters")


def extract_app_session(parsed, context):
    app_session = parsed.get("app_session")
    if not isinstance(app_session, dict):
        raise SystemExit(f"{context} app_session must be an object")
    return app_session


def assert_session_contract(app_session, context):
    if app_session.get("entitlement") != expected_entitlement:
        # 같은 IP 반복 스모크는 _ensure_device_user IP 앵커링으로 기존 유저
        # (구독/보상 이력 보유)에 바인딩될 수 있다 — 그 경우만 SKIP.
        # 진짜 신규 유저가 pro면 force_pro 누출이므로 여전히 하드 FAIL.
        anchor_bound = (
            int(app_session.get("rewarded_analysis_credits") or 0) > 0
            or int(app_session.get("free_analysis_used") or 0) > 0
            or bool(app_session.get("subscription"))
        )
        if anchor_bound:
            print(f"SKIP {context}: entitlement={app_session.get('entitlement')} via IP-anchored existing user")
        else:
            raise SystemExit(f"{context} app-session entitlement must be {expected_entitlement}")
    # 한도는 RACELENS_FREE_DAILY_ANALYSIS_LIMIT env로 운영 조정 가능 — 양의 정수 계약만 강제
    limit = app_session.get("free_analysis_limit")
    if not isinstance(limit, int) or limit < 1:
        raise SystemExit(f"{context} free_analysis_limit must be a positive integer")
    remaining = app_session.get("free_analysis_remaining")
    if not isinstance(remaining, int) or not 0 <= remaining <= limit:
        raise SystemExit(f"{context} free_analysis_remaining must be an integer from 0 to the limit")
    return remaining


def fetch_app_session(context):
    result = fetch("/api/app-session", {"device_id": smoke_device_id})
    parsed = parse_json(result, context)
    app_session = extract_app_session(parsed, context)
    assert_session_contract(app_session, context)
    return app_session


health = parse_json(fetch("/healthz"), "/healthz")
if health.get("ok") is not True:
    raise SystemExit("/healthz ok must be true")
if "entitlement_mode" not in health:
    raise SystemExit("healthz missing entitlement_mode")
if health.get("entitlement_mode") != "production":
    raise SystemExit("healthz entitlement_mode must be production")

root = fetch("/")
if "RaceLens" not in root["body"] or "�" in root["body"]:
    raise SystemExit("mobile web root did not render RaceLens shell cleanly")

legal_required = {
    "/legal/privacy": "개인정보",
    "/legal/terms": "이용약관",
    "/legal/account-deletion": "삭제",
}
for legal_path, required_text in legal_required.items():
    assert_clean_text(fetch(legal_path), legal_path, required_text)

initial_session = fetch_app_session("initial")
initial_remaining = initial_session["free_analysis_remaining"]

# live-decision 검사는 무료쿼터를 소모한다. 같은 IP에서 반복 실행(CI·재배포)하면
# IP 앵커링으로 remaining이 0일 수 있으므로 쿼터가 있을 때만 강한 어서션을 건다.
if initial_remaining > 0:
    settled_params = {
        "sport": "keirin",
        "date": "2026-07-03",
        "meet": "광명",
        "race_no": "1",
        "device_id": smoke_device_id,
    }
    settled = fetch("/api/live-decision", settled_params)
    settled_json = parse_json(settled, "settled live-decision")
    if settled_json.get("decision") != "settled":
        raise SystemExit("settled live-decision decision must be settled")
    extract_app_session(settled_json, "settled live-decision")
    if any(name in settled["body"] for name in forbidden_names):
        raise SystemExit("2026-07-03 광명 1R response contains stale demo participant names")
else:
    print("settled live-decision skipped (quota anchored)")

current_remaining = fetch_app_session("post-settled")["free_analysis_remaining"]
if current_remaining > 0:
    future_params = {
        "sport": "keirin",
        # 출주표는 경주 2~3일 전 발표 — +21일이면 미발표(no_race fast path) 보장
        "date": (date.today() + timedelta(days=21)).isoformat(),
        "meet": "광명",
        "race_no": "1",
        "device_id": smoke_device_id,
    }
    future = fetch("/api/live-decision", future_params, timeout=5)
    if future["elapsed"] >= 5:
        raise SystemExit("future no-race live-decision exceeded 5s fast-path budget")
    future_json = parse_json(future, "future no-race live-decision")
    if future_json.get("decision") != "hold":
        raise SystemExit("future no-race live-decision decision must be hold")
    if future_json.get("status") == "blocked":
        raise SystemExit("future no-race live-decision status must not be blocked")
    extract_app_session(future_json, "future no-race live-decision")

    post_future_session = fetch_app_session("post-future")
    if post_future_session.get("free_analysis_remaining") != current_remaining:
        raise SystemExit("free quota changed after future no-race live-decision")
else:
    print("future no-race live-decision skipped (quota anchored)")

print("SMOKE2_DONE")
PY
