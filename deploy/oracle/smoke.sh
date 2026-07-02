#!/usr/bin/env bash
set -eu

base_url="${1:-${ORACLE_BASE_URL:-}}"
if [ -z "$base_url" ]; then
  echo "Usage: $0 https://your-domain-or-ip" >&2
  exit 2
fi

python - "$base_url" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

base = sys.argv[1].rstrip("/")
checks = [
    ("/healthz", {}),
    ("/api/live-decision", {"sport": "keirin", "date": "2026-06-28", "meet": "광명", "race_no": "5"}),
    ("/predict", {"sport": "keirin", "date": "2026-06-28", "meet": "광명", "race_no": "5"}),
]
for path, params in checks:
    url = f"{base}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read().decode("utf-8", "replace")
        parsed = None
        if path == "/api/live-decision":
            parsed = json.loads(body)
            required = ["market_used", "market_risk", "market_odds", "poll_delay_ms", "snapshot_phase"]
            missing = [key for key in required if key not in parsed]
            if missing:
                raise SystemExit(f"live-decision missing fields: {missing}")
            if not isinstance(parsed["market_odds"], list):
                raise SystemExit("live-decision market_odds must be a list")
            if parsed["market_used"] and parsed.get("market_risk", {}).get("level") != "odds_live":
                raise SystemExit("live market_used response must expose market_risk.level=odds_live")
            if urllib.parse.urlparse(base).hostname not in {"127.0.0.1", "localhost"}:
                if parsed.get("market_risk", {}).get("level") == "live_market_blocked":
                    raise SystemExit("Oracle smoke must not expose Render-specific live_market_blocked risk")
        print(json.dumps({
            "url": url,
            "status": response.status,
            "bytes": len(body),
            "has_live_blocked": "live_market_blocked" in body,
            "has_trifecta_text": "삼쌍" in body,
            "has_old_50_candidate": "삼쌍 50% 후보" in body or "market_trifecta_50_candidate" in body,
            "live_contract_ok": bool(parsed is not None),
        }, ensure_ascii=False))
PY
