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
        print(json.dumps({
            "url": url,
            "status": response.status,
            "bytes": len(body),
            "has_live_blocked": "live_market_blocked" in body,
            "has_trifecta_text": "삼쌍" in body,
            "has_old_50_candidate": "삼쌍 50% 후보" in body or "market_trifecta_50_candidate" in body,
        }, ensure_ascii=False))
PY
