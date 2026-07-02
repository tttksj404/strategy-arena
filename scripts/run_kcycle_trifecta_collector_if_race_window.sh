#!/usr/bin/env bash
set -u

dow="$(date +%u)"
hour="$(date +%H)"

if [ "$dow" -lt 5 ] || [ "$dow" -gt 7 ] || [ "$hour" -lt 10 ] || [ "$hour" -gt 19 ]; then
  exit 0
fi

cd "/Users/tttksj/github-portfolio-docs-work/strategy-arena" || exit 1
exec /usr/bin/python3 scripts/collect_kcycle_trifecta_snapshots.py --date "$(date +%Y-%m-%d)" --races all
