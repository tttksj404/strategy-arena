#!/usr/bin/env bash
set -u

dow="$(date +%u)"
hour="$(date +%H)"

if [ "$dow" -lt 5 ] || [ "$dow" -gt 7 ] || [ "$hour" -lt 10 ] || [ "$hour" -gt 19 ]; then
  exit 0
fi

repo_dir="${STRATEGY_ARENA_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"

cd "$repo_dir" || exit 1
exec "$python_bin" scripts/collect_kcycle_trifecta_snapshots.py --date "$(date +%Y-%m-%d)" --races all
