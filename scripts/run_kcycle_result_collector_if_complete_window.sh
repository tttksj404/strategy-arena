#!/usr/bin/env bash
set -eu

dow="$(date +%u)"
hour="$(date +%H)"
result_hour="${KCYCLE_RESULT_COLLECTION_START_HOUR:-18}"

if [ "$dow" -lt 5 ] || [ "$dow" -gt 7 ] || [ "$hour" -lt "$result_hour" ] || [ "$hour" -gt 23 ]; then
  exit 0
fi

repo_dir="${STRATEGY_ARENA_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"

cd "$repo_dir"
exec "$python_bin" scripts/collect_kcycle_result_outcomes.py \
  --date "$(date +%Y-%m-%d)" \
  --meet "${KCYCLE_RESULT_MEET:-광명}" \
  --races "${KCYCLE_RESULT_RACES:-1-16}"
