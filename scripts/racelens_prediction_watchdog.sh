#!/usr/bin/env bash
set -u

mode="${1:-search}"
repo_dir="${STRATEGY_ARENA_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"

cd "$repo_dir" || exit 1
mkdir -p logs .runtime/prediction-search

case "$mode" in
  search)
    interval_sec="${SEARCH_LOOP_INTERVAL_SEC:-0}"
    marker="racelens-prediction-watchdog-search"
    pid_path=".runtime/prediction-search/search-loop.pid"
    ;;
  collector)
    interval_sec="${KCYCLE_COLLECTOR_INTERVAL_SEC:-10}"
    marker="racelens-prediction-watchdog-collector"
    pid_path=".runtime/prediction-search/collector.pid"
    ;;
  *)
    echo "{\"event\":\"watchdog_invalid_mode\",\"mode\":\"$mode\"}" >&2
    exit 2
    ;;
esac

echo "$$" > "$pid_path"
echo "{\"event\":\"watchdog_start\",\"mode\":\"$mode\",\"marker\":\"$marker\",\"interval_sec\":$interval_sec,\"repo\":\"$repo_dir\"}"

while true; do
  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "{\"event\":\"watchdog_cycle_start\",\"mode\":\"$mode\",\"started_at\":\"$started_at\"}"

  rc=0
  if [ "$mode" = "search" ]; then
    PYTHON_BIN="$python_bin" scripts/run_prediction_search_loop.sh || rc=$?
    "$python_bin" scripts/update_prediction_feedback.py || rc=$?
  else
    PYTHON_BIN="$python_bin" scripts/run_kcycle_trifecta_collector_if_race_window.sh || rc=$?
    "$python_bin" scripts/collect_kcycle_result_outcomes.py --date "$(date +%Y-%m-%d)" --meet "${KCYCLE_RESULT_MEET:-광명}" --races "${KCYCLE_RESULT_RACES:-1-16}" || rc=$?
    "$python_bin" scripts/update_prediction_feedback.py || rc=$?
  fi

  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "{\"event\":\"watchdog_cycle_done\",\"mode\":\"$mode\",\"finished_at\":\"$finished_at\",\"rc\":$rc,\"next_delay_sec\":$interval_sec}"
  sleep "$interval_sec"
done
