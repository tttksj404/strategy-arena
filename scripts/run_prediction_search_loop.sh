#!/usr/bin/env bash
set -eu

repo_dir="${STRATEGY_ARENA_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
snapshot_path="${KCYCLE_TRIFECTA_SNAPSHOT_PATH:-$repo_dir/data/kcycle_trifecta_snapshots.jsonl}"

cd "$repo_dir"
mkdir -p "$(dirname "$snapshot_path")" data logs

started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "{\"event\":\"prediction_search_start\",\"started_at\":\"$started_at\",\"snapshot_path\":\"$snapshot_path\"}"

if [ ! -s "$snapshot_path" ]; then
  cat > data/kcycle_trifecta_snapshot_audit_latest.json <<JSON
{
  "ok": false,
  "status": "waiting_for_snapshots",
  "records": 0,
  "snapshot_path": "$snapshot_path",
  "message": "No KCYCLE trifecta snapshots have been collected yet."
}
JSON
  cat > data/kcycle_trifecta_rule_search_results.json <<JSON
{
  "records": 0,
  "status": "waiting_for_snapshots",
  "risk_flags": {
    "requires_more_outcome_linked_snapshots": true
  },
  "rows": []
}
JSON
  cat > data/kcycle_trifecta_rule_search_results.md <<MD
# KCYCLE trifecta rule search

records: 0
status: waiting_for_snapshots
risk_flags: {"requires_more_outcome_linked_snapshots": true}
MD
  echo "{\"event\":\"prediction_search_waiting_for_snapshots\",\"snapshot_path\":\"$snapshot_path\"}"
  exit 0
fi

"$python_bin" scripts/audit_kcycle_trifecta_snapshot_corpus.py \
  --snapshots "$snapshot_path" \
  --out data/kcycle_trifecta_snapshot_audit_latest.json

"$python_bin" scripts/search_kcycle_trifecta_rules.py \
  --snapshots "$snapshot_path" \
  --out-json data/kcycle_trifecta_rule_search_results.json \
  --out-md data/kcycle_trifecta_rule_search_results.md

if [ "${KCYCLE_LATE_MARKET_PULL_EXPERIMENT_ENABLED:-1}" = "1" ]; then
  "$python_bin" scripts/experiment_kcycle_late_market_pull.py \
    --snapshots "$snapshot_path" \
    --out-json data/kcycle_late_market_pull_results.json \
    --out-md docs/kcycle_late_market_pull_results.md
fi

if [ "${KCYCLE_MARKET_TIMING_EXPERIMENT_ENABLED:-1}" = "1" ]; then
  "$python_bin" scripts/experiment_kcycle_market_timing_policy.py \
    --snapshots "$snapshot_path" \
    --out-json data/kcycle_market_timing_policy_results.json \
    --out-md docs/kcycle_market_timing_policy_results.md
fi

if [ "${KCYCLE_FAST_EVOLUTION_SEARCH_ENABLED:-1}" = "1" ]; then
  "$python_bin" scripts/search_kcycle_fast_evolution_trifecta.py \
    --snapshots "$snapshot_path" \
    --out-json data/kcycle_fast_evolution_trifecta_results.json \
    --out-md data/kcycle_fast_evolution_trifecta_results.md
fi

if [ "${KCYCLE_GLOBAL_BREAKTHROUGH_SEARCH_ENABLED:-1}" = "1" ]; then
  "$python_bin" scripts/search_kcycle_global_breakthrough.py \
    --snapshots "$snapshot_path" \
    --out-json data/kcycle_global_breakthrough_results.json \
    --out-md data/kcycle_global_breakthrough_results.md \
    --state-json data/kcycle_global_search_state.json \
    --auto-seed-count "${KCYCLE_GLOBAL_AUTO_SEED_COUNT:-2}" \
    --merge-existing "${KCYCLE_GLOBAL_MERGE_EXISTING:-1}"
fi

if [ "${KCYCLE_DRUG_DISCOVERY_SEARCH_ENABLED:-0}" = "1" ]; then
  "$python_bin" scripts/search_kcycle_drug_discovery_trifecta.py \
    --snapshots "$snapshot_path" \
    --out-json data/kcycle_drug_discovery_trifecta_results.json \
    --out-md data/kcycle_drug_discovery_trifecta_results.md
fi

finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "{\"event\":\"prediction_search_done\",\"finished_at\":\"$finished_at\"}"
