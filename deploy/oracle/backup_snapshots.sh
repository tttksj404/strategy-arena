#!/usr/bin/env bash
set -eu

repo_dir="${STRATEGY_ARENA_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}"
backup_dir="${STRATEGY_ARENA_BACKUP_DIR:-$repo_dir/backups}"
stamp="$(date -u +"%Y%m%dT%H%M%SZ")"

mkdir -p "$backup_dir"
cd "$repo_dir"

tar -czf "$backup_dir/strategy-arena-data-$stamp.tar.gz" \
  data/kcycle_trifecta_snapshots.jsonl \
  data/kcycle_trifecta_snapshots.jsonl.keys \
  data/kcycle_trifecta_rule_search_results.json \
  data/kcycle_trifecta_snapshot_audit_latest.json \
  data/kcycle_trifecta_rule_search_results.md 2>/dev/null || true

find "$backup_dir" -name "strategy-arena-data-*.tar.gz" -mtime +14 -delete
echo "{\"event\":\"backup_done\",\"backup_dir\":\"$backup_dir\",\"stamp\":\"$stamp\"}"
