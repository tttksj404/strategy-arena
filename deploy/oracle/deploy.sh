#!/usr/bin/env bash
set -eu

if [ -z "${ORACLE_HOST:-}" ]; then
  echo "ORACLE_HOST is required" >&2
  exit 2
fi

ORACLE_USER="${ORACLE_USER:-ubuntu}"
ORACLE_PATH="${ORACLE_PATH:-/opt/strategy-arena}"
SSH_TARGET="$ORACLE_USER@$ORACLE_HOST"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude "__pycache__" \
  --exclude ".env" \
  --exclude "deploy/oracle/.env.oracle" \
  --exclude "data/kcycle_trifecta_snapshots*.jsonl" \
  --exclude "data/kcycle_trifecta_snapshots*.jsonl.keys" \
  "$repo_root/" "$SSH_TARGET:$ORACLE_PATH/"

ssh "$SSH_TARGET" "set -eu
  cd '$ORACLE_PATH'
  if [ ! -f deploy/oracle/.env.oracle ]; then
    cp deploy/oracle/.env.oracle.example deploy/oracle/.env.oracle
    echo 'Created deploy/oracle/.env.oracle from example. Fill DATAGOKR_SERVICE_KEY and domain if needed.' >&2
    exit 3
  fi
  docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle up -d --build
  docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle ps
  docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle exec -T app python - <<'PY'
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=10) as response:
    print(response.status, response.read().decode('utf-8', 'replace')[:300])
PY"
