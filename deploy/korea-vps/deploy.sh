#!/usr/bin/env bash
set -eu

if [ -z "${VPS_HOST:-}" ]; then
  echo "VPS_HOST is required" >&2
  exit 2
fi

VPS_USER="${VPS_USER:-ubuntu}"
VPS_PATH="${VPS_PATH:-/opt/strategy-arena}"
SSH_TARGET="$VPS_USER@$VPS_HOST"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
tmp_env=""

cleanup() {
  if [ -n "$tmp_env" ] && [ -f "$tmp_env" ]; then
    rm -f "$tmp_env"
  fi
}
trap cleanup EXIT

read_env_value() {
  file="$1"
  key="$2"
  if [ ! -f "$file" ]; then
    return 1
  fi
  awk -F= -v wanted="$key" '$1 == wanted { sub(/^[^=]*=/, ""); print; exit }' "$file"
}

resolve_datagokr_key() {
  if [ -n "${DATAGOKR_SERVICE_KEY:-}" ]; then
    printf '%s\n' "$DATAGOKR_SERVICE_KEY"
    return 0
  fi
  for candidate in "${LOCAL_DATAGOKR_ENV:-}" "$HOME/keirin/.env" "$HOME/kra/.env"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
      value="$(read_env_value "$candidate" DATAGOKR_SERVICE_KEY || true)"
      if [ -n "$value" ]; then
        printf '%s\n' "$value"
        return 0
      fi
    fi
  done
  return 1
}

if ! ssh -o BatchMode=yes -o ConnectTimeout="${VPS_CONNECT_TIMEOUT:-12}" "$SSH_TARGET" "true"; then
  echo "Korea VPS SSH is not reachable: $SSH_TARGET" >&2
  exit 4
fi

if ! ssh "$SSH_TARGET" "command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1"; then
  if [ "${VPS_BOOTSTRAP_DOCKER:-1}" = "1" ]; then
    ssh "$SSH_TARGET" "bash -s" < "$repo_root/deploy/korea-vps/bootstrap_ubuntu_docker.sh"
  fi
fi

if ! ssh "$SSH_TARGET" "command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1"; then
  echo "Docker Engine and the Docker Compose plugin are required on $SSH_TARGET" >&2
  exit 6
fi

datagokr_key="$(resolve_datagokr_key || true)"
if [ -z "$datagokr_key" ]; then
  echo "DATAGOKR_SERVICE_KEY is required locally or in \$HOME/keirin/.env / \$HOME/kra/.env" >&2
  exit 5
fi

tmp_env="$(mktemp)"
chmod 600 "$tmp_env"
cat > "$tmp_env" <<EOF
ORACLE_SITE_ADDRESS=${VPS_SITE_ADDRESS:-${ORACLE_SITE_ADDRESS:-:80}}
ACME_EMAIL=${ACME_EMAIL:-admin@example.com}

DATAGOKR_SERVICE_KEY=$datagokr_key

KCYCLE_ENABLED=1
KCYCLE_TRIFECTA_ENABLED=1
KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC=${KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC:-10}
KCYCLE_COLLECTOR_INTERVAL_SEC=${KCYCLE_COLLECTOR_INTERVAL_SEC:-15}
SEARCH_LOOP_INTERVAL_SEC=${SEARCH_LOOP_INTERVAL_SEC:-3600}

WEB_CONCURRENCY=${WEB_CONCURRENCY:-2}
WEB_THREADS=${WEB_THREADS:-2}
WEB_TIMEOUT=${WEB_TIMEOUT:-120}
EOF

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude "__pycache__" \
  --exclude ".env" \
  --exclude "deploy/oracle/.env.oracle" \
  --exclude "data/kcycle_trifecta_snapshots*.jsonl" \
  --exclude "data/kcycle_trifecta_snapshots*.jsonl.keys" \
  "$repo_root/" "$SSH_TARGET:$VPS_PATH/"

ssh "$SSH_TARGET" "set -eu; mkdir -p '$VPS_PATH/deploy/oracle'; umask 077; cat > '$VPS_PATH/deploy/oracle/.env.oracle'" < "$tmp_env"

ssh "$SSH_TARGET" "set -eu
  cd '$VPS_PATH'
  docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle up -d --build
  docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle ps
  docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle exec -T app python - <<'PY'
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=10) as response:
    print(response.status, response.read().decode('utf-8', 'replace')[:300])
PY"

smoke_url="${VPS_SMOKE_URL:-http://$VPS_HOST}"
"$repo_root/deploy/oracle/smoke.sh" "$smoke_url"
