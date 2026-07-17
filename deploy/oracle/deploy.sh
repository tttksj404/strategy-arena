#!/usr/bin/env bash
set -eu

if [ -z "${ORACLE_HOST:-}" ]; then
  echo "ORACLE_HOST is required" >&2
  exit 2
fi

ORACLE_USER="${ORACLE_USER:-ubuntu}"
ORACLE_PATH_CONFIGURED="${ORACLE_PATH:-}"
ORACLE_PATH="${ORACLE_PATH_CONFIGURED:-}"
ORACLE_SSH_PORT="${ORACLE_SSH_PORT:-}"
SSH_TARGET="$ORACLE_USER@$ORACLE_HOST"
SSH_OPTS=()
RSYNC_SSH="ssh"
if [ -n "$ORACLE_SSH_PORT" ]; then
  SSH_OPTS=(-p "$ORACLE_SSH_PORT")
  RSYNC_SSH="ssh -p $ORACLE_SSH_PORT"
fi

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

read_remote_env_value() {
  remote_file="$1"
  key="$2"
  ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$SSH_TARGET" "if [ -f '$remote_file' ]; then awk -F= -v wanted='$key' '\$1 == wanted { sub(/^[^=]*=/, \"\"); print; exit }' '$remote_file'; fi" || true
}

detect_remote_oracle_path() {
  ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$SSH_TARGET" "set -eu
    for candidate in /opt/strategy-arena /home/ubuntu/strategy-arena; do
      if [ -d \"\$candidate\" ]; then
        printf '%s\n' \"\$candidate\"
        exit 0
      fi
    done
    exit 44
  "
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

if ! ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} -o BatchMode=yes -o ConnectTimeout="${ORACLE_CONNECT_TIMEOUT:-12}" "$SSH_TARGET" "true"; then
  echo "Oracle SSH is not reachable: $SSH_TARGET${ORACLE_SSH_PORT:+ port $ORACLE_SSH_PORT}" >&2
  exit 4
fi

if [ -z "$ORACLE_PATH_CONFIGURED" ]; then
  ORACLE_PATH="$(detect_remote_oracle_path || true)"
  if [ -z "$ORACLE_PATH" ]; then
    echo "ORACLE_PATH is not set and no existing remote path was found (/opt/strategy-arena, /home/ubuntu/strategy-arena)" >&2
    exit 7
  fi
  echo "Detected ORACLE_PATH=$ORACLE_PATH"
fi

remote_compose_cmd="$(ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$SSH_TARGET" "if docker compose version >/dev/null 2>&1; then printf 'docker compose'; elif command -v docker-compose >/dev/null 2>&1; then printf 'docker-compose'; else exit 1; fi")" || {
  echo "Docker Compose is required on Oracle host: $SSH_TARGET" >&2
  exit 6
}

datagokr_key="$(resolve_datagokr_key || true)"
if [ -z "$datagokr_key" ]; then
  echo "DATAGOKR_SERVICE_KEY is required locally or in \$HOME/keirin/.env / \$HOME/kra/.env" >&2
  exit 5
fi

remote_postgres_password="$(read_remote_env_value "$ORACLE_PATH/deploy/oracle/.env.oracle" POSTGRES_PASSWORD)"
postgres_password="${POSTGRES_PASSWORD:-$remote_postgres_password}"
if [ -z "$postgres_password" ]; then
  postgres_password="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
fi
remote_site_address="$(read_remote_env_value "$ORACLE_PATH/deploy/oracle/.env.oracle" ORACLE_SITE_ADDRESS)"
site_address="${ORACLE_SITE_ADDRESS:-$remote_site_address}"
if [ -z "$site_address" ]; then
  site_address=":80"
fi
remote_acme_email="$(read_remote_env_value "$ORACLE_PATH/deploy/oracle/.env.oracle" ACME_EMAIL)"
acme_email="${ACME_EMAIL:-$remote_acme_email}"
if [ -z "$acme_email" ]; then
  acme_email="admin@example.com"
fi

racelens_admin_token="${RACELENS_ADMIN_TOKEN:-$(read_remote_env_value "$ORACLE_PATH/deploy/oracle/.env.oracle" RACELENS_ADMIN_TOKEN)}"
if [ -z "$racelens_admin_token" ]; then
  racelens_admin_token="$(LC_ALL=C tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 40)"
fi

tmp_env="$(mktemp)"
chmod 600 "$tmp_env"
cat > "$tmp_env" <<EOF
ORACLE_SITE_ADDRESS=$site_address
ACME_EMAIL=$acme_email

DATAGOKR_SERVICE_KEY=$datagokr_key
POSTGRES_PASSWORD=$postgres_password

KCYCLE_ENABLED=1
KCYCLE_TRIFECTA_ENABLED=1
KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC=${KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC:-10}
KCYCLE_COLLECTOR_INTERVAL_SEC=${KCYCLE_COLLECTOR_INTERVAL_SEC:-15}
SEARCH_LOOP_INTERVAL_SEC=${SEARCH_LOOP_INTERVAL_SEC:-30}
RACELENS_ENV=${RACELENS_ENV:-production}
RACELENS_FORCE_PRO=${RACELENS_FORCE_PRO:-0}
RACELENS_ADMIN_TOKEN=$racelens_admin_token
RACELENS_SUPPORT_EMAIL=${RACELENS_SUPPORT_EMAIL:-tttksj@gmail.com}
# Per-IP request ceiling. Safe for normal use while protecting the shared API key.
RACELENS_LIVE_DECISION_IP_PER_MIN_CAP=${RACELENS_LIVE_DECISION_IP_PER_MIN_CAP:-60}
RACELENS_FREE_DAILY_ANALYSIS_LIMIT=${RACELENS_FREE_DAILY_ANALYSIS_LIMIT:-3}
RACELENS_REWARDED_ADS_ENABLED=${RACELENS_REWARDED_ADS_ENABLED:-0}
RACELENS_ADMOB_REWARDED_AD_UNIT_ID=${RACELENS_ADMOB_REWARDED_AD_UNIT_ID:-}

WEB_CONCURRENCY=${WEB_CONCURRENCY:-2}
WEB_THREADS=${WEB_THREADS:-2}
WEB_TIMEOUT=${WEB_TIMEOUT:-120}
EOF

rsync -az --delete \
  -e "$RSYNC_SSH" \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude ".runtime" \
  --exclude ".codex-fable5" \
  --exclude ".env" \
  --exclude "deploy/oracle/.env.oracle" \
  --exclude "logs" \
  --exclude "backups" \
  --exclude "mobile/node_modules" \
  --exclude "mobile/dist" \
  --exclude "mobile/.expo" \
  --exclude "data/kcycle_trifecta_snapshots*.jsonl" \
  --exclude "data/kcycle_trifecta_snapshots*.jsonl.keys" \
  "$repo_root/" "$SSH_TARGET:$ORACLE_PATH/"

ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$SSH_TARGET" "set -eu; mkdir -p '$ORACLE_PATH/deploy/oracle'; umask 077; cat > '$ORACLE_PATH/deploy/oracle/.env.oracle'" < "$tmp_env"

ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$SSH_TARGET" "set -eu
  cd '$ORACLE_PATH'
  $remote_compose_cmd -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle up -d --build
  $remote_compose_cmd -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle ps
  $remote_compose_cmd -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle exec -T app python - <<'PY'
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=10) as response:
    print(response.status, response.read().decode('utf-8', 'replace')[:300])
PY"

# bare IPv4는 Caddy 인증서 SNI와 안 맞아 TLS가 거부되므로 sslip.io 도메인으로 스모크
if [ -z "${ORACLE_SMOKE_URL:-}" ] && printf '%s' "$ORACLE_HOST" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
  ORACLE_SMOKE_URL="https://$(printf '%s' "$ORACLE_HOST" | tr '.' '-').sslip.io"
fi
smoke_url="${ORACLE_SMOKE_URL:-http://$ORACLE_HOST}"
smoke_attempt=1
while :; do
  if "$repo_root/deploy/oracle/smoke.sh" "$smoke_url"; then
    break
  fi
  if [ "$smoke_attempt" -ge 3 ]; then
    exit 1
  fi
  echo "Smoke failed on attempt $smoke_attempt/3; retrying in 5s" >&2
  sleep 5
  smoke_attempt=$((smoke_attempt + 1))
done
