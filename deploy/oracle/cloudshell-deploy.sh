#!/usr/bin/env bash
set -euo pipefail

HOST="${RACELENS_ORACLE_HOST:-168.107.2.218}"
USER="${RACELENS_ORACLE_USER:-ubuntu}"
KEY="${RACELENS_ORACLE_KEY:-$HOME/.ssh/racelens_oracle_rsa}"
REPO_URL="${RACELENS_REPO_URL:-https://github.com/tttksj404/strategy-arena.git}"
BRANCH="${RACELENS_BRANCH:-main}"
BASE="${RACELENS_REMOTE_BASE:-/home/ubuntu}"
APP_DIR="${RACELENS_REMOTE_APP_DIR:-$BASE/strategy-arena}"
FREE_LIMIT="${RACELENS_FREE_DAILY_ANALYSIS_LIMIT:-3}"
IP_CAP="${RACELENS_LIVE_DECISION_IP_PER_MIN_CAP:-60}"
REWARDED_DAILY_CAP="${RACELENS_REWARDED_AD_DAILY_CAP:-20}"
REWARDED_IP_CAP="${RACELENS_REWARDED_AD_IP_PER_MIN_CAP:-20}"
REWARDED_ADS_ENABLED="${RACELENS_REWARDED_ADS_ENABLED:-0}"
REWARDED_AD_UNIT_ID="${RACELENS_ADMOB_REWARDED_AD_UNIT_ID:-}"
SUPPORT_EMAIL="${RACELENS_SUPPORT_EMAIL:-tttksj@gmail.com}"
BASE_URL="${RACELENS_BASE_URL:-https://168-107-2-218.sslip.io}"

ssh_opts=(
  -i "$KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=12
  -o PubkeyAcceptedKeyTypes=+ssh-rsa,rsa-sha2-256,rsa-sha2-512
)

usage() {
  cat <<'EOF'
Usage: cloudshell-deploy.sh [--check|--deploy]

Run this from OCI Cloud Shell or a GitHub Actions runner. Default action is --deploy.

Environment overrides:
  RACELENS_BRANCH
  RACELENS_REPO_URL
  RACELENS_ORACLE_HOST
  RACELENS_ORACLE_USER
  RACELENS_ORACLE_KEY
  RACELENS_FREE_DAILY_ANALYSIS_LIMIT
  RACELENS_LIVE_DECISION_IP_PER_MIN_CAP
  RACELENS_REWARDED_AD_DAILY_CAP
  RACELENS_REWARDED_AD_IP_PER_MIN_CAP
  RACELENS_REWARDED_ADS_ENABLED
  RACELENS_ADMOB_REWARDED_AD_UNIT_ID
  RACELENS_SUPPORT_EMAIL
  RACELENS_BASE_URL
EOF
}

run_ssh() {
  ssh "${ssh_opts[@]}" "$USER@$HOST" "$@"
}

action="${1:---deploy}"
if [ "$action" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f "$KEY" ]; then
  echo "missing Oracle deploy key: $KEY" >&2
  echo "See docs/oracle_cloudshell_deploy.md for the Cloud Shell and GitHub Actions deploy paths." >&2
  exit 2
fi

if [ "$action" = "--check" ]; then
  run_ssh 'echo SSH_OK; hostname; whoami; test -d /home/ubuntu/strategy-arena && echo APP_DIR_OK'
  curl -skf "$BASE_URL/healthz" >/dev/null
  curl -skf "$BASE_URL/legal/support" >/dev/null
  echo "CHECK_OK $BASE_URL"
  exit 0
fi

if [ "$action" != "--deploy" ]; then
  usage >&2
  exit 2
fi

if [ -n "$REWARDED_AD_UNIT_ID" ] && ! [[ "$REWARDED_AD_UNIT_ID" =~ ^ca-app-pub-[0-9]+/[0-9]+$ ]]; then
  echo "RACELENS_ADMOB_REWARDED_AD_UNIT_ID has an invalid format" >&2
  exit 2
fi
if [ "$REWARDED_ADS_ENABLED" = "1" ] && [ -z "$REWARDED_AD_UNIT_ID" ]; then
  echo "RACELENS_ADMOB_REWARDED_AD_UNIT_ID is required when rewarded ads are enabled" >&2
  exit 2
fi

remote_script="$(mktemp)"
cat > "$remote_script" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

repo_url="$1"
branch="$2"
base="$3"
old="$4"
free_limit="$5"
ip_cap="$6"
support_email="$7"
rewarded_daily_cap="$8"
rewarded_ip_cap="$9"
rewarded_ads_enabled="${10}"
rewarded_ad_unit_id="${11}"

stamp="$(date +%Y%m%d%H%M%S)"
next="$base/strategy-arena.next"
backup="$base/strategy-arena.backup.$stamp"
env_old="$old/deploy/oracle/.env.oracle"
env_next="$next/deploy/oracle/.env.oracle"

set_env() {
  key="$1"
  value="$2"
  file="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
}

echo "DEPLOY_START $(date -Is)"
echo "repo=$repo_url branch=$branch old=$old"
if [ ! -f "$env_old" ]; then
  echo "missing env: $env_old" >&2
  exit 2
fi

rm -rf "$next"
git clone --depth 1 --branch "$branch" "$repo_url" "$next"
install -m 600 "$env_old" "$env_next"
set_env RACELENS_LIVE_DECISION_IP_PER_MIN_CAP "$ip_cap" "$env_next"
set_env RACELENS_FREE_DAILY_ANALYSIS_LIMIT "$free_limit" "$env_next"
set_env RACELENS_FORCE_PRO "0" "$env_next"
set_env RACELENS_REWARDED_ADS_ENABLED "$rewarded_ads_enabled" "$env_next"
set_env RACELENS_ADMOB_REWARDED_AD_UNIT_ID "$rewarded_ad_unit_id" "$env_next"
set_env RACELENS_REWARDED_AD_DAILY_CAP "$rewarded_daily_cap" "$env_next"
set_env RACELENS_REWARDED_AD_IP_PER_MIN_CAP "$rewarded_ip_cap" "$env_next"
set_env RACELENS_SUPPORT_EMAIL "$support_email" "$env_next"

cd "$next"
echo "NEW_HEAD=$(git rev-parse --short HEAD)"
compose="docker compose -p oracle -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle"
echo "BUILD_START $(date -Is)"
$compose build

echo "SWAP_START $(date -Is)"
mv "$old" "$backup"
mv "$next" "$old"

cd "$old"
compose="docker compose -p oracle -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle"
echo "UP_START $(date -Is)"
$compose up -d
echo "PS_START"
$compose ps
echo "LOCAL_HEALTH"
$compose exec -T app python - <<'PY'
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=15) as response:
    print(response.status, response.read().decode("utf-8", "replace")[:500])
PY
echo "ENV_QUOTA"
grep -E '^(RACELENS_FORCE_PRO|RACELENS_LIVE_DECISION_IP_PER_MIN_CAP|RACELENS_FREE_DAILY_ANALYSIS_LIMIT|RACELENS_REWARDED_ADS_ENABLED|RACELENS_ADMOB_REWARDED_AD_UNIT_ID|RACELENS_REWARDED_AD_DAILY_CAP|RACELENS_REWARDED_AD_IP_PER_MIN_CAP)=' deploy/oracle/.env.oracle
echo "DEPLOY_DONE $(date -Is) backup=$backup"
REMOTE_SCRIPT
trap 'rm -f "$remote_script"' EXIT

cat "$remote_script" | ssh "${ssh_opts[@]}" "$USER@$HOST" \
  "cat > /tmp/racelens-remote-deploy.sh; chmod +x /tmp/racelens-remote-deploy.sh; /tmp/racelens-remote-deploy.sh '$REPO_URL' '$BRANCH' '$BASE' '$APP_DIR' '$FREE_LIMIT' '$IP_CAP' '$SUPPORT_EMAIL' '$REWARDED_DAILY_CAP' '$REWARDED_IP_CAP' '$REWARDED_ADS_ENABLED' '$REWARDED_AD_UNIT_ID'"

curl -skf "$BASE_URL/healthz" >/dev/null
curl -skf "$BASE_URL/legal/support" >/dev/null
run_ssh "cd '$APP_DIR' && bash deploy/oracle/smoke.sh '$BASE_URL'"
echo "RACELENS_DEPLOY_OK $BASE_URL branch=$BRANCH free_limit=$FREE_LIMIT ip_cap=$IP_CAP rewarded_ads=$REWARDED_ADS_ENABLED rewarded_daily_cap=$REWARDED_DAILY_CAP rewarded_ip_cap=$REWARDED_IP_CAP"
