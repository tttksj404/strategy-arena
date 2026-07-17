#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$MOBILE_DIR/.." && pwd)"
STATE_DIR="${RACELENS_PREVIEW_STATE_DIR:-$REPO_DIR/.runtime/racelens-preview}"
BACKEND_PORT="${RACELENS_BACKEND_PORT:-8010}"
PROXY_PORT="${RACELENS_PREVIEW_PORT:-4173}"
BACKEND_URL="http://127.0.0.1:$BACKEND_PORT"
PROXY_URL="http://127.0.0.1:$PROXY_PORT"
PYTHON_BIN="${RACELENS_PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${RACELENS_PYTHON_BIN:-python3}"
fi

mkdir -p "$STATE_DIR"
USE_LAUNCHD=0
if [[ "$(uname -s)" == "Darwin" && "${RACELENS_PREVIEW_USE_LAUNCHD:-1}" == "1" && -x /bin/launchctl ]]; then
  USE_LAUNCHD=1
fi
LAUNCH_DOMAIN="gui/$(id -u)"

load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    set -a
    source "$file"
    set +a
  fi
}

load_env_file "$HOME/kra/.env"
load_env_file "$HOME/keirin/.env"

pid_file() {
  printf '%s/%s.pid\n' "$STATE_DIR" "$1"
}

log_file() {
  printf '%s/%s.log\n' "$STATE_DIR" "$1"
}

shell_quote() {
  printf '%q' "$1"
}

xml_escape() {
  /usr/bin/python3 -c 'import html, sys; print(html.escape(sys.stdin.read(), quote=False), end="")'
}

launch_label() {
  printf 'com.racelens.preview.%s\n' "$1"
}

launch_plist() {
  printf '%s/Library/LaunchAgents/%s.plist\n' "$HOME" "$(launch_label "$1")"
}

launch_pid() {
  local label
  label="$(launch_label "$1")"
  launchctl print "$LAUNCH_DOMAIN/$label" 2>/dev/null | awk -F'= ' '/pid =/{print $2; exit}' | tr -d ';'
}

write_launch_agent() {
  local name="$1"
  local command="$2"
  local label plist escaped
  label="$(launch_label "$name")"
  plist="$(launch_plist "$name")"
  escaped="$(printf '%s' "$command" | xml_escape)"
  mkdir -p "$(dirname "$plist")"
  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>$escaped</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$(log_file "$name")</string>
  <key>StandardErrorPath</key>
  <string>$(log_file "$name")</string>
</dict>
</plist>
PLIST
}

start_launch_agent() {
  local name="$1"
  local command="$2"
  local plist pid
  plist="$(launch_plist "$name")"
  write_launch_agent "$name" "$command"
  launchctl bootout "$LAUNCH_DOMAIN" "$plist" >/dev/null 2>&1 || true
  launchctl bootstrap "$LAUNCH_DOMAIN" "$plist"
  launchctl kickstart -k "$LAUNCH_DOMAIN/$(launch_label "$name")" >/dev/null 2>&1 || true
  sleep 1
  pid="$(launch_pid "$name")"
  [[ -n "$pid" ]] && remember_pid "$name" "$pid"
}

stop_launch_agent() {
  local name="$1"
  local plist
  plist="$(launch_plist "$name")"
  launchctl bootout "$LAUNCH_DOMAIN" "$plist" >/dev/null 2>&1 || true
  rm -f "$(pid_file "$name")"
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
  local file
  file="$(pid_file "$1")"
  [[ -f "$file" ]] && tr -d '[:space:]' < "$file"
}

port_pid() {
  local port="$1"
  lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -1 || true
}

remember_pid() {
  local name="$1"
  local pid="$2"
  printf '%s\n' "$pid" > "$(pid_file "$name")"
}

stop_pid_file() {
  local name="$1"
  local pid
  if [[ "$USE_LAUNCHD" == "1" ]]; then
    stop_launch_agent "$name"
    return 0
  fi
  pid="$(read_pid "$name" || true)"
  if pid_alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    sleep 0.5
    if pid_alive "$pid"; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$(pid_file "$name")"
}

force_free_port() {
  local port="$1"
  local pid
  pid="$(port_pid "$port")"
  if pid_alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    sleep 0.5
  fi
}

force_stop_tunnel() {
  pkill -f "cloudflared.*127.0.0.1:$PROXY_PORT" 2>/dev/null || true
  pkill -f "localtunnel.*--port $PROXY_PORT" 2>/dev/null || true
  pkill -f "lt .*--port $PROXY_PORT" 2>/dev/null || true
}

wait_http() {
  local url="$1"
  local tries="${2:-40}"
  local index
  for ((index = 1; index <= tries; index += 1)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "timeout waiting for $url" >&2
  return 1
}

url_reachable() {
  local url="$1"
  if curl -fsS --max-time 4 "$url" >/dev/null 2>&1; then
    return 0
  fi
  local host ip
  host="${url#https://}"
  host="${host#http://}"
  host="${host%%/*}"
  if [[ "$url" == https://* ]] && command -v dig >/dev/null 2>&1; then
    ip="$(dig +short "$host" | grep -E '^[0-9.]+$' | head -1 || true)"
    if [[ -n "$ip" ]]; then
      curl -fsS --max-time 4 --resolve "$host:443:$ip" "$url" >/dev/null 2>&1
      return $?
    fi
  fi
  return 1
}

ensure_dist() {
  if [[ ! -f "$MOBILE_DIR/dist/index.html" || "${RACELENS_PREVIEW_BUILD:-0}" == "1" ]]; then
    (cd "$MOBILE_DIR" && npm run export:web)
  fi
}

start_backend() {
  local existing
  existing="$(port_pid "$BACKEND_PORT")"
  if pid_alive "$existing"; then
    remember_pid backend "$existing"
    return 0
  fi
  if [[ -z "${DATAGOKR_SERVICE_KEY:-}" ]]; then
    echo "DATAGOKR_SERVICE_KEY is missing. Add it to ~/keirin/.env or ~/kra/.env." >&2
    return 1
  fi
  if [[ "$USE_LAUNCHD" == "1" ]]; then
    local repo python command
    repo="$(shell_quote "$REPO_DIR")"
    python="$(shell_quote "$PYTHON_BIN")"
    command="cd $repo; if [[ -f \"\$HOME/kra/.env\" ]]; then set -a; source \"\$HOME/kra/.env\"; set +a; fi; if [[ -f \"\$HOME/keirin/.env\" ]]; then set -a; source \"\$HOME/keirin/.env\"; set +a; fi; export KCYCLE_ENABLED=\"\${KCYCLE_ENABLED:-1}\"; export KCYCLE_TRIFECTA_ENABLED=\"\${KCYCLE_TRIFECTA_ENABLED:-1}\"; export FLASK_ENV=production; exec $python -m flask --app app run --host 127.0.0.1 --port $BACKEND_PORT"
    start_launch_agent backend "$command"
    wait_http "$BACKEND_URL/api/app-session"
    return 0
  fi
  (
    cd "$REPO_DIR"
    nohup env \
      DATAGOKR_SERVICE_KEY="$DATAGOKR_SERVICE_KEY" \
      KCYCLE_ENABLED="${KCYCLE_ENABLED:-1}" \
      KCYCLE_TRIFECTA_ENABLED="${KCYCLE_TRIFECTA_ENABLED:-1}" \
      FLASK_ENV=production \
      "$PYTHON_BIN" -m flask --app app run --host 127.0.0.1 --port "$BACKEND_PORT" \
      >"$(log_file backend)" 2>&1 &
    remember_pid backend "$!"
  )
  wait_http "$BACKEND_URL/api/app-session"
}

start_proxy() {
  local existing
  existing="$(port_pid "$PROXY_PORT")"
  if pid_alive "$existing"; then
    remember_pid proxy "$existing"
    return 0
  fi
  ensure_dist
  if [[ "$USE_LAUNCHD" == "1" ]]; then
    local mobile command
    mobile="$(shell_quote "$MOBILE_DIR")"
    command="cd $mobile; export RACELENS_PREVIEW_PORT=$PROXY_PORT; export RACELENS_UPSTREAM_API=$BACKEND_URL; exec node scripts/preview-proxy-server.cjs"
    start_launch_agent proxy "$command"
    wait_http "$PROXY_URL/"
    return 0
  fi
  (
    cd "$MOBILE_DIR"
    nohup env \
      RACELENS_PREVIEW_PORT="$PROXY_PORT" \
      RACELENS_UPSTREAM_API="$BACKEND_URL" \
      node scripts/preview-proxy-server.cjs \
      >"$(log_file proxy)" 2>&1 &
    remember_pid proxy "$!"
  )
  wait_http "$PROXY_URL/"
}

extract_tunnel_url() {
  local tunnel_log
  tunnel_log="$(log_file tunnel)"
  if [[ -f "$tunnel_log" ]]; then
    grep -Eo 'https://[-a-zA-Z0-9]+[.]trycloudflare[.]com' "$tunnel_log" | tail -1 || true
  fi
}

start_tunnel() {
  local pid url
  pid="$(read_pid tunnel || true)"
  if pid_alive "$pid"; then
    url="$(extract_tunnel_url)"
    if [[ -n "$url" ]]; then
      printf '%s\n' "$url" > "$STATE_DIR/public_url"
      if url_reachable "$url/healthz"; then
        return 0
      fi
    fi
    stop_pid_file tunnel
    force_stop_tunnel
  fi
  : >"$(log_file tunnel)"
  rm -f "$STATE_DIR/public_url"
  if [[ "$USE_LAUNCHD" == "1" ]]; then
    local mobile command
    mobile="$(shell_quote "$MOBILE_DIR")"
    command="cd $mobile; exec npx --yes cloudflared tunnel --protocol http2 --url $PROXY_URL"
    start_launch_agent tunnel "$command"
  else
  (
    cd "$MOBILE_DIR"
    nohup npx --yes cloudflared tunnel --protocol http2 --url "$PROXY_URL" \
      >"$(log_file tunnel)" 2>&1 &
    remember_pid tunnel "$!"
  )
  fi
  local index
  for ((index = 1; index <= 120; index += 1)); do
    url="$(extract_tunnel_url)"
    if [[ -n "$url" ]]; then
      printf '%s\n' "$url" > "$STATE_DIR/public_url"
      if url_reachable "$url/healthz"; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "cloudflared tunnel URL was not reachable" >&2
  return 1
}

public_url() {
  if [[ -f "$STATE_DIR/public_url" ]]; then
    tr -d '[:space:]' < "$STATE_DIR/public_url"
  fi
}

preview_healthy() {
  local url
  wait_http "$BACKEND_URL/api/app-session" 2 >/dev/null 2>&1 || return 1
  wait_http "$PROXY_URL/healthz" 2 >/dev/null 2>&1 || return 1
  url="$(public_url)"
  [[ -n "$url" ]] || return 1
  url_reachable "$url/healthz"
}

watch_once() {
  start_backend
  start_proxy
  if ! wait_http "$BACKEND_URL/api/app-session" 2 >/dev/null 2>&1; then
    stop_pid_file backend
    force_free_port "$BACKEND_PORT"
    start_backend
  fi
  if ! wait_http "$PROXY_URL/healthz" 2 >/dev/null 2>&1; then
    stop_pid_file proxy
    force_free_port "$PROXY_PORT"
    start_proxy
  fi
  if ! preview_healthy; then
    stop_pid_file tunnel
    force_stop_tunnel
    start_tunnel
  fi
  status
}

watch_loop() {
  local interval
  interval="${RACELENS_PREVIEW_WATCH_INTERVAL:-20}"
  while true; do
    if ! watch_once >>"$(log_file watchdog)" 2>&1; then
      printf '[%s] preview watchdog recovery failed\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$(log_file watchdog)"
    fi
    sleep "$interval"
  done
}

install_watchdog() {
  local repo command
  repo="$(shell_quote "$REPO_DIR")"
  command="cd $repo; export RACELENS_PREVIEW_USE_LAUNCHD=1; exec mobile/scripts/live-preview-control.sh watch"
  start_launch_agent watchdog "$command"
}

smoke_url() {
  local base="$1"
  "$PYTHON_BIN" - "$base" <<'PY'
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request

base = sys.argv[1].rstrip("/")
query = urllib.parse.urlencode({
    "sport": "keirin",
    "date": "2026-07-03",
    "meet": "광명",
    "race_no": "5",
})
url = f"{base}/api/live-decision?{query}"
device_id = f"preview-smoke-{int(time.time() * 1000)}"
started = time.time()
headers = {
    "X-RaceLens-Device-Id": device_id,
    "X-RaceLens-Platform": "web-preview",
}

def fetch_json(target_url, request_headers):
    request = urllib.request.Request(target_url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as first_error:
        parsed = urllib.parse.urlparse(target_url)
        if parsed.scheme != "https":
            raise
        dig = subprocess.run(
            ["dig", "+short", parsed.hostname or ""],
            check=False,
            capture_output=True,
            text=True,
        )
        ip = next((line.strip() for line in dig.stdout.splitlines() if line.strip().replace(".", "").isdigit()), "")
        if not ip:
            raise first_error
        command = [
            "curl",
            "-fsS",
            "--max-time",
            "12",
            "--resolve",
            f"{parsed.hostname}:443:{ip}",
            target_url,
        ]
        for key, value in request_headers.items():
            command.extend(["-H", f"{key}: {value}"])
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

payload = fetch_json(url, headers)
elapsed = round(time.time() - started, 3)
odds = payload.get("market_odds") or payload.get("odds") or []
actual_order = ((payload.get("actual_result") or {}).get("actual_order") or [])
if payload.get("status") == "settled":
    if actual_order != [1, 2, 4] or payload.get("market_used") is not False:
        raise SystemExit(f"bad settled payload: order={actual_order} market_used={payload.get('market_used')}")
elif payload.get("market_used") is not True or len(odds) < 7:
    raise SystemExit(f"bad market payload: market_used={payload.get('market_used')} odds={len(odds)}")
print(json.dumps({
    "url": url,
    "device_id": device_id,
    "elapsed_sec": elapsed,
    "status": payload.get("status"),
    "market_used": payload.get("market_used"),
    "market_odds_count": len(odds),
    "actual_order": actual_order,
    "poll_delay_ms": payload.get("poll_delay_ms"),
}, ensure_ascii=False))
PY
}

start_all() {
  if [[ "${RACELENS_PREVIEW_FORCE_STOP:-0}" == "1" ]]; then
    stop_all
    force_free_port "$BACKEND_PORT"
    force_free_port "$PROXY_PORT"
    force_stop_tunnel
  fi
  start_backend
  start_proxy
  start_tunnel
  status
}

stop_all() {
  if [[ "${RACELENS_PREVIEW_STOP_WATCHDOG:-1}" == "1" ]]; then
    stop_pid_file watchdog
  fi
  stop_pid_file tunnel
  stop_pid_file proxy
  stop_pid_file backend
}

status() {
  if [[ "$USE_LAUNCHD" == "1" ]]; then
    local backend proxy tunnel watchdog
    backend="$(launch_pid backend || true)"
    proxy="$(launch_pid proxy || true)"
    tunnel="$(launch_pid tunnel || true)"
    watchdog="$(launch_pid watchdog || true)"
    [[ -n "$backend" ]] && remember_pid backend "$backend"
    [[ -n "$proxy" ]] && remember_pid proxy "$proxy"
    [[ -n "$tunnel" ]] && remember_pid tunnel "$tunnel"
    [[ -n "$watchdog" ]] && remember_pid watchdog "$watchdog"
  fi
  printf 'backend_pid=%s backend_url=%s\n' "$(read_pid backend || true)" "$BACKEND_URL"
  printf 'proxy_pid=%s proxy_url=%s\n' "$(read_pid proxy || true)" "$PROXY_URL"
  printf 'tunnel_pid=%s public_url=%s\n' "$(read_pid tunnel || true)" "$(public_url)"
  printf 'watchdog_pid=%s interval_sec=%s\n' "$(read_pid watchdog || true)" "${RACELENS_PREVIEW_WATCH_INTERVAL:-20}"
}

smoke_all() {
  smoke_url "$PROXY_URL"
  local url
  url="$(public_url)"
  if [[ -z "$url" ]]; then
    echo "public URL is missing; run start first" >&2
    return 1
  fi
  smoke_url "$url"
}

case "${1:-start}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    RACELENS_PREVIEW_FORCE_STOP=1 start_all
    ;;
  watch)
    RACELENS_PREVIEW_STOP_WATCHDOG=0 watch_loop
    ;;
  install-watchdog)
    start_all
    install_watchdog
    status
    ;;
  status)
    status
    ;;
  smoke)
    smoke_all
    ;;
  *)
    echo "usage: $0 [start|stop|restart|status|smoke]" >&2
    exit 2
    ;;
esac
