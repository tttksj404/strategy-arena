# Oracle migration runbook

RaceLens should run from an Oracle Korea region host when live KCYCLE odds are required. Render is retired from the live-odds path because it has no Korea region and can leave `market_used=false` during live race windows.

## Target state

- Current fixed host: `g185` / `racelens-oracle` -> `ubuntu@168.107.2.218`.
- Current HTTPS smoke URL: `https://168-107-2-218.sslip.io`.
- Current remote path: `/home/ubuntu/strategy-arena` (legacy directory name for the RaceLens service).
- OCI region: `ap-seoul-1` first choice, `ap-chuncheon-1` second choice.
- Runtime: one VM with Docker Compose.
- Public entry: Caddy reverse proxy on ports 80/443.
- App: Flask/Gunicorn on container port 8000.
- Live collection: `collector` service runs only during the local race window check in `scripts/run_kcycle_trifecta_collector_if_race_window.sh`.
- Search loop: `search-loop` service reruns corpus audit and trifecta rule search on a fixed interval.
- Persistent data: Docker volume `strategy_data`.
- Backups: `deploy/oracle/backup_snapshots.sh` plus `strategy-arena-backup.timer`.

## One-time server setup

1. Create an OCI VM in Seoul or Chuncheon.
2. Open inbound TCP 80 and 443 in the OCI security list or NSG.
3. Install Docker Engine and the Docker Compose plugin.
4. Put this RaceLens repository at `/opt/strategy-arena` or `/home/ubuntu/strategy-arena`; if `ORACLE_PATH` is unset, `deploy/oracle/deploy.sh` auto-detects those existing paths and fails closed if neither exists.
5. Keep `DATAGOKR_SERVICE_KEY` on the local deploy machine in one of these places: current shell env, `LOCAL_DATAGOKR_ENV`, `~/keirin/.env`, or `~/kra/.env`.
6. If a domain is ready, set `ORACLE_SITE_ADDRESS=api.your-racelens-domain.com`; otherwise keep `:80` for IP smoke only. Store submission requires the domain/HTTPS mode.
7. Set `RACELENS_SUPPORT_EMAIL` to the real support email before store review so legal pages and app metadata match.

## Deploy from local machine

```bash
export ORACLE_HOST="<server-ip-or-hostname>"
export ORACLE_USER="ubuntu"
# If the VM exposes SSH on a non-standard port, for example the existing g185 VM:
# export ORACLE_USER="opc"
# export ORACLE_SSH_PORT="443"
# Optional: unset ORACLE_PATH lets deploy.sh auto-detect /opt/strategy-arena or /home/ubuntu/strategy-arena.
export ORACLE_PATH="/opt/strategy-arena"
deploy/oracle/deploy.sh
```

The script now fails closed before deployment if SSH is unreachable or if a local `DATAGOKR_SERVICE_KEY` cannot be resolved. On success it uploads a fresh `deploy/oracle/.env.oracle` with `0600`-style permissions, starts Docker Compose, checks container health, and runs `deploy/oracle/smoke.sh` against the Oracle URL.

## Enable boot recovery and daily backup

Run on the Oracle VM:

```bash
sudo STRATEGY_ARENA_HOME=/opt/strategy-arena deploy/oracle/install_systemd.sh
```

## Smoke checks

```bash
deploy/oracle/smoke.sh "http://<server-ip-or-domain>"
deploy/oracle/smoke.sh "https://<production-domain>"
```

Required outcomes:

- `/healthz` returns HTTP 200.
- `/legal/privacy`, `/legal/terms`, and `/legal/account-deletion` return Korean HTML with no replacement-character garbling.
- `/api/live-decision` returns JSON and 2026-07-03 광명 1R participants exactly match the official card: 황종대, 이흥주, 박진홍, 최건묵, 이승주, 박유찬, 김성진.
- `/predict` returns HTML with no legacy `market_trifecta_50_candidate` or `삼쌍 50% 후보` text.
- During a live race window, `market_risk.level` should not be `live_market_blocked`. If it is, check collector logs before switching traffic.
- Against any non-localhost Oracle URL, `smoke.sh` fails if `/api/live-decision` still exposes the Render-specific `live_market_blocked` risk.

## Risk closure table

| Risk | Closure mechanism |
|---|---|
| Render region cannot reach KCYCLE | App runs in OCI Korea region with `KCYCLE_ENABLED=1`. |
| Live odds silently unavailable | `/api/live-decision` exposes `market_used`, `market_risk`, and signal payloads. |
| Low-sample trifecta signal overstated | Live tier is watch-only; `expected_trio_exact` stays null until robust promotion. |
| Snapshot loss | `strategy_data` Docker volume plus daily compressed backup. |
| Collector runaway outside race windows | `run_kcycle_trifecta_collector_if_race_window.sh` exits outside Friday-Sunday 10-19 local time. |
| Algorithm search stalls | `search-loop` reruns audit/search on `SEARCH_LOOP_INTERVAL_SEC`. |
| Server reboot | Docker restart policies plus optional systemd compose service. |
| TLS/HTTP exposure | Caddy handles reverse proxy and HTTPS when `ORACLE_SITE_ADDRESS` is a domain. |
| Store policy pages absent | Flask serves `/legal/privacy`, `/legal/terms`, and `/legal/account-deletion`; `smoke.sh` and `npm run qa:store-readiness` verify them. |
| Missing production env | `deploy.sh` resolves the local data.go.kr key, uploads `.env.oracle`, and stops if the key is unavailable. |
| Secret leakage | Real `.env.oracle` is excluded from git/rsync, generated in a local temp file, uploaded over SSH, and removed locally on exit. |

## Render retirement gate

Do not route app API traffic back to Render. Keep any old Render app only as an offline rollback artifact until all are true for at least one live race window:

- Oracle `/healthz` is stable.
- Oracle `/api/live-decision` uses live market data or reports a collector state that is not tied to Render.
- `collector.log` shows fetched full-board snapshots.
- `search-loop.log` shows completed audit/search after new snapshots.
- `deploy/oracle/smoke.sh` passes against the final Oracle URL.

After that, update DNS to Oracle and disable the Render app.
