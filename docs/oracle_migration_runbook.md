# Oracle migration runbook

Strategy Arena should run from an Oracle Korea region host when live KCYCLE odds are required. Render remains useful as a temporary fallback, but it has no Korea region and can leave `market_used=false` during live race windows.

## Target state

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
4. Put this repository at `/opt/strategy-arena`.
5. Copy `deploy/oracle/.env.oracle.example` to `deploy/oracle/.env.oracle`.
6. Fill `DATAGOKR_SERVICE_KEY`.
7. If a domain is ready, set `ORACLE_SITE_ADDRESS=your.domain.example`; otherwise keep `:80`.

## Deploy from local machine

```bash
export ORACLE_HOST="<server-ip-or-hostname>"
export ORACLE_USER="ubuntu"
export ORACLE_PATH="/opt/strategy-arena"
deploy/oracle/deploy.sh
```

The script intentionally stops if `deploy/oracle/.env.oracle` does not exist on the server. That prevents silently deploying without the data.go.kr key.

## Enable boot recovery and daily backup

Run on the Oracle VM:

```bash
sudo STRATEGY_ARENA_HOME=/opt/strategy-arena deploy/oracle/install_systemd.sh
```

## Smoke checks

```bash
deploy/oracle/smoke.sh "http://<server-ip-or-domain>"
```

Required outcomes:

- `/healthz` returns HTTP 200.
- `/api/live-decision` returns JSON.
- `/predict` returns HTML with no legacy `market_trifecta_50_candidate` or `삼쌍 50% 후보` text.
- During a live race window, `market_risk.level` should not be `live_market_blocked`. If it is, check collector logs before switching traffic.

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
| Secret leakage | Real `.env.oracle` is excluded from git and rsync overwrite. |

## Render retirement gate

Keep Render online until all are true for at least one live race window:

- Oracle `/healthz` is stable.
- Oracle `/api/live-decision` uses live market data or reports a non-Render-specific collector state.
- `collector.log` shows fetched full-board snapshots.
- `search-loop.log` shows completed audit/search after new snapshots.
- `deploy/oracle/smoke.sh` passes against the final Oracle URL.

After that, update DNS to Oracle and keep Render as a fallback for 3-7 days before disabling it.
