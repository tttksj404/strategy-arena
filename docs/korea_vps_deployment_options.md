# Korea VPS deployment options

Render should not be the production surface for live KCYCLE odds. The production surface should be a Korea-region VM/VPS with Docker Compose, because the app needs outbound access to Korean racing sources and persistent collector/search data.

## Recommended order

1. Naver Cloud Platform Server or SSD Server
   - Best fit for the KCYCLE access risk because the server is plainly in Korea.
   - Use an Ubuntu VM and open 80/443. `deploy/korea-vps/deploy.sh` can install Docker automatically over SSH when the account has sudo.
   - Naver Cloud lists Compute Server/SSD Server as its VM products.

2. AWS Seoul VM, preferably EC2 or Lightsail instance
   - AWS Lightsail container services can run containers, but this repo already has a multi-service Docker Compose stack with app, Caddy, collector, and search-loop. A VM/instance keeps that stack intact.
   - Use region `ap-northeast-2`.

3. Google Cloud Compute Engine Seoul VM
   - Use region `asia-northeast3`.
   - Google currently points VM container deployment users toward normal Docker commands/startup scripts rather than the deprecated `create-with-container` path, so a Docker Compose VM is the cleanest match.

4. Domestic VPS providers such as Cafe24, Gabia, iwinv, or Hostcenter
   - Acceptable if SSH, public inbound 80/443, Docker, and stable outbound Korean network access are available.
   - Docker can be missing at first boot if sudo works; the deploy script bootstraps it.
   - Treat these as operationally simpler but verify `/api/live-decision` during an actual live race window before switching traffic.

## Generic deploy

```bash
export VPS_HOST="<server-ip-or-domain>"
export VPS_USER="ubuntu"
export VPS_PATH="/opt/strategy-arena"
export VPS_SITE_ADDRESS=":80"
deploy/korea-vps/deploy.sh
```

The script fails closed if:

- `VPS_HOST` is missing.
- SSH cannot connect.
- Docker Engine or the Docker Compose plugin is not installed and automatic bootstrap fails.
- `DATAGOKR_SERVICE_KEY` cannot be resolved from the current env, `LOCAL_DATAGOKR_ENV`, `~/keirin/.env`, or `~/kra/.env`.
- The public smoke test still exposes the Render-specific `live_market_blocked` state.

## Cutover gate

Do not move DNS or app API traffic to a new server until all are true:

- `deploy/korea-vps/deploy.sh` exits 0.
- `/healthz` returns HTTP 200 from the new server.
- `/api/live-decision` returns `market_odds` as a list and does not expose `live_market_blocked` on the new public URL.
- During a live race window, `collector.log` shows fresh full-board snapshots.
- `search-loop.log` shows completed audit/search after the new snapshots.

## Source notes

- Naver Cloud lists Compute Server and SSD Server products for VM-style deployment.
- AWS Lightsail documents container services, but its container-service model is different from this repo's existing Compose VM layout.
- Google Compute Engine docs note that `create-with-container` is deprecated and recommend equivalent Docker commands for containers on VMs.
