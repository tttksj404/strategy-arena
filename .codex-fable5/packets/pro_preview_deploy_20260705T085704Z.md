# Pro Preview Deployment Evidence

## Claim
Oracle preview deployment now serves RaceLens as a Pro-enabled experience by default.

## Changed Surface
- `deploy/oracle/docker-compose.yml`: app and mobile-web set `RACELENS_FORCE_PRO=1`.
- `deploy/oracle/deploy.sh`: generated `.env.oracle` writes `RACELENS_FORCE_PRO=${RACELENS_FORCE_PRO:-1}`.
- `deploy/oracle/.env.oracle.example`: documents `RACELENS_FORCE_PRO=1`.
- `deploy/oracle/smoke.sh`: fails unless `/api/live-decision` exposes `app_session.entitlement == "pro"` and unlocked participant algorithm reasons.
- `tests/test_oracle_deploy_artifacts.py`: locks the deploy and smoke contract.

## Verification
- `python -m pytest tests/test_oracle_deploy_artifacts.py -q` -> 7 passed.
- `python -m pytest tests/test_app_data_layer.py tests/test_live_decision.py tests/test_oracle_deploy_artifacts.py -q` -> 68 passed.
- `bash -n deploy/oracle/deploy.sh deploy/oracle/smoke.sh` -> passed.
- `cd mobile && npm run typecheck` -> passed.
- `cd mobile && npm run build` -> Expo web export passed.
- Oracle deploy command completed against `https://168-107-2-218.sslip.io`; integrated smoke passed all routes including `/api/live-decision`.
- `curl https://168-107-2-218.sslip.io/api/app-session` with a fresh device returned `entitlement=pro`, `data_ready=true`, `storage=postgresql`.
- `curl https://168-107-2-218.sslip.io/api/live-decision?sport=keirin&date=2026-07-03&meet=광명&race_no=1` returned `entitlement=pro` and every participant had `algorithm_locked=false`.
- SSH readback on Oracle returned `RACELENS_FORCE_PRO=1` in `deploy/oracle/.env.oracle` and inside the running `app` container.

## Residual Risk
- This intentionally forces Pro for the Oracle preview. Store-release billing still needs real receipt validation before charging users.
