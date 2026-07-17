# RaceLens Quota Review

- Reviewed: 2026-07-08 KST
- Scope: `1fb84f5` free daily env override, `2dba2b9` IP per-minute cap disabled
- Decision: `risk`

## Current Effective Limits

| Surface | Code default | Deploy-generated default | Live observation |
|---|---:|---:|---:|
| Free daily analysis per device/user | `3` | `100000` | `100000` |
| Live-decision IP per minute | `30` | `0` (disabled) | disabled or non-blocking |
| Billing mode | mobile release env `disabled` | disabled/no receipt gate for public purchase | disabled |

Evidence:
- `datastore.py` keeps `DEFAULT_FREE_DAILY_ANALYSIS_LIMIT = 3` and reads `RACELENS_FREE_DAILY_ANALYSIS_LIMIT`.
- `deploy/oracle/deploy.sh` writes `RACELENS_FREE_DAILY_ANALYSIS_LIMIT=${RACELENS_FREE_DAILY_ANALYSIS_LIMIT:-100000}`.
- `deploy/oracle/deploy.sh` writes `RACELENS_LIVE_DECISION_IP_PER_MIN_CAP=${RACELENS_LIVE_DECISION_IP_PER_MIN_CAP:-0}`.
- Live E2E response showed `free_analysis_limit: 100000`.

## Risk Assessment

The env-tunable mechanism is sound, but the deployed defaults are risky for production-like public traffic.

1. data.go.kr/API load risk: live analysis can trigger upstream card/result fetches. Caches reduce repeat calls, but a 100000/day per-device ceiling plus disabled IP minute cap leaves abuse bounded mainly by device-id/IP anchoring and provider timeouts.
2. Product risk: with `billing_mode=disabled`, a high free ceiling is acceptable for review/testing, but it should be explicitly labeled as beta/review mode. If billing later turns on, this default removes practical Pro differentiation unless changed during deploy.
3. Abuse risk: `IP_PER_MIN_CAP=0` removes the only fast shared-network throttle. Device IDs are client-controlled enough that they should not be the sole public throttle.

## Recommendation

Keep env override support, but change deploy defaults before public or ad-driven distribution:

| Env | Safer default | Override policy |
|---|---:|---|
| `RACELENS_FREE_DAILY_ANALYSIS_LIMIT` | `20` for beta/review, `3` for monetized release | raise temporarily for closed testing only |
| `RACELENS_LIVE_DECISION_IP_PER_MIN_CAP` | `60` | set `0` only for controlled load tests |
| `RACELENS_IP_NEW_USER_CAP` | keep `5` | raise only with monitoring |

Fable5 verdict request: treat the current deployed quota posture as `risk` until a bounded deploy default is restored or explicit beta-unlimited operating approval is recorded.
