# RaceLens E2E Resume Report

- Reviewed: 2026-07-08 KST
- Head before fix commits: `eb8b674` (`WIP: E2E server fixes`)
- Deployment: not performed
- Quota review decision: `risk`

## Regression Fixes

| Item | Result | Evidence |
|---|---|---|
| 1-A Windows template read | Fixed | `tests/test_live_decision.py` opens `templates/index.html` with `encoding="utf-8"`; `git grep` found no remaining test source/template `open()` reads without encoding. |
| 1-B unsupported keirin meet | Fixed | `_compute_base_prediction` validates keirin meet whitelist before upstream fetch/cache key work; targeted and full pytest pass. |
| 1-C mobile sport aliases | Fixed | `/predict`, `/api/live-decision`, and `/recent` use consistent entry-point sport alias handling; live-decision alias regression test passes. |

## Live E2E

| Journey group | Result | Evidence |
|---|---|---|
| Live mobile journey | PASS_WITH_EXPECTED_DEPLOY_GAPS | `runs/e2e_live_report.md`: 8/10 PASS, 2 `EXPECTED_FAIL_UNTIL_DEPLOY`. |
| Expected deploy gaps | EXPECTED_FAIL_UNTIL_DEPLOY | Live server still lacks `/recent?sport=kra -> horse` alias behavior and `/legal/support` route. |
| Quota usage | PASS | `Analysis calls: 3/3`; no quota exhaustion. |

## Store Pack

| Asset | Result | Evidence |
|---|---|---|
| `feature_graphic.png` | PASS | 1024x500 |
| `icon_512.png` | PASS | 512x512 |
| screenshots | PASS | 6 files, each 1080x1920 |
| `listing.md` policy copy | PASS | Forbidden terms zero; includes 19+ and information-analysis framing. |

## Quota Review

| Area | Decision | Evidence |
|---|---|---|
| Free daily limit | RISK | Code default remains `3`, but deploy default/live value is `100000`. |
| IP per-minute cap | RISK | Code default remains `30`, but deploy default is `0` (disabled). |
| Recommendation | BOUNDED_DEFAULTS | Keep env override; restore bounded deploy defaults before public/ad-driven traffic. See `runs/quota_review.md`. |

## Gates

| Gate | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest tests/test_app_data_layer.py tests/test_live_decision.py -q` | PASS: 84 passed |
| `npm run --prefix mobile typecheck` | PASS |
| `npm run --prefix mobile lint:design` | PASS |
| `npm run --prefix mobile lint:security` | PASS |
| `npm run --prefix mobile qa:mobile` | PASS: 66 passed |
| `npm run --prefix mobile qa:api-failure` | PASS |
| `npm run --prefix mobile qa:analytics` | PASS: 11 events |
| `npm run --prefix mobile qa:release-visual` | PASS |
| `npm run --prefix mobile qa:adversarial` | PASS: 12/12 |
| `EXPO_PUBLIC_RACELENS_API_BASE_URL=https://168-107-2-218.sslip.io npm run --prefix mobile qa:e2e-live` | PASS: 8/10 with 2 expected deploy gaps |
| `npm run --prefix mobile qa:store-pack` | PASS |

## Remaining Risks

- No push or deployment was performed by request.
- Fable5 final audit remains pending for the quota posture; local verdict packet is `risk`.
- Live server expected gaps remain until the local fixes and legal support route are deployed.
