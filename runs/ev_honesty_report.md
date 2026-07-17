# RaceLens EV Honesty and Quota Floor Report

Date: 2026-07-08

## EV honesty copy

### Compact mode

`적중률은 수익을 뜻하지 않습니다 · 정보 분석 전용입니다.`

Rendered on the Home screen through `StoreSafeNotice compact`, above the bottom tab bar.

### Full mode

`경륜·경마는 공제(약 20~28%) 때문에 아무리 정확해도 장기적으로 평균 손실입니다. RaceLens는 수익 도구가 아니라 경주 데이터를 이해하기 위한 정보 분석 도구이며, 구매·베팅 연결이나 수익 보장을 제공하지 않습니다. 만 19세 이상, 도박 중독에 유의해 책임 있게 이용하세요. 도박문제 상담은 1336에서 안내받을 수 있습니다.`

Rendered on Analyze and Pro surfaces through the existing `StoreSafeNotice` full mode. The Analyze screen keeps the notice in the analysis flow, directly after the empty/analysis state before deeper evidence boards, so probability or model signal readers encounter the -EV warning in the same screen.

## Regression coverage

`mobile/tests/store-safe-notice.spec.js` now asserts:

- Compact copy renders on Home: `적중률은 수익을 뜻하지 않습니다 · 정보 분석 전용입니다.`
- Full analysis copy renders after opening analysis:
  - `장기적으로 평균 손실입니다`
  - `RaceLens는 수익 도구가 아니라`
  - `도박문제 상담은 1336`

`mobile/tests/mobile-web.spec.js` keeps the bottom reachability assertion aligned with the updated compact copy above the tab bar.

`npm run qa:mobile` increased from 66 to 72 checks because the new EV honesty test runs across the existing six Playwright projects.

## Quota floor

`deploy/oracle/deploy.sh` now writes:

```sh
RACELENS_LIVE_DECISION_IP_PER_MIN_CAP=${RACELENS_LIVE_DECISION_IP_PER_MIN_CAP:-60}
```

The env override path is unchanged. `RACELENS_FREE_DAILY_ANALYSIS_LIMIT` stays beta-unlimited at `100000`, with the deploy comment updated to: `beta-unlimited: IAP 출시 전 임시 무료 일일 분석 한도. 공개/광고 배포 전 3~20으로 복원.`

No server deployment was run.

## Verification

| Gate | Result | Evidence |
| --- | --- | --- |
| `npm run typecheck` | PASS | `tsc --noEmit` exit 0 |
| `npm run lint:design` | PASS | design token/resource/contrast checks passed |
| `npm run lint:security` | PASS | security surface check passed |
| `npm run qa:mobile` | PASS | 72 passed |
| `npm run qa:api-failure` | PASS | API failure QA passed |
| `npm run qa:analytics` | PASS | analytics QA passed: 11 events |
| `npm run qa:release-visual` | PASS | release readiness QA passed; 5 viewport screenshots, overflow 0, broken text 0, forbidden action buttons 0, small targets 0 |
| `.venv/Scripts/python.exe -m pytest tests/test_app_data_layer.py tests/test_live_decision.py -q` | PASS | 84 passed |

Focused render capture: 390px dark Analyze screen rendered the full EV notice with console errors 0 and the three required text checks true.

GATES=8/8
