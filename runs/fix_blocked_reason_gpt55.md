# RACELENS blocked reason fix (GPT-5.5)

## Root cause

실서버의 무료 분석 소진 응답은 `decision=blocked`, `status=blocked`,
`app_session.free_analysis_remaining=0`, `rewarded_analysis_credits=0`,
`market_risk.message="오늘 무료 분석 3회를 모두 사용했습니다. Pro 권한이 확인되면 무제한 분석이 열립니다."`
형태이며 `roster_verification`은 생략된다.

기존 모바일 파서는 `roster_verification`이 없을 때 `state='unverified'`로 기본값을 만들었다. 그 결과 무료 소진 blocked 응답이 `waitingForRoster=true`가 되어 "공식 출주표를 아직 확인하지 못했습니다" 화면으로 잘못 라우팅됐다.

## Change

- `sanitizeRosterVerification(undefined)` 기본값을 `state='unknown'`으로 변경했다. 누락된 필드는 더 이상 출주표 미확인으로 해석하지 않는다.
- `AnalyzeScreen`에 명시적인 렌더 상태를 추가했다.
  - `settled` 결과 + 예측 있음: 정상 분석
  - 무료 소진: `free_quota_exhausted`
  - 명시적 `roster_verification.state='mismatch'`: 출주표 불일치 차단
  - 명시적 `roster_verification.state='unverified'`: 출주표 확인 대기
  - 예측 없음: 데이터 없음
- 무료 소진 화면은 `free-quota-exhausted-state`로 식별되며, 서버 `market_risk.message`를 그대로 보여주고 `Pro 안내 보기` 액션을 제공한다.
- 무료 소진 화면에서는 `DataStatusStrip`을 숨겨 출주표 상태로 오인될 여지를 없앴다.
- `DataStatusStrip`은 `unknown` roster 상태를 중립적인 "출주표 정보 없음"으로 표시한다.

## QA coverage

- `qa:api-failure` mock API에 실서버형 무료 소진 blocked 응답을 추가했다.
  - `roster_verification` 없음
  - `free_analysis_remaining=0`
  - `rewarded_analysis_credits=0`
  - 서버 무료 소진 메시지 포함
- 단언:
  - `free-quota-exhausted-state`가 렌더된다.
  - 서버 무료 소진 메시지가 노출된다.
  - "공식 출주표 확인 못함" 계열 문구와 `roster-waiting-state`가 노출되지 않는다.
  - `Pro 안내 보기` 액션이 있다.
- `qa:adversarial`의 출주표 미확인 케이스는 새 계약에 맞춰 `roster_verification.state='unverified'`를 명시하도록 보정했다.

## Verification

All commands were run from `mobile/` with `PATH="/c/Users/SSAFY/.local/node:$PATH"`.

| Gate | Result | Evidence |
| --- | --- | --- |
| `npm run typecheck` | PASS | TQE `20260710-144022790-d5b0e8a5`, exit 0 |
| `npm run lint:design` | PASS | TQE `20260710-144051595-de40f8b5`, inspected: design token/resource/contrast checks passed |
| `npm run lint:security` | PASS | TQE `20260710-144051607-f5a45de9`, security surface check passed |
| `npm run qa:mobile` | PASS | TQE `20260710-144111759-fec8226d`, 78 passed |
| `npm run qa:api-failure` | PASS | TQE `20260710-143904067-14d31349`, API failure QA passed |
| `npm run qa:analytics` | PASS | TQE `20260710-144221519-f61e2451`, analytics QA passed: 26 events |
| `npm run qa:release-visual` | PASS | TQE `20260710-144446010-cafb9ff6`, release readiness QA passed and rewarded ads policy QA passed |
| `npm run qa:adversarial` | PASS | TQE `20260710-144701328-412d2d78`, ADVERSARIAL=12/12 PASS |

## Notes

- TypeScript LSP diagnostics could not be run because `typescript-language-server` is not installed. `npm run typecheck` passed and is the effective type verification for this change.
- The worktree already contained unrelated dirty and untracked release/audit artifacts. Only the blocked-reason fix, matching QA scripts, and this report are intended for the commit.
