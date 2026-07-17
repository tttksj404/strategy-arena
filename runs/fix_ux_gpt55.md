# RACELENS UX follow-up fix - GPT-5.5

Date: 2026-07-07
Scope: `mobile/` only

## 변경 결과

1. 분석 화면 샘플 데이터 표식
   - `src/components/DataStatusStrip.tsx`: `decision.marketSource === 'sample'`일 때 상태 스트립 첫 슬롯을 `과거 데이터 예시`로 교체.
   - `src/screens/AnalyzeScreen.tsx`: 샘플 상태에서만 `분석일` 대신 `예시 기준일 YYYY-MM-DD` 표시.
   - 실데이터 분기: `marketSource !== 'sample'`이면 기존 출주표 검증 슬롯과 `분석일` 유지.
   - `scripts/qa-api-failure.mjs`: 스크롤 0 상태 viewport 안에 `과거 데이터 예시`가 보이는지 bounding box로 단언.

2. ProScreen 개발자 문구 제거
   - `src/screens/ProScreen.tsx`: `빌드`, `결제 기능이 꺼진 빌드` 문구를 사용자 언어로 교체.
   - 새 문구: `Pro 기능은 출시 준비 중입니다. 지금은 무료 분석을 그대로 이용할 수 있습니다.`
   - 계정 상태의 원시 내부 상태(`sample/sqlite`)도 `무료 이용 중`, `공식 데이터 확인 가능`, `예시 데이터 대기` 계열 사용자 문구로 교체.
   - `tests/mobile-web.spec.js`, `scripts/qa-full-functional-audit.mjs`: 새 문구 존재 확인으로 갱신.

3. 무료 사용량 진행바 색 의미 보정
   - `src/screens/AnalyzeScreen.tsx`: 남은 횟수 기준으로 진행바 fill 색상 분기.
   - `remaining > 1`: `accentTeal`
   - `remaining === 1`: `accentAmber`
   - `remaining === 0`: `accentRose`
   - 새 raw color 없이 `src/theme/tokens.ts` 기존 토큰만 사용.

4. 일정 실패/샘플 배당 문구 정리
   - `src/components/RaceSelector.tsx`: 공식 일정 실패 시 날짜 칩의 `경기일만 선택 가능` 보조문구를 숨기고 하단 실패 메시지만 남김.
   - `src/components/MarketOddsBoard.tsx`: `앱 검토용 샘플 배당`을 `과거 경주의 예시 배당`으로 교체.

## 검증 결과

All PASS:

- `npm.cmd run typecheck`
  - Log: `runs/typecheck_after_ux_fix.log`
- `npm.cmd run lint:design`
  - Log: `runs/lint_design_after_ux_fix.log`
- `npm.cmd run lint:security`
  - Log: `runs/lint_security_after_ux_fix.log`
- `npm.cmd run qa:mobile`
  - Log: `runs/qa_mobile_after_ux_fix.log`
  - Result: `66 passed`
- `npm.cmd run qa:api-failure`
  - Log: `runs/qa_api_failure_after_ux_fix.log`
  - Result: `API failure QA passed`
  - Regression locked: sample marker visible above fold at scroll position 0.
- `npm.cmd run qa:analytics`
  - Log: `runs/qa_analytics_after_ux_fix.log`
  - Result: `analytics QA passed: 11 events`
- `npm.cmd run qa:release-visual`
  - Log: `runs/qa_release_visual_after_ux_fix.log`
  - Result: `release readiness QA passed`
  - Fresh screenshots:
    - `C:\tmp\racelens-release-readiness\320-home-keirin-light.png`
    - `C:\tmp\racelens-release-readiness\375-home-horse-light.png`
    - `C:\tmp\racelens-release-readiness\390-analyze-horse-dark.png`
    - `C:\tmp\racelens-release-readiness\320-pro-dark-safety.png`
    - `C:\tmp\racelens-release-readiness\768-analysis-keirin-light.png`
  - Metrics for every screenshot: `badText=0`, `overflow=0`, `smallTargets=[]`, `forbiddenActionButtons=[]`.
- Extra consistency check after removing raw Pro account state:
  - `npm.cmd run qa:api-live`
  - Log: `runs/qa_api_live_after_ux_fix.log`
  - Result: `live API QA passed`

## 시각 확인

- 직접 확인: `768-analysis-keirin-light.png`
  - `예시 기준일 2026-07-07` and `과거 데이터 예시` are visible above the fold.
  - Free quota `1/3` rail is teal, not red.
- 직접 확인: `320-pro-dark-safety.png`
  - Pro header uses user-facing launch-prep copy and no internal build language.

Independent visual/CJK reviewer:

- Verdict on UI/content: PASS.
- Evidence summary: sample marker above fold, Pro copy and schedule failure copy corrected, `1/3` quota rail teal, no `앱 검토용`, no `결제 기능이 꺼진`, no `현재 빌드`, no `빌드에서는`, all requested QA PASS.
- Reviewer requested this final report to include explicit code-review/QA matrix evidence; this file records that evidence.

## 자기 검증

- Design token compliance: no new raw colors; quota colors use `accentTeal`, `accentAmber`, `accentRose`.
- Branch safety: sample label/date branch is tied only to `decision.marketSource === 'sample'`; non-sample data keeps real analysis wording.
- Copy safety: `git grep` found no visible-user matches for `빌드`, `앱 검토용`, `결제 기능이`, `현재 빌드` in tracked mobile sources excluding `node_modules` and lockfile.
- Schedule failure: no contradictory `경기일만 선택 가능` text is rendered when `scheduleUnavailable` is true.
- Internal state copy: Pro/account and advanced analysis rows no longer expose raw `sample`, `demo`, or storage names such as `sqlite` as user-facing state.
- Test strength preserved: Pro text checks still assert exact visible text; API failure QA now adds viewport-level fold visibility assertion.
- Programming review: small scoped diff, no new abstraction, no new dependency, no `any`, no `as`, no `!`, no `@ts-ignore`, no destructive side effects.
- Overfit/slop review: no broad selector text weakening; assertions still check exact Pro copy and viewport-visible sample marker. No new placeholder copy, dead helper, redundant visual layer, or one-off dependency was added. The sample-state branch is source-of-truth based (`marketSource === 'sample'`) rather than text-sniffed.

## 남은 리스크

- PowerShell profile execution-policy warning appears before commands in this environment, but commands succeeded via `npm.cmd`.
- Prior unrelated untracked files under `.codex-fable5/` and `runs/` existed before this fix; commit scope is limited to the requested code/spec changes and `runs/fix_ux_gpt55.md`.
