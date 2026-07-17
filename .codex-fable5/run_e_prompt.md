# RUN E — Run D 후속: 빈상태 회귀 수정 + 스펙 테스트 갱신 (mobile/ 전용)

너는 빌더다. **mobile/만 수정**(*.py 금지 — 병렬 작업 중). 커밋 금지. 완료 조건: `npx tsc --noEmit` 0에러 + `npm run qa:mobile` **0 failed** + `npm run lint:design` PASS.

현황: qa:mobile 66개 중 24 failed. 원인 두 갈래 — ① Run D가 UI를 바꿔 구식 스펙 기대가 깨짐(광고 placeholder 제거 등 의도된 변경) ② **실제 회귀**: API 불가/빈 데이터에서 PredictionSummary가 "0번 / 모델 추정 0% / 예측 순서 0" 가짜 시상대를 렌더함(스크린샷 확인됨). ②가 우선.

## E1. 빈상태/오류 가드 (실제 버그 수정)
- picks/top 데이터가 없거나 API 오류면 시상대·확률·번호 배지를 **렌더하지 않는다**. 대신 명확한 오류/빈 상태 카드("데이터를 가져오지 못했습니다 — 잠시 후 다시 시도" / no_race면 "해당 날짜에는 경주가 없습니다")를 표시. 0%·0번 같은 가짜 수치 노출 금지.
- 2·3착 후보 데이터가 있으면 시상대 좌우 슬롯이 반드시 렌더되는지 확인하고, 안 되면 고쳐라(1착 단독 노출 금지 — 1·2·3착이 핵심 요구).
- "PODIUM" 영문 뱃지 → "예측 순서" 한글로. 시상대 하단 "1착 후보 N번" 중복 캡션 제거.

## E2. 스펙 테스트 갱신 (안전 단언 약화 금지)
`mobile/tests/mobile-web.spec.js` 4개 시나리오를 새 UI 스펙으로:
- "analysis does not invent demo racers when the official API is unavailable": 새 단언 = 오류 카드 표시 + 시상대/참가자 카드 부재 + "0번"류 텍스트 부재. (제일 중요 — 절대 삭제 금지, 강화만.)
- "free analysis limit becomes a real disabled state": 새 쿼터 카드/차단 상태 셀렉터로 갱신, 의미 유지.
- "race selector updates sport, race, and opens analysis": 새 selector 구조(종목 accent 스왑 포함)로 갱신.
- "store safety and pro surfaces avoid betting actions": 광고 placeholder 부재를 전제로 갱신, 베팅 유도 문구 금지 단언은 유지·강화. "무료 플랜 광고" 기대는 monetization off에서는 제거가 맞다.
- 테스트가 특정 문구에 과결합이면 data-testid 추가로 안정화해도 된다.

## E3. 검증
- `npm run qa:mobile` 전 뷰포트 0 failed 출력과 tsc·lint:design 결과를 `.codex-fable5/run_e_evidence.txt`에 저장.
- 최종 보고 ≤60자.
