# RaceLens Adversarial Runtime QA

- 실행 시각: 2026-07-11T01:40:55.643Z
- Web export: /var/folders/b9/yvr3_kh17ng5bwr9bhfzzg_00000gn/T/racelens-qa-adversarial-dist-VCwTTA
- Evidence screenshots: /var/folders/b9/yvr3_kh17ng5bwr9bhfzzg_00000gn/T/racelens-qa-adversarial-QN7nHy
- 기준: crash 0, console/pageerror 0, 깨진문자 0, 가로 오버플로 <=1px, 정직한 한국어 상태 안내

| # | 케이스 | 결과 | 발견결함/검증내용 | 수정내역 |
|---:|---|---|---|---|
| 1 | 01-html-live-decision | PASS | HTML 응답이 JSON 파서에서 오류 상태로 전환되어 후보/가짜 데이터가 숨겨짐 | live-decision JSON 파싱 실패를 unavailableDecision으로 변환 |
| 2 | 02-missing-required-fields | PASS | 필수 예측 필드 누락 시 빈 분석 상태로 전환 | predictionAvailable가 1·2·3착 pick과 top1 확률을 동시에 요구 |
| 3 | 03-out-of-range-values | PASS | 확률/이름/번호 이상값이 sanitizer에서 거부 또는 클램프됨 | racePayload 확률·텍스트 sanitizer와 app-session NaN 보정 확인 |
| 4 | 04-long-special-text | PASS | 초장문 한글+이모지는 안전하게 줄이고, 서버 meet 특수문자는 선택 조건 밖이라 표시하지 않음 | safeText truncation, keep-all/anywhere wrapping, overlap gate |
| 5 | 05-timeout-live-decision | PASS | 15초 지연 응답은 10초 AbortController timeout 후 안내로 종료 | live-decision timeout을 unavailableDecision으로 변환하고 스피너 종료 |
| 6 | 06-status-5xx-429-401 | PASS | 5xx/429/401 상태별 한국어 안내를 구분 | HTTP status mapper 추가 |
| 7 | 07-network-flap-retry | PASS | 첫 네트워크 실패 후 다시 시도로 정상 렌더 복구 | 빈 분석 상태에 재시도 액션 추가 |
| 8 | 08-quota-boundary | PASS | remaining 0은 분석 시도 시 한도 안내+Pro 화면으로 이동 | 한도 소진 CTA를 비활성 대신 안내 액션으로 유지 |
| 9 | 09-app-session-failure | PASS | app-session 실패에도 앱 시작과 기능 제한 안내 유지 | HomeScreen에 session dataLayer error 안내 추가 |
| 10 | 10-font-scale-320 | PASS | 320px + 130% 폰트 스케일에서 홈과 분석 화면 오버플로 없음 | 기존 토큰과 wrap 규칙을 실제 브라우저에서 검증 |
| 11 | 11-analytics-down | PASS | ux-events 전송 실패는 기능에 영향 없이 조용히 드랍 | trackUxEvent catch-and-drop 경로 Playwright 검증 |
| 12 | 12-rapid-tap-inflight | PASS | 분석 버튼 5연타에도 live-decision 중복 요청 폭주 없음 | App executeAnalyze ref guard와 raceApi request-key dedupe |
