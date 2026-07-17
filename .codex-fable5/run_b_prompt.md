# RUN B — 모바일/표현 P1 수정 (Fable5 감사 20260706 기반, RUN A 후속)

너는 이 repo의 빌더다. 아래를 전부 구현하고 논리 단위 커밋(푸시 금지). RUN A가 이미 적용됨: roster_guard(응답에 roster_verification 필드, status "roster_mismatch" 가능), 에러 error_kind 분리, 경마 settled, force_pro 가드. 작업 후 `python -m pytest -q` + `cd mobile && npx tsc --noEmit` 둘 다 PASS 확인.

## B1. 배당 신선도 표시 (mobile)
- `MarketOddsBoard` 헤더에 `updated_at`(KST HH:MM) + `odds_age_sec`("n초 전 갱신") 상시 표시. 배당 미수집/폴백이면 그 상태 문구를 같은 자리에. AnalyzeScreen에서 props 연결.

## B2. 출주표 검증 배지 (mobile)
- 서버 `roster_verification.state`를 받아 분석 화면 상단에 배지: verified="공식 출주표 대조 완료", unverified="공식 대조 미완료 — 참고용", mismatch면 전체 화면을 정직한 차단 카드("공식 출주표와 달라 예측을 중단했습니다")로. types/race.ts 갱신.

## B3. 설명 템플릿 다양화
- 서버 `engine.py`의 참가자 algorithm_note 생성부: 문장 프레임 3종 이상 로테이션 + "그 선수의 가장 특이한 지표(경주 내 z-score 최대 편차) 1개를 첫 문장에" 규칙. 결정론적(선수번호+날짜 시드)이어야 테스트 가능.
- 클라 `mobile/src/services/participantInsight.ts`(경마)도 동일 원칙.
- 테스트: 같은 경주 참가자 2명의 첫 문장 프레임이 동일하지 않음(서버 pytest + 간단 node 테스트 또는 tsc 수준 검증).

## B4. 종목 전용 카드 (mobile)
- 경륜 전용: "전개 구도" 카드 — 참가자 입상전법 분포(선행/젖히기/추입/마크)를 가로 바로 시각화 + 축 후보 표시.
- 경마 전용: "게이트·부담중량" 카드 — 게이트 순 정렬 미니보드(게이트, 부담중량, 마체중 증감). 기존 profile/stats 데이터만 사용, 새 API 금지.
- AnalyzeScreen에서 sport로 분기 렌더. 두 종목 화면 구성요소 목록이 달라야 한다.

## B5. 수익화 스캐폴딩 (기능 플래그, 기본 OFF)
- 서버: `POST /api/iap/verify` — body {platform, receipt}. `RACELENS_APPLE_SHARED_SECRET`/`RACELENS_GOOGLE_SA_JSON` 미설정이면 `{ok:false, reason:"not_configured"}` (절대 pro 부여 금지). 설정 시 각 스토어 검증 후 billing.subscriptions upsert. 테스트는 not_configured 경로 + mock 검증 경로.
- 모바일: ProScreen에 구매 CTA 버튼 추가하되 `EXPO_PUBLIC_MONETIZATION` off(기본)면 "스토어 심사 준비 중" 비활성 상태. IAP/AdMob 네이티브 모듈은 설치하지 말고 `src/services/monetization.ts` 인터페이스(purchase/restore/adSlot)만 정의 + TODO env 문서화(`mobile/MONETIZATION_SETUP.md`).
- 무료 광고 슬롯 placeholder 카드는 monetization off일 때 렌더하지 않는다(심사 리스크 제거).

## B6. 소정리
- mobile types/UX에서 'lab' 잔재 제거. 경마 화면 문구에서 "KCYCLE" 노출 제거 확인. QNL 표기를 "4·1 조합"(무순) 형태로.

## 완료 조건
- pytest 전체 PASS, `npx tsc --noEmit` 0 에러, 커밋 prefix: feat(mobile-odds)/feat(mobile-roster)/feat(explain)/feat(mobile-cards)/feat(iap)/chore(cleanup).
- 최종 보고 ≤80자: 커밋 수 + 두 검증 결과만.
