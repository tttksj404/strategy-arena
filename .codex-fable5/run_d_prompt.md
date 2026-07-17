# RUN D — RaceLens 디자인 개편 (Fable5 디자인 디렉션, mobile/ 전용)

너는 빌더다. **mobile/ 디렉토리만 수정**(백엔드 *.py 금지 — 다른 작업이 병렬 진행 중). 커밋 금지(내가 한다). 완료 조건: `cd mobile && npx tsc --noEmit` 0에러 + `node scripts/check-design-contrast.mjs` PASS + 기존 테스트 유지.

## 디자인 원칙 (Fable5 확정 — 임의 변경 금지)
현 상태 진단: 다크+라임 토큰은 좋은 기반이나 ① 경륜/경마가 같은 옷 ② 1·2·3착 후보의 시각 위계 부족 ③ 신뢰 신호(검증·신선도)가 텍스트로 흩어짐 ④ 참가자 카드 정보 과밀.

### D1. 종목별 비주얼 아이덴티티 (토큰 레벨)
- `theme/tokens.ts`에 sport-scoped accent 추가: keirin=기존 라임 `accentSignal` 유지, horse=골드 `accentGold`+터프 그린(새 토큰 `accentTurf` light `#1E6B4F` / dark `#4FC08D`).
- 종목 전환 시 헤더 틴트·활성 칩·픽 하이라이트·탭 활성색이 sport accent로 일괄 스왑되는 `sportPalette(mode, sport)` 헬퍼. 하드코딩 색 사용처를 이 헬퍼로 통일.

### D2. 번호 배지 시스템 (도메인 신뢰감 핵심)
- 경륜 공식 번호색 배지: 1=흰(검정 글자), 2=검정(흰 글자), 3=빨강, 4=파랑, 5=노랑(검정 글자), 6=초록, 7=연분홍(검정 글자). `theme/numberColors.ts`로 정의, 참가자 카드·픽·배당판·시상대 전부 이 배지 사용.
- 경마는 게이트 번호를 중립 다크 배지 + 골드 테두리로 (경륜과 확실히 다르게).

### D3. 시상대(Podium) 예측 요약
- `PredictionSummary`를 개편: 1착 후보를 크게 중앙(번호 배지+이름+모델 확률 대형 수치 1개), 2·3착 후보는 좌우 낮게 — 시상대 메타포. 확률 수치엔 항상 "모델 추정" 마이크로 라벨. 삼쌍(TRI)은 별도 소형 행으로 분리(과신 방지, 확률 그대로 정직 표기).
- 대형 수치는 tabular-nums, 화면당 대형 수치는 이 1개만.

### D4. 데이터 상태 스트립
- 분석 화면 상단 고정 1줄 스트립: [출주표 검증 배지] [배당 신선도(n초 전/미수집)] [시점 phase(마감 전/종료)]를 아이콘+짧은 라벨 3슬롯으로 통합. 상태색: 양호=teal, 주의=amber, 차단=rose. 기존 흩어진 StatusPill들을 이 스트립으로 정리.

### D5. 참가자 카드 접힘 구조
- 기본: 번호 배지+이름+핵심 3지표(경륜: 득점·200m·입상률 / 경마: 복승률·게이트·부담중량)만. 탭하면 전체 profile+근거 reasons 펼침(accordion). 접힘 상태에서도 터치 영역 ≥44px.

### D6. 마감 품질
- 한글 줄바꿈 `keep-all` 적용(긴 이름/설명 깨짐 방지), 카드 radius·간격 토큰 일관화, 라임 on 다크 대비는 contrast 스크립트로 검증, Pro 화면 가격은 강조 절제(대형 배너 금지 — 심사 리스크).

## 검증 산출물
- `npx tsc --noEmit` + `node scripts/check-design-contrast.mjs` 결과를 `.codex-fable5/run_d_evidence.txt`에 저장.
- 홈(경륜)/홈(경마)/분석/Pro 4화면의 구성요소가 어떻게 달라졌는지 5줄 요약을 같은 파일에 append.
- 최종 보고 ≤80자.
