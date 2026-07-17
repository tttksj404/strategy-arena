# RUN C2C — roster_guard 공식 provider 완성 (오프라인, 네트워크 불필요)

너는 빌더다. **mobile/ 수정 금지**(병렬 작업 중). 커밋 금지. 네트워크 접근 금지 — 아래 실측 정보와 fixture만으로 구현하라. 완료 조건: `python -m pytest -q` 전체 PASS.

## 실측 확정 정보 (Fable5가 실제 kcycle.or.kr에서 검증 완료 — 그대로 구현)
1. 베이스 페이지 `GET https://www.kcycle.or.kr/race/card/decision` (HTML, ~1.7MB): `<select id="tmsDayOrd">` 옵션이 날짜 매핑을 제공. 형식: `<option value="27-1">(27회 1일) 07월 03일 (금요일)</option>`. 연도는 `stndYear` select. 과거 연도는 `GET /race/card/decision/tmsDayOrd/{year}`가 해당 연도 옵션 fragment를 반환(common-race.js의 search.year 로직 — 이 엔드포인트는 미검증이므로 실패 시 무시하고 당해 연도만 지원).
2. 실데이터 fragment: `GET https://www.kcycle.or.kr/race/card/decision/{year}/{tms}/{dayOrd}` (예: /2026/27/1) → 3개 경륜장 전체 카드 HTML. 경주 헤더 텍스트 예: `광명 01경주 (선발 12:55) 5주회 선두고정(1691m)`. 선수 행 셀 예: `1황종대 09기/48세` (번호+이름 결합, 뒤에 기수/나이).
3. 2026-07-03 광명 01경주 공식 명단(fixture로 저장됨): 1황종대 2이흥주 3박진홍 4최건묵 5이승주 6박유찬 7김성진. **주의: 이 명단은 앱의 data.go.kr 데이터와 일치함이 확인됐다** — 즉 이 경주의 기대 state는 "verified"다.

## Fixtures (이미 repo에 있음)
- `tests/fixtures/kcycle_card_20260703_gm_r1.html` — 광명 01경주 헤더+테이블 포함 fragment 조각
- `tests/fixtures/kcycle_tmsdayord_options.html` — tmsDayOrd select 원문

## 작업
1. `roster_guard.py`의 `_kcycle_official_names(ymd, meet, race_no)` 재구현:
   - `https://www.kcycle.or.kr` 도메인 직접 사용(기존 하드코딩 IP 210.90.29.27, Host 헤더 트릭, `_resolve_kcycle_tms` 주차 계산 휴리스틱 제거).
   - 단계: ① 베이스 페이지 fetch → tmsDayOrd 옵션 파싱(`(\d+)-(\d+)` value + 옵션 텍스트의 `MM월 DD일`)으로 date→(tms,dayOrd) 매핑, stndYear 옵션의 당해 연도 확인. 매핑 결과 12h 캐시. ② 목표 날짜 매핑 존재 시 fragment URL fetch. ③ fragment에서 `{meet} {race_no:02d}경주` 헤더 구획을 찾고 그 다음 테이블에서 `(\d)([가-힣]{2,4})\s+\d+기` 패턴으로 (번호,이름) 추출. 이름 리스트 반환. 실패·부재 시 None.
   - 모든 fetch timeout 8s, 예외는 None(→unverified). 절대 이름을 지어내지 마라.
2. pytest (fixture 기반, 네트워크 없이):
   - options fixture 파싱 → ("2026-07-03")→(27,1) 매핑 확인
   - card fixture 파싱 → 광명 1R 7명 = [황종대,이흥주,박진홍,최건묵,이승주,박유찬,김성진] 순서/집합 확인
   - verify_roster: 같은 7명 starters → "verified" / 2명 이상 다른 starters(방종대·박진응 등 삽입) → "mismatch" / provider None → "unverified"
   - fetch 함수는 monkeypatch로 fixture 주입.
3. 기존 테스트 깨지면 수정. 최종 보고 ≤60자: pytest 결과만.
