# RUN C2 — roster_guard 공식 provider 실동작화 (마지막 P0 조각)

현재 `roster_guard.py`의 `_kcycle_official_names`는 존재하지 않는 URL 패턴(`/race/card/decision/{year}/{tms}/{day}`, 하드코딩 IP 210.90.29.27)을 추측해서 항상 실패하고, 그 결과 live에서 state가 전부 "unverified"라 mismatch 차단이 안 걸린다. 실제 공식 페이지를 파서 provider를 진짜로 작동시켜라.

## 실측 정찰 결과 (이미 확인됨, 활용하라)
- 진짜 페이지: `https://www.kcycle.or.kr/race/card/decision` (GET 200, 서버렌더 HTML ~1.7MB, 로컬에서 접근 가능)
- 파라미터: `stndYear`(2026~2022 select 실존) + `tmsDayOrd`(select id 실존 — 회차·일차 인코딩. 옵션 값 형식은 페이지에서 직접 확인하라)
- 페이지에 광명/창원/부산 3장 모두 포함, 선수 데이터는 `<table class="excel_table">`들로 렌더
- 기본(파라미터 없음) 페이지는 최신 회차만 보여줌 → 과거 날짜는 tmsDayOrd를 정확히 줘야 함
- 저장본 참고: /private/tmp/claude-501/-Users-tttksj-Library-Mobile-Documents-com-apple-CloudDocs/843bd60c-f3a6-49b2-97d4-8be3979497b4/scratchpad/kc.html (기본 페이지)

## 작업
1. 페이지를 직접 fetch해서(네트워크 가능; 만약 sandbox가 네트워크를 막으면 "NETWORK_BLOCKED" 출력 후 중단) `tmsDayOrd` 옵션 값 형식과 날짜 매핑을 확인하라. 옵션 텍스트에 날짜가 있으면 그걸로 date→tmsDayOrd 해석기를 만들고, 없으면 stndYear+옵션 순회로 요청해 페이지 내 날짜 표기와 대조해 캐시하라.
2. `_kcycle_official_names(ymd, meet, race_no)` 재구현: 올바른 URL/파라미터로 fetch → 해당 경륜장(광명/창원/부산) + 경주번호 구획의 선수명 리스트 파싱. www.kcycle.or.kr 도메인 직접 사용(하드코딩 IP 제거). timeout 8s, 실패 시 None(→unverified 유지).
3. **실증 acceptance (필수)**: `verify_roster`가 2026-07-03 광명 1R에 대해 official_names로 정확히 {방종대, 이홍주, 박진응, 최광목, 이승주, 방효찬, 김성진}을 얻는지 실제 네트워크로 확인하고, 그 출력을 `.codex-fable5/run_c2_evidence.txt`에 저장하라. 이 7명이 안 나오면 성공 주장 금지 — 파싱을 고쳐서 될 때까지.
4. 회귀 테스트: 실제 응답에서 해당 경주 부분만 잘라낸 fixture(수십 KB 이하)를 tests/fixtures/에 저장하고, ① fixture 파싱→7명 추출 ② 오염 starters(최건묵·황종대·이흥주·김성진·박유찬·박진홍·이승주)와 대조 시 state=="mismatch" ③ 빈 응답→"unverified" 3케이스 pytest 추가.
5. verification 결과 캐시 TTL은 기존 유지. `python -m pytest -q` 전체 PASS.
6. 커밋 금지(내가 한다). 최종 보고 ≤80자: acceptance 7/7 여부 + pytest 결과만.
