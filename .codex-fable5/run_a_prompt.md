# RUN A — 백엔드 P0 수정 (Fable5 감사 20260706 기반)

너는 이 repo의 빌더다. 아래 작업을 전부 구현하고, 각 항목을 논리 단위 커밋으로 남겨라(푸시 금지). mobile/ 디렉토리는 건드리지 마라. 작업 후 `python -m pytest -q` 전체 실행해 실패 0을 확인하라. 기존 테스트를 깨면 고쳐라.

감사 배경(사실): live에서 2026-07-03 광명 1R 출주표가 공식(방종대·이홍주·박진응·최광목·이승주·방효찬·김성진)과 5/7 불일치(최건묵·황종대·이흥주·박진홍·박유찬 표시). data.go.kr 카드 데이터가 이 키에 다른 경주 로스터를 반환하는 것으로 추정. 오염된 이름이 data/participant_learning_priors.json에 학습돼 있음.

## A1. 출주표 검증 게이트 (최우선)
- 새 파일 `roster_guard.py`: `verify_roster(sport, ymd, meet, race_no, starters) -> {"state": "verified"|"unverified"|"mismatch", "official_names": [...], "checked_at": iso}`.
- 공식 소스 provider: 경륜은 kcycle.or.kr 공식 출주표(HTTP 조회, 실패 허용), 경마는 data.go.kr 결과 API(RaceDetailResult_1)의 마명 대조(결과 존재 시). provider 실패/미구현 → state "unverified" (절대 verified로 조작 금지).
- 이름 대조: 공백 제거 후 이름 집합 비교. 2명 이상 불일치 → "mismatch".
- engine의 live-decision/predict 경로에 통합: mismatch면 picks/top/rows를 내리지 말고 `status:"roster_mismatch"`, 정직한 메시지("공식 출주표와 일치하지 않아 예측을 중단했습니다")로 응답. verified/unverified는 응답에 `roster_verification` 필드로 상태 노출.
- 캐시: 검증 결과 30분 캐시. 네트워크 오류 반복 시 음수캐시 60초.
- 런타임 invariant assert 3개: ① mismatch 응답에 picks 없음 ② roster_verification 필드는 세 상태 중 하나 ③ verified인데 official_names 비어있으면 assert 실패.

## A2. 포렌식·오염 정리 스크립트
- `scripts/audit_roster_consistency.py`: 최근 N일(기본 14) 경륜 경주에 대해 카드 API 로스터 vs 공식 provider 로스터 비교 → `data/roster_audit_report.json` + 요약 MD. 키 없거나 provider 실패 항목은 "unchecked"로 정직 기록.
- `scripts/rebuild_learning_priors.py`: roster_audit_report에서 mismatch로 판정된 (date,meet,race_no)의 예측/결과 레코드를 제외하고 priors 재계산. 기존 파일은 `data/participant_learning_priors.backup_YYYYMMDD.json`으로 보존. report 없으면 안전 중단.

## A3. force_pro 프로덕션 가드
- datastore `_forced_pro_enabled`: `RACELENS_ENV=production`이면 RACELENS_FORCE_PRO/preview 플래그 무시 + 경고 로그 1회. `RACELENS_PRO_DEVICE_IDS` 명시 목록만 허용.
- `/healthz` 응답에 `entitlement_mode: "production"|"preview"` 노출(스모크용).

## A4. 쿼터 우회 방어
- 같은 IP에서 신규 익명 user 생성 일일 상한(기본 5, env RACELENS_IP_NEW_USER_CAP) — 초과 시 기존 최근 user 재사용 또는 429. `/api/live-decision` IP당 분당 상한(기본 30). DB 테이블 재사용(analytics 또는 신규 소형 테이블).

## A5. 법적 문서
- `RACELENS_SUPPORT_EMAIL` env(기본 tttksj@gmail.com)로 support@example.invalid 전부 치환.
- 약관에 추가: 구독 자동갱신·해지 방법·환불(스토어 절차 안내)·만 19세 이상 이용 조항·시행일. 처리방침에 실연락처+보존기간 명시.

## A6. 기타 정리 (각각 소커밋)
- `/api/app-data-layer`: `RACELENS_ADMIN_TOKEN` 헤더 없으면 404.
- `/predict` GET: 루트로 301 리다이렉트(POST API 사용처 있으면 JSON은 유지).
- 에러 문구 분리: 미개최일("해당 날짜에는 경주가 없습니다")/미지원 경주장/상류 API 장애/키 미설정을 서로 다른 message+`error_kind`로.
- 경마 settled: 과거 날짜 경마는 RaceDetailResult_1 결과를 붙여 snapshot_phase "settled_result"로. 경마 메시지에서 "KCYCLE" 문구 제거(종목별 문구 분리).
- UX enum에서 'lab' 제거(app.py).

## 완료 조건
- pytest 전체 PASS + 신규 기능마다 positive/negative 테스트 (mismatch fixture로 차단 확인 포함).
- 커밋 메시지 prefix: fix(roster)/fix(entitlement)/fix(quota)/docs(legal)/fix(api)/fix(horse).
- 최종 보고 ≤80자: 커밋 수 + pytest 결과만.
