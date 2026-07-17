# strategy-arena 배포 노트 (경륜/경마 7권종 적중률 예측)

## 무엇인가
- data.go.kr 출주표를 실시간 fetch → 경륜 `keirin_cross_domain_model`/특화 모델 또는
  경마 `kra_model`(모두 win·plc Calibrated)로 채점 → **7권종(단승·연승·복승·쌍승·삼복·쌍복·삼쌍)** 추천픽 표시.
- **+EV 도구 아님.** 적중률 예측 보조 도구이며 공제 후 평균 −EV(검증 완료). 수익 보장 없음.
- 상단 고정 면책 배너 상시 노출(평균 −EV / 도박중독 주의 / 책임베팅 / 만 19세+).

## 경마(KRA) 작동 — 2026-06-23 추가
- **상태: 작동.** 종전 "모델 미탑재"에서 → KRA 예측 모델 탑재·실시간 채점으로 전환.
- 모델: `static/models/kra_model.joblib` (1.1M). keirin 과 **동일 dict 구조**
  `{win, plc, cols, med, num, rel, feats}` + KRA 전용 보조키(`kind`, `global_win_rate`,
  `jk_prior`, `tr_prior`, `meta`). 앱은 keirin 과 같은 방식(per-race 피처 frame → `cols`
  reindex → `med` 채움 → predict_proba)으로 로드·채점.
- 학습 파이프라인: `kra/model_backtest.py` 의 검증된 레시피 그대로 재사용
  (`kra/train_save_model.py`) — HistGradientBoosting(max_depth=4, iter=350) + isotonic
  calibration, within-race 상대피처(`_rel`), win/place 분리, 시간순(rcDate) 학습.
- 라이브 fetch: data.go.kr B551015 `RaceDetailResult_1` (race_result 와 동일 엔드포인트),
  파라미터 `meet`(서울/제주/부경)·`rc_date`(YYYYMMDD)·`rc_no`. 한글명 실패 시 숫자코드 재시도.
- 권종 매핑·Harville 픽 로직은 경륜과 공유(`engine.build_picks`).

## 구조 (gunicorn app:app 보존)
- `app.py` — Flask, 전역 `app`. 라우트 `/`(폼), `/predict`(POST·GET), `/healthz`.
- `engine.py` — 모델 로드 + 실시간 API fetch + cross-domain/qprep2 파이프라인 재현 + Harville 7권종 픽.
- `templates/index.html` — 폼·마번별 win/plc 표·권종 카드·면책 배너(한글).
- `static/models/keirin_cross_domain_model.joblib` — 경륜 일반 모델 3.2M. Elo·모멘텀·팩터·피로 feature 적용, OOS top1 61.6% / 연대 78.3%. 고확신 선별 tier는 `pwin>=60.7%`에서 top1 72.9%(coverage 56.2%), `win gap>=56.5%p`에서 top1 82.1%(coverage 30.3%), `win gap>=63.7%p`에서 top1 84.7%(coverage 21.7%).
- `static/models/keirin_model_final.joblib` — 경륜 모델 3.1M (repo 포함).
- `static/models/kra_model.joblib` — 경마 모델 1.1M (repo 포함).
- `data/demo_race.json` — 모델 회귀 테스트용 과거 1경주(2025.12.28 광명 16R) 캐시. 운영 응답에는 섞지 않음.
- `data/demo_kra_race.json` — 모델 회귀 테스트용 과거 1경주(2026.06.21 서울 6R) 캐시. 운영 응답에는 섞지 않음.
- `render.yaml` / `Procfile` — 유지. `gunicorn app:app` 그대로.

## 라이브러리 핀 (joblib 로드 호환 — 모델 학습 환경 일치)
모델은 sklearn 1.8.0 / numpy 2.4.4 / pandas 2.3.3 / joblib 1.5.3 에서 저장됨.
`requirements.txt` 에 정확히 핀:
```
flask==3.1.0 / gunicorn==23.0.0 / numpy==2.4.4 / pandas==2.3.3
scikit-learn==1.8.0 / joblib==1.5.3 / requests==2.32.3
```
- **render.yaml `PYTHON_VERSION` 을 3.11.0 → 3.13.4 로 변경.** numpy 2.4.4 / sklearn 1.8.0
  휠이 3.11~3.13 을 타깃하고, 모델 저장 환경(3.14)과 가장 가깝게 맞추기 위함.
  (3.11.0 그대로 두면 numpy 2.4.4 휠 해석 위험.) 시작 명령(gunicorn app:app)은 불변.

## Render 환경변수 설정
- **`DATAGOKR_SERVICE_KEY`** (필수) — data.go.kr 발급 인코딩 키.
  Render 대시보드 → Service → Environment → Add Environment Variable.
  키는 코드/커밋에 절대 없음. `os.environ` 으로만 읽음.
- `KEIRIN_CARD_URL` (선택) — 미설정 시 코드 상수 기본 엔드포인트 사용
  (`apis.data.go.kr/B551014/SRVC_OD_API_CRA_RACE_ORGAN/TODZ_API_CRA_RACE_ORGAN_I`).
- `KRA_CARD_URL` (선택) — 미설정 시 `apis.data.go.kr/B551015/API214_1/RaceDetailResult_1`.
  경마는 경륜과 **같은 `DATAGOKR_SERVICE_KEY`** 사용(B551015 활용신청 완료 계정).
- 키 미설정 시: 앱은 죽지 않고 "키 설정 필요" 안내를 반환한다. 다른 날짜의 데모 경주는 운영 응답에 섞지 않는다.

## 제약 (정직 고지)
- **경륜은 광명 경주장만.** 소스 DB(race_card)·API 데이터가 광명만 존재. 폼 경주장 선택지=광명.
- **경마(KRA) 학습 범위: 2024-01-05 ~ 2026-06-21, 서울·제주·부경 3개 경주장, 6,115경주
  / 63,205두.** 이 범위 밖(타 경주장·장기 미래)은 예측 신뢰도 보장 안 됨.
- **경마도 평균 −EV.** OOS 검증(`kra/runs/model_backtest_results.md`)에서 공제(~20%) 후
  intrinsic-feature GBM 이 parimutuel 시장을 못 이김. 본 모델은 "적중률 확률" 도구지 +EV 아님.
- 경마 win 확률은 경주 내 정규화 표시(합=1.0), 연대(plc) 확률은 모델 raw 값.
- data.go.kr OD 카드 API(경륜)는 `stnd_yr` 필터만 서버측 지원(날짜 오름차순). 날짜·경주장·경주번호는
  클라이언트에서 페이지를 넘기며 필터(목표 날짜 통과 시 중단, 최대 25페이지). 한 해 ≈18페이지.
  경마 카드 API(`RaceDetailResult_1`)는 `meet`·`rc_date`·`rc_no` 직접 지원(단일 경주 fetch).
- 권종 매핑은 keirin/pnl_exotic.py 실착순 대조 검증(n=14153)을 따름.
  복/삼복=무순, 쌍승/삼쌍=순서, 쌍복=1착 고정+2·3 무순. 조합픽은 Harville 순서모형 근사(win 내림차순).

## 로컬 검증 결과 (텍스트)
환경: venv(--system-site-packages, Python 3.14) — flask 3.1.3 / sklearn 1.8.0 /
numpy 2.4.4 / pandas 2.3.3 로 해석 성공.

### 경마(KRA) — 2026-06-23 신규 검증
1. **모델 sanity** (`kra/sanity_check.py`): kra_model.joblib 재로드 → 실제 과거 1경주
   (부경 2026-06-19 8R, 11두) 채점 → top-1 win(정규화) 0.204 / plc 0.382,
   win 확률 합=1.000, plc>win 다수, **top-1 픽(센세이셔널윈)이 실제 1착** → SANITY PASS.
2. `GET /healthz` → 200 `{"ok":true,"keirin_model":"loaded","kra_model":"loaded",...}`.
3. **경마 데모 모드** `GET /predict?sport=horse` (키 없음, 서울 2026-06-21 6R, 9두):
   → 200, 7권종 전부 렌더(단승=6번 탭딘, 연승=6/7번, 복/쌍/삼복/쌍복/삼쌍), 에러 0.
4. **경마 라이브 모드** (DATAGOKR_SERVICE_KEY 주입, 더미키): 실시간 fetch 시도 → 실패 시
   graceful 데모 폴백 → 200, 7권종 렌더, "데모 모드/실시간 조회 실패" 안내, 앱 안 죽음.

### 경륜 — 회귀 재확인
5. **경륜 데모 모드** `GET /predict?sport=keirin`: → 200, 채점 결과 + 7권종 렌더, 에러 0
   (KRA 추가로 인한 경륜 경로 회귀 없음 확인).

## 키 노출 점검
- 레포 전체(app.py·engine.py·templates·render.yaml·Procfile·requirements.txt) grep:
  serviceKey 값/40자+ 토큰 매칭 0건. 키는 전부 `os.environ` 참조뿐 (경마·경륜 공통).
- `.gitignore` 에 `.env`, `*.db`, `__pycache__/`, `venv/` 추가. `.joblib` 은 의도적 포함(앱 로드).

## 역순 페이지네이션 (느린 fetch→demo 폴백 수정, 2026-06)
- 경륜 `fetch_race_card`: 정방향 9콜/~8s → 역순(last_page→1) 변경. 최근 경주 뒤쪽이라 1~3콜로 종료, 못 찾으면 정방향 폴백으로 옛 날짜 보장. 실측: 2026-06-20 광명3R **2콜/0.32s**(7두), 옛 2026-01-02 광명1R도 정상(7두, 폴백 10콜). 경마 `fetch_kra_card`는 서버측 meet+date+rc_no 필터라 본래 1경주만 반환→경륜과 동일 totalCount 페이지 보강만, 실측 2026-06-21 서울1R **11두/~3s**.

## 푸시 보류
상위 검토 후 푸시. (이 작업에서 git push 수행하지 않음.)
