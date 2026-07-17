# RaceLens Backend and Models

RaceLens는 경륜·경마 공식 데이터, 모델 산출물, 앱 세션 데이터를 제공하는 Flask 백엔드와 모바일 앱 저장소입니다. 사용자에게 표시되는 표현은 "예측 픽"이 아니라 경주 데이터 기반의 적중률 예측 분석입니다.

## 안전 고지

RaceLens의 적중률 예측 분석은 수익을 보장하지 않습니다. 경륜·경마 OOS 검증에서 마감 배당 시장은 공제율 영향으로 평균 -EV가 확인됩니다. RaceLens는 정보 제공 및 검증 가능한 분석 보조 도구이며, 만 19세 이상 이용과 책임 있는 이용 원칙을 전제로 합니다.

## 구성

- `app.py`: RaceLens API 서버와 legal 웹 루트
- `engine.py`: 경륜·경마 모델 분석 로직
- `datastore.py`: 앱 세션, 무료 분석 quota, 영수증 검증 결과, 익명 UX 이벤트 저장
- `mobile/`: Play Store/App Store 제출용 Expo 앱
- `templates/`: 심사자가 도메인 루트에서 확인하는 웹 UI
- `tests/`: 서버 데이터 계층 및 정책 회귀 테스트

## 실행

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

실시간 공식 데이터가 필요한 환경에서는 `DATAGOKR_SERVICE_KEY`를 환경변수로 주입합니다. 키가 없거나 API가 실패하면 앱은 검증 전 데이터를 표시하지 않고 안전한 대기 상태를 반환합니다.

## 배포

Oracle Korea 리전 배포를 기본 경로로 사용합니다.

```bash
cp deploy/oracle/.env.oracle.example deploy/oracle/.env.oracle
# deploy/oracle/.env.oracle에 DATAGOKR_SERVICE_KEY 설정
docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle up -d --build
```

상세 이전 절차는 `docs/oracle_migration_runbook.md`를 참고합니다. Render는 실시간 배당 운영 경로로 사용하지 않습니다.

## 검증

```bash
.venv\Scripts\python.exe -m pytest tests/test_app_data_layer.py -q
cd mobile
npm run typecheck
npm run qa:mobile
```

스토어 제출 전에는 모바일 `release.env.example`을 기반으로 실제 HTTPS API, 개인정보처리방침, 이용약관, 계정 삭제 URL, 지원 이메일, billing mode를 주입해야 합니다. placeholder 값은 제출 readiness 게이트에서 차단됩니다.
