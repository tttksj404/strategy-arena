# 경륜·경마 7권종 예측 (Race Predictor)

data.go.kr 실시간 출주표 + 보정(isotonic) GBM 모델로 **경륜(광명)·경마(서울/제주/부경)** 의 7권종(단승·연승·복승·쌍승·삼복·쌍복·삼쌍) **최고확률 픽**을 예측하는 Flask 웹앱.

## ⚠️ 정직 고지
**적중률(예측) 도구이지 수익 도구가 아니다.** 전 권종 OOS 백테스트(경륜 N=9,558·경마 6,115경주)에서 마감배당 시장은 효율적 — 공제(20~27%) 때문에 **평균 −EV**가 확정됐다. 모델은 확률을 잘 추정하지만 *수익*은 보장하지 않는다. 도박중독 주의·책임베팅·만 19세 이상.

## 검증된 예측력 (적중률)
- 경륜 일반 모델: 교차분야 feature(Elo·모멘텀·팩터·피로) 적용 후 OOS 단승 top-1 61.6%, 연대 78.3% (기존 60.1%/77.3% 대비 +1.51pp/+1.05pp)
- 경륜 일반 모델 고확신 선별: `pwin>=60.7%` 구간 OOS top-1 72.9%(coverage 56.2%), `win gap>=56.5%p` 구간 OOS top-1 82.1%(coverage 30.3%), `win gap>=63.7%p` 구간 OOS top-1 84.7%(coverage 21.7%)
- 경륜 결승전 고확신 구간: top-1 약 77~78% (coverage 약 6.9%)
- 경마: 입상 예측 강함 (연승 demo 5/5), 단승은 시장인기마 동급

## 실행
```bash
pip install -r requirements.txt
export DATAGOKR_SERVICE_KEY="<data.go.kr 인증키>"   # 미설정 시 데모 폴백
gunicorn app:app --bind 0.0.0.0:$PORT
```
종목·날짜·경주장·경주번호 선택 → 출주표 실시간 fetch → 7권종 예측 픽 + 마번별 win·연대 확률.

## 배포

### Oracle Korea 권장
KCYCLE 실시간 배당을 쓰려면 Oracle Seoul/Chuncheon 같은 한국 리전 배포가 기본이다.

```bash
cp deploy/oracle/.env.oracle.example deploy/oracle/.env.oracle
# deploy/oracle/.env.oracle 에 DATAGOKR_SERVICE_KEY 설정
docker compose -f deploy/oracle/docker-compose.yml --env-file deploy/oracle/.env.oracle up -d --build
```

상세 이전 절차와 리스크 closure: `docs/oracle_migration_runbook.md`.

### Render fallback
`render.yaml`(gunicorn app:app) 자동배포. **Render → Environment 에 `DATAGOKR_SERVICE_KEY` 설정 필수**(실시간 데이터용; 미설정 시 데모만). Render는 한국 리전이 없어 KCYCLE live market은 `market_used=false`로 폴백될 수 있다.

## 제약
- 경륜 = 광명만 (출주표 API가 광명만 제공). 경마 = 서울·제주·부경.
- 모델 학습 범위: 경마 2024–2026. 모델 파일 `static/models/`.

상세: `DEPLOY_NOTES.md`. 연구 전말: github.com/tttksj404/keirin-ev, kra-ev.
