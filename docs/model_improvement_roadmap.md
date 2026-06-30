# 경륜·경마 예측 개선 로드맵

목표는 수익 보장이 아니라, 사용자가 보는 예측 확률과 신뢰도를 더 재현 가능하게 만드는 것이다. 기존 OOS 결과상 경마·경륜 파리뮤추얼 시장을 안정적으로 이기는 +EV는 확인되지 않았으므로, 모델 개선과 베팅 엣지 헌팅은 분리한다.

## 이미 소진한 축

경륜:
- 기본 GBM + isotonic, within-race 상대 피처, Harville 7권종 픽.
- rest_days, racer expanding, grade strength, all_features, LambdaRank 조합.
- final, 11R+, special 11R 특화 모델.
- 배당 밴드, 조건부 슬라이스, favorite/longshot, 엑조틱 Harville, 직접 EV 헌팅.
- 최신 결과상 작은 정확도 증분은 `M_all_lambdarank` 계열의 top1 +0.44pp, 연대 +0.49pp가 가장 현실적이다.

경마:
- KRA GBM + isotonic, 기수·조교사 prior, within-race 상대 피처.
- 단승·연승·복승·쌍승·삼복·삼쌍, 복연승, 필드사이즈 포켓, 구조적 페이스 포켓.
- 조건부로짓, 랭크마진, 스택까지 OOS 게이트 실패.
- 마감 배당 시장은 top1, log-loss, ECE에서 순수 모델보다 강했다.

## 다음 우선순위

1. KRA dual-phase odds router
   - 사전 예측 모델과 live odds 모델을 분리한다.
   - 배당 스냅샷이 없을 때 `winOdds/plcOdds=0`으로 들어가는 상태를 별도 사전 모델로 분리한다.
   - 배당이 있을 때만 market/implied probability를 calibration 또는 blend에 사용한다.

2. Keirin learned meta-router
   - 현재 앱은 race_no와 등급 threshold로 base/final/11R/special 모델을 고른다.
   - OOF meta-router가 race_no, day_tcnt, grade, field_size, pwin gap으로 어떤 모델을 쓸지 고르게 한다.
   - 단순 평균 앙상블은 이미 약했으므로, 평균이 아니라 레짐별 선택 문제로 검증한다.

3. Keirin rest/expanding/LambdaRank candidate
   - 이미 작은 양의 정확도 증분이 나온 조합을 seed/time split 안정성으로 재검증한다.
   - 통과하면 새 joblib 후보를 만들고 배포 모델 교체 전 current-router와 A/B 비교한다.

4. Regime-specific calibration and confidence policy
   - 종목, 경주장, race_no bucket, field_size, top1-top2 gap별 ECE/Brier를 따로 본다.
   - “모든 경주 추천”보다 “예측 가능한 경주만 final_candidate”가 조건부 적중률을 더 올릴 수 있다.

5. Direct combo ranker
   - 7권종은 현재 win 확률 기반 Harville 순서로 만든다.
   - pair/trio 직접 랭커를 만들어 복승·쌍승·삼복·삼쌍 hit-rate만 Harville 대비 검증한다.
   - ROI는 별도 게이트를 통과하기 전까지 주장하지 않는다.

## 공통 통과 기준

- 시간순 OOS 또는 walk-forward만 사용한다.
- 배당은 benchmark/settlement/live phase에서만 쓰고, 사전 모델에는 넣지 않는다.
- top1, plc-top2, log-loss, Brier/ECE를 함께 본다.
- 개선 주장은 paired bootstrap by race로 검증한다.
- +EV 표현은 ROI>0, boot_neg<30%, top10 제외 ROI>0, negative control 붕괴, 다중검정 보정까지 통과할 때만 쓴다.

## 실행 도구

다음 실험 순서는 아래 CLI가 점수화한다.

```bash
python tools/model_improvement_matrix.py --sport all
python tools/model_improvement_matrix.py --sport horse --format json
```
