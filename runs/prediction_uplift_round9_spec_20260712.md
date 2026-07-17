# Round 9 — KRA v4 학습창 확장 재학습 + 사전예측 앙상블 (2026-07-12)

전제: `/Users/tttksj/kra/data/kra.db`에 2019-2026 race_result/horse/dividend 수집 완료(2023: race_result +26,073행 확인). 현행 production = v4(dual_phase, 학습 2024-01~2026-06, fresh holdout top-1 31.3%). 픽 정책은 이미 market_if_odds(커밋 2616500)라 **이 라운드의 표적은 배당 없는 사전예측(pre-odds) 구간의 모델 품질**이다.

## 1. 학습창 확장 재학습
- v4와 동일 아키텍처·피처 파이프라인(kra/train_save_model.py 재현 경로)을 학습 데이터만 2019-01~2025-05로 확장해 재학습.
- split: train 2019-01~2025-05 / val 2025-06~2026-05 (선택용) / **fresh holdout 2026-06-22~07-11 134경주는 최종 1회만**.
- ablation 3개: (a) 2020-21 COVID 연도 제외, (b) 연도 가중(최근 가중), (c) sectional time 피처 추가(bu_*fGTime·je_*cTime 컬럼 — 결측 처리 명시).
- 평가 지표: pre-odds 조건 top-1/top-3(배당 피처 없이 예측), 전체 top-1/top-3, log-loss. baseline = 현행 v4 동표.

## 2. 사전예측 앙상블 (kcycle Round1 패턴)
- 어제 캠페인 후보 풀(tools/kra_diversified_search.py·kra_drug_discovery_* 산출물이 남아 있으면 재사용, 없으면 1의 ablation 변형 4~6개로 대체)을 **val-선택 상위 rank-average** → fresh holdout 1회.

## 3. 승격 판정 자료 (실행 금지, 수치만)
- 기준 제시: fresh holdout top-1 ≥ v4 +1.5pp AND val 비열등 AND pre-odds 구간 개선.
- 충족 후보가 있으면 아티팩트를 `static/models/kra_model_v6_candidate.joblib` + enabled=false 게이트로 보관(v5 가드 패턴). **production 교체 금지 — 최종 판정은 Fable5.**

## 산출물/규율
- `data/kra_round9_results.{json,md}`, 진행 runs/kra_corpus_progress.md append.
- 선택 train/val만·fresh holdout 후보당 1회. 음수 그대로. 기존 테스트 침해 금지. commit 금지. 키 하드코딩 금지. 보고 ≤50자 file path만.
