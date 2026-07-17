# KCYCLE 예측률 개선 Round 2 — 평가기 통일 + 라인피처 + conditional logit (2026-07-12)

Round 1 산출물(`data/kcycle_ensemble_gating_results.*`, `data/kcycle_entries.jsonl` 12,593건 조인 90.6%, `data/kcycle_model_market_blend_results.*`)을 이어받아 진행한다.

## Round 1 감사에서 발견된 결함 (최우선 수정)
`data/kcycle_model_market_blend_results.md`에서 top_k=10/20/40의 exact가 전부 동일(블렌드 0.1482/0.1619, current_axis 0.1502/0.1653) — **조합을 더 많이 살수록 exact는 단조 증가해야 하므로 불가능한 출력**. 블렌드 평가기가 top_k 구매 정책을 실제로 적용하지 않은 것으로 판단됨. Round 1의 블렌드 음수 판정은 이 결함 때문에 무효.

## 0. 평가기 통일 (게이트)
- `search_kcycle_global_breakthrough.py`의 split·top_k 평가 경로를 단일 공용 함수로 추출(신규 파일 `scripts/kcycle_eval_common.py`)해 모든 실험이 이것만 쓰게 한다. 기존 파일 수정 금지 — import/복제만.
- **재현 어서션 2개 (통과 전 실험 진행 금지)**:
  A. 새 평가기로 current_axis full test exact = 0.1606±0.003, gen2_mut_436 top_k=20 test exact = 0.1879±0.003 재현.
  B. 단조성: 모든 후보에서 exact(k=40) ≥ exact(k=20) ≥ exact(k=10).

## 1. 블렌드 재평가 (통일 평가기로)
- Round 1의 β-grid 블렌드를 통일 평가기로 재실행. val 선택, test 1회. 결과가 그래도 음수면 음수로 확정 보고.

## 2. 라인-페어 피처 (미시도 도메인 레버)
- 경륜 도메인: 같은 훈련지(trng_plc_nm)/지역 선수는 라인(선행-마크)으로 협조 주행 → 동반 입상 상관이 시장에 덜 반영될 수 있다.
- `data/kcycle_entries.jsonl`로 트리오(i,j,k)-레벨 피처 생성: 동일 훈련지 페어 수(0~3), 등급 갭(1-2위 후보 간), 기어 차, 나이 차, 200m 기록 순위 조합, 승률/연대율 순위 조합.
- 이 피처들을 기존 시장 피처(neg_log_odds, log_q, pair_gap 등 — breakthrough 캠페인과 동일 정의)와 **함께** 선형 스코어에 넣고 train 적합(릿지 또는 로지스틱), val 선택, test 1회. 시장-only 대비 라인피처 추가의 순증분(pp)을 명시 보고.
- 피처 중요도(계수) 상위 10개 보고 — 라인 피처가 실제 기여하는지.

## 3. Conditional logit (러너 단위 직적합)
- 러너별: log(시장 implied win prob — 단승 근사로 board에서 유도 가능한 rank_score 또는 best20 축 빈도) + 출주표 피처(등급 원핫, 승률, 연대율, 기어, 200m z, 나이 z, 훈련지 동료 수).
- PL 우도로 실제 착순(actual_order 1~3위) 학습 (scipy.optimize). train만 적합, val로 정칙화 선택, test 1회.
- PL 확률 → 삼쌍 스코어 → 통일 평가기 top_k 평가.

## 산출물
- `data/kcycle_round2_results.{json,md}`: ①블렌드 재평가 ②라인피처 ③cond-logit 각각 val/test lift 표 + baseline(current_axis, gen2_mut_436, Round1 앙상블) 동표.
- 진행 로그 `runs/prediction_uplift_progress.md` append (기존 파일이지만 append는 허용 — 이 파일만 예외).

## 규율 (Round 1과 동일)
- 선택은 train/val만, test는 후보당 1회. lift pp 정직 보고, 음수는 음수 그대로.
- 기존 파일 수정 금지(진행 로그 append 제외), 신규 파일만. git commit 금지. 키 하드코딩 금지.
- "시장을 이겼다/수익" 표현 금지 — 적중률 lift만.

## 보고
최종 보고 ≤50자, 결과 file path만.
