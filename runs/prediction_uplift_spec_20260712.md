# KCYCLE 삼쌍 예측률 개선 캠페인 — 출주표×시장 블렌드 (2026-07-12)

당신은 이 레포의 구현 담당이다. 아래 3개 Phase를 순서대로 구현·실행하라.
최종 감사(PASS/FAIL)는 Fable5가 별도 수행하므로 "완료" 선언 금지 — 각 Phase는 결과 파일과 수치로만 말한다.

## 배경 (사전 조사 완료 — 재조사 불필요)
- `data/kcycle_trifecta_snapshots.jsonl`: 13.9k 레이스. 스키마: date/meet/race_no + `board`(210개 삼쌍 배당) + `actual_order`. **선수/출주표 피처 없음.**
- 시장-only 수식진화(`search_kcycle_global_breakthrough.py`)는 1,841사이클 포화: best `gen2_mut_436` test exact 0.1879 (test-선택 +2.7pp, **val lift +0.16pp = noise**). current_axis(시장 baseline) test exact 0.1606.
- `engine.py`: 출주표 러너 스코어링 모델 존재(`static/models/keirin_model_final.joblib`). 리서치 코퍼스와 **미결합** — 이 결합이 이번 캠페인의 핵심.
- `data/kcycle_market_timing_policy_results.json`: strong_pull 서브셋(n=2275) exact 0.3517 vs 전체 0.1655.

## 성공 기준 (전 Phase 공통)
- **선택(selection)은 train/val만 사용. test는 최종 후보당 정확히 1회.** test로 고르고 test로 보고하는 순환 금지.
- 성공 = val board lift ≥ +1.0pp AND test 확인 lift > 0. 미달이면 미달 수치 그대로 보고.
- lift는 pp 단위 정직 보고. 음수는 음수로. "시장을 이겼다/수익" 표현 금지(공제 미반영) — 적중률 lift만 주장.

## Phase 0 — 퀵윈: 앙상블 + 게이팅 (기존 데이터만, 스크레이프 불필요)
1. `data/kcycle_global_breakthrough_results.json`의 deployable 후보 256개 로드 → **val exact 기준 상위 20개 선택** → 각 후보의 조합 랭킹을 rank-average 앙상블 → top_k∈{10,20,40} 평가. 기존 캠페인과 동일 split·동일 평가 함수 재사용(`search_kcycle_global_breakthrough.py`에서 import 또는 복제).
2. strong_pull 게이팅: `experiment_kcycle_market_timing_policy.py`의 strong_pull 정의를 재사용해 per-race 연속 신호강도 산출 → 커버리지 티어(전체 / 상위 50% / 상위 16%)별 exact·board·top1 표. 이건 lift가 아니라 **정밀도-커버리지 트레이드오프** 표임을 명시.
3. 결과: `data/kcycle_ensemble_gating_results.{json,md}`

## Phase 1 — 출주표 수집 (kcycle.or.kr 공개 페이지, 키 불필요)
1. **피저빌리티 프로브 먼저**: 2018/2022/2025 각 1개 일자에 대해 출주표(선수 정보) 페이지를 찾아 fetch — 경주결과 페이지(`race/result/general/...`)에 선수 등급·기어·승률이 포함되어 있으면 그것을 사용해도 됨. 확인 대상 필드: 등급(SS/S/A급), 기어배수, 200m기록, 승률/연대율/삼연대율, 각질, 훈련지(지역/팀), 나이. `scripts/import_kcycle_full_trifecta_archive.py`와 `engine.py` 상단의 Host/UA/IP 고정 패턴 재사용.
2. 프로브 성공 시 전 기간 수집: 코퍼스의 (date, meet, race_no) 목록 대상, **1~2 req/s 준수**, 일자 단위 페이지면 요청 수 최소화. 체크포인트/재개 가능 구조(`data/kcycle_entries.jsonl` + `.keys` 패턴 동일). 연도별 매칭 커버리지 리포트 의무.
3. 커버리지 <60%면 사실대로 보고하고 가용 연도만으로 Phase 2 진행.

## Phase 2 — 블렌드 모델 (Benter식)
1. 러너 강도 s_i: 1차는 자체 적합 — 출주표 피처를 within-race 상대화(z-score)한 뒤 train split만으로 로지스틱(1착 여부) 적합. (`static/models/keirin_model_final.joblib` 로드가 순조로우면 그 스코어도 피처로 추가.)
2. PL 분해: p_model(i,j,k) = s_i/Σ · s_j/(Σ−s_i) · s_k/(Σ−s_i−s_j) (s는 exp(score)).
3. 시장 prior: 배당 역수 정규화 → p_mkt(i,j,k).
4. 블렌드: score(i,j,k) = β·log p_mkt + (1−β)·log p_model, β ∈ {0.50, 0.55, …, 0.95} — train 적합, **val로 β 선택**.
5. (스트레치, 시간 남으면) conditional logit: 러너별 log(시장 implied win prob) + 출주표 피처로 PL 우도 직접 최적화(scipy).
6. 평가: 기존과 동일 split(train ≤2024 / val 2025 / test 2026 — 기존 캠페인 split 함수 그대로), 동표 비교 baseline = current_axis + gen2_mut_436 (동일 레이스·동일 top_k). 지표: exact / board_exact / top1 × top_k{10,20,40} × (조인된 서브셋, 전체).
7. 결과: `data/kcycle_model_market_blend_results.{json,md}`

## 제약
- **기존 파일 수정 금지 — 신규 파일만 생성.** (검색 루프가 주기 실행 중이라 기존 결과 파일과 충돌 금지. 단 기존 모듈 import는 허용.)
- 키/시크릿 하드코딩 금지. 외부 API 키가 필요한 경로는 쓰지 않는다.
- 진행 상태를 `runs/prediction_uplift_progress.md`에 Phase별 append (타임스탬프 + 1줄).
- git commit 하지 않는다 (감사 후 커밋).

## 보고
최종 보고 ≤50자, 결과 file path만.
