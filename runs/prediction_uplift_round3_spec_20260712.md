# KCYCLE 예측률 개선 Round 3 — 롤링 폼(선수 시계열) 피처 (2026-07-12)

Round 2까지 확정: 출주표 요약 스탯(통산 승률·등급·기어 절대값)은 시장에 완전 반영 → 블렌드/라인/cond-logit 전부 음수 kill. 마지막 미시도 로컬 레버 = **선수별 시계열(롤링 폼)**. `scripts/kcycle_eval_common.py`(Round 2에서 통일된 평가기)와 `data/kcycle_entries.jsonl`(12,593 레이스)을 그대로 재사용한다.

## 1. 롤링 폼 피처 구축 (누출 방지 최우선)
`data/kcycle_entries.jsonl` × 스냅샷 `actual_order`로 선수별 경주 이력 시계열 구성 (racer_nm 기준, 날짜 오름차순):
- **top3_rate_last5 / last10**: 직전 5·10출주 입상(top3)률 — actual_order의 back_no→racer_nm 매핑으로 전 선수 top3 여부 도출 가능
- **days_since_last**: 직전 출주와의 간격 (휴장 복귀 신호)
- **gear_delta**: 이번 기어배수 − 직전 경주 기어배수 (기어 인상 = 컨디션 자신감, 경륜 고전 신호)
- **rec200_delta**: 200m 기록 직전 대비 변화 (기록 갱신 = 상승 폼)
- **grade_change**: racer_grd_cur_cd vs racer_grd_bef_cd (승급/강급 직후)
- **streak**: 직전 연속 입상/연속 미입상 길이
- **meet_top3_rate**: 해당 경주장(meet) 한정 입상률 (과거만)

**누출 금지 어서션 (통과 전 진행 금지)**: 모든 롤링 피처는 해당 레이스 date **미만**의 이력만 사용. 검증: 임의 선수 3명 샘플에서 특정 레이스 t의 피처를 t 이후 데이터 삭제 후 재계산해 동일함을 assert.

## 2. 평가 (2 트랙, 통일 평가기)
- **트랙 A — 러너 레벨**: Round 2 conditional logit 코드 재사용, 피처를 롤링 폼으로 교체(+시장 log_first_mass 유지). train 적합, val 정칙화 선택, test 1회.
- **트랙 B — 트리오 레벨 가산**: 시장 피처(Round 2 line_features의 시장부분과 동일) + 트리오 요약 롤링 피처(트리오 3인의 form 합/최소/기어인상 인원수)를 릿지로 적합. val 선택, test 1회.
- 지표: selection/purchase exact × top_k{10,20,40}, baseline 동표(market_rank, gen2_mut_436, round1_ensemble).
- **추가 리포트**: form 피처 상위 계수 10개 + 서브그룹 분석 — 휴장 복귀(days_since_last>30)·기어 인상 선수 포함 레이스에서의 국소 lift (전체 lift가 0이어도 국소 신호가 있으면 게이팅 재료가 됨).

## 3. (보너스, 시간 남으면) 앙상블 확장
round1_ensemble(현 1등, test +1.0pp)에 트랙 B 스코어를 rank-average 멤버로 추가했을 때 val/test 변화 1줄.

## 산출물
- `data/kcycle_round3_results.{json,md}` + `runs/prediction_uplift_progress.md` append.

## 규율 (동일)
선택은 train/val만·test 1회 / lift pp 정직(음수 그대로) / 기존 파일 수정 금지(progress append 제외) / commit 금지 / "시장 이김·수익" 표현 금지.

## 보고
≤50자, file path만.
