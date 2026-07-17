# Round 4 — 확정 승자 프로덕트화: 앙상블 픽 + 신뢰도 티어 (서버측만, 2026-07-12)

리서치 캠페인 확정 결과(Round 1~3, 통일 평가기·val-선택·test 1회):
- **round1_ensemble**: 단일픽 test exact 0.1821~0.1828 (market_rank 0.1720 대비 +1.0pp, current_axis 0.1606 대비 +2.2pp)
- **강쏠림 게이팅**: strong_pull(n=2,275) exact 35.2% / 상위50% 38.2% / 상위16% 43.4%, top1 85.4%
- 나머지 접근(블렌드·라인·cond-logit·롤링폼) 전부 kill — 재시도 금지.

이 두 승자를 RaceLens 서버(Flask)에 탑재하라. **모바일 UI 변경 금지** — API 응답 additive 필드만.

## 1. 앙상블 아티팩트 동결
- `data/kcycle_global_breakthrough_results.json`의 deployable 후보에서 **val exact 상위 20개**(Round 1과 동일 선택 — `data/kcycle_ensemble_gating_results.json`에 선택 목록 있으면 그대로 재사용)의 수식(피처 항+가중치)을 추출 → `static/models/kcycle_trifecta_ensemble_v1.json`.
- 메타데이터 포함: 선택 기준(val-only), test 수치, 생성일, 코퍼스 크기, strong_pull 티어 임계값(코퍼스 percentile 고정값)과 티어별 역사 적중률(16.5/35.2/38.2/43.4%).
- joblib 금지 — 선형 수식이므로 순수 JSON.

## 2. engine.py 신규 함수 (기존 함수 수정 최소화)
- `kcycle_ensemble_trifecta_rank(board)`: 아티팩트 로드(모듈 캐시) → 210조합 피처 산출(breakthrough 캠페인과 동일 정의 — `scripts/kcycle_eval_common.py` 재사용) → 20수식 rank-average → 정렬된 트리오 리스트.
- `kcycle_trifecta_confidence_tier(board)`: strong_pull 신호(`experiment_kcycle_market_timing_policy.py` 정의 재사용) → 티어 {"T2_top16","T1_strong","T0_base"} + 해당 티어 역사 exact.
- live-decision 삼쌍 경로(board 존재 시): 픽 = 앙상블 1위 트리오로 교체, 응답에 additive 필드 `trifecta_ensemble`: {pick, top5, tier, tier_historical_exact, source:"ensemble_v1"}. board 없으면 기존 Harville 경로 그대로(무변경).
- 표기 정직성: "백테스트 13,900경주 기준 티어별 적중률"로만 서술. 수익/시장초과 표현 금지. 엔진 docstring의 -EV 정직 고지 유지.

## 3. 테스트 (pytest, tests/test_kcycle_ensemble.py 신규)
- 픽 결정성: 고정 fixture board → 동일 픽/티어 재현.
- 티어 경계: 임계값 ±ε board fixture로 T0/T1/T2 각 1개.
- board 없음/불완전(조합 누락) → 기존 경로 fallback, 예외 없음.
- 아티팩트 무결성: JSON 로드, 수식 20개, 가중치 유한값.
- 기존 스위트 침해 없음: `python -m pytest tests -q` 전체 green (기준 185).

## 규율
- 기존 테스트/기존 응답 필드를 깨지 않는다(additive만). ensemble 아티팩트 외 static/models 수정 금지.
- git commit 금지 (Fable5 감사 후 별도 수행). 진행 로그 append 유지.
- 완료 조건: 전체 pytest green + 신규 테스트 green. "완료" 선언 대신 수치·경로만.

## 보고
≤50자, file path만.
