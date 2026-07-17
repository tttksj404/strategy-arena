# Round 6 — KRA(경마) 신뢰도 티어 서버 탑재 (2026-07-12)

Round 5B 확정 결과(`data/kra_market_corpus_results.md`, 6,249경주/64,548출주 측정): 시장 쏠림 티어가 경마 전 권종에서 재현 — 단승 top1 all 36.7% → very_strong_pull 54.2%(커버리지 25.6%), 삼복승 11.0→20.4%, 쌍승 10.7→21.2%, weak_or_open은 24.4%(회피 신호). 경륜 Round 4(커밋 995b862)와 동일 패턴으로 서버에 탑재하라. **모바일 UI 변경 금지, additive 필드만.**

## 1. 티어 아티팩트 동결
- `static/models/kra_confidence_tiers_v1.json`: Round 5B의 티어 정의(very_strong_pull/strong_pull/price_short/gap_wide/weak_or_open 임계값 — `scripts/kra_market_corpus_round5b.py`의 정의 그대로)와 티어×두수버킷(≤7/8-10/11+)별 역사 top1/top3, 측정 코퍼스 크기·기간 메타데이터.
- 티어 판정 입력 = 출주마 단승 배당 목록(이미 live 경로에 존재: winOdds). 두수 버킷 필수.

## 2. engine.py 신규 함수
- `kra_confidence_tier(starters)`: winOdds에서 1·2번인기 배당, 배당비, 두수 → 티어 + 해당 (티어×두수) 역사 top1/top3 반환. winOdds 없음/불완전(2두 미만 유효 배당) 시 None(필드 미포함) — 기존 경로 무변경.
- KRA live-decision 응답에 additive `market_confidence`: {tier, field_bucket, historical_top1, historical_top3, coverage, source:"kra_tiers_v1"}.
- 표기 정직성: "과거 6,249경주(2024-2026) 측정 기준"로만 서술. 수익/시장초과 표현 금지. 엔진 -EV 정직 고지 유지.

## 3. 테스트 (tests/test_kra_confidence_tier.py 신규)
- 티어 경계 fixture(임계 ±ε) 각 티어 1개 + 두수 버킷 경계(7/8, 10/11).
- winOdds 없음/부분 결측/0·음수 배당 → None 폴백, 예외 없음.
- 아티팩트 무결성(로드·임계 단조성·역사율 [0,1]).
- 전체 `python -m pytest tests -q` green (기준 274+5).

## 규율
- 기존 응답 필드·기존 테스트 침해 금지. `data/kcycle_trifecta_snapshots_expansion.jsonl`과 관련 스크레이프 파일 절대 접근 금지(병행 수집 중).
- git commit 금지. 진행은 runs/kra_corpus_progress.md append.
- 완료 보고 ≤50자, file path만.
