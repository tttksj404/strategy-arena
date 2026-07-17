# Round 10b — 전 권종 시장 앵커 픽 확장 (2026-07-12)

R10 측정 확정(`data/round10_pick_extension_results.md`): 시장(보드 주변화/plcOdds) 픽이 전 권종에서 모델 픽 압도 — 경륜 복승 +7.96pp(2026 +11.27pp)·쌍승 +7.84pp·삼복승 +6.02pp, 경마 연승 +7.07pp(전 티어 우위, 하이브리드 무이득 switch_n=0). R8b(경륜 단승·연승, 커밋 065a7c7)와 R7(경마 단승, 2616500) 패턴을 그대로 확장하라.

## 구현
- **경륜**: 기존 `KEIRIN_PICK_POLICY=market_if_board` 게이트 아래에서 복승·쌍승·삼복승·쌍복승 픽도 삼쌍 보드 주변화로 교체(R10 측정 코드의 주변화 정의 재사용 — 복승=1-2슬롯 unordered pair mass 최대, 쌍승=ordered pair mass, 삼복승=unordered trio mass, 쌍복승=1-2착 unordered pair mass). 보드 없음/불완전 시 각 권종 기존 model_final 경로. pick_source 필드는 권종별로 정확히 표기.
- **경마**: `KRA_PICK_POLICY=market_if_odds` 아래에서 연승 픽 = plcOdds 최저(유효 양수). plcOdds 결측 시 기존 v4 pplc 경로.
- 표기: 경륜 시장 픽은 "삼쌍 배당 기준 근사" 명시(네이티브 pool 배당 아님). 수익 표현 금지.

## 테스트
- tests/test_keirin_pick_policy.py 확장 + tests/test_kra_prediction_phase.py 확장: 권종별 market/model 폴백 각 1, 주변화 결정성 fixture 1, env=model_always 회귀.
- 전체 `python -m pytest tests -q` green (기준 289).

## 규율
기존 필드·테스트 침해 금지 / commit 금지 / progress append / 보고 ≤50자 file path만.
