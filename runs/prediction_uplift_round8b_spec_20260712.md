# Round 8b — 경륜 픽 소스 정책 구현 (2026-07-12)

R8 측정 확정(`data/kcycle_pick_policy_results.md`): 보드 존재 시 시장 1번인기가 모델 픽 대비 2026 단승 top1 +4.76pp(59.0→63.8), 불일치 서브셋 +24.7pp. R7의 KRA 구현(`KRA_PICK_POLICY`, 커밋 2616500)과 동일 패턴으로 경륜에 구현하라.

## 구현
- `KEIRIN_PICK_POLICY` env (기본 `market_if_board`, 대안 `model_always`). 삼쌍 보드가 존재·완전(210조합 양수)할 때 단승 픽 = 보드 first_mass 1위, 연승 픽 = first_mass 상위 2. 보드 없음/불완전 시 기존 model_final 경로 그대로.
- 응답에 additive `pick_source`(market|model) — KRA와 동일 네이밍. 기존 필드·포맷 불변.
- 다른 권종(복승·쌍승·삼복승·쌍복)은 이번에 건드리지 않는다(보드 주변화 근사의 검증이 단승·연승만 돼 있음).
- Round 4의 trifecta_ensemble 경로와 독립 — 상호 간섭 금지.
- 표기: "시장 배당 기준 픽" 명시. 수익 표현 금지, -EV 고지 유지.

## 테스트 (tests/test_keirin_pick_policy.py 신규)
- 보드 완전 → market 픽 + pick_source=market / 보드 없음·불완전(<210, 0·음수 포함) → model 경로 + pick_source=model / env=model_always → 항상 model.
- 결정성 fixture 1개. 전체 `python -m pytest tests -q` green (기준 285).

## 규율
기존 테스트 침해 금지 / commit 금지 / data/kcycle_trifecta_snapshots_expansion.jsonl 쓰기 금지 / progress append / 보고 ≤50자 file path만.
