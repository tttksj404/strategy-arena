# Round 10 — 잔여 권종 픽 소스 측정 (경륜 4권종 + 경마 연승 + 약신호 하이브리드, 2026-07-12)

확정 지식: 픽 소스 교정이 최대 점프였다(경마 단승 +6.7pp, 경륜 단승 +4.8pp — 커밋 2616500/065a7c7). 아직 모델 픽으로 남은 권종들에 같은 결함이 있는지 측정한다. **순수 측정만, engine 수정 금지.**

## A. 경륜 잔여 4권종 (데이터: snapshots 17,312 + entries 12,593)
- 삼쌍 보드 주변화로 각 권종의 시장 픽 유도:
  복승(2착내 2마리): unordered pair mass 상위 / 쌍승(1-2 순서): ordered pair mass 상위 / 삼복승(3착내 3마리): unordered trio mass 상위 / 쌍복승(1-2착 unordered): pair mass 상위.
- 각 권종 적중 정의로 model_final 픽 vs 보드 주변화 픽 vs 불일치 서브셋 (R8 포맷 그대로, 연도 split 표기).
- 주의: 보드 주변화는 삼쌍 pool 기준이라 해당 pool 자체 배당과 다를 수 있음 — "삼쌍보드 근사" 명시.

## B. 경마 연승(place) 픽 (데이터: kra_market_corpus 6,249)
- plcOdds 최저(시장 연승 1위) vs v4 place 픽(pplc 최고) — top1이 3착내 드는 비율, 불일치 서브셋 분해, fresh 구간 분리.

## C. 약신호 하이브리드 미세 최적화
- weak_or_open 티어(경마)·비강쏠림(경륜)에서만 모델이 시장을 이기는지: 티어×(모델/시장) 매트릭스. 이긴다면 P3-refined 정책(약신호만 모델) 시뮬레이션 수치 제시.

## 판정 기준
- 각 항목: 시장(또는 하이브리드)이 2026/fresh 구간 +1.0pp 이상이면 교정 후보로 표기.

## 산출물/규율
- `data/round10_pick_extension_results.{json,md}`, progress append. 신규 파일만. commit 금지. 음수 그대로. 보고 ≤50자 file path만.
