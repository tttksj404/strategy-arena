# Round 8 — 경륜 픽 소스 정책 측정 (단승·연승 등 단일권종, 2026-07-12)

KRA에서 확정된 결함 패턴(모델-시장 불일치 시 시장 압승: 30.0% vs 19.3%)의 **경륜 버전을 측정**한다. 순수 측정만 — engine.py 수정 금지(Round 7이 병행 수정 중, 충돌 방지).

## 배경 (재조사 불필요)
- 경륜 앱 픽은 `keirin_model_final.joblib` 출주표 스코어링(7권종). 삼쌍은 Round 4에서 앙상블로 교체됨. 단승 등 단일권종은 여전히 모델 픽.
- 시장 baseline: 삼쌍 보드 주변화 first_mass 기준 top1 = 61.5% (13.9k). 보드가 있으면 단승 시장 1번인기를 유도 가능.
- 데이터: `data/kcycle_trifecta_snapshots.jsonl`(+`_expansion.jsonl` — **읽기만**, 쓰기 절대 금지, 병행 수집 중) × `data/kcycle_entries.jsonl`(12,593 조인) × keirin.db.

## 측정
1. 조인된 레이스에서 model_final로 역사 출주표 스코어링(엔진 score 함수 재사용 — import만, 수정 금지) → 모델 1위/2위 픽.
2. 시장 픽: 보드 first_mass 1위(=삼쌍 보드 주변화 단승 근사). 가능하면 keirin.db payoff로 실제 단승 1번인기 검증 표본 1개 연도.
3. 표 (전체 + 연도별 + 등급전별 가능하면):
   - model top1 vs market top1 vs 일치/불일치 서브셋별 (KRA와 동일 포맷)
   - 연승(top2 내), 복승-쌍승류 근사 가능하면 추가 — 불가하면 단승·연승만.
4. 정책 시뮬레이션: P0=모델 항상 / P1=보드 있으면 시장, 없으면 모델 / P2=P1+불일치·약신호 시 모델. 연도 split(≤2024 train 불필요 — 순수 측정이나, 정책 비교는 전 구간+2026 분리 표기).
5. 판정 기준 제시: P1/P2가 2026 구간에서 P0 대비 +1.0pp 이상이면 교정 후보로 보고.

## 산출물/규율
- `data/kcycle_pick_policy_results.{json,md}`, progress는 runs/prediction_uplift_progress.md append.
- engine.py·app.py·기존 테스트 수정 금지. 신규 파일만. commit 금지. 음수 그대로. 보고 ≤50자 file path만.
