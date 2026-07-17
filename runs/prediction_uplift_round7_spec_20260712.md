# Round 7 — KRA 픽 정책 교정 + 학습창 확장 (2026-07-12)

확정 측정(`data/kra_market_corpus_results.md`): 모델-시장 불일치 레이스(coverage 45.8%, n=2,859)에서 market_top1 30.0% vs model_top1 19.3% (2026: 31.9 vs 22.1 / fresh: 32.2 vs 16.9). v4는 fresh holdout top-1 31.3%, 시장 1번인기 전체 36.7%. 학습창은 2024-01~2026-06뿐.

## 1. 픽 정책 시뮬레이션 (측정 먼저, 코드 변경은 그 다음)
- 현행 정책 문서화: `engine.score_kra`의 use_market 게이트가 정확히 언제 시장 경로를 쓰는지(완전 배당? 신선도? phase?) 코드 기준으로 1문단.
- 6,249 코퍼스(+fresh 구간 분리)에서 정책별 top-1/top-3 표:
  P0 = v4 항상 / P1 = 현행 게이트 재현 / P2 = 배당 존재 시 무조건 시장 1번인기, 부재 시 v4 / P3 = P2 + 불일치 레이스에서 티어가 weak_or_open이면 v4 유지(약신호 시장은 못 믿는다는 가설 검증).
- 판정 기준: P2/P3가 P1 대비 fresh 구간에서 +1.0pp 이상이면 정책 교정 후보.

## 2. 정책 교정 구현 (시뮬레이션이 지지할 때만)
- engine에 config-gated 정책(`KRA_PICK_POLICY` env, 기본 = 시뮬레이션 승자). 기존 응답 구조 불변, 픽 소스만 교정. 픽 소스 필드(`pick_source`: model|market) additive 추가.
- 테스트: 배당 유/무/부분/스테일 각 경로 + 정책 env 스위치. 전체 pytest green (기준 279).

## 3. v4 학습창 확장 (2019-2023, 미시도 레버)
- 어제 캠페인과 동일한 data.go.kr 경로로 2019-01~2023-12 서울/제주/부경 출주·결과 수집(체크포인트/재개, rate limit 준수, `data/kra_history_extension.jsonl`). API가 과거 연도를 안 주면 가용 시작점을 보고하고 거기까지만.
- v4와 동일 아키텍처(dual_phase + history features)로 재학습: train=2019~2025-05, val=2025-06~2026-06 [선택용], **fresh holdout(2026-06-22..07-11 134경주) 은 최종 1회만**.
- 승격 기준(같은 아키텍처·데이터만 확장이므로 v5 때의 +5.0pp 아닌): fresh holdout top-1 ≥ v4 +1.5pp AND val 비열등. 미달 시 미달 수치 보고 후 아티팩트는 enabled=false 보관(v5 가드와 동일 패턴). **승격 실행은 금지 — 수치만 보고, 최종 판정은 Fable5.**
- COVID 연도(2020-21) 분포 이질성: 연도 더미 또는 제외 ablation 1개 포함.

## 산출물/규율
- `data/kra_round7_results.{json,md}`, 진행 runs/kra_corpus_progress.md append.
- 선택은 train/val만·fresh holdout 1회. 음수 그대로. `data/kcycle_trifecta_snapshots_expansion.jsonl` 및 스크레이프 관련 파일 접근 금지(병행 수집 중). 기존 테스트 침해 금지. commit 금지. 키 하드코딩 금지. 보고 ≤50자 file path만.
