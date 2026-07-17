# Round 5A — kcycle 코퍼스 확장 + 캘리브레이션 (2026-07-12)

전제(재조사 불필요): 코퍼스 13,900 = 광명 12,796 / 창원 1,116 / 부산 201, 연도 갭 2024(770)·2025(1,753 부분). 원본 DB `/Users/tttksj/keirin/data/keirin.db`: race_result·race_card·payoff(59,202행: pool1=단승 등 멀티풀 확정배당)·racer_info. 통일 평가기 `scripts/kcycle_eval_common.py`. 기존 결론: 펀더멘털 kill, 생존=앙상블(+1.0pp)·쏠림 티어. Round 4 아티팩트 v1 탑재 완료(커밋 995b862).

## 1. 코퍼스 확장 (최우선 — 표본 +60~80% 기대)
- keirin.db race_result에서 (연도,경주장)별 경주 수 산출 → 현 스냅샷 코퍼스와 대조표 = 갭 목록.
- `scripts/import_kcycle_full_trifecta_archive.py` 로직 재사용(신규 스크립트로 복제)해 **갭 레이스의 dividendra 삼쌍 보드 스크레이프**: 창원·부산 전 연도 + 광명 2024·2025 갭. 1~2 req/s, 체크포인트/재개, `data/kcycle_trifecta_snapshots_expansion.jsonl`에 기록(기존 snapshots 파일 수정 금지 — 평가 시 두 파일 concat).
- 시간이 오래 걸리면(예상 2~4h) 끝까지 돌린다. 진행률을 progress에 30분마다 append.
- 확장 후: 연도×경주장 커버리지 표 + '2999' 같은 이상 레코드 정리 규칙 보고.

## 2. 확장 코퍼스 재평가 (앙상블 v2 후보)
- 동일 연도 split(train ≤2023 / val 2024-2025 / test 2026 — 확장으로 2024 val이 실질화됨)·통일 평가기.
- deployable 후보 풀에서 val-상위 20 재선택 → rank-average → test 1회. v1(현행)과 동표 비교.
- 쏠림 티어 임계·티어별 정밀도 재추정 (경주장별 분리표 포함 — 광명/창원/부산 티어 정밀도가 다르면 경주장별 임계).
- val에서 v1 대비 개선 시 `static/models/kcycle_trifecta_ensemble_v2_candidate.json` 생성 (배포 금지 — Fable5 감사 대기).

## 3. 캘리브레이션 (신규 신호 — 보드 자체 편향)
- **A. 보드 내 등화**: train에서 삼쌍 combo implied prob(배당 역수 정규화) 버킷별 실현율 → isotonic 보정 곡선 → 보정 후 재랭킹 → val/test (통일 평가기).
- **B. 크로스풀 앵커**: keirin.db payoff의 단승 확정배당(pool1_val) vs 삼쌍 보드 주변화 win prob(first_mass) — 승자 표본 기반 버킷 보정 곡선(train 연도만) → 보드 주변화 확률 보정 → 재랭킹 실험. (승자 표본 선택편향 한계를 결과에 명시.)
- 각각 val 선택·test 1회, 음수면 음수 그대로.

## 산출물/규율
- `data/kcycle_round5a_results.{json,md}`, progress append. 선택은 train/val만·test 1회. 단조성 어서션 유지. 기존 파일 수정 금지(progress 제외). commit 금지. 보고 ≤50자 file path만.
