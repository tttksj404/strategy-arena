# Round 5B — KRA(경마) 시장 코퍼스 신설 + 티어 이식 (2026-07-12)

전제: 어제(7/11) KRA 캠페인이 data.go.kr 공식 결과를 로컬에서 수집했다(runs/kra_fresh_holdout_20260622.json, 1,358행/134경주) — 그때 사용한 API 접근 경로(키 포함)를 재사용하라. 현행 KRA production = v4(top-1 31.3% fresh holdout), pairwise candidate enabled=false. 앱은 어제부터 live 배당 스냅샷 축적 시작(datastore.record_market_odds_snapshot_safely). 경륜 쪽 확정 지식: 쏠림 티어(정밀도-커버리지)가 사용자 가치 최대, 앙상블 +1.0pp.

## 1. 역사 배당 소스 확보 (프로브 먼저)
- data.go.kr KRA API 카탈로그에서 **역사 배당률/확정배당(exotic 포함: 복승·쌍승·삼복승·삼쌍승) API** 존재 확인. 어제의 RaceDetailResult_1 응답에 배당 필드가 이미 있는지도 확인(있으면 그걸로 충분).
- API 불가 시 race.kra.co.kr 공개 결과 페이지 프로브(3개 일자 샘플).
- 프로브 결과(가능/불가, 필드 목록)를 먼저 progress에 기록 후 진행.

## 2. 코퍼스 수집
- 범위: 서울·제주·부경, 2024-01 ~ 2026-07 (kra_model 학습기간과 정합). 일일 트래픽 한도 고려해 rate limit 준수, 체크포인트/재개.
- 저장: `data/kra_market_corpus.jsonl` (race key + 권종별 배당/확정 + 착순). 경마는 출주 두수 가변(7~14두) — 두수 필드 필수.
- 수집량이 크면 우선순위: ①단승 배당+착순(전 경주) ②삼쌍승/삼복승 배당(가능한 만큼).

## 3. 측정 (경륜 파이프라인 이식)
- 단승 기준: 시장 1번인기 top-1 적중률(전체 baseline) + **쏠림 신호(1-2번인기 배당비, 1번인기 절대배당)로 티어 분할** → 티어별 top-1/top-3 정밀도-커버리지 표. 두수별(7두 이하/8-10/11+) 분리표.
- exotic 보드가 확보되면: 경륜과 동일 market_rank baseline + 쏠림 티어 측정. 시장 효율이 경륜과 다른지(버킷 캘리브레이션 곡선) 보고.
- v4 모델 예측과 시장의 결합 여지: v4 top-1 pick이 시장 1번인기와 불일치하는 경주 서브셋의 정밀도 (모델-시장 disagreement 신호) — 측정만, 승격 주장 금지.

## 산출물/규율
- `data/kra_market_corpus_results.{json,md}`, 진행은 **`runs/kra_corpus_progress.md`**(신규, 5A와 파일 분리).
- 학습/선택 개입 없는 순수 측정이 기본. 모델 적합이 필요하면 날짜 split(≤2025 train / 2026 val·test 분리) 규율. 음수/불가는 그대로 보고. 기존 파일 수정 금지. commit 금지. 키 하드코딩·커밋 절대 금지(어제와 동일 방식으로 환경에서만 읽기). 보고 ≤50자 file path만.
