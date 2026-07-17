# Round 9b — KRA 말별 과거 페이스 지표 (마지막 펀더멘털 레버, 2026-07-12)

R9 판정 반영: sectional 피처를 **현재 경주 행에서 읽는 것은 label leakage로 FAIL 확정**. 학습창 확장은 이득 0 확정. 이번 라운드는 단 하나만 검증한다 — **과거 경주들의 구간기록으로 만든 말별 페이스 폼**이 pre-odds 예측을 올리는가.

## 누출 금지 (게이트 — 통과 전 학습 금지)
- 모든 페이스 피처는 해당 경주 rc_date **미만**의 그 말(hrNo 기준) 이력만 사용.
- 검증 어서션: 말 3마리 샘플에서 특정 경주 t의 피처를, t 이후 행 삭제 후 재계산해 동일함을 assert (kcycle R3와 동일 프로토콜). 결과 md에 어서션 통과 명시.

## 피처 (kra.db race_result, hrNo·rc_date 시계열)
- 최근 3·5출주: 평균 최종구간(je_1cTime 상당) z-score(거리·경주장 조건부), 최고 기록, 기록 추세(직전 대비)
- 최근 3출주 평균 착순·top3율, days_since_last, 부담중량 변화(wgBudam delta), rating 변화
- 거리 적성: 해당 거리대(±200m) 과거 top3율
- 결측(신마 등): 학습 split median + is_first_start 플래그

## 학습/평가 (R9와 동일 프로토콜)
- v4 아키텍처에 페이스 피처 추가(odds 컬럼 배제 유지). train 2019-01~2025-05 / val 2025-06~2026-05 선택 / **fresh 134경주 1회**.
- baseline 동표: v4 pre-odds, R9의 expanded_2019_202505(31.34%).
- 판정 기준: fresh top-1 ≥ 33.0%(+1.7pp) AND val 비열등이면 후보 보고. 미달 시 미달 그대로 — 이것으로 KRA 펀더멘털 레버 전체 소진 선언 자료가 된다.

## 산출물/규율
- 결과를 repo `data/kra_round9b_results.{json,md}`에 직접 쓴다(/Users/tttksj/kra 쓰기 금지 시 tmp 스테이징 후 repo 복사). 아티팩트는 enabled=false 보관. production 교체·commit 금지. 진행 runs/kra_corpus_progress.md append. 보고 ≤50자 file path만.
