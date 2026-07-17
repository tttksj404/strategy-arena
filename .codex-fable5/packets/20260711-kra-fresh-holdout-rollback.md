# Fable5 audit packet — KRA fresh-holdout rollback

## Verdict requested

`PASS`, `NEEDS_FIX`, `FAIL` 중 하나로 판정한다. 핵심 판단은 새 미접촉 데이터가 기존 pairwise 승격 주장을 뒤집었을 때 production을 v4로 안전 복귀시킨 것이 타당한지다.

## New evidence

- 공식 data.go.kr 결과 수집: 2026-06-22..2026-07-11, 1,358 rows, 134 races.
- v4: top-1 31.343%, top-3 66.418%, log-loss 1.9135.
- v5: top-1 29.851%, top-3 66.418%, log-loss 1.9137.
- v5 minus v4: -2 net wins, -1.493%p, bootstrap 95% CI -3.731..0.000%p.
- Guard decision: `keep_v4_baseline`, `promotion_pass=false`.
- Hard promotion floor: current production 대비 OOS top-1 절대 `+5.0%p`.
- 재실행된 제한 pairwise: `+0.549%p`; context 결합: `+0.976%p`; fresh holdout: `-1.493%p`. 모두 승격 실패.

## Production change

- `kra_pairwise_artifact.build_pairwise_artifact` stores the research candidate with `enabled=false`.
- `engine.score_kra` executes pairwise reranking only when `enabled is True`.
- retrained artifact kind: `kra_dual_phase_v4_history_fresh_holdout_guard`.
- model retrained through 2026-07-11; latest complete odds still use the market path.

## Evidence paths

- `runs/kra_fresh_holdout_20260622.json`
- `docs/kra_prediction_fresh_holdout_guard.md`
- `tests/test_kra_fresh_holdout_guard.py`
- `tests/test_kra_prediction_phase.py`
- `static/models/kra_model.joblib`

## Audit questions

1. 새 134경주가 후보 선택 이후 수집된 미접촉 구간이라는 주장이 데이터 시점상 맞는가?
2. pairwise candidate를 artifact에 보관하되 `enabled=false`로 차단하는 방식이 fail-closed인가?
3. DB 갱신 이후 재학습 artifact가 동일일·미래 결과를 inference feature에 누출하지 않는가?
4. 향후 승격 기준인 top-1 절대 `+5.0%p`, `net_wins > 0`, bootstrap CI 하한 `> 0`의 결합이 올바르게 fail-closed 되는가?

## Stop condition

Fable5 판정 전 커밋·푸시·배포 금지. 현재 로컬 production artifact는 v4 기준 순위를 사용한다.
