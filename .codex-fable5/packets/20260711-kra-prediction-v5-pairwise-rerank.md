# Fable5 audit packet — KRA prediction v5 pairwise reranker

## Verdict requested

`PASS`, `NEEDS_FIX`, `FAIL` 중 하나로 판정한다. 금융·예측·베팅 필수 감사이며 이 판정 전에는 커밋·푸시·배포하지 않는다.

## Superseded claim under audit

이 패킷의 초기 승격 주장은 후속 미접촉 검증과 `+5.0%p` 절대 승격 하한 도입으로 철회됐다. pairwise HGB는 production에서 비활성화 상태다.

## Evidence

- 4개 시간순 fold top-1 개선: `+0.328`, `+0.725`, `+0.329`, `+0.841`%p.
- 4개 fold top-3 변화: 모두 `0.000`%p.
- pooled 4,865 races: `+0.555`%p, paired bootstrap 95% CI `+0.206..+0.904`%p, positive probability `0.9989`.
- 확률을 바꾸지 않으므로 winner log-loss delta `0.0000`.
- 재정렬 범위: 경주의 약 `3.54..4.59%`.
- 후보 선택은 첫 3개 fold의 최소/평균 top-1 개선으로 수행하고 2026H1을 마지막 확인 구간으로 사용했다. 다만 2026 자체는 과거 연구에서 이미 관찰됐으므로 pristine holdout 주장은 하지 않는다.

Primary artifacts:

- `runs/kra_pairwise_rerank_v5_results.json`
- `.omx/specs/autoresearch-kra-v5/result.json` (`status=failed`)
- `docs/kra_prediction_v5.md`
- `static/models/kra_model.joblib` (`kind=kra_dual_phase_v4_history_fresh_holdout_guard`, pairwise disabled)

## Rejected alternatives

- 최근/거리/경마장/기수 등 추가 피처: fold top-3 또는 bootstrap gate 실패.
- HGB/ExtraTrees/RandomForest/ensemble: top-3 또는 bootstrap gate 실패.
- unrestricted pairwise: pooled top-1 CI는 양수였지만 2025H2 top-3 `-0.247`%p, log-loss `+0.0256` 악화로 기각.
- restricted pairwise d2도 통과했으나 첫 3개 fold 선택 규칙에서 d3가 우세해 d3 선택.

## Production boundary

- `engine.score_kra`: 사전 단계에서만 artifact의 pairwise estimator를 실행한다.
- `restricted_rerank`: v4 top-3 밖 후보는 top-1로 승격할 수 없다.
- `pwin`/`pplc`는 변경하지 않고 `rank_score`만 단승 순위와 단승 pick에 사용한다.
- 최신 완전 배당판이 확인된 live 단계는 pairwise reranker를 우회한다.
- 재정렬된 경주는 기존 고확신 라벨을 비활성화해 일반 예측으로 표시한다.

## Audit questions

1. fold 생성과 말/기수/조교사 이력이 동일일·미래 결과를 확실히 배제하는가?
2. 후보 탐색·선택 과정에 다중비교 또는 반복 관찰 편향을 감안하면 CI가 과도하게 낙관적인가?
3. `rank_score` 도입이 top/picks 외 다른 순위 경로에 의도치 않은 회귀를 만들지 않는가?
4. 저장 artifact의 feature column/median 계약과 production inference가 학습 시점과 일치하는가?
5. 개선 폭이 작으므로 추가 미래 데이터가 쌓일 때 자동 롤백/재검증 게이트가 필요한가?

## Current stop condition

구현과 자체 검증만 완료한다. Fable5 독립 판정 수신 전 커밋·푸시·배포 금지.
