# Fable5 audit packet: KRA prediction v4 history

Requested verdict: `PASS`, `NEEDS_FIX`, or `FAIL`.

## Claim

날짜 안전한 말 과거 전적 피처가 배당 없는 KRA 사전 순위 정확도를 개선했다. 수익률 또는 시장 초과 수익은 주장하지 않는다.

## 변경 경계

- 학습 시 `hrNo`별 승률, 연대율, 평균 착순 백분위, 과거 출전 수, 직전 출전 후 경과일을 현재 날짜보다 앞선 날짜만 사용해 계산한다.
- 추론 시 모델 산출물에 저장된 확정 과거 스냅샷을 경주일 기준으로 적용한다.
- 기수와 조교사 prior도 같은 날 결과가 섞이지 않도록 날짜 단위 shift로 수정했다.
- 연승 선두 역사용 후보는 신뢰구간이 0을 포함해 배포하지 않았다.
- 최신 완전 배당이 명시적으로 증명된 경우에만 시장확률 경로를 사용한다.

## 독립 시간 구간 결과

| 구간 | top-1 개선 | top-3 개선 |
|---|---:|---:|
| 2024H2 | +2.71%p | +4.10%p |
| 2025H1 | +4.75%p | +1.69%p |
| 2025H2 | +3.95%p | +4.85%p |
| 2026H1 | +5.21%p | +4.37%p |

4,865경주 pooled paired bootstrap: +4.15%p, 95% CI +3.04~+5.28%p, 양수 확률 1.0. 역연승 후보는 +0.21%p, 95% CI -0.68~+1.09%p로 기각했다.

## 검토 파일

- `kra_history_features.py`
- `kra_training_features.py`
- `kra_model_evaluation.py`
- `engine.py`
- `tools/kra_dual_phase_experiment.py`
- `tools/kra_reverse_candidate_audit.py`
- `tests/test_kra_history_features.py`
- `tests/test_kra_prediction_phase.py`
- `runs/kra_reverse_candidate_audit.json`
- `runs/kra_dual_phase_results.json`
- `static/models/kra_model.joblib`

## 자체 검증

- 백엔드 unittest 198개 통과.
- 말 이력 날짜 shift 및 추론 스냅샷 단위 테스트 통과.
- 4구간 감사 스크립트 재실행과 10,000회 paired bootstrap 통과.
- 데모 경주에서 `pre_race`와 `live_odds` 경로, v4 산출물, 말 이력 스냅샷 6,170건을 실제 로드해 확인.
- 모바일 TypeScript 검사와 보안 표면 검사 통과.
- Python compile, diff whitespace, 비밀값 패턴 검사 통과.

## 감사 질문

1. 동일 날짜 결과 또는 미래 결과가 어떤 학습 피처에도 유입되는가?
2. 학습 피처와 실제 추론 스냅샷의 정의가 일치하는가?
3. 2026 재사용 한계를 문서가 정확히 고지하고 있는가?
4. 네 구간 승격 게이트와 bootstrap이 적절한가?
5. 수익성 주장으로 오해될 UI 또는 API 표현이 남아 있는가?
