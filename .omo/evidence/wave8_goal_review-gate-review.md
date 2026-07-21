# Final gate review: Wave 8-10 crypto method expansion

## recommendation

**APPROVE**

현재 파일과 산출물을 직접 재검증한 결과, 이전 게이트를 막았던 Wave 8 Q8 거래량 신호의 당일 정보 사용이 수정되었고 관련 하위 산출물도 다시 생성되었습니다. 명시된 성공 기준을 위반하는 남은 blocker는 없습니다. 이 승인은 연구 산출물의 완결성에 대한 것이며 실제 $100 운용 승인이 아닙니다.

## blockers

없음.

## originalIntent

이전 웨이브에서 승격 가능한 후보가 나오지 않은 뒤, 고정 $100 계좌·현실적 비용·최소 주문 금액·시간 순서를 지키는 OOS·엄격한 연구 전용 fail-closed 규칙 아래에서 여러 추가 암호화폐 전략 방법을 탐색하는 것.

## desiredOutcome

추가 방법을 재현 가능하게 평가하여 자본 제약을 통과하는 후보가 있으면 근거와 함께 식별하고, 없으면 어떤 문턱도 완화하지 않은 채 생존 후보 0개를 정직하게 보고하는 것. 라이브 배포, 미래 정보 사용, 파라미터 탐색 결과의 독립 검증 오인, $100 실행 계약의 암묵적 완화는 허용하지 않는다.

## success criteria

| ID | Criterion | Verdict | 직접 확인한 근거 |
|---|---|---|---|
| C1 | 초기 자본 $100, $10 reserve 및 0.60 gross cap 유지 | PASS | 세 SPEC와 결과 JSON의 계약 및 `initial_equity`를 확인했고 최대 gross는 약 0.6000000000000002였다. |
| C2 | 수수료·슬리피지 및 2배 슬리피지 스트레스 반영 | PASS | 세 runner의 비용 계산과 결과의 비용/스트레스 게이트 일관성을 확인했다. |
| C3 | $5 최소 주문 실행 가능성이 승격 조건에 참여 | PASS | start-of-day equity 기반 동적 notionals와 capital gate를 확인했고, $5 미만 관측 후보는 모두 해당 게이트를 실패했다. |
| C4 | 연구 전용이며 자격증명·실주문·네트워크 fetch·배포 없음 | PASS | 새 세 패키지의 import와 실행 경로를 검사했으며 exchange SDK/HTTP/credential/order/deploy 경로가 없다. |
| C5 | 신호와 OOS가 시간 순서를 지키며 day-t/미래 feature 누출 없음 | PASS | Q8가 `volume.shift(1)`을 사용한다. 독립 변이 probe에서 day-t volume 변경 시 과거와 day-t position은 동일하고 t+1만 변했다. funding cash와 Wave 10 throttle도 직전 보유/실현 상태만 사용한다. |
| C6 | 모든 게이트를 통과하지 않으면 승격하지 않는 fail-closed 판정 | PASS | Wave 8/9/10 모두 `selection_independent=false`, `eligible=[]`; stored gates와 `all_gates_pass`를 raw metrics에 대조했다. |
| C7 | 이전 웨이브를 넘어 실제로 여러 방법을 추가 탐색 | PASS | Wave 8 16개, Wave 9 16개, Wave 10 6개 후보를 각각 독립 결과와 aggregate에서 확인했다. |

## userOutcomeReview

사용자가 기대한 추가 방법 탐색은 38개 후보 산출물로 구현되어 있으며, 현 결과는 세 웨이브 모두 적격 후보 0개다. 이는 실패가 아니라 fail-closed 계약에 맞는 결론이다. Q8 수정 후 전체 Wave 8과 이를 소비하는 후속 산출물이 재실행되었고, aggregate·standalone·report·manifest가 서로 일치한다. 따라서 연구 확장 결과는 전달 가능하지만 실제 자본 투입 판단으로 승격해서는 안 된다.

## reproduced evidence

- 현재 validator를 직접 실행:
  - `WAVE8_VALIDATION_PASS candidates=16 eligible=[]`
  - `WAVE9_VALIDATION_PASS candidates=16 eligible=[] selection_independent=false`
  - `WAVE10_VALIDATION_PASS candidates=6 eligible=[] selection_independent=false`
- bytecode/cache 쓰기를 비활성화한 scoped test 재현: `13 passed in 0.69s` (Wave 8: 7, Wave 9: 4, Wave 10: 2).
- `.omx/evidence/wave8-qa/`의 full-run evidence에서 세 runner 모두 exit 0, 각각 16/16/6개 후보, `eligible=[]`를 확인했다. post-full validator도 모두 exit 0이다.
- 직접 JSON 감사:
  - 후보 ID는 각 웨이브에서 유일하고 standalone 객체는 aggregate 항목과 동일하다.
  - report/results/spec/runner hash는 manifest와 일치한다.
  - gate booleans와 `all_gates_pass`는 기록된 raw metric과 일치한다.
  - 세 웨이브 모두 selection independence가 false이며 적격 후보는 없다.
- Q8 adversarial probe:
  - `past_equal=True`
  - `same_day_equal=True`
  - day-t position: `0.0 -> 0.0`
  - t+1 position: `0.0 -> 0.6`
  - 당일 거래량이 당일 포지션에 유입되지 않고 다음 날에만 반영됨을 확인했다.
- 핵심 소스 포인터:
  - Q8 prior-day volume: `research/wave8_alternative/run_wave8.py:187-193`
  - Q8 adversarial regression: `research/wave8_alternative/tests/test_wave8.py:34-43`
  - Wave 8 funding alignment: `research/wave8_alternative/run_wave8.py:316-317`
  - Wave 10 causal throttle: `research/wave10_ensemble/run_wave10.py:80-110`
  - 정확히 4개 OOS block: `research/wave8_alternative/run_wave8.py:279`, `research/wave9_methods/run_wave9.py:313`
  - 동적 최소 주문 계산: `research/wave8_alternative/run_wave8.py:341-347`, `research/wave9_methods/run_wave9.py:362-372`, `research/wave10_ensemble/run_wave10.py:134-144`

## remove-ai-slops and programming review

현재 production code, validator, tests에 대해 직접 overfit/slop 및 programming 관점 검사를 수행했다.

- 새 Q8 변이 회귀, funding alignment, causal throttle 테스트는 요청 삭제 여부나 구현 문자열을 검사하는 테스트가 아니라 입력 변화에 대한 행위 불변식을 검증한다.
- deletion-only 테스트, 요청된 제거만 확인하는 테스트, tautological 테스트, 테스트를 위한 불필요한 production extraction/parsing/normalization은 발견하지 못했다.
- 기존 `test_no_future_timestamp...`는 일부 implementation-mirroring 성격이 있지만 새 Q8 경로 직접 회귀가 이전 false-confidence 공백을 닫았다.
- Wave 8/9 runner의 큰 함수와 variant dispatch는 유지보수성 NOTE다. 명시된 성공 기준 실패를 입증하지 않으므로 blocker가 아니다.
- 기존 코드 리뷰 보고서 `.omo/evidence/wave8-wave10-research-code-review.md:67-70`도 `omo:remove-ai-slops`와 `omo:programming` 관점 및 overfit/구현 미러링 문제를 명시했다. 다만 해당 보고서의 최종 판정과 hash는 수정 전 snapshot이라 최신 승인 근거로 사용하지 않았고, 이번 직접 검사와 현 산출물 재현으로 판정했다.

## checked artifact paths

- `.omx/specs/autoresearch-wave8/mission.md`
- `.omx/specs/autoresearch-wave8/sandbox.md`
- `.debug-journal.md`
- `.codex-fable5/packets/PACKET-20260721-method-expansion.md`
- `.omx/evidence/wave8-qa/`
- `.omo/evidence/wave8-wave10-research-code-review.md`
- `.omx/evidence/wave8-qa/wave8-qa-manual-qa.md`
- `research/wave8_alternative/SPEC.md`
- `research/wave8_alternative/run_wave8.py`
- `research/wave8_alternative/validate_wave8.py`
- `research/wave8_alternative/tests/test_wave8.py`
- `research/wave8_alternative/results/wave8_results.json` 및 16개 standalone result JSON
- `research/wave8_alternative/report/wave8_report.md`
- `research/wave8_alternative/report/wave8_manifest.json`
- `research/wave9_methods/` 아래의 대응 SPEC, runner, validator, tests, aggregate, standalone results, report, manifest
- `research/wave10_ensemble/` 아래의 대응 SPEC, runner, validator, tests, aggregate, standalone results, report, manifest

## exact evidence gaps

명시된 성공 기준에 연결되는 미충족 증거 공백은 없다. 다음은 승인 비차단 NOTE다.

1. `.omo/evidence/wave8-wave10-research-code-review.md`의 verdict와 기록 hash는 Q8 수정 전 snapshot이라 stale하다. 이번 게이트는 그 결론을 신뢰하지 않고 현재 파일·runner evidence·validator·tests·adversarial probe를 독립 재현했다.
2. 전체 저장소 pytest collection은 `research/wave1-rwa`의 기존 `src` import 오류 3건이 있고, 해당 경로 제외 시 기존 Wave 4의 `research/wave1/cache/universe.json` 부재로 2건이 실패한다. 현재 Wave 8-10 scoped suite 13건은 모두 통과하며 이 오류들은 변경 범위나 성공 기준과 연결되지 않는다.
3. validator가 모든 feature timing 불변식을 독립 재계산하지는 않는다. 다만 명시된 Q8 누출 경로는 전용 행위 테스트와 독립 probe로 직접 검증되어 현재 C5의 공백은 없다.
4. `.debug-journal.md`의 이전 pending 표기는 최신 상태가 아니다. 사용자 산출물이나 명시된 완료 기준은 아니므로 blocker가 아니다.
