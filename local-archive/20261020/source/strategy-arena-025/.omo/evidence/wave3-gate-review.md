# Wave-3 gate review

- recommendation: REJECT
- reviewRole: final gate reviewer, read-only source audit
- networkUsed: false
- reportPathMode: fallback (`omo ulw-loop status --json` equivalent returned `ULW_LOOP_PLAN_MISSING`)

## originalIntent

`research/wave3/SPEC.md`의 동결 계약대로 listing-aware Wave-3 파이프라인을 구현한다. Binance UM `onboardDate` 기준 상장 60일 후 편입, 과거 시점 직전 30일 quote-volume 상위 150만 사용, W3a-f 전 후보 구현, 네트워크는 fetch stage에만 격리, wave1/2와 같은 `fetch/run/gates/report` CLI, 필수 합성 pytest 3종을 제공한다.

## desiredOutcome

오케스트레이터가 fetch를 실행한 뒤 로컬 run/gates/report 단계가 미래정보 없이 재현 가능하고, W3a-f의 동결 전략·비용·게이트·W3e/W3f 비교 산출물이 실제 계산 근거로 생성되어야 한다.

## userOutcomeReview

필수 테스트 함수와 4-stage CLI 표면은 존재하지만, 동결 후보 정의와 wave1 게이트 계약을 실제로 계산하지 않거나 허위 상수로 통과시키는 경로가 있어 사용자 기대 결과를 충족하지 못한다.

## blockers

1. violatedCriterion: `W3a-f / W3c frozen definition`
   - observation: W3c에 명시된 변동성 타게팅이 구현되지 않았다. 엔진은 30일 모멘텀과 균등가중 수익만 계산하며 realized-vol 스케일링이 없다.
   - evidencePointer: `research/wave3/SPEC.md:18`; `research/wave3/engine.py:40`; `research/wave3/engine.py:184-197`; `research/wave3/engine.py:214-221`

2. violatedCriterion: `wave-2 cost contract + wave-1 gate table`
   - observation: 테이커 0.06% 스트레스 변형을 재실행하지 않고, 기본 실행 총수익을 `stress_total_return` 자리에 넣어 gate 7 입력으로 사용한다.
   - evidencePointer: `research/wave3/SPEC.md:11`; `research/wave3/engine.py:20`; `research/wave3/engine.py:175-180`; `research/wave3/engine.py:223-247`; `research/wave1/gate_reporting.py:103`; `research/wave1/gates.py:153`

3. violatedCriterion: `wave-1 gates / evidence integrity`
   - observation: fetch manifest는 존재 여부만 확인되고 무결성 결과는 run에서 검증되지 않는다. 대신 `data_valid=True`, `cost_model_valid=True`, `neighbor_is_sharpes=[1.0, 1.1]`를 하드코딩해 gate 1·2·7에 실제 검증이 아닌 상수를 공급한다.
   - evidencePointer: `research/wave3/fetch.py:110-120`; `research/wave3/run_wave3.py:40-50`; `research/wave3/engine.py:247-259`; `research/wave1/gates.py:126-153`

4. violatedCriterion: `wave-1 gate table / W3e-f hysteresis capacity`
   - observation: 히스테리시스는 순위 10위 이내 기존 종목을 top-3에 추가 보유할 수 있는데, 메타데이터는 최대 동시 포지션을 항상 `top_k`로 기록한다. gate 8의 `concurrent <= 3` 판정이 W3e/f에서 거짓 PASS가 될 수 있다.
   - evidencePointer: `research/wave3/strategy.py:55-70`; `research/wave3/engine.py:157-171`; `research/wave3/engine.py:254`; `research/wave1/gate_reporting.py:58-69`

5. violatedCriterion: `maximum wave1 reuse`
   - observation: wave1 캐시 fallback 경로가 `research/wave3/wave1/cache`로 계산되어 실제 `research/wave1/cache`를 찾지 못한다.
   - evidencePointer: `research/wave3/engine.py:99-106`; `research/wave1/common.py:22-25`

6. violatedCriterion: `W3e/W3f comparison and stock-token contribution reporting`
   - observation: SPEC가 요구한 ETF 포함 가치 정량 비교, 주식토큰 기여분 별도 표기, 연도별 워크포워드 표가 report writer에 없다. 현재 출력은 후보별 verdict와 OOS label뿐이다.
   - evidencePointer: `research/wave3/SPEC.md:22-26`; `research/wave3/reporting.py:22-46`

## mandatoryTestsReview

- listing-aware 60일 경계: 존재, `research/wave3/tests/test_wave3.py:10-22`
- W3e max-z 선택: 존재, `research/wave3/tests/test_wave3.py:45-58`
- hysteresis 순위 10위 밖 청산: 존재, `research/wave3/tests/test_wave3.py:61-76`
- no-lookahead 직전 30일 필터 테스트: 존재, `research/wave3/tests/test_wave3.py:25-42`
- NOTE: `test_w3e_runs_a_synthetic_cross_asset_market`는 이름과 달리 전부 crypto이며 결과 길이/NaN만 확인해 cross-asset 계약에 대한 강한 증거가 아니다 (`research/wave3/tests/test_wave3.py:79-96`). 삭제 전용·요청 제거만 검증·명백한 tautology 테스트는 발견하지 못했다.

## slopAndProgrammingPass

- BLOCKING false-confidence slop: 실제 계산을 대신한 `neighbor_is_sharpes`, `data_valid`, `cost_model_valid` 상수.
- BLOCKING scope gap: W3c volatility target과 보고 계약 누락.
- NOTE: pandas/requests 사용과 일부 장문 함수는 프로젝트 기존 관례 및 비요구 품질 항목이므로 별도 blocker로 승격하지 않았다.

## checkedArtifacts

- `research/wave3/SPEC.md`
- `research/wave3/__init__.py`
- `research/wave3/universe.py`
- `research/wave3/strategy.py`
- `research/wave3/fetch.py`
- `research/wave3/engine.py`
- `research/wave3/reporting.py`
- `research/wave3/run_wave3.py`
- `research/wave3/tests/test_wave3.py`
- `research/wave1/SPEC.md`
- `research/wave1/run_wave1.py`
- `research/wave1/common.py`
- `research/wave1/costs.py`
- `research/wave1/gates.py`
- `research/wave1/gate_reporting.py`
- `research/wave2/SPEC.md`
- `research/wave2/run_wave2.py`
- `research/wave2/funding.py`
- `research/wave2/gates.py`
- `research/wave2/reporting.py`

## exactEvidenceGaps

- 현재 ulw-loop plan 없음. 따라서 attempt directory가 없고 fallback 보고서 경로를 사용했다.
- 현재 wave3 executor evidence, code-review report, manual QA matrix, notepad는 존재하지 않았다. `.omo/evidence`에는 무관한 wave2 CLI/manual-QA 파일 2개만 있었다.
- pytest 재현은 완료하지 못했다. `uv run --offline`은 기본 cache 경로에서 OS error 183, `C:\tmp`에서 access denied, workspace 임시경로에서는 uv managed-Python 디렉터리 접근 거부로 종료했다. 이 실행 공백과 무관하게 위 blocker는 직접 소스 대조로 확정된다.
- 독립 code-reviewer/architect lane을 시작했으나 사용자의 즉시 종료 지시로 결과를 기다리지 않았고 판정 근거로 사용하지 않았다.
- fetch stage 및 기타 네트워크 호출은 실행하지 않았다.
