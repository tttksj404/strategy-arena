# Wave-3 code quality review

- Verdict: **FAIL**
- codeQualityStatus: **BLOCK**
- recommendation: **REQUEST_CHANGES**
- Scope: `research/wave3/` 신규 모듈 8개와 `tests/test_wave3.py`, `SPEC.md`, wave1/wave2 재사용 API
- Verification note: 현재 소스·SPEC·wave1/wave2 구현은 직접 읽었다. 요청된 `compileall`/`pytest` 재실행은 Python 실행 경로를 찾는 중 사용자가 즉시 종료를 지시해 완료하지 못했다. 따라서 과거의 성공 transcript는 승인 근거로 사용하지 않았다. `research/wave3/__init__.py`의 현재 1바이트 소스는 그 자체로 결정적인 구문 오류다.
- Skill-perspective check: `omo:remove-ai-slops`와 `omo:programming` 원문 및 Python 타입·데이터·오류·데이터처리 기준을 명시적으로 적용했다. **두 관점 모두 위반**한다(약한/구현 미러링 테스트, raw string variant, 비 exhaustive 분기, 과다 인자, 누락된 경계 계약, 불필요한 자체 재구현).

## CRITICAL

없음.

## HIGH

1. `research/wave3/__init__.py:1` — 파일 내용이 닫히지 않은 `"` 하나뿐이라 `research.wave3.*` import, pytest collection, CLI가 모두 `SyntaxError`로 차단된다. **반드시 올바른 패키지 초기화 파일로 수정 후 compileall/pytest를 새로 실행해야 한다.**

2. `research/wave3/engine.py:99-106` — wave1 캐시 fallback을 `research/wave3/wave1/cache`로 계산한다. 실제 재사용 원본은 `research/wave1/cache`이므로 wave3 캐시에 없는 기존 종목은 `PipelineError`가 난다. **경로를 repo의 wave1 캐시 상수/API로 연결해야 한다.**

3. `research/wave3/engine.py:139-172` — W3c에 SPEC의 변동성 타게팅이 전혀 없고, `previous`가 비어 있으면 금요일이 아닌 날에도 최초 진입한다(161-162). 또한 종목 수가 6 미만이면 top/bottom 집합이 겹쳐 동일 심볼 롱·숏이 중복 생성된다(149-156). **W3c 후보 계약을 구현하고 경계 테스트를 추가해야 한다.**

4. `research/wave3/engine.py:222-236` — 비용은 부호와 factor를 무시한 `abs(new_weight-old_weight)`만 본다. 동일 비중에서 롱→숏 또는 W3e carry(2 legs)→momentum(1 leg) 전환 시 turnover/cost가 0이 된다. `research/wave3/engine.py:247`은 별도 taker stress run 없이 기본 실행 수익을 `stress_total_return`으로 기록한다. **wave2와 같은 독립 stress 경로 및 leg/direction-aware turnover가 필요하다.**

5. `research/wave3/universe.py:123-137` — “직전 30d”를 시간 구간이 아니라 마지막 30개 행으로 계산한다. 긴 결측/거래중단 뒤에도 오래된 30개 관측을 현재 유동성처럼 사용한다. 현재 테스트도 1월 자료로 3월 2일 편입을 허용해 이 오류를 고정한다(`tests/test_wave3.py:10-22`). **`[as_of-30d, as_of)` 달력 구간과 충분한 coverage 계약을 검증해야 한다.**

6. `research/wave3/engine.py:242-258` — 히스테리시스로 W3e 보유 종목은 최대 rank 10까지 늘 수 있는데 metadata는 `max_concurrent_positions=3`, `min_position_weight=1/3`, `max_position_weight=1.0`으로 허위 기록한다. 게이트의 capacity 판정이 잘못된다. **실제 시계열에서 관측한 포지션 수/비중을 기록해야 한다.**

## MEDIUM

1. `research/wave3/reporting.py:26-42` — 한 후보의 누락이 `complete=False`로 누적되어 이후 정상 후보도 FAIL 처리된다. 후보별 verdict를 독립 계산해야 한다.

2. `research/wave3/engine.py:130-172` — 8개 인자, raw `candidate_id`/`factor` 문자열과 `if/elif/else` fallback으로 후보 typo가 W3e 경로로 조용히 들어간다. 타입화된 variant와 exhaustive dispatch가 필요하다.

3. `research/wave3/tests/test_wave3.py:79-96` — 통합 smoke는 ID·길이·NaN 부재만 확인해 W3e 선택, 다음날 체결, 비용, 히스테리시스, 후보별 수익 동작을 전혀 증명하지 않는다. W3a~W3f, 비용 stress, factor 전환, 주간 경계, pandas 시점 정렬 회귀 테스트가 없다.

4. `research/wave3/fetch.py:36-43`은 wave1의 `cached_frame`을 재구현하고, `run_wave3.py:46-56`은 manifest 존재만 확인해 hash/frozen_end/source_integrity를 검증하지 않는다. 재사용 목표와 데이터 경계 무결성을 모두 약화한다.

## LOW

1. `research/wave3/universe.py:29,34-40` — `asset_type` 공개 타입이 `AssetType | str`인 채 내부에서 강제 변환되어 타입 검사자는 이후 상태를 증명할 수 없다. boundary parser와 내부 `AssetType` 모델을 분리하는 편이 안전하다.

## Blockers

- `__init__.py` 구문 오류 제거 후 fresh `compileall`/focused+full `pytest` 증거 확보
- wave1 캐시 fallback 경로/API 수정
- W3c 변동성 타게팅·주간 최초 진입·롱숏 중복 경계 수정
- leg/direction-aware 비용과 실제 taker stress 결과 구현
- 달력 기준 직전 30일 유동성 정렬 및 회귀 테스트
- W3e 실제 포지션 capacity metadata 수정

