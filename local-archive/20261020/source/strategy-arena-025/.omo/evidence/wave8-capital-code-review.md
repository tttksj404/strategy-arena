# Wave-8 Capital 코드 품질 리뷰

- 리뷰 일자: 2026-07-21
- 목표: `research/wave8_capital/run_capital100.py`의 lookahead, RNG/seed, 자본 스케일링·reserve, source gate fail-closed, 최소 주문, 출력 스키마, 네트워크·실주문 안전성을 SPEC·상류 산출물과 대조한다.
- 코드 품질 상태: **WATCH**
- 권고: **APPROVE**
- 종합 판정: **PASS**

## 검토 범위와 최종 스냅샷

- `research/wave8_capital/run_capital100.py`: SHA-256 `37083E9D38F4CA4EA62E158F1373EC2C92BDE104671340CCE3DA4292BD3D19F9`
- `research/wave8_capital/tests/test_capital100.py`: SHA-256 `C9F29938903C0736F720C7A19F61E6904AB706A1BD044FB0215ECFFC8F177FEA`
- `research/wave8_capital/results/capital100.json`: SHA-256 `115CA02BD3B3295C73F313071127A1D9D76C103A328B946CF5525E3DC2CD8CB9`
- `research/wave8_capital/report/capital100_report.md`: SHA-256 `3A24C8C07C2AC27E0639D16E3449B6D338C9D2B9ECC3000D05F2A272B284AA7F`
- `research/wave8_capital/SPEC.md`: SHA-256 `F0332EE701BF6775E48FB2701CA186A6373A688FD61CAC11D7195A0A9E461231`
- `.codex-fable5/packets/PACKET-20260721-wave8-capital100.md`: SHA-256 `5271AAD2043A14F4F72B60F59835DEC679FDDEEBBC9639372D9C505E43EEAFBA`

`omo ulw-loop status --json`은 이 저장소에서 `The syntax of the command is incorrect.`로 실패했고 `.omo/evidence/ulw` 계획도 없으므로, 지정된 fallback 경로에 본 리뷰를 기록했다.

## Findings

### CRITICAL

없음.

### HIGH

없음.

### MEDIUM

1. **W7 blend weight의 비유한 값과 JSON의 비표준 수치에 대한 마지막 방어선이 없다.**
   - 위치: `research/wave8_capital/run_capital100.py:125`, `research/wave8_capital/run_capital100.py:167`, `research/wave8_capital/run_capital100.py:286`
   - `min_position_weight`는 유한한 양수로 검증하도록 보강됐지만, blend weight 자체는 같은 검증 helper를 거치지 않는다. `value=Infinity` fixture에서 `_min_order` 결과의 blend/gross 필드가 `Infinity`가 되었고 기본 `json.dumps`가 이를 그대로 직렬화했다.
   - 현재 고정 입력은 모두 유한하고 현재 결과 JSON도 유효하며, 이 경우 gross gate가 false라 잘못 승격되지는 않는다. 따라서 현재 판정을 뒤집는 blocker는 아니지만, 입력 경계와 RFC 8259 출력 보장을 위해 blend weight 검증 및 `allow_nan=False`가 바람직하다.

2. **모듈 크기·타입 경계가 programming 유지보수 기준을 넘는다.**
   - 위치: `research/wave8_capital/run_capital100.py:8`, `research/wave8_capital/run_capital100.py:25`, `research/wave8_capital/run_capital100.py:120`, `research/wave8_capital/run_capital100.py:189`, `research/wave8_capital/run_capital100.py:224`
   - programming checker는 295 pure LOC의 oversized module과 generic exception 7건을 검출했다. payload 전반이 `dict[str, Any]`라 입력 스키마가 정적 타입으로 고정되지 않는다.
   - 고정 입력을 읽는 단일 연구 CLI라 현재 계산 정확성에는 영향이 없지만, 다음 wave에서 스키마가 바뀔 때 회귀를 찾기 어렵다. 책임별 분리 또는 명시적 입력 모델이 권장된다.

### LOW

1. **파일 경로와 payload 후보 ID의 일대일 대응·중복 집합은 별도로 검증하지 않는다.**
   - 위치: `research/wave8_capital/run_capital100.py:199`, `research/wave8_capital/run_capital100.py:259`, `research/wave8_capital/run_capital100.py:262`
   - 미등록 ID(`W7x`)는 이제 fail-closed로 거부하지만, 예를 들어 `W7a.json` 내부 ID가 등록된 다른 ID `W7b`로 바뀌는 경우 경로/ID 불일치나 중복 후보 집합은 별도로 거부하지 않는다.
   - 현재 다섯 입력은 경로와 ID가 모두 일치하고 입력 SHA도 결과에 묶여 있어 현재 결론 영향은 없다.

2. **정렬된 equity 경로와 역사 최종값의 기준이 다르다.**
   - 위치: `research/wave8_capital/run_capital100.py:37`, `research/wave8_capital/run_capital100.py:227`
   - 수익률은 timestamp로 정렬하지만 역사 최종값은 원래 배열의 마지막 원소를 사용한다. 현재 입력은 정렬돼 있어 결과 영향은 없으나, 비정렬 payload에서는 MC와 역사 스케일링이 서로 다른 종점을 사용할 수 있다.

## 확인된 정상 동작

- **Lookahead**: wave-8은 저장된 equity만 재표본화하며 신규 signal 계산이 없다. 상류 `research/wave7/engine_w7.py:126`~`156`에서 funding regime과 MA200 guard가 각각 `shift(1)`을 적용한다. 검토 범위에서 lookahead를 찾지 못했다.
- **RNG/seed**: 고정 seed `20260721`, 후보별 MC `+101*i`, block `+103*i`가 결정적이다. 현재 산출물과 메모리 내 재계산 결과가 정확히 일치했다.
- **스케일링/reserve**: 역사·MC·block 모두 `$10 reserve + $90 active * growth` 규약을 일관되게 사용하고 MDD에도 reserve를 포함한다.
- **최소 주문/gross**: W2c 최소 레그 `$22.50`은 주문만 PASS지만 별도 레그 gross `$180 > $90`으로 FAIL이다. W7 후보는 최소 주문, gross, source gate에서 차단된다.
- **source gate fail-closed**: 미등록 후보 ID, 누락·추가·중복 gate 이름, 미등록 status, `FAIL`과 `UNDETERMINED`를 모두 거부한다. 현재 W2c는 FAIL/UNDETERMINED를 포함하고 W7a~d는 모두 `sharpe_vs_carry_alone=FAIL`이므로 `eligible_candidates=[]`가 타당하다.
- **수치 경계**: W2c `min_position_weight=0`과 `NaN`은 모두 `ValueError`로 거부된다.
- **안전성**: 네트워크·거래소 SDK·API key·order 실행 경로를 찾지 못했다. 실행은 로컬 JSON 읽기와 결과/보고서 쓰기만 수행한다.
- **산출물 정합성**: 현 코드로 다섯 후보를 메모리에서 재계산한 `results`와 저장 JSON이 exact match다. input manifest, `spec_sha256`, `runner_sha256`도 현재 파일과 모두 exact match다.
- **보고 계보**: 최종 결과·보고서는 코드와 SPEC보다 늦게, audit packet은 결과보다 늦게 갱신됐다. packet의 수치와 판정도 현재 JSON과 일치한다.

## 테스트와 증거

- `python -B -m pytest -q -p no:cacheprovider research\wave8_capital\tests research\wave5\tests research\wave7\tests research\validation\tests` → **25 passed in 4.15s**
- 메모리 내 `_candidate` 재계산 대 저장 JSON 비교 → `results_exact_match=true`, `manifest_exact_match=true`
- 결과 provenance 비교 → `spec_hash_match=true`, `runner_hash_match=true`
- 미등록 ID 경계 → `W7x`의 `all_pass=false`
- W2c 수치 경계 → `0.0`, `NaN` 모두 `ValueError`
- 네트워크·API·order 키워드 검색 → 코드에서 일치 없음
- programming `check-no-excuse-rules.py research\wave8_capital\run_capital100.py` → **8 violations**: generic exception 7건, oversized module 1건

## 스킬 관점 점검

- `omo:programming` 관점 점검: **실행함**. oversized module, 광범위한 `Any`, generic exception은 유지보수 관점 위반이다. 경계 검증은 후보 ID와 핵심 position 값에서 보강됐고, 현재 고정 입력의 결과 정확성은 독립 재현됐다.
- `omo:remove-ai-slops` 관점 점검: **실행함**. 네 테스트는 삭제 전용·동어반복·요청 문구 미러링 테스트가 아니며 실제 행동 경계를 검증한다. production code에는 250 LOC 초과 단일 모듈과 raw-dict 처리라는 유지보수 slop이 남아 있으나, 목표와 무관한 데이터 추출·정규화는 발견하지 못했다.

## Blockers

없음. 위 MEDIUM/LOW 항목은 후속 hardening 대상으로 추적할 수 있으나, 현재 동결 입력·현 SPEC·현재 출력에 대한 정확성과 안전성 판정을 뒤집지 않는다.

최종 결론은 **PASS / APPROVE**, 잔여 코드 품질 상태는 **WATCH**다.
