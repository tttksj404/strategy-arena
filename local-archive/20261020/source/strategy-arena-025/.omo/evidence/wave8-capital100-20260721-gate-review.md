# Wave-8 $100 Final Gate Review

- recommendation: `APPROVE`
- verdict: `PASS`
- blockers: `[]`
- reviewed_at: `2026-07-21 Asia/Seoul`

## originalIntent

고정된 W2c 및 W7a-W7d 결과만 사용해 총자본 $100, 현금 버퍼 10%, 1x 조건에서 실제 주문 단위와 위험·상류 게이트를 모두 만족하는 후보가 있는지 fail-closed로 판정하고, 통과 후보가 없으면 실행 추천을 만들지 않는다.

## desiredOutcome

결정론적이고 재현 가능한 JSON/Markdown 증거가 모든 입력 출처와 계산을 추적하며, `$100` 실행·승격 후보가 없다는 결론을 수치와 게이트별 차단 사유로 입증해야 한다.

## userOutcomeReview

의도한 결과를 충족한다. 독립 재계산은 저장 JSON과 최대 절대오차 `5.684341886080802e-14` 이내로 일치했고, 최종 무쓰기 드라이런은 생성 시각만 정규화한 뒤 JSON과 보고서 모두 저장본과 정확히 일치했다. `eligible_candidates=[]`이다. W2c는 최소주문 게이트만 통과하지만 별도 레그 gross `$180 > $90` 및 source gate에서 실패한다. W7a-W7d는 최소 주문, gross, source gate가 모두 실패한다. 따라서 $100 실행·승격 가능 후보는 없다.

## checkedArtifacts

- `research/wave8_capital/SPEC.md`
- `research/wave8_capital/run_capital100.py`
- `research/wave8_capital/results/capital100.json`
- `research/wave8_capital/report/capital100_report.md`
- `.codex-fable5/packets/PACKET-20260721-wave8-capital100.md`
- `research/STRATEGY_CARD.md`
- `research/wave2/results/W2c.json`
- `research/wave7/results/W7a.json`
- `research/wave7/results/W7b.json`
- `research/wave7/results/W7c.json`
- `research/wave7/results/W7d.json`
- `research/wave8_capital/tests/test_capital100.py`
- `.omo/evidence/wave8-review/a1/qa-w8-execution-evidence.md`
- `.omo/evidence/wave8-review/a1/wave8-capital100-20260721-manual-qa.md`

## reproducedEvidence

- 최종 테스트: `25 passed in 4.08s`.
- 현재 runner AST 컴파일: `AST_COMPILE_PASS`.
- 무쓰기 드라이런: `json_match=true`, `report_match=true`, `eligible=[]`.
- provenance: 입력 JSON 5개, SPEC, runner의 현재 SHA-256가 `capital100.json` 기록과 모두 일치.
- 독립 수치 재계산:
  - W7c: scaled `$282.28097667939124`, MC p05 `$223.62305964104647`, MDD p95 `7.897261141255506%`, min leg `$0.10160055929243465`, gross `$180`.
  - W2c: scaled `$263.3834574901392`, MC p05 `$229.89975119132242`, MDD p95 `1.741899518379444%`, min leg `$22.50`, gross `$180`.
  - W7d: scaled `$259.7697176535001`, MC p05 `$218.08883305609905`, MDD p95 `4.2269848100528515%`.
  - W7a: scaled `$216.947759261844`, MC p05 `$176.9960933043174`, MDD p95 `8.939358717095228%`.
  - W7b: scaled `$202.80914916528528`, MC p05 `$157.83237266027075`, MDD p95 `12.482632246823518%`.
- source roster는 전 후보에서 완전하며, W2c에는 `FAIL`/`UNDETERMINED`, W7 전 후보에는 Sharpe `FAIL`이 존재한다.

## removeAiSlopsAndProgrammingPass

- 과잉·무용·삭제 전용·요청 제거만 검증하는 테스트 없음.
- 테스트는 equity 일수익 산출, source roster fail-closed, 2-leg gross, 미등록 후보/잘못된 position 입력을 직접 검증한다.
- 구현을 그대로 복제한 tautological assertion이나 불필요한 production extraction/normalization으로 성공을 꾸민 흔적 없음.
- `run_capital100.py`는 pure LOC 약 295로 programming 기준의 250 LOC 권고를 넘는다. 이는 유지보수 NOTE지만, 현재의 명시된 성공 기준이나 재현된 사용자 결과를 위반하지 않아 blocker가 아니다.

## exactEvidenceGaps

- 별도 최신 code-review 보고서는 없고, 기존 QA 두 파일은 최종 provenance/fail-closed 보강 전의 `21 passed` 스냅샷이다. 본 gate에서 최종 코드에 대해 25개 테스트, AST 컴파일, 무쓰기 재생성, 해시·수치·source roster, remove-ai-slops/programming 관점을 직접 재검증했으므로 성공 기준 위반으로 연결되지 않는 NOTE다.
- 전체 `research` compile 실패가 기존 QA에 기록되어 있으나 원인은 범위 밖의 기존 `research/wave1-rwa/live/booster_live.py` 구문 오류다. Wave-8 범위 AST/테스트는 통과했으며 본 성공 기준의 blocker가 아니다.
- W2c OOS `n_trades=2`, 0.02% maker/leg·zero-slippage 가정, 외부 Fable5 감사 대기는 보고서와 패킷에 명시돼 있다. 이들은 실행 추천을 막는 보수적 잔여 리스크이며, `$100` 후보 없음 결론을 약화하지 않는다.

## blockers

없음.
