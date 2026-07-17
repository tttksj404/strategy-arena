# FABLE5 AUDIT PACKET — kcycle-trifecta-lift-application
- 날짜/작업 ID: 2026-07-04 / kcycle-trifecta-lift-application
- 시도 횟수: 1회
- 전달 경로: [ ] 직접 호출  [x] 사용자 수동 전달

## 1. 사용자 목표
경마/경륜 예측률을 실제로 높이는 알고리즘을 가져와 반영하라는 요청. 단순 리스크 개선이 아니라, 검증된 후보가 최종 예측 출력에 들어가야 한다.

## 2. 현재 구현 요약
- 기존 `_market_trifecta_lift_signal`은 holdout에서 `PROMOTE_STAT_STRICT_LIFT`를 받은 순서 재랭킹 후보였지만, 앱 최종 `picks`에는 실질 반영되지 않았다.
- `xdom_bradley_terry_position` 기반 삼쌍 순서 재랭킹을 `TRI` pick에 반영하도록 연결했다.
- 반영 조건은 `market_timing.allow_trifecta_axis == true`일 때만이다.
- early/mid/post-start 구간은 관찰 신호로만 두고 `blocked_by_timing`을 붙인다.
- 50% 절대 적중 후보는 여전히 저표본 watch로 유지하며 배포 신호로 쓰지 않았다.

## 3. 변경 파일
- `engine.py` — `_mobile_live_picks`가 `trifecta_order_signal`을 받아 `TRI`/`TRB` 순서에 반영.
- `engine.py` — `trifecta_lift_signal`에 `applied`, `timing_phase`, `blocked_by_timing`을 부여하고 timing gate 적용.
- `tests/test_live_decision.py` — lift 신호가 실제 TRI pick에 반영되는지, mid 구간에서는 차단되는지 검증.
- `data/kcycle_trifecta_rule_search_results.json` / `.md` — 깨끗한 코퍼스 기준 재탐색 결과 갱신.

## 4. 주요 diff
```diff
-def _mobile_live_picks(rows, trifecta_axis_signal=None):
+def _mobile_live_picks(rows, trifecta_axis_signal=None, trifecta_order_signal=None):
...
+            tri_order = [int(x) for x in (order_signal or {}).get("order", []) ...]
+            tri_prob = float((order_signal or trifecta_axis_signal).get("expected_trio_exact") or 0.0)
+            "basis": (order_signal or trifecta_axis_signal).get("tier"),
...
+        allow_lift = bool(market_timing.get("allow_trifecta_axis"))
+        if allow_lift:
+            trifecta_lift_signal["applied"] = True
+            trifecta_lift_used = True
+            suffix = " · 삼쌍 순서 재랭킹 반영(holdout 26.09%, +0.55pp)"
+        else:
+            trifecta_lift_signal["blocked_by_timing"] = True
...
+            trifecta_lift_signal if trifecta_lift_used else None,
```

## 5. 실행한 테스트/명령
```text
$ .venv/bin/python -m py_compile engine.py tests/test_live_decision.py
exit code: 0
```

```text
$ .venv/bin/python -m pytest -q tests/test_live_decision.py::LiveDecisionTestCase::test_live_decision_exposes_trifecta_lift_signal_when_full_board_matches tests/test_live_decision.py::LiveDecisionTestCase::test_trifecta_lift_signal_is_blocked_before_late_timing
..                                                                       [100%]
2 passed in 0.28s
exit code: 0
```

```text
$ .venv/bin/python -m pytest -q tests/test_live_decision.py tests/test_trifecta_snapshot_audit.py tests/test_prediction_feedback.py
..................................................                       [100%]
50 passed in 0.34s
SNAPSHOT_LINES_BEFORE=14020 AFTER=14020
exit code: 0
```

```text
$ .venv/bin/python scripts/audit_kcycle_trifecta_snapshot_corpus.py --snapshots data/kcycle_trifecta_snapshots.jsonl --out /tmp/kcycle_audit_after_cleanup.json
"ok": true,
"records": 14020,
"actual_count": 13900,
"critical_failures": {"duplicate_keys": 0, "hash_mismatch": 0, "board_count_mismatch": 0, "stored_signal_mismatch": 0, "missing_index_tokens": 0, "extra_index_tokens": 0, "actual_joined_live_like_risk": 0, "missing_timing_for_live_like_v2": 0, "post_start_marked_pre_result": 0}
exit code: 0
```

```text
$ .venv/bin/python scripts/search_kcycle_trifecta_rules.py --snapshots data/kcycle_trifecta_snapshots.jsonl --out-json data/kcycle_trifecta_rule_search_results.json --out-md data/kcycle_trifecta_rule_search_results.md --limit 300
{"records": 13900, "predicate_count": 7887, "xdom_predicate_count": 11, "evaluated_candidates": 80359, "fifty_watch_or_promote_count": 153, "promotion_count": 0, "xdom_fifty_watch_or_promote_count": 99, "xdom_promotion_count": 0, "directional_lift_count": 79, "directional_lift_promote_count": 2, "stat_strict_lift_count": 2, "risk_flags": {"no_robust_promotion": true, "low_sample_watch_only": true, "xdom_duplicate_inflation": true, "requires_more_outcome_linked_snapshots": true, "directional_lift_not_stat_strict": false}}
exit code: 0
```

## 6. 실패 로그
초기 테스트에서 `test_trifecta_lift_signal_is_blocked_before_late_timing`가 실패했다. 원인은 코드가 아니라 테스트 fixture 시간이 5경주 시작 추정 12:32 기준 mid가 아닌 early였기 때문이다. fetched_at을 12:05로 수정해 mid 구간 테스트로 재검증했다.

또한 테스트가 기본 `data/kcycle_trifecta_snapshots.jsonl`에 인공 스냅샷 1건을 쓰는 문제가 발견됐다. `setUp` 기본값을 `KCYCLE_TRIFECTA_SNAPSHOT_ENABLED=0`으로 바꾸고 저장 기능 테스트만 명시적으로 활성화했다. 생성된 인공 hash `480b97ea...` 한 줄은 JSONL/keys에서 제거했고, 이후 pytest 전후 line count가 14020으로 동일함을 확인했다.

## 7. 남은 리스크
- 이번 반영은 전체 삼쌍 순서 적중을 50%로 끌어올린 것이 아니다. 통계 엄격 후보의 적용 범위에서 holdout exact 26.09%, same-slice board 대비 +0.55%p 개선이다.
- 50% 절대 적중 후보는 holdout year별 표본이 작아 계속 watch only다.
- Fable5 직접 모델 감사는 이 환경에서 호출하지 못했고, 본 패킷을 사용자 수동 전달용으로 남겼다.

## 8. 의심되는 설계 결함
- `market_timing.phase == "unknown"`에서 lift 반영을 허용하는 현재 정책은 운영상 과감할 수 있다. 실시간 v2 스냅샷이 쌓이면 unknown을 별도 그룹으로 검증해야 한다.
- UI가 `TRI` basis를 충분히 드러내지 않으면 사용자가 왜 1착 후보와 삼쌍 순서가 다를 수 있는지 혼동할 수 있다.
- 전체 예측률 50%급 목표는 현재 코퍼스에서 검증된 후보가 없으므로, 추가 데이터원 또는 더 강한 live-like closing trajectory 데이터가 필요하다.

## 9. Fable5에게 묻는 질문
1. `xdom_bradley_terry_position` 순서 재랭킹을 TRI pick에만 반영하는 범위가 적절한가?
2. same-slice board 대비 +0.55%p이지만 current full-board-axis 대비 +10.03%p인 설명은 UI에서 어떻게 표현해야 오해가 없는가?
3. `unknown` timing에서는 lift 적용을 차단하는 편이 나은가?
4. 50% watch 후보를 유료 사용자에게 노출할 경우 어떤 경고/표본 기준이 필요한가?
5. 다음 후보 탐색은 HGB surrogate, closing trajectory, participant priors 중 어디를 우선해야 하는가?

## 10. 원하는 출력 형식
판정 → finding 목록(위치·근거·심각도·수정 방향) → 개선 제안 → 재검증 계획

## 11. 판정 기준
- PASS: 필수 게이트 전부 evidence 첨부, 미해결 finding 0, 남은 리스크는 자동 제거 불가 항목만
- NEEDS_FIX: 동작하나 구체 finding ≥1 — finding이 닫히기 전까지 완료 아님
- FAIL: 목표 미충족, 데이터 누수/보안 결함 발견, 또는 검증 evidence 부재·조작
