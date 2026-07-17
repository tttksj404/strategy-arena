# FABLE5 AUDIT PACKET — kcycle-market-timing-snapshots
- 날짜/작업 ID: 2026-07-04 / kcycle-market-timing-snapshots
- 시도 횟수: 1회
- 전달 경로: [ ] 직접 호출  [x] 사용자 수동 전달

## 1. 사용자 목표
경마/경륜 예측률 개선을 위해 현재 구현된 알고리즘을 Fable5식 절차로 판단하고, 실제 개선을 코드에 반영하라는 요청. 성공 기준은 예측률 숫자 과장 없이, 실시간/마감직전 배당 데이터가 사후 데이터 누수 없이 학습·검증에 쌓이도록 만드는 것.

## 2. 현재 구현 요약
- 경륜 삼쌍 배당판 스냅샷 저장 시 `market_timing` 메타를 저장한다.
- `early/mid/late/post_start/unknown` 시점과 `minutes_to_start`, 시장 가중치, 축 신호 허용 여부를 JSONL에 남긴다.
- 삼쌍 신호 묶음(`axis`, `watch`, `lift`)도 함께 저장해 후보 비교가 가능하게 했다.
- 사후 시점으로 판단된 스냅샷은 `post_start_market_blocked`로 저장되도록 했다.
- 수집기와 live decision 경로 모두 새 v2 스냅샷을 쓰도록 연결했다.
- 감사 스크립트가 v2 live-like 스냅샷의 시점 메타 누락과 post-start/pre-result 오표기를 실패로 잡는다.

## 3. 변경 파일
- `engine.py` — 삼쌍 스냅샷 v2 메타 저장, live decision 경로에서 삼쌍판 fetched_at 기준 timing 계산.
- `scripts/collect_kcycle_trifecta_snapshots.py` — 수동/자동 수집기에도 timing/signals 저장.
- `scripts/audit_kcycle_trifecta_snapshot_corpus.py` — timing quality와 누수 방지 critical failure 추가.
- `tests/test_live_decision.py` — v2 스냅샷 저장, timing 필드 필터링, live decision 저장 검증.
- `tests/test_trifecta_snapshot_audit.py` — v2 timing 누락과 post-start 오표기 실패 검증.

## 4. 주요 diff
```diff
+def _snapshot_market_timing_payload(market_timing):
+    ...
+def _snapshot_phase_from_market_timing(market_timing):
+    if phase == "post_start":
+        return "post_start_market_blocked"
+    return "pre_result_market_snapshot"
...
+    market_timing=None,
+    signals=None,
...
+        "schema": "kcycle_trifecta_snapshot_v2" if timing_payload or signal_payloads else "kcycle_trifecta_snapshot_v1",
+        "signals": signal_payloads,
+        "market_timing": timing_payload,
...
+                trifecta_timing = _kcycle_market_timing_policy(ymd, race_no, trifecta_fetched_at)
+                    snapshot_phase=_snapshot_phase_from_market_timing(trifecta_timing),
+                    market_timing=trifecta_timing,
+                    signals={"axis": trifecta_axis_signal, "watch": trifecta_signal, "lift": trifecta_lift_signal}
```

## 5. 실행한 테스트/명령
```text
$ .venv/bin/python -m py_compile engine.py scripts/audit_kcycle_trifecta_snapshot_corpus.py scripts/collect_kcycle_trifecta_snapshots.py
exit code: 0
```

```text
$ .venv/bin/python -m pytest -q tests/test_trifecta_snapshot_audit.py tests/test_live_decision.py
.........................................                                [100%]
41 passed in 0.55s
exit code: 0
```

```text
$ .venv/bin/python -m pytest -q tests/test_prediction_feedback.py
........                                                                 [100%]
8 passed in 0.06s
exit code: 0
```

```text
$ .venv/bin/python scripts/audit_kcycle_trifecta_snapshot_corpus.py --snapshots data/kcycle_trifecta_snapshots.jsonl --out /tmp/kcycle_audit_after_timing.json
"ok": true,
"records": 14021,
"actual_count": 13900,
"training_classes": {"post_result_archive_join": 13951, "unsettled_or_unknown": 35, "live_like_pre_result": 35},
"timing_quality": {"live_like_with_timing": 0, "live_like_missing_timing": 35, "pre_start_snapshots": 0, "post_start_snapshots": 0},
"critical_failures": {"duplicate_keys": 0, "hash_mismatch": 0, "board_count_mismatch": 0, "stored_signal_mismatch": 0, "missing_index_tokens": 0, "extra_index_tokens": 0, "actual_joined_live_like_risk": 0, "missing_timing_for_live_like_v2": 0, "post_start_marked_pre_result": 0}
exit code: 0
```

```text
$ codex-fable5 status
codex-fable5: 0 findings
codex-fable5: no goal plan
exit code: 0
```

## 6. 실패 로그
없음. 기본 `python3 -m pytest ...`는 이 환경에서 Flask/Numpy 의존성이 없어 실패 가능성이 있어, 프로젝트 venv인 `.venv/bin/python`으로 검증했다.

## 7. 남은 리스크
- 기존 live-like v1 스냅샷 35개는 생성 당시 timing 메타가 없어 `live_like_missing_timing`으로 남는다. v2 critical failure 대상은 아니며, 새 저장분부터 timing 메타가 강제된다.
- 이 패치는 예측률을 직접 10%p 올렸다는 증거가 아니다. 안전한 알고리즘 탐색을 위해 마감 전/사후 배당을 분리하는 데이터 품질 개선이다.
- Fable5 직접 모델 감사는 이 환경에서 호출하지 못했고, 본 패킷을 사용자 수동 전달용으로 남겼다.

## 8. 의심되는 설계 결함
- `_market_trifecta_lift_signal`의 설명에는 current full-board-axis 대비 +10.03%p가 들어가지만 same-slice baseline 대비 lift는 +0.55%p다. 사용자에게 10%p 절대 개선으로 보고하면 과장이다.
- `phase == "unknown"`은 date mismatch나 시작시각 산정 실패에서도 시장 가중치 0.30을 허용한다. 운영에서는 unknown을 late처럼 강하게 쓰는 정책이 과감할 수 있어 별도 실험이 필요하다.
- 기존 레거시 스냅샷의 timing 결손은 복원 불가하므로, 향후 성능 평가는 v2 이후 구간과 archive_import 구간을 분리해야 한다.

## 9. Fable5에게 묻는 질문
1. `unknown` timing의 시장 가중치를 0.30으로 유지하는 것이 안전한가, 아니면 별도 실험 전 0.05~0.15로 제한해야 하는가?
2. v2 timing 메타가 없으면 live-like 학습에서 완전 제외하는 현재 방향이 충분한가?
3. post-start 스냅샷을 저장은 하되 학습에서 제외하는 정책이 적절한가, 아니면 별도 파일로 분리해야 하는가?
4. `signals` bundle 구조가 향후 후보 탐색/리플레이에 충분한가?
5. 다음 개선 우선순위는 closing-odds trajectory, participant weekly priors, expert sheet NLP 중 무엇인가?

## 10. 원하는 출력 형식
판정 → finding 목록(위치·근거·심각도·수정 방향) → 개선 제안 → 재검증 계획

## 11. 판정 기준
- PASS: 필수 게이트 전부 evidence 첨부, 미해결 finding 0, 남은 리스크는 자동 제거 불가 항목만
- NEEDS_FIX: 동작하나 구체 finding ≥1 — finding이 닫히기 전까지 완료 아님
- FAIL: 목표 미충족, 데이터 누수/보안 결함 발견, 또는 검증 evidence 부재·조작
