# RaceLens live-decision P0/P1/P2 implementer report

## Changed files

- `app.py`
- `engine.py`
- `datastore.py`
- `deploy/oracle/deploy.sh`
- `docs/oracle_migration_runbook.md`
- `tests/test_live_decision.py`
- `tests/test_app_data_layer.py`
- `tests/test_oracle_deploy_artifacts.py`

Pre-existing/unrelated working tree entries preserved:

- `runs/qa_adversarial_report.md`
- `runs/release_finalize_report.md`

## Design rationale

### P0 live-decision provider budget and negative cache

- `/api/live-decision` now calls `_compute_base_prediction_cached(..., live_decision=True)`.
- Live-decision provider fetches use `_LIVE_DECISION_PROVIDER_TIMEOUT = 1.5` and `_LIVE_DECISION_PROVIDER_MAX_PAGES = 1`.
- For keirin, worst cold provider path is bounded to the first count page plus one card page, each at 1.5 seconds, leaving app-side processing under the 4 second provider budget.
- For KRA, live-decision passes the same 1.5 second timeout and 1 page limit.
- No-card/no-race base errors are cached in `_NEGATIVE_BASE_PREDICTION_CACHE` for 600 seconds, keyed by `(sport, normalized ymd, meet, race_no)`.
- Negative cache hits return the same honest hold path immediately and do not fabricate model rows, participants, top picks, or market usage.
- `engine.compute_live_decision` no longer applies KCYCLE official fallback when the explicit base error kind is `no_race` or `upstream_api_error`; legacy generic card-model failure fallback without those error kinds remains covered by existing tests.

### P1 quota fairness

- The existing pre-compute claim remains in place, preserving the concurrency guard.
- `datastore.release_live_decision_session[_safely]` decrements only free-user usage and never changes Pro usage.
- `/api/live-decision` releases the claimed free quota for no-card holds and prediction failures before recording the view result.
- Blocked, rate-limited, settled, and normal prediction responses keep the claimed quota.
- Response `app_session.free_analysis_remaining` is read after release/record and reflects the refund.

### P2 deploy hardening

- `deploy/oracle/deploy.sh` keeps explicit `ORACLE_PATH` behavior unchanged.
- If `ORACLE_PATH` is unset, the script detects `/opt/strategy-arena` first, then `/home/ubuntu/strategy-arena`, echoes `Detected ORACLE_PATH=...`, and fails closed if neither exists.
- Built-in smoke now retries 3 times with 5 second sleeps after the compose health check block, reducing false failures from transient TLS readiness.
- `docs/oracle_migration_runbook.md` documents the auto-detect behavior.

## Verification

Commands run:

```bash
python3 -m pytest tests/ -q
bash -n deploy/oracle/deploy.sh
python3 -m py_compile app.py datastore.py engine.py tests/test_live_decision.py tests/test_app_data_layer.py tests/test_oracle_deploy_artifacts.py
git diff --check
```

Final pytest tail 20:

```text
........................................................................ [ 44%]
........................................................................ [ 88%]
...................                                                      [100%]
=============================== warnings summary ===============================
tests/test_cross_domain_model.py::CrossDomainModelTestCase::test_demo_keirin_uses_cross_domain_fallback
  /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py:131: UserWarning: Could not find the number of physical cores for the following reason:
  invalid literal for int() with base 10: ''
  Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
163 passed, 1 warning in 3.79s
```

Additional cleanup/guard evidence:

- `data/` and `mobile/` are clean after verification; pytest-generated data/log changes were restored without writing `.git`.
- `deploy/oracle/deploy.sh` passed `bash -n`.
- No git commit, index write, or push was performed.

## Self-verification limits

- I did not run the real Oracle deploy or remote smoke against production, because that would be an external deployment action and final release judgment is reserved for Fable5.
- I did not commit; repository `.git` write was intentionally avoided.
- Final claim remains implementer-scoped: implementation plus self-verification completed, Fable5 audit pending.

## deflake 후속

Change scope:

- `tests/test_live_decision.py` only.
- Disabled keirin prewarm during this test module/import path and patched `engine.prewarm_keirin_card_pages` to no-op in `LiveDecisionTestCase.setUp`, restoring `_PREWARM_STARTED` in `tearDown`.
- Product code in `app.py` / `engine.py` unchanged.

Command run 5 consecutive times:

```bash
python3 -m pytest tests/ -q
```

Run 1 tail:

```text
........................................................................ [ 44%]
........................................................................ [ 88%]
...................                                                      [100%]
=============================== warnings summary ===============================
tests/test_cross_domain_model.py::CrossDomainModelTestCase::test_demo_keirin_uses_cross_domain_fallback
  /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py:131: UserWarning: Could not find the number of physical cores for the following reason:
  invalid literal for int() with base 10: ''
  Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
163 passed, 1 warning in 4.35s
```

Run 2 tail:

```text
........................................................................ [ 44%]
........................................................................ [ 88%]
...................                                                      [100%]
=============================== warnings summary ===============================
tests/test_cross_domain_model.py::CrossDomainModelTestCase::test_demo_keirin_uses_cross_domain_fallback
  /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py:131: UserWarning: Could not find the number of physical cores for the following reason:
  invalid literal for int() with base 10: ''
  Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
163 passed, 1 warning in 3.88s
```

Run 3 tail:

```text
........................................................................ [ 44%]
........................................................................ [ 88%]
...................                                                      [100%]
=============================== warnings summary ===============================
tests/test_cross_domain_model.py::CrossDomainModelTestCase::test_demo_keirin_uses_cross_domain_fallback
  /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py:131: UserWarning: Could not find the number of physical cores for the following reason:
  invalid literal for int() with base 10: ''
  Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
163 passed, 1 warning in 4.16s
```

Run 4 tail:

```text
........................................................................ [ 44%]
........................................................................ [ 88%]
...................                                                      [100%]
=============================== warnings summary ===============================
tests/test_cross_domain_model.py::CrossDomainModelTestCase::test_demo_keirin_uses_cross_domain_fallback
  /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py:131: UserWarning: Could not find the number of physical cores for the following reason:
  invalid literal for int() with base 10: ''
  Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
163 passed, 1 warning in 3.89s
```

Run 5 tail:

```text
........................................................................ [ 44%]
........................................................................ [ 88%]
...................                                                      [100%]
=============================== warnings summary ===============================
tests/test_cross_domain_model.py::CrossDomainModelTestCase::test_demo_keirin_uses_cross_domain_fallback
  /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py:131: UserWarning: Could not find the number of physical cores for the following reason:
  invalid literal for int() with base 10: ''
  Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
163 passed, 1 warning in 3.82s
```

## smoke production 계약

- `deploy/oracle/smoke.sh`를 production 계약 기준으로 재작성했다.
- `/healthz`는 HTTP 200, `ok == true`, `entitlement_mode == "production"`을 강제한다.
- 신규 device의 `/api/app-session`은 기본 `free`와 `free_analysis_limit == 3`, `free_analysis_remaining == 3`을 강제하며, `SMOKE_EXPECT_ENTITLEMENT=pro`일 때만 entitlement 기대값을 `pro`로 전환한다.
- 2026-07-03 광명 1R `/api/live-decision`은 settled 축약 JSON 계약에 맞춰 `decision == "settled"`와 stale demo names 부재만 확인한다.
- 요청시점 +3일 광명 1R `/api/live-decision`은 wall time 5초 미만, `decision == "hold"`, `status != "blocked"`를 확인하고 같은 device의 free quota remaining이 3으로 유지되는지 재검증한다.
- 스모크 성공 stdout 마지막 줄은 `SMOKE_DONE`이다.
