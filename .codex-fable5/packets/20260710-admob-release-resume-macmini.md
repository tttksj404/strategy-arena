# Fable5 Audit Packet: RaceLens AdMob Release Resume

Date: 2026-07-10 KST
Repository: `/Users/tttksj/github-portfolio-docs-work/strategy-arena`
HEAD: `ee4f70c`

## Requested outcome

- Keep preview test ads completely separated from production AdMob inventory.
- Fail closed for production AAB builds without real AdMob app and rewarded-unit IDs.
- Carry Oracle deployment configuration and release documentation.
- Re-run the final Playwright rewarded-ad QA on macOS.
- Re-verify backend tests, TypeScript, diff hygiene, and secret-pattern hygiene.
- Connect real AdMob IDs when the console and credentials are available.

## Fresh evidence

- Git sync: `git pull --ff-only origin main` fast-forwarded `3a892f8..d1f3344` while preserving the pre-existing local data/log changes.
- Mobile install: `npm install` completed; 598 packages audited, 0 vulnerabilities.
- Rewarded-ad Playwright QA: `node scripts/qa-rewarded-ads-policy.mjs` passed.
  - ads off / Firebase off: `adTextCount=0`, `rewardRequestCount=0`.
  - preview ads on / Firebase on: `PREVIEW TEST AD` present, `rewardRequestCount=1`.
- Independent visual QA: two read-only passes both returned PASS for all three fresh 390x844 captures; no CJK clipping or state leakage.
- Backend: `.venv/bin/python -m pytest tests -q` -> `181 passed in 7.06s` after installing the newly pinned `cryptography==49.0.0` and other requirements into the project venv.
- TypeScript: `npm run typecheck` -> exit 0.
- Diff hygiene: `git diff --check` -> exit 0.
- Secret patterns: high-entropy token/private-key scan passed with runtime logs excluded; unexpected non-test AdMob ID scan passed. Only Google's official public sample IDs remain in the allowed release-gate sources.
- Production fail-closed: `node scripts/check-store-release-env.mjs` with rewarded ads enabled and blank IDs exited 1 with explicit invalid-app-ID, invalid-unit-ID, and mobile/server SSV unit mismatch failures.
- Oracle deploy path: GitHub repository secret `ORACLE_SSH_PRIVATE_KEY` exists; workflow inputs and Oracle env examples include rewarded-ad enablement and unit-ID propagation.

## External-state findings

1. Google Play Console app `RaceLens` was created for `com.tttksj.racelens`; policy, Play App Signing, export, ads, content-rating, target-audience, government, finance, health, category, contact, and privacy declarations were saved.
2. Firebase Android app `1:1065670671903:android:f0ec09dd3a1110b0f15f2a` and ignored local `google-services.json` are connected.
3. AdMob app ID `ca-app-pub-6215167272534219~9434798589` and rewarded unit `ca-app-pub-6215167272534219/6412636739` were created. Reward configuration is amount `1`, item `analysis_credit`, matching the server verifier.
4. The real IDs, production ad/test-mode gates, Firebase values, and sensitive Google services file are present in the EAS production environment. `npm run qa:store-readiness` and the full `npm run qa:release` passed with those values.
5. Fresh backend verification is `.venv/bin/python -m pytest -q` -> `184 passed in 3.899s`. A suite-order-only mock assertion was corrected to inspect direct calls instead of nested mock history.
6. Preview APK build `45cea2bc-3646-442a-a904-866b954163bd` finished. Production AAB build `c38bf144-1446-4fc6-b8d5-bf633f0de299` finished with version code 9 and SHA-256 `501cf1a8a47fdbd614ad06e3b6ea6507d12950d1184665b952be6a3e795804ec`. The AAB ZIP structure is valid and includes a JAR signature block.
7. Artifact inspection confirms preview uses Google sample app ID with test mode `1`, while production uses app ID `ca-app-pub-6215167272534219~9434798589`, rewarded unit `ca-app-pub-6215167272534219/6412636739`, rewarded ads `1`, and test mode `0`.
8. Rewarded-enabled Oracle deploy run `29132680693` completed successfully. Live `/healthz` reports production entitlement mode, the Oracle smoke suite returns `SMOKE2_DONE`, and an unsigned `/api/rewarded-ad/ssv` request returns HTTP 400 `invalid callback` instead of the previous 404, proving the guarded verifier is live.
9. Google Play production access is externally gated by the required closed test with at least 12 opted-in testers for at least 14 days.
10. AdMob payment-profile completion is externally gated and real ad serving remains held until the account owner supplies payment details.
11. Play submission through EAS remains unavailable because no Google Play service-account key is configured, but the browser path is now working. Chrome extension file-URL access is persistently enabled (`newAllowFileAccess: true`) and was proven with real Play Console file chooser uploads.
12. Google AdMob's live callback verifier reached Oracle and exposed two official-console formats: the numeric suffix of the configured unit ID and Google's UI verification placeholder `1234567890`. The verifier now accepts the exact configured full ID/suffix plus that placeholder only after Google signature verification. The route returns HTTP 200 for the placeholder before persistence, so the console probe cannot mint credit. Regression tests failed before the fix and pass afterward. Commits through `ee4f70c` are pushed; Oracle deployment run `29135022133` succeeded in 2m30s. The AdMob console then reported `완료 콜백 URL이 확인되었습니다.`, the callback URL was saved on the rewarded unit, and a live session read for `racelens_ssv_test_001` showed `rewarded_analysis_credits: 0` with PostgreSQL ready.
13. Google Play Data safety is saved with the production Firebase/AdMob collection and sharing declarations. The store listing is saved with the 512x512 icon, 1024x500 feature graphic, and six 1080x1920 phone screenshots. Production AAB version code 9 was uploaded to closed testing Alpha, passed Play processing with target SDK 36 and no blocking bundle error, and the Korean release notes were saved. South Korea is the configured test region. The track is now blocked only on tester selection and Google review submission; 0 of the required 12 testers have opted in.

## Audit request

Review the pulled rewarded-ad/SSV implementation and the evidence above. Return one verdict:

- `PASS` only if production release may proceed after supplying the real IDs and credentials without code changes.
- `NEEDS_FIX` if code/config/documentation changes are still required.
- `FAIL` for a security, replay, preview/production isolation, or SSV verification defect.

Current local status: production AAB and the SSV compatibility fix have passed self-verification and are pushed/deployed. Full mobile release QA, 78 Playwright cases, 12/12 adversarial cases, six release visual sizes, rewarded-ad policy QA, store-readiness, server QA, TypeScript, Expo doctor, npm audit, diff hygiene, secret-pattern scan, live health smoke, and live AdMob SSV console verification all pass. Play assets and AAB are uploaded and the Alpha release is saved. Closed-test email selection/12-person enrollment, Google review submission, payment-profile completion, and independent Fable5 verdict remain external gates.
