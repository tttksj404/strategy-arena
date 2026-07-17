# RACELENS Firebase Disclosure and Ads Honesty Fix

Date: 2026-07-09

## Result

- FIREBASE_DISCLOSED: yes
  - `/legal/privacy` now discloses Google Firebase Analytics and Crashlytics use, app instance identifiers, event parameters, crash diagnostics, device/OS/app version/stack information, Google transfer/processing, and purposes.
  - Store listing/Data safety notes now disclose Firebase Analytics/Crashlytics, app activity events, diagnostics, app instance/Firebase installation identifiers, Google Firebase provider processing, encrypted transfer, and deletion path.
  - `release.env.example` documents that Firebase-enabled releases require matching privacy/Data safety disclosure.
- ADS_HONEST: yes
  - Production release env defaults rewarded ads to `EXPO_PUBLIC_RACELENS_REWARDED_ADS=0`.
  - Rewarded-ad UI and reward claim calls are gated behind `rewardAdsEnabled`.
  - Rewarded ads off QA asserts DOM contains zero `광고` strings and `/api/rewarded-ad/claim` is not called.
  - Rewarded ads on QA still snapshots the on path separately for future integration work.

## Changed Files

- `app.py`
- `tests/test_app_data_layer.py`
- `mobile/App.tsx`
- `mobile/src/screens/AnalyzeScreen.tsx`
- `mobile/tests/mobile-web.spec.js`
- `mobile/release.env.example`
- `mobile/scripts/finalize-release.mjs`
- `mobile/scripts/generate-brand-assets.mjs`
- `mobile/scripts/check-store-pack.mjs`
- `mobile/scripts/qa-release-visual.mjs`
- `mobile/scripts/qa-rewarded-ads-policy.mjs`
- `runs/store_pack/listing.md`

## Verification

- PASS: `cd mobile && npm.cmd run typecheck`
- PASS: `cd mobile && npm.cmd run lint:design`
- PASS: `cd mobile && npm.cmd run lint:security`
- PASS: `cd mobile && npm.cmd run qa:mobile` (`72 passed`)
- PASS: `cd mobile && npm.cmd run qa:api-failure`
- PASS: `cd mobile && npm.cmd run qa:analytics` (`analytics QA passed: 11 events`)
- PASS: `cd mobile && npm.cmd run qa:release-visual`
  - release readiness: no bad text, forbidden action buttons, overflow, or small targets
  - rewardedAds off/firebase off: `adTextCount=0`, `rewardRequestCount=0`
  - rewardedAds on/firebase on: `adTextCount=7`, `rewardRequestCount=0`
- PASS: `cd mobile && npm.cmd run qa:adversarial` (`ADVERSARIAL=12/12 PASS`)
- PASS: `cd mobile && npm.cmd run qa:store-pack`
- PASS: `.venv\Scripts\python.exe -m pytest tests/test_app_data_layer.py::AppDataLayerTestCase::test_legal_pages_use_real_default_contact_and_required_subscription_terms -q`
- PASS: Flask test client render check for `/legal/privacy`, `/legal/terms`, and `/legal/account-deletion`
- PASS: `git check-ignore -v mobile/google-services.json mobile/GoogleService-Info.plist`

## DEPLOY_REQUIRED

- `DEPLOY_REQUIRED=1`
- Server legal pages changed in `app.py`; server redeploy is required for production `/legal/privacy`, `/legal/terms`, and `/legal/account-deletion` to reflect this fix.
- Server deployment was intentionally not performed.

FIREBASE_DISCLOSED=y ADS_HONEST=y GATES=11/11 commits=1 DEPLOY_REQUIRED=1
