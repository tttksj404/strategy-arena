# RaceLens Mobile

RaceLens is a store-ready Expo app shell for the RaceLens API server. The app is positioned as a race data analysis tool, not a betting app.

## Product Position

- App name: RaceLens / 레이스렌즈
- Store category target: Sports / Data analysis, not gambling.
- No betting links, purchase prompts, stake sizing, guaranteed-return language, or real-money gambling flow.
- All prediction surfaces must show data-source state, sample size, and risk/uncertainty.
- Premium plan copy is prepared, but payment is intentionally inactive until store policy review.

## Run

```bash
npm install
npm run typecheck
npm run build
npm run web
```

## Connect the Engine

Set the API base URL before launching:

```bash
EXPO_PUBLIC_RACELENS_API_BASE_URL="https://your-api.example.com" npm run start
```

The API adapter reads `/api/live-decision` with RaceLens query params. If no API URL or official race card is available, it shows an unavailable state instead of mixing in demo racers or odds from another race.

## UX Risk Telemetry

Set a self-hosted analytics endpoint before beta or store release:

```bash
EXPO_PUBLIC_RACELENS_ANALYTICS_URL="https://your-api.example.com/api/ux-events" npm run start
```

The app sends only anonymous product events: app open, screen view, tab selection, race context changes, analysis request/result/error, latency, market state, and coarse confidence percentages. It does not send participant names, horse/rider names, user IDs, device IDs, betting selections, or meet names. Keep the endpoint server-side so beta retention, session length, analysis-start rate, detail-view rate, and Pro-screen drop-off can be measured without adding third-party tracking.

## Store Build

```bash
npm run qa:release
npm run qa:store-readiness
npm run qa:submission
npx eas build --platform android --profile production
npx eas build --platform ios --profile production
```

The production profile already fixes the public app contract to `https://strategy-arena.onrender.com`: API, anonymous analytics, legal pages, support page, support email, disabled billing, disabled Firebase Auth, and disabled offline examples. `qa:store-readiness` reads those profile defaults, so local release shells only need the non-public submission credentials. It rejects placeholder domains, HTTP URLs, localhost URLs, unreachable policy pages, unreachable `/health`, and broken Korean text.

Keep Firebase service files and real AdMob/SSV IDs in EAS production secrets or a local ignored `release.env`; never commit them. Production builds stay blocked until Firebase Analytics/Crashlytics and real rewarded-ad IDs are present.

Run `node scripts/finalize-release.mjs --domain <domain> --support-email <email>` to write `release.env` and smoke-test production.
Review `runs/release_finalize_report.md` for the PASS/WARN/FAIL table before EAS submission.
Treat `unsupported_event` on `live_odds_refresh` as a stale-server WARN and redeploy `deploy/oracle/deploy.sh` from Mac.

Production EAS builds must inject the same values through the production profile environment, for example:

```json
{
  "build": {
    "production": {
      "env": {
        "EXPO_PUBLIC_RACELENS_API_BASE_URL": "https://api.example.invalid",
        "EXPO_PUBLIC_RACELENS_ANALYTICS_URL": "https://api.example.invalid/api/ux-events",
        "EXPO_PUBLIC_RACELENS_PRIVACY_URL": "https://api.example.invalid/legal/privacy",
        "EXPO_PUBLIC_RACELENS_TERMS_URL": "https://api.example.invalid/legal/terms",
        "EXPO_PUBLIC_RACELENS_ACCOUNT_DELETION_URL": "https://api.example.invalid/legal/account-deletion",
        "EXPO_PUBLIC_RACELENS_SUPPORT_EMAIL": "support@example.invalid",
        "EXPO_PUBLIC_RACELENS_BILLING_MODE": "disabled",
        "EXPO_PUBLIC_RACELENS_REWARDED_ADS": "1",
        "EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE": "0",
        "EXPO_PUBLIC_RACELENS_ADMOB_ANDROID_APP_ID": "ca-app-pub-PUBLISHER~APP",
        "EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID": "ca-app-pub-PUBLISHER/UNIT"
      }
    }
  }
}
```

Keep placeholder values out of release builds; the snippet shows structure only.

`npm run build:android:preview` creates an internal APK using Google's official test ad. `npm run build:android:production` first runs the production environment gate and refuses to build if rewarded ads are disabled, IDs are missing, test mode is enabled, or Google test IDs are present. See `docs/admob_rewarded_release.md` for the exact AdMob/SSV handoff.

The `app.json` package IDs are placeholders:

- iOS: `com.tttksj.racelens`
- Android: `com.tttksj.racelens`

Change them before first store submission if a final publisher namespace is chosen.

## Design References

- curated.design, godly.website, awwwards.com, landing.love, saaspo.com, onepagelove.com: product, landing, and high-polish visual benchmarks.
- navbar.gallery, cta.gallery, collectui.com, mobbin.com: navigation, CTA, common UI flows, and native mobile screen references.
- 60fps.design, 21st.dev, component.gallery: mobile motion, component quality, and standard interface pattern checks.
- rebrand.gallery, logofolio.com, svgl.app, coolors.co, fontpair.co, hugeicons.com: brand, logo, SVG, palette, font, and icon research inputs.
- dezignheroes.framer.website: curated discovery layer for design resources and tool references.
- open-design.ai and getdesign.md: DESIGN.md-driven reusable design systems.
- Apple Sports/Revolut style synthesis: calm sports terminal plus finance-grade trust.

Run `npm run lint:design` to confirm design token discipline and required resource registry coverage.

## Pre-Release QA

```bash
npm run qa:release
```

This gate checks design-token discipline, required design-resource registry coverage, WCAG AA contrast for the current palette, TypeScript, npm audit, Expo Doctor, light/dark mobile and tablet Playwright smoke tests, mocked live API, anonymous UX telemetry, API-failure paths, and release visual readiness. The app is not ready for store submission unless this command passes cleanly.

Run `npm run qa:store-readiness` only with the real submission environment loaded. It blocks submission unless production API, privacy policy, terms, account deletion, support contact, and billing mode are explicitly configured and reachable over HTTPS.

Run `npm run qa:submission` before TestFlight, Google closed testing, or public store submission. It chains the full local release gate, store environment gate, and backend data-layer/privacy tests so the app cannot ship with localhost URLs, missing policy pages, disabled receipt validation configuration, unsafe CORS assumptions, or traceback-leaking API errors.
