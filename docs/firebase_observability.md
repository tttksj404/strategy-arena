# RaceLens Firebase Observability

Firebase is used only for production observability:

- Crashlytics: native crash reports and recovered JavaScript error reports.
- Google Analytics for Firebase: minimal product health events already sanitized by `uxAnalytics`.

Firebase Authentication is intentionally disabled. RaceLens does not need login for the current anonymous free tier, rewarded-ad credits, or same-device purchase session. Add Auth only when account-based Pro restore is implemented across devices or platforms.

## Android Play Store setup

1. Create or open the Firebase project.
2. Add Android app `com.tttksj.racelens`.
3. Download `google-services.json`.
4. Place it at `mobile/google-services.json` or set `RACELENS_FIREBASE_ANDROID_SERVICES_FILE`.
5. Set `RACELENS_FIREBASE_PROJECT_ID=racelens-tttksj`.
6. Build with `EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED=1`.

Keep `EXPO_PUBLIC_RACELENS_FIREBASE_AUTH_ENABLED` unset or `0`.

The Play Store release target is Android. Do not add iOS Firebase config until an App Store build is in scope.

## Event policy

Allowed Firebase event fields are coarse product telemetry only: tab, sport, race number, latency, market-used flag, market-risk level, and rounded confidence percentages.

Do not log participant names, selected betting combinations, device IDs, user IDs, email addresses, purchase tokens, or free-form API errors.
