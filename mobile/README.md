# RaceLens Mobile

RaceLens is a store-ready Expo app shell for the Strategy Arena prediction engine. The app is positioned as a race data analysis tool, not a betting app.

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
npm run web
```

## Connect the Engine

Set the API base URL before launching:

```bash
EXPO_PUBLIC_RACELENS_API_BASE_URL="https://your-api.example.com" npm run start
```

The API adapter currently reads `/api/live-decision` with Strategy Arena query params and falls back to demo data when no API URL is configured.

## Store Build

```bash
npx eas build --platform android --profile production
npx eas build --platform ios --profile production
```

The `app.json` package IDs are placeholders:

- iOS: `com.tttksj.racelens`
- Android: `com.tttksj.racelens`

Change them before first store submission if a final publisher namespace is chosen.

## Design References

- 60fps.design: mobile motion and micro-interaction detail.
- open-design.ai and getdesign.md: DESIGN.md-driven reusable design systems.
- component.gallery: named component patterns and accessibility expectations.
- Dezign Heroes: curated motion, icon, mockup, and design resource discovery.
- Apple Sports/Revolut style synthesis: calm sports terminal plus finance-grade trust.
