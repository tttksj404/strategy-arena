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
npm run qa:preflight
```

This gate checks design-token discipline, required design-resource registry coverage, WCAG AA contrast for the current palette, TypeScript, npm audit, Expo Doctor, light/dark mobile and tablet Playwright smoke tests, and a mocked live API path. The app is not ready for store submission unless this command passes cleanly.
