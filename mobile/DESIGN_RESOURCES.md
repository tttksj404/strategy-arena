# RaceLens Design Resource Registry

This registry is the mandatory reference shelf for RaceLens and future app work. Use these sources as inspiration and validation inputs, not as copy-paste asset sources unless the license is explicitly compatible.

## Product And Screen Inspiration

| Source | Use In RaceLens |
|---|---|
| curated.design | Broad product/site category scan, especially apps, finance, AI, and data tools. |
| godly.website | Recent visual direction, app icons, app screenshots, typography, and motion taste checks. |
| awwwards.com | High-polish interaction and visual craft benchmark; keep only practical patterns for native app UX. |
| landing.love | Motion and landing-section timing ideas for web marketing surfaces. |
| saaspo.com | SaaS layout density, dashboard hierarchy, pricing/subscription framing. |
| onepagelove.com | One-screen onboarding, concise landing copy, and launch page structure. |
| collectui.com | Common UI flows and compact component variants. |
| mobbin.com | Native mobile app navigation, paywall, onboarding, analytics, and subscription patterns. |

## Component And Conversion Patterns

| Source | Use In RaceLens |
|---|---|
| navbar.gallery | Navigation patterns, tab density, active states, and mobile-safe hierarchy. |
| cta.gallery | CTA wording, button grouping, and conversion-safe hierarchy without betting pressure. |
| 21st.dev | React component quality benchmark and reusable primitive references. |
| component.gallery | Component naming, standard affordances, and accessibility expectations. |
| 60fps.design | Micro-interaction timing, rail fills, tab transitions, and motion restraint. |

## Brand, Assets, And Tokens

| Source | Use In RaceLens |
|---|---|
| rebrand.gallery | Visual identity maturity, color-system restraint, and before/after brand coherence. |
| logofolio.com | Logo mark exploration and app icon quality benchmark. |
| svgl.app | SVG/logo source discovery only when license and trademark usage allow it. |
| coolors.co | Palette exploration; final colors must still land in `src/theme/tokens.ts`. |
| fontpair.co | Font pairing exploration; default remains system font unless app-store build needs a bundled family. |
| hugeicons.com | Icon style benchmark; app implementation should prefer native/icon-library equivalents unless licensed. |
| dezignheroes.framer.website | Curated resource discovery across mockups, icons, UI kits, motion, and design tools. |

## Application Rules

- Before a new app screen is considered complete, check at least one source from product inspiration, one from component/conversion patterns, and one from brand/assets/tokens.
- Do not clone copyrighted screenshots, icons, logos, UI kits, or templates into the repo without a compatible license.
- Record source influence in `DESIGN.md` and keep raw color tokens confined to `src/theme/tokens.ts`.
- Native mobile verification must include at least one mobile viewport screenshot or equivalent device/simulator proof.
