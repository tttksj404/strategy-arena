# RaceLens Design System

## 1. Atmosphere & Identity

RaceLens is a calm race-intelligence terminal, not a gambling product. It should feel like Apple Sports met a premium finance dashboard: quick, precise, restrained, and honest about uncertainty. The signature is the "lens stack": translucent race-state panels layered over a quiet graphite canvas, with probability rails and risk labels that make model confidence inspectable instead of magical.

Reference synthesis:
- curated.design: category-level scan for data apps, AI products, finance tools, and mobile app structure.
- godly.website: recent design taste checks across web interface, branding, product, typography, motion, and app icons.
- awwwards.com: high-craft interaction benchmark; borrow restraint and polish, not heavy web spectacle.
- landing.love: motion and launch-page references for any RaceLens web marketing surface.
- saaspo.com: SaaS dashboard hierarchy, subscription framing, and compact operational UI density.
- onepagelove.com: concise one-page launch/onboarding structure.
- navbar.gallery and cta.gallery: navigation and call-to-action hierarchy, especially for safe non-betting conversion.
- collectui.com and mobbin.com: practical mobile flows, paywall patterns, onboarding, and recurring app screen structure.
- 60fps.design: micro-interactions should be small, tactile, and useful: rail fill, chip press, tab transition, loading shimmer.
- 21st.dev: component quality benchmark for React primitives and templates.
- open-design.ai and getdesign.md: every app screen derives from this DESIGN.md, so the visual language is reusable instead of one-off.
- component.gallery: use standard interface patterns with names users already understand: tabs, carousel-like chips, popovers/sheets later, accordions for legal details, pagination only for long archives.
- rebrand.gallery, logofolio.com, svgl.app, coolors.co, fontpair.co, hugeicons.com: brand, logo, SVG, palette, font, and icon research inputs; final app tokens and assets must remain license-safe.
- Dezign Heroes: use curated icon/motion resources as taste signals, but keep implementation native and store-safe.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/base | surfaceBase | #E8F0E3 | #0B0D0C | Moss-tinted app background |
| Surface/raised | surfaceRaised | #FFF8EA | #151310 | Warm porcelain/obsidian panels |
| Surface/inset | surfaceInset | #DDE8D2 | #221B15 | Inputs, chips, metric wells |
| Surface/glass | surfaceGlass | #FFF8EAE8 | #1C1712E8 | Overlay panels |
| Text/primary | textPrimary | #17130F | #F7EFE2 | Headlines, body |
| Text/secondary | textSecondary | #5E554B | #B9AC9D | Captions, helper text |
| Text/muted | textMuted | #746557 | #958776 | Disabled, quiet metadata |
| Border/subtle | borderSubtle | #D7CFC1 | #3A3027 | Dividers, cards |
| Accent/primary | accentPrimary | #A9431F | #FF8B55 | Main action, focus, selected state |
| Accent/teal | accentTeal | #006B5D | #4ED1B6 | Positive verified states |
| Accent/amber | accentAmber | #9B6A00 | #FFC35A | Caution, market missing |
| Accent/rose | accentRose | #B8324B | #FF6F8A | Error, restriction |
| Accent/violet | accentViolet | #5940B5 | #B79CFF | Pro, model lab |
| Rail/base | railBase | #D6D0C1 | #3A332D | Probability rail track |

### Rules
- The signature color is oxidized copper, not default app blue. It marks primary action, selected state, and the hero confidence number.
- Verdigris teal means verified or healthy data path. Amber means uncertain/missing data. Rose means blocked or unsafe.
- Surfaces use warm porcelain, moss tint, and obsidian so the app feels like a race-intelligence instrument rather than a generic SaaS screen.
- Keep accent colors purposeful. No casino neon, no purple-blue AI gradient, and no raw colors outside `src/theme/tokens.ts`.
- Any legal/safety notice uses amber or rose with plain language.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Tracking | Usage |
|-------|------|--------|-------------|----------|-------|
| Display | 34px | 700 | 1.08 | 0 | App title, hero metric |
| H1 | 28px | 700 | 1.14 | 0 | Screen heading |
| H2 | 22px | 700 | 1.22 | 0 | Section heading |
| H3 | 18px | 700 | 1.30 | 0 | Card title |
| Body | 15px | 500 | 1.45 | 0 | Default text |
| Body/strong | 15px | 700 | 1.35 | 0 | Values and labels |
| Body/sm | 13px | 500 | 1.40 | 0 | Secondary copy |
| Caption | 11px | 700 | 1.25 | 0.08em | Pills, overlines |
| Mono | 12px | 700 | 1.35 | 0.03em | Status IDs and percentages |

### Font Stack
- Primary: system San Francisco / Apple SD Gothic Neo / Noto Sans KR fallback.
- Mono: SF Mono / ui-monospace fallback.

### Rules
- No display text inside compact cards unless it is the single hero metric.
- Korean labels must fit within small controls without truncating meaning.
- Do not use emoji as icons; use vector line icons or text labels.

## 4. Spacing & Layout

### Base Unit
All spacing derives from 4px.

| Token | Value | Usage |
|-------|-------|-------|
| space1 | 4px | Tight inline spacing |
| space2 | 8px | Chip gap |
| space3 | 12px | Compact padding |
| space4 | 16px | Default card padding |
| space5 | 20px | Screen horizontal padding |
| space6 | 24px | Large card padding |
| space8 | 32px | Section separation |
| space10 | 40px | Hero spacing |

### Grid
- Mobile-first single column.
- Screen padding is 20px.
- Cards use 16px radius unless they are pills.
- Bottom navigation height reserves safe-area inset.

### Rules
- No nested cards. Use inset wells inside cards when necessary.
- Lists must remain scan-friendly: label left, value right, explanation below.

## 5. Components

### LensCard
- **Structure**: outer rounded panel with optional header, body, footer.
- **Variants**: base, glass, warning, danger, verified, pro.
- **Spacing**: space4 default, space6 for hero panels.
- **States**: default, pressed, loading, empty.
- **Accessibility**: card actions must expose a role and label.
- **Motion**: fade-up on mount, press scale 0.985.

### StatusPill
- **Structure**: compact pill with optional dot and uppercase/caption label.
- **Variants**: verified, caution, blocked, pro, neutral.
- **Spacing**: horizontal space3, vertical space2.
- **States**: default, focus.
- **Accessibility**: text never color-only; label includes semantic state.
- **Motion**: opacity + transform only.

### ProbabilityRail
- **Structure**: label row, fixed-height rail, percentage.
- **Variants**: primary, teal, amber, rose, violet.
- **Spacing**: space2 between label and rail.
- **States**: default, loading with shimmer, empty.
- **Accessibility**: exposes percentage in text.
- **Motion**: rail fill animates with spring-like cubic easing.

### RaceSelector
- **Structure**: segmented sport control, date chip row, venue/race picker.
- **Variants**: compact, loading, disabled.
- **Spacing**: space3 between controls, space4 around group.
- **States**: default, selected, disabled, loading, error.
- **Accessibility**: each control is a button with selected state.
- **Touch target**: interactive chips and buttons must be at least 44px tall on mobile.

### ParticipantBoard
- **Structure**: information-first list for each rider or horse: number, name, supporting profile, recent flow, style, and neutral note.
- **Variants**: compact for home, detailed for analysis.
- **Spacing**: rows use space3 gap and inset surface; no nested cards.
- **States**: default, empty later, live data later.
- **Accessibility**: every row includes a combined label so the source data is readable without relying on model picks.
- **Purpose**: keep RaceLens from forcing a judgement by showing the underlying provided race materials next to the model signal.

### MarketOddsBoard
- **Structure**: market-source board for win, quinella/exacta, and trifecta odds: code/label left, selection and odds right, change text below in detailed mode.
- **Variants**: compact for home, detailed for analysis, live, fallback, empty.
- **Spacing**: rows use space3 gap and inset surface; the live/fallback pill stays 32px+ high.
- **States**: live data, fallback/waiting, empty.
- **Accessibility**: every row exposes label, selection, odds, and change text in one readable label.
- **Purpose**: show real-time odds as neutral market material, separate from model recommendations, so users can compare source signals without being pushed toward an action.

### BottomTab
- **Structure**: four fixed tabs with line icons and labels.
- **Variants**: home, analyze, lab, pro.
- **Spacing**: safe-area aware.
- **States**: active, inactive, pressed, focus.
- **Accessibility**: role tab, selected state.

### StoreSafeNotice
- **Structure**: legal card with short headline and body.
- **Variants**: compact, full, blocked.
- **Spacing**: space4.
- **States**: default.
- **Accessibility**: plain language; no small text below 13px for critical warnings.

## 6. Motion & Interaction

### Timing

| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Tap | 120ms | cubic-bezier(0.2, 0.8, 0.2, 1) | Press feedback |
| Standard | 240ms | cubic-bezier(0.32, 0.72, 0, 1) | Tab and chip changes |
| Emphasis | 520ms | cubic-bezier(0.16, 1, 0.3, 1) | Screen entry, rail fill |

### Rules
- Animate only opacity and transform in shared components.
- Respect reduced motion by keeping animations short and non-essential.
- Use 60fps-style microdetails sparingly: rail fill, pill press, bottom-tab spring.

## 7. Depth & Surface

### Strategy
Mixed: tonal-shift plus restrained glass and hairline borders.

| Level | Treatment | Usage |
|-------|-----------|-------|
| Base | surfaceBase | Screen background |
| Raised | surfaceRaised + borderSubtle | Cards |
| Inset | surfaceInset | Metric wells, selectors |
| Glass | surfaceGlass + blur where native supports it | Hero lens panels |

Depth must not look like casino lighting. Use copper-tinted borders, warm porcelain panels, obsidian dark surfaces, and compact motion instead of heavy drop shadows or generic blue glow.
