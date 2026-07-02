# RaceLens Design System

## 1. Atmosphere & Identity

RaceLens is a calm race-intelligence terminal, not a gambling product. It should feel like Apple Sports met a premium finance dashboard: quick, precise, restrained, and honest about uncertainty. The signature is the "lens stack": translucent race-state panels layered over a quiet graphite canvas, with probability rails and risk labels that make model confidence inspectable instead of magical.

Reference synthesis:
- 60fps.design: micro-interactions should be small, tactile, and useful: rail fill, chip press, tab transition, loading shimmer.
- open-design.ai and getdesign.md: every app screen derives from this DESIGN.md, so the visual language is reusable instead of one-off.
- component.gallery: use standard interface patterns with names users already understand: tabs, carousel-like chips, popovers/sheets later, accordions for legal details, pagination only for long archives.
- Dezign Heroes: use curated icon/motion resources as taste signals, but keep implementation native and store-safe.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/base | surfaceBase | #F6F7FA | #080A0D | App background |
| Surface/raised | surfaceRaised | #FFFFFF | #11151B | Primary panels |
| Surface/inset | surfaceInset | #EDF0F5 | #171D25 | Inputs, chips, metric wells |
| Surface/glass | surfaceGlass | #F7F9FCE6 | #151B24E6 | Overlay panels |
| Text/primary | textPrimary | #101318 | #F5F7FB | Headlines, body |
| Text/secondary | textSecondary | #596270 | #AAB3C2 | Captions, helper text |
| Text/muted | textMuted | #87909E | #697282 | Disabled, quiet metadata |
| Border/subtle | borderSubtle | #E2E6EE | #252D38 | Dividers, cards |
| Accent/primary | accentPrimary | #276EF1 | #68A0FF | Main action, focus |
| Accent/teal | accentTeal | #008F72 | #4AD6B0 | Positive verified states |
| Accent/amber | accentAmber | #B66A00 | #FFB64D | Caution, market missing |
| Accent/rose | accentRose | #C7334D | #FF6B86 | Error, restriction |
| Accent/violet | accentViolet | #6547D9 | #9E8CFF | Pro, model lab |
| Rail/base | railBase | #DDE3EC | #25303D | Probability rail track |

### Rules
- Blue is only for primary action and focus, not decoration.
- Teal means verified or healthy data path. Amber means uncertain/missing data. Rose means blocked or unsafe.
- Backgrounds stay graphite or near-white; racing content should not become casino neon.
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

Depth must not look like casino lighting. No heavy drop shadows. Use surface stepping, thin borders, and compact motion.
