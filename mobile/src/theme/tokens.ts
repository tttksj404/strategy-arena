export const colors = {
  surfaceBase: { light: '#F6F7FA', dark: '#080A0D' },
  surfaceRaised: { light: '#FFFFFF', dark: '#11151B' },
  surfaceInset: { light: '#EDF0F5', dark: '#171D25' },
  surfaceGlass: { light: '#F7F9FCE6', dark: '#151B24E6' },
  textPrimary: { light: '#101318', dark: '#F5F7FB' },
  textSecondary: { light: '#596270', dark: '#AAB3C2' },
  textMuted: { light: '#87909E', dark: '#697282' },
  borderSubtle: { light: '#E2E6EE', dark: '#252D38' },
  accentPrimary: { light: '#276EF1', dark: '#68A0FF' },
  accentTeal: { light: '#008F72', dark: '#4AD6B0' },
  accentAmber: { light: '#B66A00', dark: '#FFB64D' },
  accentRose: { light: '#C7334D', dark: '#FF6B86' },
  accentViolet: { light: '#6547D9', dark: '#9E8CFF' },
  railBase: { light: '#DDE3EC', dark: '#25303D' }
} as const;

export const gradients = {
  appTop: { light: '#FFFFFF', dark: '#11151B' },
  appBottom: { light: '#F6F7FA', dark: '#080A0D' }
} as const;

export const space = {
  space1: 4,
  space2: 8,
  space3: 12,
  space4: 16,
  space5: 20,
  space6: 24,
  space8: 32,
  space10: 40
} as const;

export const radius = {
  small: 10,
  medium: 16,
  large: 22,
  pill: 999
} as const;

export const typography = {
  display: { fontSize: 34, fontWeight: '700' as const, lineHeight: 37 },
  h1: { fontSize: 28, fontWeight: '700' as const, lineHeight: 32 },
  h2: { fontSize: 22, fontWeight: '700' as const, lineHeight: 27 },
  h3: { fontSize: 18, fontWeight: '700' as const, lineHeight: 23 },
  body: { fontSize: 15, fontWeight: '500' as const, lineHeight: 22 },
  bodyStrong: { fontSize: 15, fontWeight: '700' as const, lineHeight: 20 },
  bodySm: { fontSize: 13, fontWeight: '500' as const, lineHeight: 18 },
  caption: { fontSize: 11, fontWeight: '700' as const, lineHeight: 14, letterSpacing: 0.88 },
  mono: { fontSize: 12, fontWeight: '700' as const, lineHeight: 16, letterSpacing: 0.36 }
} as const;

export type ThemeMode = 'light' | 'dark';

export function palette(mode: ThemeMode) {
  return Object.fromEntries(
    Object.entries(colors).map(([key, value]) => [key, value[mode]])
  ) as { [K in keyof typeof colors]: string };
}

export function gradient(mode: ThemeMode) {
  return {
    app: [gradients.appTop[mode], gradients.appBottom[mode]] as const
  };
}
