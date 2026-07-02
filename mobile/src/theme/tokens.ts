export const colors = {
  surfaceBase: { light: '#E8F0E3', dark: '#0B0D0C' },
  surfaceRaised: { light: '#FFF8EA', dark: '#151310' },
  surfaceInset: { light: '#DDE8D2', dark: '#221B15' },
  surfaceGlass: { light: '#FFF8EAE8', dark: '#1C1712E8' },
  textPrimary: { light: '#17130F', dark: '#F7EFE2' },
  textSecondary: { light: '#5E554B', dark: '#B9AC9D' },
  textMuted: { light: '#746557', dark: '#958776' },
  borderSubtle: { light: '#D7CFC1', dark: '#3A3027' },
  accentPrimary: { light: '#A9431F', dark: '#FF8B55' },
  accentTeal: { light: '#006B5D', dark: '#4ED1B6' },
  accentAmber: { light: '#9B6A00', dark: '#FFC35A' },
  accentRose: { light: '#B8324B', dark: '#FF6F8A' },
  accentViolet: { light: '#5940B5', dark: '#B79CFF' },
  railBase: { light: '#D6D0C1', dark: '#3A332D' }
} as const;

export const gradients = {
  appTop: { light: '#FFF8EA', dark: '#151310' },
  appBottom: { light: '#E8F0E3', dark: '#0B0D0C' }
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
