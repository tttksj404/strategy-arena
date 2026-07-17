export const colors = {
  surfaceBase: { light: '#F6F8F1', dark: '#080A08' },
  surfaceRaised: { light: '#FFFFFF', dark: '#111611' },
  surfaceInset: { light: '#EBF0E8', dark: '#1C241C' },
  surfaceGlass: { light: '#FFFFFFEB', dark: '#131A14EB' },
  overlayScrim: { light: '#11151266', dark: '#000000B8' },
  surfaceBoard: { light: '#101512', dark: '#030604' },
  textOnBoard: { light: '#FFFFFF', dark: '#FFFFFF' },
  textBoardMuted: { light: '#DDE7D8', dark: '#DDE7D8' },
  textBoardQuiet: { light: '#A8B6A4', dark: '#A8B6A4' },
  textPrimary: { light: '#111512', dark: '#F7FAF2' },
  textSecondary: { light: '#455047', dark: '#C9D5C3' },
  textMuted: { light: '#687168', dark: '#98A590' },
  borderSubtle: { light: '#D2DBCF', dark: '#2F3A2F' },
  accentPrimary: { light: '#1E6B4F', dark: '#4FC08D' },
  accentSignal: { light: '#C9F24A', dark: '#D2FF5A' },
  accentTeal: { light: '#007F72', dark: '#58DEC8' },
  accentGold: { light: '#A86500', dark: '#F4B83F' },
  accentGoldSurface: { light: '#F0B429', dark: '#F4B83F' },
  accentTurf: { light: '#1E6B4F', dark: '#4FC08D' },
  accentAmber: { light: '#A86500', dark: '#F4B83F' },
  accentRose: { light: '#B4233F', dark: '#FF6F8A' },
  accentViolet: { light: '#5A45C8', dark: '#AA98FF' },
  brandBackground: { light: '#0B0D0C', dark: '#0B0D0C' },
  brandLensInner: { light: '#10150F', dark: '#10150F' },
  brandWhite: { light: '#F7FAF2', dark: '#F7FAF2' },
  shadowTint: { light: '#152018', dark: '#020402' },
  railBase: { light: '#D6E0D3', dark: '#2A342A' }
} as const;

export const gradients = {
  appTop: { light: '#FCFDF6', dark: '#111611' },
  appBottom: { light: '#E9EFE7', dark: '#080A08' },
  heroTop: { light: '#161B17', dark: '#151B16' },
  heroBottom: { light: '#050706', dark: '#030604' }
} as const;

export const keirinOfficialNumberColors = {
  1: { backgroundColor: '#FFFFFF', borderColor: '#D8DED4', color: '#090B09' },
  2: { backgroundColor: '#090B09', borderColor: '#090B09', color: '#FFFFFF' },
  3: { backgroundColor: '#C91528', borderColor: '#C91528', color: '#FFFFFF' },
  4: { backgroundColor: '#0C4EB8', borderColor: '#0C4EB8', color: '#FFFFFF' },
  5: { backgroundColor: '#F2D13D', borderColor: '#D2AD00', color: '#090B09' },
  6: { backgroundColor: '#007A3D', borderColor: '#007A3D', color: '#FFFFFF' },
  7: { backgroundColor: '#F3B6C8', borderColor: '#E58AA7', color: '#090B09' }
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
export type SportTone = 'keirin' | 'horse';

export function palette(mode: ThemeMode) {
  return Object.fromEntries(
    Object.entries(colors).map(([key, value]) => [key, value[mode]])
  ) as { [K in keyof typeof colors]: string };
}

export function gradient(mode: ThemeMode) {
  return {
    app: [gradients.appTop[mode], gradients.appBottom[mode]] as const,
    hero: [gradients.heroTop[mode], gradients.heroBottom[mode]] as const
  };
}

export function sportPalette(mode: ThemeMode, sport: SportTone) {
  const colors = palette(mode);
  if (sport === 'horse') {
    return {
      accent: colors.accentGold,
      accentOn: mode === 'light' ? colors.textPrimary : colors.surfaceBoard,
      secondary: colors.accentTurf,
      headerTint: colors.accentTurf,
      chipActive: colors.accentGoldSurface,
      pickHighlight: colors.accentGoldSurface,
      tabActive: colors.accentGoldSurface,
      ctaBackground: colors.accentGoldSurface,
      ctaText: mode === 'light' ? colors.textPrimary : colors.surfaceBoard
    };
  }
  return {
    accent: colors.accentSignal,
    accentOn: mode === 'light' ? colors.textPrimary : colors.surfaceBase,
    secondary: colors.accentTeal,
    headerTint: colors.accentSignal,
    chipActive: colors.accentSignal,
    pickHighlight: colors.accentSignal,
    tabActive: colors.accentSignal,
    ctaBackground: colors.accentSignal,
    ctaText: colors.textPrimary
  };
}
