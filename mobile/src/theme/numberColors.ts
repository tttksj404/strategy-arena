import { keirinOfficialNumberColors, palette } from './tokens';
import type { SportTone, ThemeMode } from './tokens';

export type NumberBadgePalette = {
  readonly backgroundColor: string;
  readonly borderColor: string;
  readonly color: string;
};

export function numberBadgePalette(mode: ThemeMode, sport: SportTone, number: number): NumberBadgePalette {
  const colors = palette(mode);
  if (sport === 'horse') {
    return {
      backgroundColor: colors.surfaceBoard,
      borderColor: colors.accentGold,
      color: colors.textOnBoard
    };
  }
  return keirinOfficialNumberColors[number as keyof typeof keirinOfficialNumberColors] ?? {
    backgroundColor: colors.surfaceInset,
    borderColor: colors.borderSubtle,
    color: colors.textPrimary
  };
}
