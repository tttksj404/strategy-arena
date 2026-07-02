import { StyleSheet, Text, View } from 'react-native';

import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { RiskLevel } from '../types/race';

type StatusPillProps = {
  mode: ThemeMode;
  level?: RiskLevel | 'pro';
  label: string;
};

export function StatusPill({ mode, level = 'neutral', label }: StatusPillProps) {
  const colors = palette(mode);
  const accent =
    level === 'verified' ? colors.accentTeal :
    level === 'caution' ? colors.accentAmber :
    level === 'blocked' ? colors.accentRose :
    level === 'pro' ? colors.accentViolet :
    colors.textMuted;

  return (
    <View style={[styles.pill, { backgroundColor: colors.surfaceInset, borderColor: accent }]}>
      <View style={[styles.dot, { backgroundColor: accent }]} />
      <Text style={[styles.text, { color: colors.textPrimary }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    alignItems: 'center',
    alignSelf: 'flex-start',
    borderRadius: radius.pill,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space2,
    paddingHorizontal: space.space3,
    paddingVertical: space.space2
  },
  dot: {
    borderRadius: radius.pill,
    height: 7,
    width: 7
  },
  text: {
    ...typography.caption
  }
});
