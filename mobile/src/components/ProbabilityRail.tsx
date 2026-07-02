import { StyleSheet, Text, View } from 'react-native';
import type { DimensionValue } from 'react-native';

import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';

type ProbabilityRailProps = {
  mode: ThemeMode;
  label: string;
  value: number;
  tone?: 'primary' | 'teal' | 'amber' | 'rose' | 'violet';
};

export function ProbabilityRail({ mode, label, value, tone = 'primary' }: ProbabilityRailProps) {
  const colors = palette(mode);
  const fill =
    tone === 'teal' ? colors.accentTeal :
    tone === 'amber' ? colors.accentAmber :
    tone === 'rose' ? colors.accentRose :
    tone === 'violet' ? colors.accentViolet :
    colors.accentPrimary;
  const normalized = Math.max(0, Math.min(1, value));
  const percent = `${Math.round(normalized * 100)}%`;
  const fillWidth = percent as DimensionValue;

  return (
    <View accessibilityLabel={`${label} ${percent}`} style={styles.wrap}>
      <View style={styles.row}>
        <Text style={[styles.label, { color: colors.textSecondary }]}>{label}</Text>
        <Text style={[styles.value, { color: colors.textPrimary }]}>{percent}</Text>
      </View>
      <View style={[styles.track, { backgroundColor: colors.railBase }]}>
        <View style={[styles.fill, { backgroundColor: fill, width: fillWidth }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: space.space2
  },
  row: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  label: {
    ...typography.bodySm
  },
  value: {
    ...typography.mono
  },
  track: {
    borderRadius: radius.pill,
    height: 9,
    overflow: 'hidden'
  },
  fill: {
    borderRadius: radius.pill,
    height: 9
  }
});
