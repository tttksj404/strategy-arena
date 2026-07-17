import { useEffect, useRef } from 'react';
import { Animated, Easing, StyleSheet, Text, View } from 'react-native';

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
    colors.accentSignal;
  const normalized = Math.max(0, Math.min(1, value));
  const percent = `${Math.round(normalized * 100)}%`;
  const fillPct = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.timing(fillPct, {
      duration: 220,
      easing: Easing.out(Easing.cubic),
      toValue: normalized * 100,
      // width is a layout property and cannot run on the native driver.
      useNativeDriver: false
    });
    animation.start();
    return () => animation.stop();
  }, [fillPct, normalized]);

  const animatedWidth = fillPct.interpolate({
    extrapolate: 'clamp',
    inputRange: [0, 100],
    outputRange: ['0%', '100%']
  });

  return (
    <View accessibilityLabel={`${label} ${percent}`} style={styles.wrap}>
      <View style={styles.row}>
        <Text style={[styles.label, { color: colors.textSecondary }]}>{label}</Text>
        <Text style={[styles.value, { color: colors.textPrimary }]}>{percent}</Text>
      </View>
      <View style={[styles.track, { backgroundColor: colors.railBase }]}>
        <Animated.View style={[styles.fill, { backgroundColor: fill, width: animatedWidth }]} />
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
