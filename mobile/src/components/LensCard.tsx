import type { ReactNode } from 'react';
import { useRef } from 'react';
import { Animated, Pressable, StyleSheet, View } from 'react-native';

import { radius, space } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { RiskLevel } from '../types/race';

type LensCardProps = {
  children: ReactNode;
  mode: ThemeMode;
  variant?: RiskLevel | 'pro' | 'base';
  onPress?: () => void;
  padded?: boolean;
};

export function LensCard({ children, mode, variant = 'base', onPress, padded = true }: LensCardProps) {
  const colors = palette(mode);
  const borderColor =
    variant === 'verified' ? colors.accentTeal :
    variant === 'caution' ? colors.accentAmber :
    variant === 'blocked' ? colors.accentRose :
    variant === 'pro' ? colors.accentViolet :
    colors.borderSubtle;
  const backgroundColor = variant === 'base' ? colors.surfaceRaised : colors.surfaceGlass;
  const scale = useRef(new Animated.Value(1)).current;
  const content = (
    <Animated.View
      style={[
        styles.card,
        {
          backgroundColor,
          borderColor,
          padding: padded ? space.space4 : 0,
          shadowColor: colors.shadowTint,
          transform: [{ scale }]
        }
      ]}
    >
      {children}
    </Animated.View>
  );

  if (!onPress) return content;

  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      onPressIn={() => Animated.timing(scale, { toValue: 0.985, duration: 120, useNativeDriver: true }).start()}
      onPressOut={() => Animated.timing(scale, { toValue: 1, duration: 180, useNativeDriver: true }).start()}
    >
      {content}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: radius.medium,
    borderWidth: 1,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.12,
    shadowRadius: 24,
    overflow: 'hidden'
  }
});
