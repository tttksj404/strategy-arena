import { StyleSheet, Text, View } from 'react-native';

import { numberBadgePalette } from '../theme/numberColors';
import { radius, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import type { Sport } from '../types/race';

type NumberBadgeProps = {
  readonly mode: ThemeMode;
  readonly number: number;
  readonly sport: Sport;
  readonly size?: 'small' | 'medium' | 'large';
};

export function NumberBadge({ mode, number, sport, size = 'medium' }: NumberBadgeProps) {
  const colors = numberBadgePalette(mode, sport, number);
  return (
    <View
      accessibilityLabel={`${number}번`}
      style={[
        styles.badge,
        size === 'small' ? styles.small : size === 'large' ? styles.large : styles.medium,
        {
          backgroundColor: colors.backgroundColor,
          borderColor: colors.borderColor
        }
      ]}
    >
      <Text style={[styles.text, size === 'large' && styles.largeText, { color: colors.color }]}>{number}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignItems: 'center',
    borderRadius: radius.pill,
    borderWidth: 1.5,
    justifyContent: 'center'
  },
  small: {
    height: 28,
    width: 28
  },
  medium: {
    height: 36,
    width: 36
  },
  large: {
    height: 52,
    width: 52
  },
  text: {
    ...typography.mono
  },
  largeText: {
    fontSize: 17,
    lineHeight: 22
  }
});
