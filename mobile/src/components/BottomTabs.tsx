import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { TabKey } from '../types/race';

const tabs: Array<{ key: TabKey; label: string; icon: keyof typeof Ionicons.glyphMap }> = [
  { key: 'home', label: '홈', icon: 'speedometer-outline' },
  { key: 'analyze', label: '분석', icon: 'analytics-outline' },
  { key: 'lab', label: '랩', icon: 'flask-outline' },
  { key: 'pro', label: 'Pro', icon: 'diamond-outline' }
];

type BottomTabsProps = {
  mode: ThemeMode;
  active: TabKey;
  onChange: (tab: TabKey) => void;
};

export function BottomTabs({ mode, active, onChange }: BottomTabsProps) {
  const colors = palette(mode);
  return (
    <View style={[styles.shell, { backgroundColor: colors.surfaceGlass, borderColor: colors.borderSubtle }]}>
      {tabs.map((tab) => {
        const selected = tab.key === active;
        const color = selected ? colors.accentPrimary : colors.textMuted;
        return (
          <Pressable
            accessibilityRole="tab"
            accessibilityState={{ selected }}
            key={tab.key}
            onPress={() => onChange(tab.key)}
            style={[styles.tab, selected && { backgroundColor: colors.surfaceInset }]}
          >
            <Ionicons name={tab.icon} size={20} color={color} />
            <Text style={[styles.label, { color }]}>{tab.label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  shell: {
    borderRadius: radius.large,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space1,
    margin: space.space5,
    padding: space.space1
  },
  tab: {
    alignItems: 'center',
    borderRadius: radius.medium,
    flex: 1,
    gap: space.space1,
    paddingVertical: space.space2
  },
  label: {
    ...typography.caption
  }
});
