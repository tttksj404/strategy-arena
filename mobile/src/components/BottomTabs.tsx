import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { PressableScale } from './PressableScale';
import { radius, space, sportPalette, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { Sport, TabKey } from '../types/race';

const tabs: Array<{ key: TabKey; label: string; icon: keyof typeof Ionicons.glyphMap }> = [
  { key: 'home', label: '홈', icon: 'speedometer-outline' },
  { key: 'analyze', label: '분석', icon: 'analytics-outline' },
  { key: 'pro', label: 'Pro', icon: 'diamond-outline' }
];

type BottomTabsProps = {
  mode: ThemeMode;
  active: TabKey;
  sport: Sport;
  meet?: string;
  raceNo?: number;
  onChange: (tab: TabKey) => void;
};

function accessibleTabLabel(tab: { key: TabKey; label: string }, sport: Sport, meet?: string, raceNo?: number) {
  // Screen readers get race context on the analyze tab (e.g. "경륜 광명 1R 분석 보기");
  // the visible label under the icon stays the short "분석" text.
  if (tab.key !== 'analyze' || !meet || !raceNo) return tab.label;
  const sportLabel = sport === 'keirin' ? '경륜' : '경마';
  return `${sportLabel} ${meet} ${raceNo}R 분석 보기`;
}

export function BottomTabs({ mode, active, sport, meet, raceNo, onChange }: BottomTabsProps) {
  const colors = palette(mode);
  const sportColors = sportPalette(mode, sport);
  return (
    <View style={[styles.shell, { backgroundColor: colors.surfaceGlass, borderColor: colors.borderSubtle }]}>
      {tabs.map((tab) => {
        const selected = tab.key === active;
        const color = selected ? sportColors.accentOn : colors.textMuted;
        return (
          <PressableScale
            accessibilityLabel={accessibleTabLabel(tab, sport, meet, raceNo)}
            accessibilityRole="tab"
            accessibilityState={{ selected }}
            key={tab.key}
            onPress={() => onChange(tab.key)}
            style={[styles.tab, selected && { backgroundColor: sportColors.tabActive }]}
          >
            <Ionicons
              accessibilityElementsHidden
              accessible={false}
              importantForAccessibility="no-hide-descendants"
              name={tab.icon}
              size={20}
              color={color}
            />
            <Text style={[styles.label, { color }]}>{tab.label}</Text>
          </PressableScale>
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
