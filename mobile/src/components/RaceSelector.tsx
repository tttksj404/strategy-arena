import { Pressable, StyleSheet, Text, View } from 'react-native';

import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { Sport } from '../types/race';

type RaceSelectorProps = {
  mode: ThemeMode;
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
  onSportChange: (sport: Sport) => void;
  onRaceChange: (raceNo: number) => void;
  onAnalyze: () => void;
};

const races = [1, 2, 3, 4, 5, 6, 7, 8];

export function RaceSelector({
  mode,
  sport,
  date,
  meet,
  raceNo,
  onSportChange,
  onRaceChange,
  onAnalyze
}: RaceSelectorProps) {
  const colors = palette(mode);
  return (
    <View style={[styles.wrap, { backgroundColor: colors.surfaceRaised, borderColor: colors.borderSubtle }]}>
      <View style={styles.segment}>
        {(['keirin', 'horse'] as const).map((item) => {
          const selected = item === sport;
          return (
            <Pressable
              aria-selected={selected}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              key={item}
              onPress={() => onSportChange(item)}
              style={[styles.segmentItem, selected && { backgroundColor: colors.textPrimary }]}
            >
              <Text style={[styles.segmentText, { color: selected ? colors.surfaceRaised : colors.textSecondary }]}>
                {item === 'keirin' ? '경륜' : '경마'}
              </Text>
            </Pressable>
          );
        })}
      </View>

      <View style={styles.metaRow}>
        <View>
          <Text style={[styles.caption, { color: colors.textMuted }]}>분석일</Text>
          <Text style={[styles.value, { color: colors.textPrimary }]}>{date}</Text>
        </View>
        <View>
          <Text style={[styles.caption, { color: colors.textMuted }]}>경주장</Text>
          <Text style={[styles.value, { color: colors.textPrimary }]}>{meet}</Text>
        </View>
      </View>

      <View style={styles.raceGrid}>
        {races.map((race) => {
          const selected = raceNo === race;
          return (
            <Pressable
              aria-selected={selected}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              key={race}
              onPress={() => onRaceChange(race)}
              style={[styles.raceChip, { backgroundColor: selected ? colors.accentPrimary : colors.surfaceInset }]}
            >
              <Text style={[styles.raceText, { color: selected ? colors.surfaceRaised : colors.textPrimary }]}>
                {race}R
              </Text>
            </Pressable>
          );
        })}
      </View>

      <Pressable accessibilityRole="button" onPress={onAnalyze} style={[styles.cta, { backgroundColor: colors.accentPrimary }]}>
        <Text style={[styles.ctaText, { color: colors.surfaceRaised }]}>모델 신호 보기</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    borderRadius: radius.large,
    borderWidth: 1,
    gap: space.space4,
    padding: space.space4
  },
  segment: {
    backgroundColor: 'transparent',
    flexDirection: 'row',
    gap: space.space2
  },
  segmentItem: {
    alignItems: 'center',
    borderRadius: radius.pill,
    flex: 1,
    paddingVertical: space.space3
  },
  segmentText: {
    ...typography.bodyStrong
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  caption: {
    ...typography.caption,
    marginBottom: space.space1
  },
  value: {
    ...typography.bodyStrong
  },
  raceGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2
  },
  raceChip: {
    alignItems: 'center',
    borderRadius: radius.pill,
    minWidth: 58,
    paddingHorizontal: space.space3,
    paddingVertical: space.space2
  },
  raceText: {
    ...typography.mono
  },
  cta: {
    alignItems: 'center',
    borderRadius: radius.pill,
    paddingVertical: space.space4
  },
  ctaText: {
    ...typography.bodyStrong
  }
});
