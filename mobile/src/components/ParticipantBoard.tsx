import { StyleSheet, Text, View } from 'react-native';

import { LensCard } from './LensCard';
import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { RaceParticipant, Sport } from '../types/race';

type ParticipantBoardProps = {
  mode: ThemeMode;
  sport: Sport;
  participants: RaceParticipant[];
  compact?: boolean;
};

function signalColor(colors: ReturnType<typeof palette>, signal: RaceParticipant['signal']) {
  if (signal === 'teal') return colors.accentTeal;
  if (signal === 'amber') return colors.accentAmber;
  if (signal === 'rose') return colors.accentRose;
  if (signal === 'violet') return colors.accentViolet;
  return colors.accentPrimary;
}

export function ParticipantBoard({ mode, sport, participants, compact = false }: ParticipantBoardProps) {
  const colors = palette(mode);
  const subject = sport === 'horse' ? '출전마' : '출전 선수';
  const countUnit = sport === 'horse' ? '두' : '명';
  const helper = sport === 'horse'
    ? '말명, 기수, 부담중량, 최근 흐름을 먼저 확인할 수 있게 정리했습니다.'
    : '선수명, 등급, 최근 흐름, 전개 성향을 먼저 확인할 수 있게 정리했습니다.';

  return (
    <LensCard mode={mode}>
      <View style={styles.header}>
        <View>
          <Text style={[styles.overline, { color: colors.textMuted }]}>기본 제공 자료</Text>
          <Text style={[styles.title, { color: colors.textPrimary }]}>{subject}</Text>
        </View>
        <Text style={[styles.count, { color: colors.textMuted }]}>{participants.length}{countUnit}</Text>
      </View>
      <Text style={[styles.helper, { color: colors.textSecondary }]}>{helper}</Text>

      <View style={styles.list}>
        {participants.length === 0 ? (
          <View style={[styles.empty, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
            <Text style={[styles.emptyText, { color: colors.textMuted }]}>출전 정보 대기 중</Text>
          </View>
        ) : null}
        {participants.map((participant) => {
          const accent = signalColor(colors, participant.signal);
          return (
            <View
              accessibilityLabel={`${participant.number}번 ${participant.name} ${participant.subtitle} ${participant.stats}`}
              accessible
              key={`${participant.number}-${participant.name}`}
              style={[styles.row, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}
            >
              <View style={[styles.number, { backgroundColor: accent }]}>
                <Text style={[styles.numberText, { color: colors.surfaceRaised }]}>{participant.number}</Text>
              </View>
              <View style={styles.main}>
                <View style={styles.nameRow}>
                  <Text style={[styles.name, { color: colors.textPrimary }]}>{participant.name}</Text>
                  <Text style={[styles.trait, { color: accent }]}>{participant.trait}</Text>
                </View>
                <Text style={[styles.meta, { color: colors.textSecondary }]}>{participant.subtitle}</Text>
                <Text style={[styles.meta, { color: colors.textSecondary }]}>{participant.stats}</Text>
                {!compact ? (
                  <Text style={[styles.note, { color: colors.textMuted }]}>{participant.note}</Text>
                ) : null}
              </View>
            </View>
          );
        })}
      </View>
    </LensCard>
  );
}

const styles = StyleSheet.create({
  header: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  overline: {
    ...typography.caption,
    marginBottom: space.space1
  },
  title: {
    ...typography.h3
  },
  count: {
    ...typography.mono
  },
  helper: {
    ...typography.bodySm,
    marginTop: space.space3
  },
  list: {
    gap: space.space3,
    marginTop: space.space4
  },
  row: {
    borderRadius: radius.medium,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space3,
    padding: space.space3
  },
  empty: {
    borderRadius: radius.medium,
    borderWidth: 1,
    padding: space.space4
  },
  emptyText: {
    ...typography.bodySm
  },
  number: {
    alignItems: 'center',
    borderRadius: radius.pill,
    height: 34,
    justifyContent: 'center',
    width: 34
  },
  numberText: {
    ...typography.mono
  },
  main: {
    flex: 1,
    gap: space.space1
  },
  nameRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space2,
    justifyContent: 'space-between'
  },
  name: {
    ...typography.bodyStrong,
    flexShrink: 1
  },
  trait: {
    ...typography.caption,
    flexShrink: 0
  },
  meta: {
    ...typography.bodySm
  },
  note: {
    ...typography.bodySm,
    marginTop: space.space1
  }
});
