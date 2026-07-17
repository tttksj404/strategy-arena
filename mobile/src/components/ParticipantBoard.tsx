import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { LensCard } from './LensCard';
import { NumberBadge } from './NumberBadge';
import { PressableScale } from './PressableScale';
import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { ParticipantMetric, RaceParticipant, Sport } from '../types/race';

type ParticipantBoardProps = {
  mode: ThemeMode;
  sport: Sport;
  participants: RaceParticipant[];
  compact?: boolean;
  proActive?: boolean;
};

function signalColor(colors: ReturnType<typeof palette>, signal: RaceParticipant['signal']) {
  if (signal === 'teal') return colors.accentTeal;
  if (signal === 'amber') return colors.accentAmber;
  if (signal === 'rose') return colors.accentRose;
  if (signal === 'violet') return colors.accentViolet;
  return colors.accentPrimary;
}

function metricColor(colors: ReturnType<typeof palette>, tone: ParticipantMetric['tone']) {
  if (tone === 'teal') return colors.accentTeal;
  if (tone === 'amber') return colors.accentAmber;
  if (tone === 'rose') return colors.accentRose;
  if (tone === 'violet') return colors.accentViolet;
  if (tone === 'primary') return colors.accentPrimary;
  return colors.textPrimary;
}

function findMetric(participant: RaceParticipant, label: string) {
  return [...participant.profile, ...participant.form, ...participant.tactics].find((item) => item.label === label);
}

function coreMetrics(sport: Sport, participant: RaceParticipant) {
  const labels = sport === 'horse'
    ? ['복승률', '게이트', '부담중량']
    : ['평균득점', '200m', '입상률'];
  return labels.map((label): ParticipantMetric => {
    const metric = findMetric(participant, label);
    return {
      label: label === '평균득점' ? '득점' : label,
      value: metric?.value ?? '-',
      tone: metric?.tone
    };
  });
}

function MetricGrid({
  colors,
  items,
  compact
}: {
  colors: ReturnType<typeof palette>;
  items: readonly ParticipantMetric[];
  compact: boolean;
}) {
  const visible = compact ? items.slice(0, 6) : items;
  return (
    <View style={styles.metricGrid}>
      {visible.map((item) => (
        <View key={`${item.label}-${item.value}`} style={[styles.metricCell, { borderColor: colors.borderSubtle }]}>
          <Text style={[styles.metricLabel, { color: colors.textMuted }]}>{item.label}</Text>
          <Text adjustsFontSizeToFit numberOfLines={1} style={[styles.metricValue, { color: metricColor(colors, item.tone) }]}>
            {item.value}
          </Text>
        </View>
      ))}
    </View>
  );
}

function AlgorithmReasons({
  colors,
  participant,
  proActive
}: {
  colors: ReturnType<typeof palette>;
  participant: RaceParticipant;
  proActive: boolean;
}) {
  if (participant.algorithmReasons?.length && proActive) {
    return (
      <View style={[styles.algorithmBox, { backgroundColor: colors.surfaceRaised, borderColor: colors.borderSubtle }]}>
        <Text style={[styles.blockTitle, { color: colors.textMuted }]}>Pro 알고리즘 판단</Text>
        {participant.algorithmReasons.map((reason) => (
          <View key={`${participant.number}-${reason.label}-${reason.value}`} style={styles.reasonRow}>
            <Text style={[styles.reasonLabel, { color: metricColor(colors, reason.tone) }]}>{reason.label}</Text>
            <Text style={[styles.reasonValue, { color: colors.textSecondary }]}>{reason.value}</Text>
          </View>
        ))}
      </View>
    );
  }
  if (participant.algorithmLocked) {
    return (
      <View style={[styles.lockBox, { backgroundColor: colors.surfaceRaised, borderColor: colors.borderSubtle }]}>
        <Text style={[styles.lockText, { color: colors.textSecondary }]}>
          Pro에서 모델확률·배당반영·누적학습 보정 근거를 선수별로 엽니다.
        </Text>
      </View>
    );
  }
  return null;
}

export function ParticipantBoard({ mode, sport, participants, compact = false, proActive = false }: ParticipantBoardProps) {
  const colors = palette(mode);
  const [expandedNumbers, setExpandedNumbers] = useState<readonly number[]>([]);
  const subject = sport === 'horse' ? '출전마' : '출전 선수';
  const countUnit = sport === 'horse' ? '두' : '명';
  const displayParticipants = [...participants].sort((left, right) => left.number - right.number);
  const helper = sport === 'horse'
    ? '기본 카드에는 복승률·게이트·부담중량만 두고, 펼치면 마체와 주행 근거를 봅니다.'
    : '기본 카드에는 득점·200m·입상률만 두고, 펼치면 전법과 모델 근거를 봅니다.';

  function toggleExpanded(number: number) {
    setExpandedNumbers((current) => current.includes(number)
      ? current.filter((item) => item !== number)
      : [...current, number]);
  }

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

      <View style={styles.list} testID="participant-board">
        {participants.length === 0 ? (
          <View style={[styles.empty, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
            <Text style={[styles.emptyText, { color: colors.textMuted }]}>출전 정보 대기 중</Text>
          </View>
        ) : null}
        {displayParticipants.map((participant) => {
          const accent = signalColor(colors, participant.signal);
          const expanded = expandedNumbers.includes(participant.number);
          return (
            <PressableScale
              accessibilityLabel={`${participant.number}번 ${participant.name} ${expanded ? '상세 닫기' : '상세 열기'}`}
              accessibilityRole="button"
              accessibilityState={{ expanded }}
              key={`${participant.number}-${participant.name}`}
              onPress={() => toggleExpanded(participant.number)}
              style={[
                styles.row,
                {
                  backgroundColor: colors.surfaceInset,
                  borderColor: expanded ? accent : colors.borderSubtle
                }
              ]}
            >
              <NumberBadge mode={mode} number={participant.number} sport={sport} />
              <View style={styles.main}>
                <View style={styles.nameRow}>
                  <View style={styles.nameBlock}>
                    <Text style={[styles.name, { color: colors.textPrimary }]} numberOfLines={2}>{participant.name}</Text>
                  </View>
                  <View style={styles.traitBlock}>
                    <Ionicons
                      accessibilityElementsHidden
                      accessible={false}
                      importantForAccessibility="no-hide-descendants"
                      name={expanded ? 'chevron-up' : 'chevron-down'}
                      size={16}
                      color={colors.textMuted}
                    />
                  </View>
                </View>
                <MetricGrid colors={colors} compact={false} items={coreMetrics(sport, participant)} />
                {expanded ? (
                  <>
                    <Text style={[styles.trait, { color: accent }]}>{participant.trait}</Text>
                    <Text style={[styles.meta, { color: colors.textSecondary }]}>{participant.subtitle}</Text>
                    <Text style={[styles.meta, { color: colors.textSecondary }]}>{participant.stats}</Text>
                    <View style={styles.detailBlock}>
                      <Text style={[styles.blockTitle, { color: colors.textMuted }]}>프로필</Text>
                      <MetricGrid colors={colors} compact={false} items={participant.profile} />
                    </View>
                    <View style={styles.detailBlock}>
                      <Text style={[styles.blockTitle, { color: colors.textMuted }]}>
                        {sport === 'horse' ? '최근 흐름' : '입상 흐름'}
                      </Text>
                      <MetricGrid colors={colors} compact={false} items={participant.form} />
                    </View>
                    <View style={styles.detailBlock}>
                      <Text style={[styles.blockTitle, { color: colors.textMuted }]}>
                        {sport === 'horse' ? '주행 성향' : '입상전법'}
                      </Text>
                      <MetricGrid colors={colors} compact={false} items={participant.tactics} />
                    </View>
                    <Text style={[styles.note, { color: colors.textMuted }]}>{participant.note}</Text>
                    <AlgorithmReasons colors={colors} participant={participant} proActive={proActive} />
                  </>
                ) : null}
              </View>
            </PressableScale>
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
    minHeight: 64,
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
  main: {
    flex: 1,
    gap: space.space2,
    minWidth: 0
  },
  nameRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space2,
    justifyContent: 'space-between',
    minHeight: 44
  },
  nameBlock: {
    flex: 1,
    minWidth: 0
  },
  name: {
    ...typography.bodyStrong,
    flexShrink: 1
  },
  traitBlock: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space1
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
  },
  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2,
    marginTop: space.space1
  },
  metricCell: {
    borderRadius: radius.small,
    borderWidth: 1,
    minHeight: 46,
    minWidth: 72,
    paddingHorizontal: space.space2,
    paddingVertical: space.space2
  },
  metricLabel: {
    ...typography.caption,
    marginBottom: space.space1
  },
  metricValue: {
    ...typography.bodyStrong,
    maxWidth: 104
  },
  detailBlock: {
    gap: space.space2,
    marginTop: space.space2
  },
  blockTitle: {
    ...typography.caption
  },
  algorithmBox: {
    borderRadius: radius.small,
    borderWidth: 1,
    gap: space.space2,
    marginTop: space.space2,
    padding: space.space3
  },
  reasonRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: space.space2,
    justifyContent: 'space-between'
  },
  reasonLabel: {
    ...typography.caption,
    flexShrink: 0
  },
  reasonValue: {
    ...typography.bodySm,
    flex: 1,
    textAlign: 'right'
  },
  lockBox: {
    borderRadius: radius.small,
    borderWidth: 1,
    marginTop: space.space2,
    padding: space.space3
  },
  lockText: {
    ...typography.bodySm
  }
});
