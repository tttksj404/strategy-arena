import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { FadeInUp } from './FadeInUp';
import { LensCard } from './LensCard';
import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { MarketSource, ParticipantMetric, RaceDecision, RaceParticipant, Sport } from '../types/race';

type EvidenceGuideProps = {
  mode: ThemeMode;
  decision: RaceDecision;
  compact?: boolean;
};

type EvidenceItem = {
  label: string;
  value: string;
  detail: string;
  tone?: ParticipantMetric['tone'];
};

function metricValue(items: ParticipantMetric[], label: string) {
  return items.find((item) => item.label === label)?.value ?? '-';
}

function sourceCopy(source: MarketSource) {
  if (source === 'live') return '실시간 배당';
  if (source === 'historical') return '과거 배당';
  if (source === 'sample') return '샘플 배당';
  return '배당 없음';
}

function toneColor(colors: ReturnType<typeof palette>, tone: ParticipantMetric['tone']) {
  if (tone === 'teal') return colors.accentTeal;
  if (tone === 'amber') return colors.accentAmber;
  if (tone === 'rose') return colors.accentRose;
  if (tone === 'violet') return colors.accentViolet;
  if (tone === 'primary') return colors.accentPrimary;
  return colors.textPrimary;
}

function findPrimaryParticipant(decision: RaceDecision) {
  const firstNumber = Number(decision.picks[0]?.selection.split('-')[0]);
  return decision.participants.find((participant) => participant.number === firstNumber) ?? decision.participants[0];
}

function glossary(sport: Sport) {
  if (sport === 'keirin') {
    return [
      '200m: 짧을수록 순발력',
      '기어: 높을수록 무거운 가속',
      '입상률: 3착권 안정성'
    ];
  }
  return [
    '부담중량: 체력 소모 변수',
    '마체: 컨디션 변화 단서',
    '게이트: 초반 자리 변수'
  ];
}

function buildItems(decision: RaceDecision, primary?: RaceParticipant): EvidenceItem[] {
  const sport = decision.sport;
  if (!primary) {
    return [
      {
        label: '자료 상태',
        value: sourceCopy(decision.marketSource),
        detail: '출전 기본 자료가 들어오면 판단 근거가 자동으로 채워집니다.',
        tone: 'amber'
      }
    ];
  }

  if (sport === 'keirin') {
    return [
      {
        label: '중심 후보',
        value: `${primary.number}번 ${primary.name}`,
        detail: `${metricValue(primary.profile, '평균득점')}점 · 200m ${metricValue(primary.profile, '200m')} · 입상률 ${metricValue(primary.form, '입상률')}`,
        tone: 'teal'
      },
      {
        label: '전법 근거',
        value: primary.trait,
        detail: `${primary.trait} 비중 ${metricValue(primary.tactics, primary.trait)} · 최근 ${metricValue(primary.form, '최근 3주')} 흐름`,
        tone: primary.trait === '젖히기' || primary.trait === '선행' ? 'primary' : 'violet'
      },
      {
        label: '변수 확인',
        value: sourceCopy(decision.marketSource),
        detail: `${decision.meet} ${decision.raceNo}R · 배당 출처와 모델 확률을 분리해서 봅니다.`,
        tone: decision.marketSource === 'sample' ? 'amber' : 'teal'
      }
    ];
  }

  return [
    {
      label: '말 상태',
      value: `${primary.number}번 ${primary.name}`,
      detail: `${metricValue(primary.profile, '마체')} · 거리 ${metricValue(primary.profile, '거리')} · 복승률 ${metricValue(primary.form, '복승률')}`,
      tone: 'teal'
    },
    {
      label: '기수·부담',
      value: `${metricValue(primary.profile, '기수')} · ${metricValue(primary.profile, '부담중량')}`,
      detail: `최근 ${metricValue(primary.form, '최근 4전')} · 게이트 ${metricValue(primary.form, '게이트')}`,
      tone: 'primary'
    },
    {
      label: '전개 근거',
      value: primary.trait,
      detail: `${primary.trait} 비중 ${metricValue(primary.tactics, primary.trait)} · ${sourceCopy(decision.marketSource)} 출처 확인`,
      tone: 'violet'
    }
  ];
}

export function EvidenceGuide({ mode, decision, compact = false }: EvidenceGuideProps) {
  const colors = palette(mode);
  const primary = findPrimaryParticipant(decision);
  const items = buildItems(decision, primary);
  const terms = glossary(decision.sport);
  const title = decision.sport === 'keirin' ? '경륜 판단 근거' : '경마 판단 근거';
  const helper = decision.sport === 'keirin'
    ? '처음 보는 사람도 선수 기록, 전법, 배당 출처를 따로 비교할 수 있게 정리했습니다.'
    : '처음 보는 사람도 말 상태, 기수·부담중량, 전개와 배당 출처를 따로 비교할 수 있게 정리했습니다.';

  return (
    <FadeInUp>
    <LensCard mode={mode}>
      <View style={styles.header}>
        <View style={styles.titleGroup}>
          <Text style={[styles.overline, { color: colors.textMuted }]}>초심자용 판단 순서</Text>
          <Text style={[styles.title, { color: colors.textPrimary }]}>{title}</Text>
        </View>
        <View style={[styles.iconWell, { backgroundColor: colors.surfaceBoard }]}>
          <Ionicons
            accessibilityElementsHidden
            accessible={false}
            importantForAccessibility="no-hide-descendants"
            name={decision.sport === 'keirin' ? 'bicycle-outline' : 'trail-sign-outline'}
            size={21}
            color={colors.accentSignal}
          />
        </View>
      </View>
      <Text style={[styles.helper, { color: colors.textSecondary }]}>{helper}</Text>

      <View style={styles.items}>
        {items.map((item, index) => (
          <View key={`${item.label}-${item.value}`} style={[styles.item, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
            <View style={[styles.index, { backgroundColor: toneColor(colors, item.tone) }]}>
              <Text style={[styles.indexText, { color: colors.surfaceRaised }]}>{index + 1}</Text>
            </View>
            <View style={styles.itemText}>
              <Text style={[styles.itemLabel, { color: colors.textMuted }]}>{item.label}</Text>
              <Text style={[styles.itemValue, { color: colors.textPrimary }]}>{item.value}</Text>
              <Text style={[styles.itemDetail, { color: colors.textSecondary }]}>{item.detail}</Text>
            </View>
          </View>
        ))}
      </View>

      {!compact ? (
        <View style={styles.terms}>
          {terms.map((term) => (
            <View key={term} style={[styles.term, { borderColor: colors.borderSubtle }]}>
              <Text style={[styles.termText, { color: colors.textSecondary }]}>{term}</Text>
            </View>
          ))}
        </View>
      ) : null}
    </LensCard>
    </FadeInUp>
  );
}

const styles = StyleSheet.create({
  header: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  titleGroup: {
    flex: 1
  },
  overline: {
    ...typography.caption,
    marginBottom: space.space1
  },
  title: {
    ...typography.h3
  },
  iconWell: {
    alignItems: 'center',
    borderRadius: radius.medium,
    height: 44,
    justifyContent: 'center',
    width: 44
  },
  helper: {
    ...typography.bodySm,
    marginTop: space.space3
  },
  items: {
    gap: space.space3,
    marginTop: space.space4
  },
  item: {
    borderRadius: radius.medium,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space3,
    padding: space.space3
  },
  index: {
    alignItems: 'center',
    borderRadius: radius.pill,
    height: 30,
    justifyContent: 'center',
    width: 30
  },
  indexText: {
    ...typography.mono
  },
  itemText: {
    flex: 1,
    gap: space.space1
  },
  itemLabel: {
    ...typography.caption
  },
  itemValue: {
    ...typography.bodyStrong
  },
  itemDetail: {
    ...typography.bodySm
  },
  terms: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2,
    marginTop: space.space4
  },
  term: {
    borderRadius: radius.pill,
    borderWidth: 1,
    paddingHorizontal: space.space3,
    paddingVertical: space.space2
  },
  termText: {
    ...typography.caption
  }
});
