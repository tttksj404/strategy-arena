import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { LensCard } from '../components/LensCard';
import { MarketOddsBoard } from '../components/MarketOddsBoard';
import { ParticipantBoard } from '../components/ParticipantBoard';
import { ProbabilityRail } from '../components/ProbabilityRail';
import { StatusPill } from '../components/StatusPill';
import { StoreSafeNotice } from '../components/StoreSafeNotice';
import { palette, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import type { RaceDecision } from '../types/race';

type AnalyzeScreenProps = {
  mode: ThemeMode;
  decision: RaceDecision;
};

export function AnalyzeScreen({ mode, decision }: AnalyzeScreenProps) {
  const colors = palette(mode);
  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <Text style={[styles.title, { color: colors.textPrimary }]}>분석 상세</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>{decision.headline}</Text>
      </View>

      <LensCard mode={mode} variant={decision.marketRisk.level}>
        <View style={styles.riskTop}>
          <StatusPill mode={mode} level={decision.marketRisk.level} label={decision.marketRisk.title} />
          <Text style={[styles.sample, { color: colors.textMuted }]}>표본 {decision.confidence.sample.toLocaleString('ko-KR')}</Text>
        </View>
        <Text style={[styles.riskMessage, { color: colors.textSecondary }]}>{decision.marketRisk.message}</Text>
      </LensCard>

      <MarketOddsBoard marketUsed={decision.marketUsed} mode={mode} odds={decision.marketOdds} />

      <ParticipantBoard mode={mode} participants={decision.participants} sport={decision.sport} />

      <View style={styles.pickList}>
        {decision.picks.map((pick) => (
          <LensCard key={pick.code} mode={mode}>
            <View style={styles.pickHeader}>
              <View>
                <Text style={[styles.pickCode, { color: colors.textMuted }]}>{pick.code}</Text>
                <Text style={[styles.pickTitle, { color: colors.textPrimary }]}>{pick.label}</Text>
              </View>
              <Text style={[styles.selection, { color: colors.textPrimary }]}>{pick.selection}</Text>
            </View>
            <ProbabilityRail
              mode={mode}
              label={`등급 ${pick.grade}`}
              value={pick.probability}
              tone={pick.grade === '강' ? 'teal' : pick.grade === '중' ? 'primary' : 'amber'}
            />
          </LensCard>
        ))}
      </View>

      <StoreSafeNotice mode={mode} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    gap: space.space5,
    padding: space.space5,
    paddingBottom: 120
  },
  header: {
    gap: space.space2,
    paddingTop: space.space4
  },
  title: {
    ...typography.h1
  },
  subtitle: {
    ...typography.body
  },
  riskTop: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  sample: {
    ...typography.mono
  },
  riskMessage: {
    ...typography.bodySm,
    marginTop: space.space3
  },
  pickList: {
    gap: space.space3
  },
  pickHeader: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: space.space4
  },
  pickCode: {
    ...typography.caption,
    marginBottom: space.space1
  },
  pickTitle: {
    ...typography.h3
  },
  selection: {
    ...typography.h2
  }
});
