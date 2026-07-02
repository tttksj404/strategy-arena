import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { LensCard } from '../components/LensCard';
import { ParticipantBoard } from '../components/ParticipantBoard';
import { ProbabilityRail } from '../components/ProbabilityRail';
import { RaceSelector } from '../components/RaceSelector';
import { StatusPill } from '../components/StatusPill';
import { StoreSafeNotice } from '../components/StoreSafeNotice';
import { palette, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import type { RaceDecision, Sport } from '../types/race';

type HomeScreenProps = {
  mode: ThemeMode;
  decision: RaceDecision;
  sport: Sport;
  raceNo: number;
  onSportChange: (sport: Sport) => void;
  onRaceChange: (raceNo: number) => void;
  onAnalyze: () => void;
};

export function HomeScreen({
  mode,
  decision,
  sport,
  raceNo,
  onSportChange,
  onRaceChange,
  onAnalyze
}: HomeScreenProps) {
  const colors = palette(mode);
  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.hero}>
        <View style={styles.heroTop}>
          <StatusPill mode={mode} level={decision.marketRisk.level} label={decision.marketUsed ? 'LIVE MARKET' : 'MODEL ONLY'} />
          <Ionicons name="scan-outline" size={24} color={colors.accentPrimary} />
        </View>
        <Text style={[styles.title, { color: colors.textPrimary }]}>RaceLens</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          경륜·경마 경주 데이터를 렌즈처럼 분해해서 모델 신호, 표본, 리스크를 한 화면에 보여줍니다.
        </Text>
      </View>

      <LensCard mode={mode} variant={decision.marketRisk.level}>
        <View style={styles.metricHeader}>
          <View>
            <Text style={[styles.overline, { color: colors.textMuted }]}>현재 분석</Text>
            <Text style={[styles.metricTitle, { color: colors.textPrimary }]}>{decision.meet} {decision.raceNo}R</Text>
          </View>
          <Text style={[styles.metricValue, { color: colors.accentPrimary }]}>{Math.round(decision.confidence.top1 * 100)}%</Text>
        </View>
        <View style={styles.railGroup}>
          <ProbabilityRail mode={mode} label="1순위 신호" value={decision.confidence.top1} />
          <ProbabilityRail mode={mode} label="삼쌍 순서 신호" value={decision.confidence.trifecta} tone="amber" />
        </View>
      </LensCard>

      <RaceSelector
        mode={mode}
        sport={sport}
        date={decision.date}
        meet={sport === 'keirin' ? '광명' : '서울'}
        raceNo={raceNo}
        onAnalyze={onAnalyze}
        onRaceChange={onRaceChange}
        onSportChange={onSportChange}
      />

      <ParticipantBoard compact mode={mode} participants={decision.participants} sport={sport} />

      <StoreSafeNotice mode={mode} compact />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    gap: space.space5,
    padding: space.space5,
    paddingBottom: 120
  },
  hero: {
    gap: space.space3,
    paddingTop: space.space4
  },
  heroTop: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  title: {
    ...typography.display
  },
  subtitle: {
    ...typography.body
  },
  metricHeader: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  overline: {
    ...typography.caption,
    marginBottom: space.space1
  },
  metricTitle: {
    ...typography.h2
  },
  metricValue: {
    ...typography.display
  },
  railGroup: {
    gap: space.space4,
    marginTop: space.space4
  }
});
