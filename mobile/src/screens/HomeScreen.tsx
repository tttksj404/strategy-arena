import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

import { BrandMark } from '../components/BrandMark';
import { RaceSelector } from '../components/RaceSelector';
import { StatusPill } from '../components/StatusPill';
import { StoreSafeNotice } from '../components/StoreSafeNotice';
import { gradient, palette, radius, space, sportPalette, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import type { RaceDecision, Sport } from '../types/race';

type HomeScreenProps = {
  mode: ThemeMode;
  decision: RaceDecision;
  freeAnalysisLimit: number;
  freeAnalysisUsed: number;
  rewardedAnalysisCredits: number;
  rewardAdsEnabled: boolean;
  sport: Sport;
  raceCount: number;
  raceNo: number;
  raceDates: string[];
  onSportChange: (sport: Sport) => void;
  onDateChange: (date: string) => void;
  onMeetChange: (meet: string) => void;
  onRaceChange: (raceNo: number) => void;
  onAnalyze: () => void;
};

export function HomeScreen({
  mode,
  decision,
  freeAnalysisLimit,
  freeAnalysisUsed,
  rewardedAnalysisCredits,
  rewardAdsEnabled,
  sport,
  raceCount,
  raceNo,
  raceDates,
  onSportChange,
  onDateChange,
  onMeetChange,
  onRaceChange,
  onAnalyze
}: HomeScreenProps) {
  const colors = palette(mode);
  const gradients = gradient(mode);
  const sportColors = sportPalette(mode, sport);
  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false} style={styles.scroll}>
      <LinearGradient colors={gradients.hero} style={[styles.heroPanel, { borderColor: colors.borderSubtle }]}>
        <View style={[styles.scanBand, { backgroundColor: sportColors.headerTint }]} />
        <View style={styles.heroTop}>
          <StatusPill mode={mode} level="neutral" label="조건 선택" />
        </View>
        <View style={styles.brandLockup}>
          <BrandMark mode="dark" size={28} />
          <Text style={[styles.title, { color: colors.textOnBoard }]}>RaceLens</Text>
        </View>
        <Text style={[styles.subtitle, { color: colors.textBoardMuted }]}>
          {sport === 'keirin'
            ? '경륜장과 경주 번호를 고른 뒤 1착·2착·3착 예측 순서를 확인합니다.'
            : '경마장과 경주 번호를 고른 뒤 1착·2착·3착 예측 순서를 확인합니다.'}
        </Text>
      </LinearGradient>

      <RaceSelector
        mode={mode}
        sport={sport}
        date={decision.date}
        freeAnalysisLimit={freeAnalysisLimit}
        freeAnalysisUsed={freeAnalysisUsed}
        meet={decision.meet}
        proActive={decision.appSession.entitlement === 'pro'}
        raceCount={raceCount}
        raceNo={raceNo}
        raceDates={raceDates}
        rewardedAnalysisCredits={rewardedAnalysisCredits}
        rewardAdsEnabled={rewardAdsEnabled}
        onAnalyze={onAnalyze}
        onDateChange={onDateChange}
        onMeetChange={onMeetChange}
        onRaceChange={onRaceChange}
        onSportChange={onSportChange}
      />

      {decision.dataLayer.error ? (
        <View style={[styles.sessionNotice, { backgroundColor: colors.surfaceRaised, borderColor: colors.accentAmber }]}>
          <Text style={[styles.sessionNoticeTitle, { color: colors.textPrimary }]}>세션 정보를 확인하지 못했습니다</Text>
          <Text style={[styles.sessionNoticeCopy, { color: colors.textSecondary }]}>
            무료 기능으로 시작합니다. 네트워크가 회복되면 사용 가능 횟수와 데이터 상태를 다시 확인합니다.
          </Text>
        </View>
      ) : null}

      <StoreSafeNotice mode={mode} compact />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flex: 1
  },
  content: {
    gap: space.space3,
    padding: space.space5,
    paddingBottom: 120
  },
  heroPanel: {
    borderRadius: radius.large,
    borderWidth: 1,
    gap: space.space3,
    overflow: 'hidden',
    padding: space.space4
  },
  scanBand: {
    height: 3,
    left: 0,
    opacity: 0.9,
    position: 'absolute',
    right: 0,
    top: 0
  },
  heroTop: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  brandLockup: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space2
  },
  title: {
    ...typography.display,
    fontWeight: '800',
    letterSpacing: 0.4
  },
  subtitle: {
    ...typography.body,
    maxWidth: 310
  },
  sessionNotice: {
    borderRadius: radius.medium,
    borderWidth: 1,
    gap: space.space1,
    padding: space.space4
  },
  sessionNoticeTitle: {
    ...typography.bodyStrong
  },
  sessionNoticeCopy: {
    ...typography.bodySm
  },
});
