import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { PressableScale } from './PressableScale';
import { formatRaceDateLabel, raceVenues } from '../services/raceSchedule';
import { radius, space, sportPalette, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { Sport } from '../types/race';

type RaceSelectorProps = {
  mode: ThemeMode;
  sport: Sport;
  date: string;
  freeAnalysisLimit: number;
  freeAnalysisUsed: number;
  meet: string;
  proActive: boolean;
  raceCount: number;
  raceNo: number;
  raceDates: string[];
  rewardedAnalysisCredits: number;
  rewardAdsEnabled: boolean;
  onSportChange: (sport: Sport) => void;
  onDateChange: (date: string) => void;
  onMeetChange: (meet: string) => void;
  onRaceChange: (raceNo: number) => void;
  onAnalyze: () => void;
};

export function RaceSelector({
  mode,
  sport,
  date,
  freeAnalysisLimit,
  freeAnalysisUsed,
  meet,
  proActive,
  raceCount,
  raceNo,
  raceDates,
  rewardedAnalysisCredits,
  rewardAdsEnabled,
  onSportChange,
  onDateChange,
  onMeetChange,
  onRaceChange,
  onAnalyze
}: RaceSelectorProps) {
  const colors = palette(mode);
  const sportColors = sportPalette(mode, sport);
  const selectedText = sportColors.accentOn;
  const freeRemaining = Math.max(0, freeAnalysisLimit - freeAnalysisUsed);
  const rewardReady = rewardAdsEnabled && !proActive && freeRemaining === 0 && rewardedAnalysisCredits > 0;
  const needsReward = rewardAdsEnabled && !proActive && freeRemaining === 0 && rewardedAnalysisCredits === 0;
  const needsProGuidance = !rewardAdsEnabled && !proActive && freeRemaining === 0;
  const ctaBackground = needsReward ? colors.accentGold : sportColors.ctaBackground;
  // The reward CTA uses the darker accentGold fill (light) / bright accentGold (dark),
  // so it needs its own foreground: white ink on the dark light-mode fill, dark ink on
  // the bright dark-mode fill. The normal CTA keeps the sport ctaText (dark on bright gold).
  const ctaForeground = needsReward
    ? (mode === 'dark' ? colors.surfaceBoard : colors.textOnBoard)
    : sportColors.ctaText;
  const dates = raceDates;
  const scheduleUnavailable = dates.length === 0;
  const races = Array.from({ length: Math.max(1, Math.min(24, raceCount)) }, (_, index) => index + 1);
  const dateIndex = dates.indexOf(date);
  const previousDisabled = dateIndex <= 0;
  const nextDisabled = dateIndex === -1 || dateIndex >= dates.length - 1;
  return (
    <View style={[styles.wrap, { backgroundColor: colors.surfaceRaised, borderColor: colors.borderSubtle }]}>
      <View style={styles.segment}>
        {(['keirin', 'horse'] as const).map((item) => {
          const selected = item === sport;
          return (
            <PressableScale
              aria-selected={selected}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              key={item}
              onPress={() => onSportChange(item)}
              style={[styles.segmentItem, selected && { backgroundColor: sportColors.chipActive }]}
              testID={`sport-${item}`}
            >
              <Text style={[styles.segmentText, { color: selected ? selectedText : colors.textSecondary }]}>
                {item === 'keirin' ? '경륜' : '경마'}
              </Text>
            </PressableScale>
          );
        })}
      </View>

      <View style={styles.metaRow}>
        <View style={styles.dateBlock}>
          <Text style={[styles.caption, { color: colors.textMuted }]}>분석일</Text>
          <View style={styles.dateRow}>
            <PressableScale
              accessibilityLabel="이전 경기일"
              accessibilityRole="button"
              accessibilityState={{ disabled: previousDisabled }}
              disabled={previousDisabled}
              onPress={() => onDateChange(dates[Math.max(0, dateIndex - 1)] ?? date)}
              style={[styles.dateStepper, { backgroundColor: colors.surfaceInset, opacity: previousDisabled ? 0.45 : 1 }]}
            >
              <Ionicons
                accessibilityElementsHidden
                accessible={false}
                importantForAccessibility="no-hide-descendants"
                name="chevron-back"
                size={18}
                color={colors.textPrimary}
              />
            </PressableScale>
            <View style={[styles.selectedDate, { backgroundColor: colors.surfaceInset }]}>
              <Text accessibilityLabel="선택된 경기일" style={[styles.selectedDateText, { color: colors.textPrimary }]}>
                {date}
              </Text>
              {scheduleUnavailable ? null : (
                <Text style={[styles.selectedDateMeta, { color: colors.textMuted }]}>경기일만 선택 가능</Text>
              )}
            </View>
            <PressableScale
              accessibilityLabel="다음 경기일"
              accessibilityRole="button"
              accessibilityState={{ disabled: nextDisabled }}
              disabled={nextDisabled}
              onPress={() => onDateChange(dates[Math.min(dates.length - 1, dateIndex + 1)] ?? date)}
              style={[styles.dateStepper, { backgroundColor: colors.surfaceInset, opacity: nextDisabled ? 0.45 : 1 }]}
            >
              <Ionicons
                accessibilityElementsHidden
                accessible={false}
                importantForAccessibility="no-hide-descendants"
                name="chevron-forward"
                size={18}
                color={colors.textPrimary}
              />
            </PressableScale>
          </View>
          <View style={styles.dateChipRow}>
            {dates.map((raceDate) => {
              const selected = raceDate === date;
              return (
                <PressableScale
                  aria-selected={selected}
                  accessibilityRole="button"
                  accessibilityState={{ selected }}
                  key={raceDate}
                  onPress={() => onDateChange(raceDate)}
                  style={[styles.dateChip, { backgroundColor: selected ? sportColors.chipActive : colors.surfaceInset }]}
                >
                  <Text style={[styles.dateChipText, { color: selected ? selectedText : colors.textPrimary }]}>
                    {formatRaceDateLabel(raceDate)}
                  </Text>
                </PressableScale>
              );
            })}
          </View>
          {scheduleUnavailable ? (
            <Text style={[styles.scheduleWarning, { color: colors.accentAmber }]}>
              공식 일정을 불러오지 못했습니다
            </Text>
          ) : null}
        </View>
      </View>

      <View>
        <Text style={[styles.caption, { color: colors.textMuted }]}>
          {sport === 'keirin' ? '경륜장' : '경마장'}
        </Text>
        <View style={styles.venueGrid}>
          {raceVenues[sport].map((venue) => {
            const selected = venue === meet;
            return (
              <PressableScale
                aria-selected={selected}
                accessibilityRole="button"
                accessibilityState={{ selected }}
                key={venue}
                onPress={() => onMeetChange(venue)}
                style={[styles.venueChip, { backgroundColor: selected ? sportColors.chipActive : colors.surfaceInset }]}
              >
                <Text style={[styles.venueText, { color: selected ? selectedText : colors.textPrimary }]}>{venue}</Text>
              </PressableScale>
            );
          })}
        </View>
      </View>

      <View style={styles.raceGrid}>
        {races.map((race) => {
          const selected = raceNo === race;
          return (
            <PressableScale
              aria-selected={selected}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              key={race}
              onPress={() => onRaceChange(race)}
              style={[
                styles.raceChip,
                {
                  backgroundColor: selected ? sportColors.chipActive : colors.surfaceInset,
                  borderColor: selected ? sportColors.chipActive : colors.borderSubtle
                }
              ]}
            >
              <Text style={[styles.raceText, { color: selected ? selectedText : colors.textPrimary }]}>
                {race}R
              </Text>
            </PressableScale>
          );
        })}
      </View>

      <PressableScale
        accessibilityLabel={needsReward ? '광고 보고 분석 1회 추가' : rewardReady ? '광고 보상으로 분석 보기' : needsProGuidance ? '무료 분석 한도 안내 보기' : '모델 신호 보기'}
        accessibilityRole="button"
        onPress={onAnalyze}
        style={[styles.cta, { backgroundColor: ctaBackground }]}
        testID="analyze-cta"
      >
        <Ionicons
          accessibilityElementsHidden
          accessible={false}
          importantForAccessibility="no-hide-descendants"
          name="analytics-outline"
          size={20}
          color={ctaForeground}
        />
        <Text style={[styles.ctaText, { color: ctaForeground }]}>
          {needsReward ? '광고 보고 1회 추가' : rewardReady ? '광고 보상으로 분석 보기' : needsProGuidance ? '한도 안내 보기' : '모델 신호 보기'}
        </Text>
      </PressableScale>
      <Text style={[styles.freeQuota, { color: colors.textMuted }]}>
        {proActive
          ? 'Pro 무제한 분석 활성'
          : freeRemaining > 0
          ? `무료 분석 ${Math.min(freeAnalysisUsed, freeAnalysisLimit)}/${freeAnalysisLimit} 사용 · 오늘 ${freeRemaining}회 남음`
          : rewardAdsEnabled && rewardedAnalysisCredits > 0
          ? `무료 3회 사용 완료 · 광고 보상 ${rewardedAnalysisCredits}회 보유`
          : rewardAdsEnabled
          ? '무료 3회 사용 완료 · 광고 보면 계속 이용 가능'
          : `무료 분석 ${Math.min(freeAnalysisUsed, freeAnalysisLimit)}/${freeAnalysisLimit} 사용 · 오늘 ${freeRemaining}회 남음`}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    borderRadius: radius.large,
    borderWidth: 1,
    gap: space.space2,
    padding: space.space3
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
    alignItems: 'stretch'
  },
  caption: {
    ...typography.caption,
    marginBottom: space.space1
  },
  dateBlock: {
    width: '100%'
  },
  dateRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space1
  },
  selectedDate: {
    borderRadius: radius.small,
    flex: 1,
    gap: space.space1,
    justifyContent: 'center',
    minHeight: 44,
    paddingHorizontal: space.space3
  },
  selectedDateText: {
    ...typography.bodyStrong
  },
  selectedDateMeta: {
    ...typography.caption
  },
  dateStepper: {
    alignItems: 'center',
    borderRadius: radius.small,
    height: 44,
    justifyContent: 'center',
    width: 44
  },
  dateChipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2,
    marginTop: space.space2
  },
  dateChip: {
    alignItems: 'center',
    borderRadius: radius.pill,
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: space.space3
  },
  dateChipText: {
    ...typography.bodyStrong
  },
  scheduleWarning: {
    ...typography.caption,
    marginTop: space.space1
  },
  raceGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2
  },
  venueGrid: {
    flexDirection: 'row',
    gap: space.space2,
    marginTop: space.space2
  },
  venueChip: {
    alignItems: 'center',
    borderRadius: radius.pill,
    flex: 1,
    justifyContent: 'center',
    minHeight: 44,
    paddingHorizontal: space.space3
  },
  venueText: {
    ...typography.bodyStrong
  },
  raceChip: {
    alignItems: 'center',
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: 'center',
    minHeight: 44,
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
    flexDirection: 'row',
    gap: space.space2,
    justifyContent: 'center',
    minHeight: 56,
    paddingVertical: space.space3
  },
  ctaText: {
    ...typography.bodyStrong
  },
  freeQuota: {
    ...typography.mono,
    textAlign: 'center'
  }
});
