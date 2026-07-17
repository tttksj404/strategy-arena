import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { RaceDecision, RiskLevel, RosterVerification } from '../types/race';

type DataStatusStripProps = {
  readonly decision: RaceDecision;
  readonly mode: ThemeMode;
};

type StripSlot = {
  readonly icon: keyof typeof Ionicons.glyphMap;
  readonly label: string;
  readonly level: RiskLevel;
};

function sampleSlot(): StripSlot {
  return { icon: 'flask-outline', label: '과거 데이터 예시', level: 'caution' };
}

function rosterSlot(verification: RosterVerification): StripSlot {
  if (verification.state === 'verified') {
    return { icon: 'shield-checkmark-outline', label: '출주표 검증', level: 'verified' };
  }
  if (verification.state === 'mismatch') {
    return { icon: 'alert-circle-outline', label: '출주표 차단', level: 'blocked' };
  }
  if (verification.state === 'unknown') {
    return { icon: 'help-circle-outline', label: '출주표 정보 없음', level: 'neutral' };
  }
  return { icon: 'shield-outline', label: '검증 대기', level: 'caution' };
}

function oddsSlot(decision: RaceDecision): StripSlot {
  if (!decision.marketUsed || decision.marketSource === 'unavailable') {
    return { icon: 'time-outline', label: '배당 미수집', level: 'caution' };
  }
  if (decision.marketSource === 'sample') {
    return { icon: 'flask-outline', label: '샘플 배당', level: 'caution' };
  }
  if (decision.marketSource === 'settled') {
    return { icon: 'checkmark-done-outline', label: '확정 배당', level: 'verified' };
  }
  const age = decision.oddsAgeSec === null ? '갱신 확인' : `${decision.oddsAgeSec}초 전`;
  return { icon: 'pulse-outline', label: age, level: 'verified' };
}

function phaseSlot(decision: RaceDecision): StripSlot {
  if (decision.status === 'settled') return { icon: 'flag-outline', label: '종료', level: 'verified' };
  if (decision.status === 'blocked') return { icon: 'ban-outline', label: '차단', level: 'blocked' };
  if (decision.status === 'hold') return { icon: 'hourglass-outline', label: '대기', level: 'caution' };
  return { icon: 'timer-outline', label: '마감 전', level: 'verified' };
}

function accentForLevel(colors: ReturnType<typeof palette>, level: RiskLevel) {
  if (level === 'verified') return colors.accentTeal;
  if (level === 'blocked') return colors.accentRose;
  if (level === 'caution') return colors.accentAmber;
  return colors.textMuted;
}

export function DataStatusStrip({ decision, mode }: DataStatusStripProps) {
  const colors = palette(mode);
  const slots = decision.marketSource === 'sample'
    ? [sampleSlot(), oddsSlot(decision), phaseSlot(decision)] as const
    : [rosterSlot(decision.rosterVerification), oddsSlot(decision), phaseSlot(decision)] as const;
  return (
    <View style={[styles.strip, { backgroundColor: colors.surfaceRaised, borderColor: colors.borderSubtle }]}>
      {slots.map((slot) => {
        const accent = accentForLevel(colors, slot.level);
        return (
          <View key={slot.label} style={styles.slot}>
            <Ionicons
              accessibilityElementsHidden
              accessible={false}
              importantForAccessibility="no-hide-descendants"
              name={slot.icon}
              size={15}
              color={accent}
            />
            <Text numberOfLines={2} style={[styles.label, { color: colors.textPrimary }]}>{slot.label}</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  strip: {
    borderRadius: radius.pill,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space1,
    minHeight: 44,
    paddingHorizontal: space.space2,
    paddingVertical: space.space2
  },
  slot: {
    alignItems: 'center',
    flex: 1,
    flexDirection: 'row',
    gap: space.space1,
    justifyContent: 'center',
    minWidth: 0
  },
  label: {
    ...typography.caption,
    flexShrink: 1
  }
});
