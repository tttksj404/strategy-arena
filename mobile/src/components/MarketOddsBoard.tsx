import { StyleSheet, Text, View } from 'react-native';

import { LensCard } from './LensCard';
import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { MarketOddsEntry } from '../types/race';

type MarketOddsBoardProps = {
  mode: ThemeMode;
  odds: MarketOddsEntry[];
  marketUsed: boolean;
  compact?: boolean;
};

function signalColor(colors: ReturnType<typeof palette>, signal: MarketOddsEntry['signal']) {
  if (signal === 'teal') return colors.accentTeal;
  if (signal === 'amber') return colors.accentAmber;
  if (signal === 'rose') return colors.accentRose;
  if (signal === 'violet') return colors.accentViolet;
  return colors.accentPrimary;
}

function formatOdds(value: number) {
  return `${value.toLocaleString('ko-KR', {
    maximumFractionDigits: 2,
    minimumFractionDigits: value % 1 === 0 ? 0 : 1
  })}배`;
}

export function MarketOddsBoard({ mode, odds, marketUsed, compact = false }: MarketOddsBoardProps) {
  const colors = palette(mode);
  const helper = marketUsed
    ? '마감 직전 배당 변화를 모델 신호와 분리해서 확인할 수 있습니다.'
    : '현재는 실시간 배당 미사용 상태입니다. 연결 전에는 대기값으로만 표시됩니다.';

  return (
    <LensCard mode={mode} variant={marketUsed ? 'verified' : 'caution'}>
      <View style={styles.header}>
        <View style={styles.heading}>
          <Text style={[styles.overline, { color: colors.textMuted }]}>시장 원자료</Text>
          <Text style={[styles.title, { color: colors.textPrimary }]}>실시간 배당</Text>
        </View>
        <View style={[styles.livePill, { borderColor: marketUsed ? colors.accentTeal : colors.accentAmber }]}>
          <View style={[styles.dot, { backgroundColor: marketUsed ? colors.accentTeal : colors.accentAmber }]} />
          <Text style={[styles.liveText, { color: marketUsed ? colors.accentTeal : colors.accentAmber }]}>
            {marketUsed ? 'LIVE' : '대기'}
          </Text>
        </View>
      </View>
      <Text style={[styles.helper, { color: colors.textSecondary }]}>{helper}</Text>

      <View style={styles.list}>
        {odds.length === 0 ? (
          <View style={[styles.empty, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
            <Text style={[styles.emptyText, { color: colors.textMuted }]}>배당 정보 대기 중</Text>
          </View>
        ) : null}
        {odds.map((entry) => {
          const accent = signalColor(colors, entry.signal);
          return (
            <View
              accessibilityLabel={`${entry.label} ${entry.selection} ${formatOdds(entry.odds)} ${entry.change}`}
              accessible
              key={`${entry.code}-${entry.selection}`}
              style={[styles.row, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}
            >
              <View style={styles.marketName}>
                <Text style={[styles.code, { color: colors.textMuted }]}>{entry.code}</Text>
                <Text style={[styles.label, { color: colors.textPrimary }]} numberOfLines={1}>{entry.label}</Text>
              </View>
              <View style={styles.marketValue}>
                <Text style={[styles.selection, { color: colors.textPrimary }]} numberOfLines={1}>{entry.selection}</Text>
                <Text style={[styles.odds, { color: accent }]}>{formatOdds(entry.odds)}</Text>
                {!compact ? (
                  <Text style={[styles.change, { color: colors.textMuted }]} numberOfLines={1}>{entry.change}</Text>
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
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space3,
    justifyContent: 'space-between'
  },
  heading: {
    flex: 1
  },
  overline: {
    ...typography.caption,
    marginBottom: space.space1
  },
  title: {
    ...typography.h3
  },
  livePill: {
    alignItems: 'center',
    borderRadius: radius.pill,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space2,
    minHeight: 32,
    paddingHorizontal: space.space3
  },
  dot: {
    borderRadius: radius.pill,
    height: 7,
    width: 7
  },
  liveText: {
    ...typography.caption
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
    alignItems: 'center',
    borderRadius: radius.medium,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space3,
    justifyContent: 'space-between',
    padding: space.space3
  },
  marketName: {
    flex: 1,
    minWidth: 0
  },
  code: {
    ...typography.caption,
    marginBottom: space.space1
  },
  label: {
    ...typography.bodyStrong
  },
  marketValue: {
    alignItems: 'flex-end',
    flex: 1,
    minWidth: 0
  },
  selection: {
    ...typography.h3
  },
  odds: {
    ...typography.bodyStrong,
    marginTop: space.space1
  },
  change: {
    ...typography.bodySm,
    marginTop: space.space1
  },
  empty: {
    borderRadius: radius.medium,
    borderWidth: 1,
    padding: space.space4
  },
  emptyText: {
    ...typography.bodySm
  }
});
