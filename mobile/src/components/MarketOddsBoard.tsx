import { StyleSheet, Text, View } from 'react-native';

import { LensCard } from './LensCard';
import { NumberBadge } from './NumberBadge';
import { radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { MarketOddsEntry, MarketSource, Sport } from '../types/race';

type MarketOddsBoardProps = {
  mode: ThemeMode;
  odds: MarketOddsEntry[];
  marketUsed: boolean;
  marketSource: MarketSource;
  updatedAt: string;
  oddsAgeSec: number | null;
  sport: Sport;
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

function sourceLabel(source: MarketSource) {
  if (source === 'live') return 'LIVE';
  if (source === 'historical') return '과거';
  if (source === 'settled') return '확정';
  if (source === 'sample') return '샘플';
  return '없음';
}

function sourceHelper(source: MarketSource, marketUsed: boolean) {
  if (source === 'live') return '마감 직전 배당 변화를 모델 신호와 분리해서 확인할 수 있습니다.';
  if (source === 'historical') return '이미 끝난 경주는 저장된 과거 배당을 기준으로 표시합니다.';
  if (source === 'settled') return '경주 종료 후 확정 결과가 확인되어 최종 배당은 예측 신호가 아니라 검증 자료로만 봅니다.';
  if (source === 'sample') return '현재 화면은 공식 배당이 아니라 과거 경주의 예시 배당입니다.';
  return marketUsed ? '배당 응답을 해석하지 못했습니다.' : '배당 원자료가 아직 연결되지 않았습니다.';
}

function formatKstTime(value: string) {
  const date = new Date(value);
  if (!Number.isNaN(date.getTime())) {
    return new Intl.DateTimeFormat('ko-KR', {
      hour: '2-digit',
      hour12: false,
      minute: '2-digit',
      timeZone: 'Asia/Seoul'
    }).format(date);
  }
  const match = value.match(/T(\d{2}):(\d{2})/);
  return match ? `${match[1]}:${match[2]}` : '--:--';
}

function freshnessText(source: MarketSource, updatedAt: string, oddsAgeSec: number | null, hasOdds: boolean) {
  if (!hasOdds || source === 'unavailable') return '배당 미수집';
  if (source === 'sample') return '폴백 샘플 배당';
  if (source === 'settled') return '확정 결과 자료';
  const ageText = oddsAgeSec === null ? '갱신 시각 확인 중' : `${oddsAgeSec}초 전 갱신`;
  return `${formatKstTime(updatedAt)} KST · ${ageText}`;
}

function selectionNumbers(selection: string) {
  return selection
    .split(/[-·,\s]+/)
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
}

export function MarketOddsBoard({ mode, odds, marketUsed, marketSource, updatedAt, oddsAgeSec, sport, compact = false }: MarketOddsBoardProps) {
  const colors = palette(mode);
  const helper = sourceHelper(marketSource, marketUsed);
  const freshness = freshnessText(marketSource, updatedAt, oddsAgeSec, odds.length > 0);
  const statusAccent = marketSource === 'live' || marketSource === 'historical' || marketSource === 'settled'
    ? colors.accentTeal
    : marketSource === 'sample' ? colors.accentGold : colors.accentAmber;

  return (
    <LensCard mode={mode} variant={marketSource === 'live' || marketSource === 'historical' || marketSource === 'settled' ? 'verified' : 'caution'}>
      <View style={styles.header}>
        <View style={styles.heading}>
          <Text style={[styles.overline, { color: colors.textMuted }]}>시장 원자료</Text>
          <Text style={[styles.title, { color: colors.textPrimary }]}>배당 자료</Text>
        </View>
        <View style={[styles.livePill, { borderColor: statusAccent }]}>
          <View style={[styles.dot, { backgroundColor: statusAccent }]} />
          <Text style={[styles.liveText, { color: statusAccent }]}>
            {sourceLabel(marketSource)}
          </Text>
        </View>
      </View>
      <Text style={[styles.freshness, { color: statusAccent }]}>{freshness}</Text>
      <Text style={[styles.helper, { color: colors.textSecondary }]}>{helper}</Text>

      <View style={styles.list}>
        {odds.length === 0 ? (
          <View style={[styles.empty, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
            <Text style={[styles.emptyText, { color: colors.textMuted }]}>
              {marketSource === 'settled' ? '확정 배당은 실제 결과 카드에서 확인' : '배당 정보 대기 중'}
            </Text>
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
                <View style={styles.selectionBadges}>
                  {selectionNumbers(entry.selection).map((number) => (
                    <NumberBadge key={`${entry.code}-${entry.selection}-${number}`} mode={mode} number={number} size="small" sport={sport} />
                  ))}
                  {selectionNumbers(entry.selection).length === 0 ? (
                    <Text style={[styles.selection, { color: colors.textPrimary }]} numberOfLines={1}>{entry.selection}</Text>
                  ) : null}
                </View>
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
    marginTop: space.space2
  },
  freshness: {
    ...typography.mono,
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
  selectionBadges: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space1,
    justifyContent: 'flex-end'
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
