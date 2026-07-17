import { StyleSheet, Text, View } from 'react-native';

import { FadeInUp } from './FadeInUp';
import { LensCard } from './LensCard';
import { NumberBadge } from './NumberBadge';
import { radius, space, sportPalette, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import type { RaceActualRacer, RaceDecision, RaceParticipant, TrifectaEnsembleTier } from '../types/race';

type PredictionSummaryProps = {
  mode: ThemeMode;
  decision: RaceDecision;
};

function selectionNumbers(selection: string) {
  return selection
    .split(/[-·,\s]+/)
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
}

function orderedNumbers(decision: RaceDecision) {
  const orderedPick = decision.picks.find((pick) => pick.code === 'TRI' && selectionNumbers(pick.selection).length >= 3)
    ?? decision.picks.find((pick) => selectionNumbers(pick.selection).length >= 3);
  return (orderedPick?.selection ?? '')
    .split(/[-·,\s]+/)
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
}

function findParticipant(participants: RaceParticipant[], number: number) {
  return participants.find((participant) => participant.number === number);
}

function findActualRacer(racers: RaceActualRacer[], number: number) {
  return racers.find((racer) => racer.number === number);
}

function rankLabel(index: number, settled: boolean) {
  if (settled && index === 0) return '1착 확정';
  if (settled && index === 1) return '2착 확정';
  if (settled) return '3착 확정';
  if (index === 0) return '1착 후보';
  if (index === 1) return '2착 후보';
  return '3착 후보';
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

const ensembleTierLabels: Record<TrifectaEnsembleTier, string> = {
  T0_base: '기본 앙상블',
  T1_strong: '강신호 후보',
  T2_top16: '강신호 경주'
};

export function PredictionSummary({ mode, decision }: PredictionSummaryProps) {
  const colors = palette(mode);
  const sportColors = sportPalette(mode, decision.sport);
  const predictedNumbers = orderedNumbers(decision).slice(0, 3);
  const actualResult = decision.actualResult;
  const settled = decision.status === 'settled' && actualResult !== undefined;
  const numbers = settled ? actualResult.actualOrder.slice(0, 3) : predictedNumbers;
  const hasRenderablePodium = numbers.length >= 3 && (settled || decision.confidence.top1 > 0);
  if (!hasRenderablePodium) {
    return (
      <FadeInUp>
        <LensCard mode={mode} variant="caution">
          <Text style={[styles.title, { color: colors.textPrimary }]}>데이터를 가져오지 못했습니다 — 잠시 후 다시 시도</Text>
          <Text style={[styles.helper, { color: colors.textSecondary }]}>
            공식 출전표와 예측 순서가 확인되기 전에는 후보 번호와 모델 확률을 표시하지 않습니다.
          </Text>
        </LensCard>
      </FadeInUp>
    );
  }
  const ordered = numbers.map((number) => ({
    number,
    participant: findParticipant(decision.participants, number),
    actualRacer: settled ? findActualRacer(actualResult.racers, number) : undefined
  }));
  const orderedText = numbers.length ? numbers.join('-') : '대기';
  const predictedText = predictedNumbers.length ? predictedNumbers.join('-') : '대기';
  const subject = decision.sport === 'horse' ? '말' : '선수';
  const trifectaPayout = actualResult?.payouts.find((payout) => payout.label === '삼쌍승식');
  const trifectaPick = decision.picks.find((pick) => pick.code === 'TRI') ?? decision.picks[0];
  const podiumOrder = [ordered[1], ordered[0], ordered[2]].filter((item) => item !== undefined);

  return (
    <FadeInUp>
    <LensCard mode={mode} variant={decision.marketRisk.level}>
      <View style={styles.header}>
        <View>
          <Text style={[styles.overline, { color: colors.textMuted }]}>
            {decision.meet} {decision.raceNo}R {settled ? '확정 결과' : '모델 결론'}
          </Text>
          <Text style={[styles.title, { color: colors.textPrimary }]}>
            {settled ? `실제 착순 ${orderedText}` : `예측 순서 ${orderedText}`}
          </Text>
        </View>
        <View style={[styles.badge, { backgroundColor: colors.surfaceBoard, borderColor: sportColors.headerTint }]}>
          <Text style={[styles.badgeText, { color: sportColors.headerTint }]}>예측 순서</Text>
        </View>
      </View>

      <Text style={[styles.helper, { color: colors.textSecondary }]}>
        {settled
          ? `경주가 종료되어 확정 착순을 먼저 표시합니다. 모델 예측은 ${predictedText}였고, 이 결과는 다음 튜닝 검증에만 반영됩니다.`
          : `1착 후보를 중심에 크게 두고, 2·3착 후보는 좌우에서 각 ${subject}의 근거와 함께 비교합니다.`}
      </Text>

      <View testID="prediction-podium" style={styles.podium}>
        {podiumOrder.map(({ number, participant, actualRacer }) => {
          const originalIndex = numbers.indexOf(number);
          const displayName = actualRacer?.name || participant?.name;
          const isWinner = originalIndex === 0;
          return (
            <View
              key={`${originalIndex}-${number}`}
              style={[
                styles.podiumSlot,
                isWinner ? styles.winnerSlot : styles.sideSlot,
                {
                  backgroundColor: isWinner ? colors.surfaceBoard : colors.surfaceInset,
                  borderColor: isWinner ? sportColors.pickHighlight : colors.borderSubtle
                }
              ]}
            >
              <Text style={[styles.rankLabel, { color: isWinner ? colors.textBoardQuiet : colors.textMuted }]}>
                {rankLabel(originalIndex, settled)}
              </Text>
              <NumberBadge mode={mode} number={number} size={isWinner ? 'large' : 'medium'} sport={decision.sport} />
              <Text
                numberOfLines={2}
                style={[styles.rankName, { color: isWinner ? colors.textOnBoard : colors.textPrimary }]}
              >
                {displayName ?? `${number}번`}
              </Text>
              {isWinner ? (
                <View style={styles.heroProbability}>
                  <Text style={[styles.probabilityLabel, { color: colors.textBoardQuiet }]}>모델 추정</Text>
                  <Text style={[styles.probabilityValue, { color: sportColors.pickHighlight }]}>
                    {percent(decision.confidence.top1)}
                  </Text>
                </View>
              ) : (
                <Text style={[styles.rankMeta, { color: colors.textSecondary }]} numberOfLines={2}>
                  {participant ? participant.trait : `${subject} 상세 대기`}
                </Text>
              )}
              <View style={[styles.podiumBase, { backgroundColor: isWinner ? sportColors.pickHighlight : colors.borderSubtle }]} />
            </View>
          );
        })}
      </View>

      <View style={[styles.trifectaRow, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
        <View style={styles.trifectaCopy}>
          <Text style={[styles.rankLabel, { color: colors.textMuted }]}>TRI 삼쌍</Text>
          <Text style={[styles.rankMeta, { color: colors.textSecondary }]}>
            {trifectaPick ? trifectaPick.selection : predictedText} · 순서 확률은 별도 표기
          </Text>
        </View>
        <View style={styles.trifectaValue}>
          <Text style={[styles.probabilityLabel, { color: colors.textMuted }]}>모델 추정</Text>
          <Text style={[styles.trifectaPercent, { color: sportColors.accent }]}>
            {percent(decision.confidence.trifecta)}
          </Text>
        </View>
      </View>

      {decision.sport === 'keirin' && decision.trifectaEnsemble ? (
        <View testID="trifecta-ensemble-card" style={[styles.ensembleCard, { backgroundColor: colors.surfaceBoard, borderColor: sportColors.secondary }]}>
          <View style={styles.ensembleHeader}>
            <View style={styles.trifectaCopy}>
              <Text style={[styles.rankLabel, { color: colors.textBoardQuiet }]}>앙상블 삼쌍</Text>
              <Text style={[styles.ensembleTitle, { color: colors.textOnBoard }]}>시장 신호 앙상블 픽</Text>
            </View>
            <Text style={[styles.ensemblePick, { color: sportColors.pickHighlight }]}>
              {decision.trifectaEnsemble.pick}
            </Text>
          </View>
          <View style={styles.ensembleMeta}>
            <Text style={[styles.rankMeta, { color: colors.textBoardQuiet }]}>
              {ensembleTierLabels[decision.trifectaEnsemble.tier]}
            </Text>
            <Text style={[styles.rankMeta, { color: colors.textBoardQuiet }]}>
              과거 적중 {percent(decision.trifectaEnsemble.tierHistoricalExact)}
            </Text>
            {decision.trifectaEnsemble.coverage !== undefined ? (
              <Text style={[styles.rankMeta, { color: colors.textBoardQuiet }]}>
                커버리지 {percent(decision.trifectaEnsemble.coverage)}
              </Text>
            ) : null}
          </View>
          <Text style={[styles.ensembleHelper, { color: colors.textBoardQuiet }]}>과거 검증 기반 보조 신호이며 결과를 보장하지 않습니다.</Text>
        </View>
      ) : null}

      {settled ? (
        <View style={styles.rankList}>
          {ordered.map(({ number, participant, actualRacer }, index) => (
            <Text key={`${index}-${number}-meta`} style={[styles.detailMeta, { color: colors.textMuted }]}>
              {settled && actualRacer ? `${index + 1}착 ${number}번 ${actualRacer.name}` :
                participant ? `${index + 1}착 후보 ${number}번 ${participant.name} · ${participant.trait}` : `${index + 1}착 후보 ${number}번`}
            </Text>
          ))}
        </View>
      ) : null}

      {settled && trifectaPayout ? (
        <View style={[styles.resultPayout, { backgroundColor: colors.surfaceBoard }]}>
          <Text style={[styles.payoutLabel, { color: colors.textBoardQuiet }]}>삼쌍승식 확정</Text>
          <Text style={[styles.payoutValue, { color: sportColors.pickHighlight }]}>
            {trifectaPayout.winner.replace(/\s*·\s*/g, '-')} · {trifectaPayout.odds.toLocaleString('ko-KR', {
              maximumFractionDigits: 2,
              minimumFractionDigits: trifectaPayout.odds % 1 === 0 ? 0 : 1
            })}배
          </Text>
        </View>
      ) : null}
    </LensCard>
    </FadeInUp>
  );
}

const styles = StyleSheet.create({
  header: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: space.space3,
    justifyContent: 'space-between'
  },
  overline: {
    ...typography.caption,
    marginBottom: space.space1
  },
  title: {
    ...typography.h2
  },
  badge: {
    alignItems: 'center',
    borderRadius: radius.medium,
    borderWidth: 1,
    minHeight: 42,
    justifyContent: 'center',
    paddingHorizontal: space.space3
  },
  badgeText: {
    ...typography.mono
  },
  helper: {
    ...typography.bodySm,
    marginTop: space.space3
  },
  podium: {
    alignItems: 'flex-end',
    flexDirection: 'row',
    gap: space.space2,
    marginTop: space.space4
  },
  podiumSlot: {
    alignItems: 'center',
    borderRadius: radius.medium,
    borderWidth: 1,
    flex: 1,
    gap: space.space2,
    padding: space.space3
  },
  winnerSlot: {
    minHeight: 180
  },
  sideSlot: {
    minHeight: 136
  },
  rankLabel: {
    ...typography.caption
  },
  rankName: {
    ...typography.bodyStrong,
    textAlign: 'center'
  },
  rankMeta: {
    ...typography.bodySm,
    textAlign: 'center'
  },
  heroProbability: {
    alignItems: 'center',
    gap: space.space1
  },
  probabilityLabel: {
    ...typography.caption
  },
  probabilityValue: {
    ...typography.display,
    fontVariant: ['tabular-nums']
  },
  podiumBase: {
    borderRadius: radius.pill,
    height: 5,
    marginTop: 'auto',
    width: '80%'
  },
  trifectaRow: {
    alignItems: 'center',
    borderRadius: radius.medium,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space3,
    justifyContent: 'space-between',
    marginTop: space.space3,
    padding: space.space3
  },
  trifectaCopy: {
    flex: 1,
    gap: space.space1
  },
  trifectaValue: {
    alignItems: 'flex-end'
  },
  trifectaPercent: {
    ...typography.h3,
    fontVariant: ['tabular-nums']
  },
  ensembleCard: {
    borderRadius: radius.medium,
    borderWidth: 1,
    gap: space.space2,
    marginTop: space.space3,
    padding: space.space3
  },
  ensembleHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  ensembleTitle: {
    ...typography.bodyStrong,
    marginTop: space.space1
  },
  ensemblePick: {
    ...typography.h2,
    fontVariant: ['tabular-nums']
  },
  ensembleMeta: {
    flexDirection: 'row',
    gap: space.space3
  },
  ensembleHelper: {
    ...typography.caption,
    fontWeight: '500'
  },
  rankList: {
    gap: space.space1,
    marginTop: space.space3
  },
  detailMeta: {
    ...typography.bodySm
  },
  resultPayout: {
    borderRadius: radius.medium,
    marginTop: space.space3,
    padding: space.space3
  },
  payoutLabel: {
    ...typography.caption,
    marginBottom: space.space1
  },
  payoutValue: {
    ...typography.h3
  }
});
