import { ScrollView, StyleSheet, Text, View } from 'react-native';
import type { DimensionValue } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { DataStatusStrip } from '../components/DataStatusStrip';
import { EvidenceGuide } from '../components/EvidenceGuide';
import { LensCard } from '../components/LensCard';
import { MarketOddsBoard } from '../components/MarketOddsBoard';
import { NumberBadge } from '../components/NumberBadge';
import { ParticipantBoard } from '../components/ParticipantBoard';
import { PressableScale } from '../components/PressableScale';
import { ProbabilityRail } from '../components/ProbabilityRail';
import { PredictionSummary } from '../components/PredictionSummary';
import { StatusPill } from '../components/StatusPill';
import { StoreSafeNotice } from '../components/StoreSafeNotice';
import { adSlot } from '../services/monetization';
import { palette, radius, space, sportPalette, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import type { ParticipantMetric, RaceDecision, RaceParticipant, RacePick, RosterVerification, Sport } from '../types/race';

type AnalyzeScreenProps = {
  mode: ThemeMode;
  decision: RaceDecision;
  freeAnalysisLimit: number;
  freeAnalysisUsed: number;
  rewardAdsEnabled: boolean;
  rewardedAnalysisCredits: number;
  onChooseRace: () => void;
  onRetry: () => void;
  onViewPro: () => void;
};

type AnalysisRenderState = 'analysis' | 'empty' | 'official_data_pending' | 'free_quota_exhausted' | 'roster_mismatch' | 'roster_unverified';

export function AnalyzeScreen({
  mode,
  decision,
  freeAnalysisLimit,
  freeAnalysisUsed,
  rewardedAnalysisCredits,
  rewardAdsEnabled,
  onChooseRace,
  onRetry,
  onViewPro
}: AnalyzeScreenProps) {
  const colors = palette(mode);
  const sportColors = sportPalette(mode, decision.sport);
  const proActive = decision.appSession.entitlement === 'pro';
  const sampleDecision = decision.marketSource === 'sample';
  const predictionAvailable = hasPredictionData(decision);
  const renderState = analysisRenderState(decision, predictionAvailable);
  const freeAdSlot = !proActive ? adSlot('free_analysis_gate') : null;
  const advancedRows = [
    {
      label: '고확신 후보 필터',
      value: `${Math.round(decision.confidence.top1 * 100)}% 이상 후보 ${decision.picks.filter((pick) => pick.grade !== '약').length}개`
    },
    {
      label: '마감 직전 배당 반영',
      value: decision.marketUsed ? `${sourceLabel(decision.marketSource)} 배당 연결` : '배당 폴백 감지'
    },
    {
      label: '유사 경주 복기',
      value: `표본 ${decision.confidence.sample.toLocaleString('ko-KR')}건 기준`
    },
    {
      label: '검증 로그',
      value: decision.dataLayer.ready ? '공식 데이터 레이어 준비' : '예시 데이터 레이어 대기'
    }
  ];

  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false} style={styles.scroll}>
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <View style={styles.headerTitleBlock}>
            <Text style={[styles.title, { color: colors.textPrimary }]}>{decision.meet} {decision.raceNo}R 분석</Text>
            <Text style={[styles.dateLabel, { color: colors.textMuted }]}>
              {sampleDecision ? '예시 기준일' : '분석일'} {decision.date}
            </Text>
          </View>
          <StatusPill mode={mode} level={proActive ? 'pro' : 'neutral'} label={proActive ? 'PRO 활성' : '무료 이용'} />
        </View>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          {decision.status === 'settled'
            ? '경주 종료 후에는 실제 착순을 먼저 보고, 아래에서 모델 판단과 근거 데이터를 복기합니다.'
            : '먼저 1착·2착·3착 예측 순서를 보고, 아래에서 근거 데이터를 확인합니다.'}
        </Text>
      </View>

      {renderState !== 'free_quota_exhausted' ? <DataStatusStrip decision={decision} mode={mode} /> : null}

      {renderState === 'free_quota_exhausted' ? (
        <>
          <FreeQuotaExhaustedBlock
            decision={decision}
            mode={mode}
            onViewPro={onViewPro}
          />
          <StoreSafeNotice mode={mode} />
        </>
      ) : renderState === 'roster_mismatch' || renderState === 'roster_unverified' ? (
        <>
          <RosterMismatchBlock
            mode={mode}
            verification={decision.rosterVerification}
            waiting={renderState === 'roster_unverified'}
            onChooseRace={onChooseRace}
            onRetry={onRetry}
          />
          <StoreSafeNotice mode={mode} />
        </>
      ) : renderState === 'official_data_pending' ? (
        <>
          <OfficialDataPendingCard decision={decision} mode={mode} />
          <StoreSafeNotice mode={mode} />
        </>
      ) : renderState === 'empty' ? (
        <>
          <EmptyAnalysisCard decision={decision} mode={mode} onRetry={onRetry} />
          <StoreSafeNotice mode={mode} />
        </>
      ) : (
        <>
      <PredictionSummary decision={decision} mode={mode} />

      <EntitlementPanel
        advancedRows={advancedRows}
        freeAnalysisLimit={freeAnalysisLimit}
        freeAnalysisUsed={freeAnalysisUsed}
        mode={mode}
        onViewPro={onViewPro}
        proActive={proActive}
        rewardAdsEnabled={rewardAdsEnabled}
        rewardedAnalysisCredits={rewardedAnalysisCredits}
      />

      {freeAdSlot?.enabled ? <FreeAdSlot mode={mode} /> : null}

      <LensCard mode={mode} variant={decision.marketRisk.level}>
        <View style={styles.riskTop}>
          <StatusPill mode={mode} level={decision.marketRisk.level} label={decision.marketRisk.title} />
          <Text style={[styles.sample, { color: colors.textMuted }]}>표본 {decision.confidence.sample.toLocaleString('ko-KR')}</Text>
        </View>
        <Text style={[styles.riskMessage, { color: colors.textSecondary }]}>{decision.marketRisk.message}</Text>
      </LensCard>

      <EvidenceGuide decision={decision} mode={mode} />

      <MarketOddsBoard
        marketSource={decision.marketSource}
        marketUsed={decision.marketUsed}
        mode={mode}
        odds={decision.marketOdds}
        oddsAgeSec={decision.oddsAgeSec}
        sport={decision.sport}
        updatedAt={decision.updatedAt}
      />

      {decision.sport === 'keirin' ? (
        <KeirinDevelopmentCard decision={decision} mode={mode} />
      ) : (
        <HorseGateWeightCard decision={decision} mode={mode} />
      )}

      <ParticipantBoard mode={mode} participants={decision.participants} proActive={proActive} sport={decision.sport} />

      <View style={styles.pickList}>
        {decision.picks.map((pick) => (
          <LensCard key={pick.code} mode={mode}>
            <View style={styles.pickHeader}>
              <View>
                <Text style={[styles.pickCode, { color: colors.textMuted }]}>{pick.code}</Text>
                <Text style={[styles.pickTitle, { color: colors.textPrimary }]}>{formatPickLabel(pick.code, pick.label)}</Text>
              </View>
              <PickSelection mode={mode} pick={pick} sport={decision.sport} />
            </View>
            <ProbabilityRail
              mode={mode}
              label={`등급 ${pick.grade}`}
              value={pick.probability}
              tone={pick.grade === '강' ? 'teal' : pick.grade === '중' ? 'primary' : 'amber'}
            />
            <View style={[styles.pickAccent, { backgroundColor: sportColors.pickHighlight }]} />
          </LensCard>
        ))}
      </View>

      <StoreSafeNotice mode={mode} />
        </>
      )}
    </ScrollView>
  );
}

function sourceLabel(source: RaceDecision['marketSource']) {
  if (source === 'live') return '실시간';
  if (source === 'historical') return '과거';
  if (source === 'sample') return '샘플';
  return '대기';
}

function selectionNumbers(selection: string) {
  return selection
    .split(/[-·,\s]+/)
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
}

function hasPredictionData(decision: RaceDecision) {
  if (decision.status === 'settled') {
    return (decision.actualResult?.actualOrder.length ?? 0) >= 3;
  }
  const orderedPick = decision.picks.find((pick) => pick.code === 'TRI' && selectionNumbers(pick.selection).length >= 3)
    ?? decision.picks.find((pick) => selectionNumbers(pick.selection).length >= 3);
  return orderedPick !== undefined && decision.confidence.top1 > 0;
}

function hasNoRaceSignal(decision: RaceDecision) {
  const combined = [
    decision.headline,
    decision.marketRisk.title,
    decision.marketRisk.message,
    decision.dataLayer.error ?? ''
  ].join(' ');
  return /no[_\s-]?race|경주가\s*없|경기가\s*없/.test(combined);
}

function isFreeQuotaExhausted(decision: RaceDecision) {
  const session = decision.appSession;
  const quotaFieldsExhausted = session.entitlement !== 'pro' &&
    session.freeAnalysisRemaining <= 0 &&
    session.rewardedAnalysisCredits <= 0;
  const quotaMessage = /무료\s*분석.*(3회|모두\s*사용|소진)/.test(decision.marketRisk.message);
  return quotaFieldsExhausted || quotaMessage;
}

function analysisRenderState(decision: RaceDecision, predictionAvailable: boolean): AnalysisRenderState {
  if (decision.status === 'settled' && predictionAvailable) return 'analysis';
  if (isFreeQuotaExhausted(decision)) return 'free_quota_exhausted';
  if (decision.officialDataPending) return 'official_data_pending';
  if (decision.rosterVerification.state === 'mismatch') return 'roster_mismatch';
  if (
    decision.rosterVerification.state === 'unverified' &&
    decision.marketSource !== 'sample' &&
    decision.status !== 'settled' &&
    !decision.dataLayer.error
  ) {
    return 'roster_unverified';
  }
  if (!predictionAvailable) return 'empty';
  return 'analysis';
}

function OfficialDataPendingCard({ decision, mode }: { decision: RaceDecision; mode: ThemeMode }) {
  const colors = palette(mode);
  return (
    <LensCard mode={mode} variant="caution">
      <View testID="analysis-pending-state" style={styles.emptyState}>
        <Ionicons
          accessibilityElementsHidden
          accessible={false}
          importantForAccessibility="no-hide-descendants"
          name="hourglass-outline"
          size={24}
          color={colors.accentAmber}
        />
        <View style={styles.emptyCopyBlock}>
          <Text style={[styles.blockTitle, { color: colors.textPrimary }]}>{decision.marketRisk.title}</Text>
          <Text style={[styles.blockCopy, { color: colors.textSecondary }]}>{decision.marketRisk.message}</Text>
        </View>
      </View>
    </LensCard>
  );
}

function FreeQuotaExhaustedBlock({
  decision,
  mode,
  onViewPro
}: {
  decision: RaceDecision;
  mode: ThemeMode;
  onViewPro: () => void;
}) {
  const colors = palette(mode);
  return (
    <LensCard mode={mode} variant="pro">
      <View style={styles.blockHeader} testID="free-quota-exhausted-state">
        <Ionicons
          accessibilityElementsHidden
          accessible={false}
          importantForAccessibility="no-hide-descendants"
          name="lock-closed-outline"
          size={24}
          color={colors.accentViolet}
        />
        <Text style={[styles.blockTitle, { color: colors.textPrimary }]}>무료 3회 소진 · Pro 안내</Text>
      </View>
      <Text style={[styles.blockCopy, { color: colors.textSecondary }]}>
        {decision.marketRisk.message}
      </Text>
      <View style={styles.blockActions}>
        <PressableScale
          accessibilityLabel="Pro 안내 보기"
          accessibilityRole="link"
          onPress={onViewPro}
          style={[styles.retryButton, { backgroundColor: colors.accentViolet }]}
        >
          <Text style={[styles.retryButtonText, { color: colors.textOnBoard }]}>Pro 안내 보기</Text>
        </PressableScale>
      </View>
    </LensCard>
  );
}

function EmptyAnalysisCard({
  decision,
  mode,
  onRetry
}: {
  decision: RaceDecision;
  mode: ThemeMode;
  onRetry: () => void;
}) {
  const colors = palette(mode);
  const message = decision.dataLayer.error
    ? decision.dataLayer.error
    : hasNoRaceSignal(decision)
    ? '해당 날짜에는 경주가 없습니다'
    : decision.analysisError
    ? decision.marketRisk.message
    : '데이터를 가져오지 못했습니다 — 잠시 후 다시 시도';
  return (
    <LensCard mode={mode} variant="caution">
      <View testID="analysis-empty-state" style={styles.emptyState}>
        <Ionicons
          accessibilityElementsHidden
          accessible={false}
          importantForAccessibility="no-hide-descendants"
          name="cloud-offline-outline"
          size={24}
          color={colors.accentAmber}
        />
        <View style={styles.emptyCopyBlock}>
          <Text style={[styles.blockTitle, { color: colors.textPrimary }]}>{message}</Text>
          <Text style={[styles.blockCopy, { color: colors.textSecondary }]}>
            공식 출전표와 1·2·3착 예측 순서가 확인될 때까지 후보 번호, 모델 확률, 참가자 카드를 표시하지 않습니다.
          </Text>
          <PressableScale
            accessibilityRole="button"
            onPress={onRetry}
            style={[
              styles.retryButton,
              {
                alignSelf: 'flex-start',
                backgroundColor: colors.accentAmber,
                marginTop: space.space3
              }
            ]}
          >
            <Text style={[styles.retryButtonText, { color: colors.textOnBoard }]}>다시 시도</Text>
          </PressableScale>
        </View>
      </View>
    </LensCard>
  );
}

function PickSelection({ mode, pick, sport }: { mode: ThemeMode; pick: RacePick; sport: Sport }) {
  const colors = palette(mode);
  const numbers = selectionNumbers(pick.selection);
  if (numbers.length === 0) {
    return <Text style={[styles.selection, { color: colors.textPrimary }]}>{formatPickSelection(pick.code, pick.selection)}</Text>;
  }
  return (
    <View style={styles.pickBadges}>
      {numbers.map((number) => (
        <NumberBadge key={`${pick.code}-${pick.selection}-${number}`} mode={mode} number={number} sport={sport} />
      ))}
      {pick.code === 'QNL' ? (
        <Text style={[styles.pickSuffix, { color: colors.textMuted }]}>조합</Text>
      ) : null}
    </View>
  );
}

function RosterMismatchBlock({
  mode,
  verification,
  waiting = false,
  onChooseRace,
  onRetry
}: {
  mode: ThemeMode;
  verification: RosterVerification;
  waiting?: boolean;
  onChooseRace: () => void;
  onRetry: () => void;
}) {
  const colors = palette(mode);
  const title = waiting
    ? '공식 출주표를 아직 확인하지 못했습니다'
    : '공식 출주표와 달라 예측을 중단했습니다';
  const copy = waiting
    ? '공식 출주표 대조가 끝나기 전에는 모델 순서와 후보 카드를 표시하지 않습니다. 잠시 후 다시 요청하거나 다른 경주를 선택하세요.'
    : verification.message ?? '서버가 받은 출전 명단이 공식 출주표와 일치하지 않아 모델 결과를 숨깁니다. 다른 경주를 선택하거나 출주표 갱신 후 다시 시도하세요.';
  return (
    <LensCard mode={mode} variant={waiting ? 'caution' : 'blocked'}>
      <View style={styles.blockHeader} testID={waiting ? 'roster-waiting-state' : 'roster-blocked-state'}>
        <Ionicons
          accessibilityElementsHidden
          accessible={false}
          importantForAccessibility="no-hide-descendants"
          name="alert-circle-outline"
          size={24}
          color={waiting ? colors.accentAmber : colors.accentRose}
        />
        <Text style={[styles.blockTitle, { color: colors.textPrimary }]}>{title}</Text>
      </View>
      <Text style={[styles.blockCopy, { color: colors.textSecondary }]}>
        {copy}
      </Text>
      <View style={styles.blockActions}>
        <PressableScale
          accessibilityRole="button"
          onPress={onRetry}
          style={[
            styles.retryButton,
            { backgroundColor: waiting ? colors.accentAmber : colors.accentRose }
          ]}
        >
          <Text style={[styles.retryButtonText, { color: colors.textOnBoard }]}>다시 시도</Text>
        </PressableScale>
        <PressableScale
          accessibilityRole="button"
          onPress={onChooseRace}
          style={[styles.secondaryButton, { borderColor: colors.borderSubtle }]}
        >
          <Text style={[styles.secondaryButtonText, { color: colors.textPrimary }]}>다른 경주 선택</Text>
        </PressableScale>
      </View>
    </LensCard>
  );
}

function metricValue(items: ParticipantMetric[], label: string) {
  return items.find((item) => item.label === label)?.value ?? '-';
}

function metricNumber(items: ParticipantMetric[], label: string) {
  const match = metricValue(items, label).match(/-?\d+(?:\.\d+)?/);
  return match ? Number(match[0]) : null;
}

function tacticPercent(participant: RaceParticipant, label: string) {
  return metricNumber(participant.tactics, label) ?? 0;
}

function KeirinDevelopmentCard({ decision, mode }: { decision: RaceDecision; mode: ThemeMode }) {
  const colors = palette(mode);
  const tactics = ['선행', '젖히기', '추입', '마크'] as const;
  const totals = tactics.map((label) => ({
    label,
    value: decision.participants.reduce((sum, participant) => sum + tacticPercent(participant, label), 0)
  }));
  const total = totals.reduce((sum, item) => sum + item.value, 0) || 1;
  const axisNumber = Number(decision.picks[0]?.selection.split('-')[0]);
  const axis = decision.participants.find((participant) => participant.number === axisNumber) ?? decision.participants[0];

  return (
    <LensCard mode={mode}>
      <View style={styles.sportCardHeader}>
        <View>
          <Text style={[styles.overline, { color: colors.textMuted }]}>경륜 전용</Text>
          <Text style={[styles.sportCardTitle, { color: colors.textPrimary }]}>전개 구도</Text>
        </View>
        <StatusPill mode={mode} level="verified" label={axis ? `${axis.number}번 축 후보` : '축 후보 대기'} />
      </View>
      <View style={styles.tacticBar}>
        {totals.map((item, index) => (
          <View
            key={item.label}
            style={[
              styles.tacticSegment,
              {
                backgroundColor: tacticColor(colors, index),
                flexGrow: Math.max(0.08, item.value / total)
              }
            ]}
          />
        ))}
      </View>
      <View style={styles.tacticLegend}>
        {totals.map((item, index) => (
          <View key={item.label} style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: tacticColor(colors, index) }]} />
            <Text style={[styles.legendText, { color: colors.textSecondary }]}>
              {item.label} {Math.round((item.value / total) * 100)}%
            </Text>
          </View>
        ))}
      </View>
      <Text style={[styles.sportCardCopy, { color: colors.textSecondary }]}>
        {axis ? `${axis.name}의 ${axis.trait} 신호를 기준으로 입상전법 분포를 비교합니다.` : '출전 선수 전법 자료가 도착하면 축 후보와 분포를 표시합니다.'}
      </Text>
    </LensCard>
  );
}

function tacticColor(colors: ReturnType<typeof palette>, index: number) {
  const paletteItems = [colors.accentTeal, colors.accentSignal, colors.accentPrimary, colors.accentViolet] as const;
  return paletteItems[index] ?? colors.accentAmber;
}

function HorseGateWeightCard({ decision, mode }: { decision: RaceDecision; mode: ThemeMode }) {
  const colors = palette(mode);
  const rows = [...decision.participants]
    .sort((left, right) => left.number - right.number)
    .map((participant) => ({
      number: participant.number,
      name: participant.name,
      gate: metricValue(participant.form, '게이트'),
      weight: metricValue(participant.profile, '부담중량'),
      body: metricValue(participant.profile, '마체')
    }));

  return (
    <LensCard mode={mode}>
      <View style={styles.sportCardHeader}>
        <View>
          <Text style={[styles.overline, { color: colors.textMuted }]}>경마 전용</Text>
          <Text style={[styles.sportCardTitle, { color: colors.textPrimary }]}>게이트·부담중량</Text>
        </View>
        <StatusPill mode={mode} level="caution" label="게이트 순" />
      </View>
      <View style={styles.horseMiniBoard}>
        {rows.length === 0 ? (
          <Text style={[styles.emptyMiniBoard, { color: colors.textMuted }]}>출전마 자료 대기 중</Text>
        ) : rows.map((row) => (
          <View key={`${row.number}-${row.name}`} style={[styles.horseRow, { borderColor: colors.borderSubtle }]}>
            <NumberBadge mode={mode} number={row.number} size="small" sport={decision.sport} />
            <Text style={[styles.horseGate, { color: colors.textPrimary }]}>{row.gate}</Text>
            <View style={styles.horseNameBlock}>
              <Text style={[styles.horseName, { color: colors.textPrimary }]} numberOfLines={1}>{row.number}번 {row.name}</Text>
              <Text style={[styles.horseMeta, { color: colors.textMuted }]}>{row.weight} · 마체 {row.body}</Text>
            </View>
          </View>
        ))}
      </View>
    </LensCard>
  );
}

function formatPickLabel(code: string, label: string) {
  return code === 'QNL' ? '복승 조합(무순)' : label;
}

function formatPickSelection(code: string, selection: string) {
  if (code !== 'QNL') return selection;
  return `${selection.split('-').join('·')} 조합`;
}

function EntitlementPanel({
  advancedRows,
  freeAnalysisLimit,
  freeAnalysisUsed,
  mode,
  onViewPro,
  proActive,
  rewardAdsEnabled,
  rewardedAnalysisCredits
}: {
  advancedRows: { label: string; value: string }[];
  freeAnalysisLimit: number;
  freeAnalysisUsed: number;
  mode: ThemeMode;
  onViewPro: () => void;
  proActive: boolean;
  rewardAdsEnabled: boolean;
  rewardedAnalysisCredits: number;
}) {
  const colors = palette(mode);
  const safeUsed = Math.min(freeAnalysisLimit, freeAnalysisUsed);
  const remaining = Math.max(0, freeAnalysisLimit - safeUsed);
  const quotaPct = (freeAnalysisLimit > 0 ? `${Math.round((safeUsed / freeAnalysisLimit) * 100)}%` : '0%') as DimensionValue;
  const rewardReady = rewardAdsEnabled && rewardedAnalysisCredits > 0;
  const quotaExhausted = remaining === 0 && !rewardReady;
  const quotaFillColor = rewardReady ? colors.accentGold : remaining === 0 ? colors.accentRose : remaining === 1 ? colors.accentAmber : colors.accentTeal;
  const freeCopy = remaining > 0
    ? '기본 무료 분석을 사용 중입니다. 무료 한도 이후에는 Pro 준비 화면에서 이용 한도를 안내합니다.'
    : rewardReady
    ? '광고 보상 분석권을 보유 중입니다. 다음 분석 요청에 1회가 사용됩니다.'
    : '오늘 무료 분석 한도를 모두 사용했습니다. Pro 기능은 준비 중이며, 아래에서 한도 안내를 바로 확인하세요.';
  const freeStatusLabel = remaining > 0
    ? `오늘 ${remaining}회 남음`
    : rewardReady
    ? `광고 보상 ${rewardedAnalysisCredits}회`
    : '무료 한도 소진';
  return (
    <LensCard mode={mode} variant={proActive ? 'pro' : 'caution'}>
      <View style={styles.entitlementHeader}>
        <View>
          <Text style={[styles.entitlementTitle, { color: colors.textPrimary }]}>
            {proActive ? 'Pro 고급 분석 열림' : `무료 분석 ${safeUsed}/${freeAnalysisLimit} 사용`}
          </Text>
          <Text style={[styles.entitlementCopy, { color: colors.textSecondary }]}>
            {proActive ? '전체 신호와 검증 로그를 바로 확인합니다.' : freeCopy}
          </Text>
        </View>
        <StatusPill mode={mode} level={proActive ? 'pro' : 'neutral'} label={proActive ? '무제한' : freeStatusLabel} />
      </View>

      {!proActive ? (
        <View style={[styles.quotaTrack, { backgroundColor: colors.railBase }]}>
          <View style={[styles.quotaFill, { backgroundColor: quotaFillColor, width: quotaPct }]} />
        </View>
      ) : null}

      {quotaExhausted ? (
        <PressableScale
          accessibilityLabel="Pro 안내 보기"
          accessibilityRole="link"
          onPress={onViewPro}
          style={styles.proGuidanceLink}
        >
          <Text style={[styles.proGuidanceLinkText, { color: colors.accentViolet }]}>Pro 안내 보기</Text>
          <Ionicons
            accessibilityElementsHidden
            accessible={false}
            importantForAccessibility="no-hide-descendants"
            name="arrow-forward"
            size={16}
            color={colors.accentViolet}
          />
        </PressableScale>
      ) : null}

      <View style={styles.advancedList}>
        {advancedRows.map((row) => (
          <View key={row.label} style={[styles.advancedRow, { borderColor: colors.borderSubtle }]}>
            <Ionicons
              accessibilityElementsHidden
              accessible={false}
              importantForAccessibility="no-hide-descendants"
              name={proActive ? 'checkmark-circle-outline' : 'lock-closed-outline'}
              size={19}
              color={proActive ? colors.accentTeal : colors.textMuted}
            />
            <View style={styles.advancedText}>
              <Text style={[styles.advancedLabel, { color: colors.textPrimary }]}>
                {proActive ? row.label : `${row.label} 잠금`}
              </Text>
              <Text style={[styles.advancedValue, { color: colors.textSecondary }]}>
                {proActive ? row.value : 'Pro에서 실제 값과 복기 내역을 엽니다.'}
              </Text>
            </View>
          </View>
        ))}
      </View>
    </LensCard>
  );
}

function FreeAdSlot({ mode }: { mode: ThemeMode }) {
  const colors = palette(mode);
  return (
    <LensCard mode={mode}>
      <View style={styles.adHeader}>
        <View style={[styles.adIcon, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
          <Ionicons
            accessibilityElementsHidden
            accessible={false}
            importantForAccessibility="no-hide-descendants"
            name="megaphone-outline"
            size={20}
            color={colors.accentGold}
          />
        </View>
        <View style={styles.adText}>
          <Text style={[styles.adTitle, { color: colors.textPrimary }]}>광고 보고 계속 이용</Text>
          <Text style={[styles.adCopy, { color: colors.textSecondary }]}>무료 3회 이후에는 광고 1회 시청마다 분석 1회가 충전됩니다.</Text>
        </View>
      </View>
    </LensCard>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flex: 1
  },
  content: {
    gap: space.space5,
    padding: space.space5,
    paddingBottom: 120
  },
  header: {
    gap: space.space2,
    paddingTop: space.space4
  },
  headerTop: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: space.space3,
    justifyContent: 'space-between'
  },
  headerTitleBlock: {
    flex: 1,
    gap: space.space1
  },
  title: {
    ...typography.h1
  },
  dateLabel: {
    ...typography.mono
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
  blockHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space3
  },
  blockTitle: {
    ...typography.h3,
    flex: 1
  },
  blockCopy: {
    ...typography.body,
    marginTop: space.space3
  },
  blockActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2,
    marginTop: space.space4
  },
  retryButton: {
    alignItems: 'center',
    borderRadius: radius.pill,
    justifyContent: 'center',
    minHeight: 44,
    paddingHorizontal: space.space4,
    paddingVertical: space.space3
  },
  retryButtonText: {
    ...typography.bodyStrong
  },
  secondaryButton: {
    alignItems: 'center',
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: 'center',
    minHeight: 44,
    paddingHorizontal: space.space4,
    paddingVertical: space.space3
  },
  secondaryButtonText: {
    ...typography.bodyStrong
  },
  emptyState: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: space.space3
  },
  emptyCopyBlock: {
    flex: 1,
    gap: space.space1
  },
  overline: {
    ...typography.caption,
    marginBottom: space.space1
  },
  sportCardHeader: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: space.space3,
    justifyContent: 'space-between'
  },
  sportCardTitle: {
    ...typography.h3
  },
  tacticBar: {
    borderRadius: radius.pill,
    flexDirection: 'row',
    gap: 2,
    height: 18,
    marginTop: space.space4,
    overflow: 'hidden'
  },
  tacticSegment: {
    minWidth: 12
  },
  tacticLegend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2,
    marginTop: space.space3
  },
  legendItem: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space1
  },
  legendDot: {
    borderRadius: radius.pill,
    height: 8,
    width: 8
  },
  legendText: {
    ...typography.mono
  },
  sportCardCopy: {
    ...typography.bodySm,
    marginTop: space.space3
  },
  horseMiniBoard: {
    gap: space.space2,
    marginTop: space.space4
  },
  horseRow: {
    alignItems: 'center',
    borderRadius: radius.small,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space3,
    minHeight: 52,
    padding: space.space3
  },
  horseGate: {
    ...typography.bodyStrong,
    width: 42
  },
  horseNameBlock: {
    flex: 1,
    minWidth: 0
  },
  horseName: {
    ...typography.bodyStrong
  },
  horseMeta: {
    ...typography.bodySm,
    marginTop: space.space1
  },
  emptyMiniBoard: {
    ...typography.bodySm
  },
  entitlementHeader: {
    alignItems: 'flex-start',
    flexDirection: 'column',
    gap: space.space3,
    justifyContent: 'space-between'
  },
  entitlementTitle: {
    ...typography.h3
  },
  entitlementCopy: {
    ...typography.bodySm,
    marginTop: space.space1
  },
  quotaTrack: {
    borderRadius: radius.pill,
    height: 9,
    marginTop: space.space4,
    overflow: 'hidden'
  },
  quotaFill: {
    borderRadius: radius.pill,
    height: 9
  },
  proGuidanceLink: {
    alignItems: 'center',
    alignSelf: 'flex-start',
    flexDirection: 'row',
    gap: space.space1,
    marginTop: space.space3,
    minHeight: 44,
    paddingVertical: space.space2
  },
  proGuidanceLinkText: {
    ...typography.bodyStrong
  },
  advancedList: {
    gap: space.space2,
    marginTop: space.space4
  },
  advancedRow: {
    alignItems: 'center',
    borderRadius: radius.small,
    borderWidth: 1,
    flexDirection: 'row',
    gap: space.space3,
    minHeight: 58,
    padding: space.space3
  },
  advancedText: {
    flex: 1
  },
  advancedLabel: {
    ...typography.bodyStrong
  },
  advancedValue: {
    ...typography.bodySm,
    marginTop: space.space1
  },
  adHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space3
  },
  adIcon: {
    alignItems: 'center',
    borderRadius: radius.pill,
    borderWidth: 1,
    height: 42,
    justifyContent: 'center',
    width: 42
  },
  adText: {
    flex: 1
  },
  adTitle: {
    ...typography.bodyStrong
  },
  adCopy: {
    ...typography.bodySm,
    marginTop: space.space1
  },
  pickList: {
    gap: space.space3
  },
  pickHeader: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: space.space3,
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
  },
  pickBadges: {
    alignItems: 'center',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space1,
    justifyContent: 'flex-end'
  },
  pickSuffix: {
    ...typography.caption,
    marginLeft: space.space1
  },
  pickAccent: {
    borderRadius: radius.pill,
    height: 4,
    marginTop: space.space3
  }
});
