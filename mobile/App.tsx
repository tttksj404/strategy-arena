import { useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, AppState, StyleSheet, Text, useColorScheme, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { StatusBar } from 'expo-status-bar';

import { BottomTabs } from './src/components/BottomTabs';
import { FadeInUp } from './src/components/FadeInUp';
import { PressableScale } from './src/components/PressableScale';
import { fetchAppSession, fetchRaceDates, fetchRaceDecision, hostedPublicPro } from './src/services/raceApi';
import { isRewardedAdsEnabled } from './src/services/monetization';
import { isRewardedAdPreview, showRewardedAd } from './src/services/rewardedAds';
import { availableRaceDates, defaultRaceCount, defaultRaceVenue, nearestRaceDate, todayInKorea } from './src/services/raceSchedule';
import { AnalyzeScreen } from './src/screens/AnalyzeScreen';
import { HomeScreen } from './src/screens/HomeScreen';
import { ProScreen } from './src/screens/ProScreen';
import { gradient, palette, space, typography } from './src/theme/tokens';
import type { ThemeMode } from './src/theme/tokens';
import { recordFirebaseError } from './src/services/firebaseTelemetry';
import { trackUxEvent } from './src/services/uxAnalytics';
import type { AppSession, DataLayerStatus, RaceDecision, Sport, TabKey } from './src/types/race';
import type { RaceDateSchedule } from './src/services/raceSchedule';

const freeAnalysisLimit = 3;
const calendarRefreshMs = 5 * 60 * 1000;
const demoAppSession: AppSession = {
  userId: 'demo-user',
  deviceId: 'demo-device',
  entitlement: hostedPublicPro ? 'pro' : 'free',
  freeAnalysisLimit,
  freeAnalysisUsed: 0,
  freeAnalysisRemaining: freeAnalysisLimit,
  rewardedAnalysisCredits: 0
};
const demoDataLayer: DataLayerStatus = {
  ready: false,
  storage: 'demo',
  schemas: []
};

export default function App() {
  const scheme = useColorScheme();
  const mode = scheme === 'light' ? 'light' : 'dark';
  const colors = palette(mode);
  const gradients = gradient(mode);
  const rewardAdsEnabled = isRewardedAdsEnabled();
  const rewardAdPreview = isRewardedAdPreview();
  const initialSport: Sport = 'keirin';
  const initialMeet = defaultRaceVenue(initialSport);
  const initialToday = todayInKorea();
  const initialDate = nearestRaceDate(initialSport, initialMeet, initialToday);
  const [activeTab, setActiveTab] = useState<TabKey>('home');
  const [sport, setSport] = useState<Sport>(initialSport);
  const [analysisDate, setAnalysisDate] = useState(initialDate);
  const [todayKey, setTodayKey] = useState(initialToday);
  const [meet, setMeet] = useState(initialMeet);
  const [raceNo, setRaceNo] = useState(1);
  const [decision, setDecision] = useState<RaceDecision | null>(null);
  const activeDecision = decision && decision.sport === sport && decision.raceNo === raceNo && decision.meet === meet && decision.date === analysisDate ? decision : null;
  const [appSession, setAppSession] = useState<AppSession>(demoAppSession);
  const [sessionDataLayer, setSessionDataLayer] = useState<DataLayerStatus>(demoDataLayer);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [freeAnalysisUsed, setFreeAnalysisUsed] = useState(0);
  const [adGateVisible, setAdGateVisible] = useState(false);
  const [rewardingAd, setRewardingAd] = useState(false);
  const [raceSchedule, setRaceSchedule] = useState<RaceDateSchedule>({});
  const [raceCount, setRaceCount] = useState(defaultRaceCount(initialSport, initialMeet));
  const [calendarRefreshToken, setCalendarRefreshToken] = useState(0);
  const userPinnedDateRef = useRef(false);
  const userPinnedRaceRef = useRef(false);
  const analyzeInFlightRef = useRef(false);
  const raceDates = useMemo(
    () => availableRaceDates(sport, meet, raceSchedule),
    [meet, raceSchedule, sport, todayKey]
  );

  useEffect(() => {
    if (typeof document === 'undefined') return undefined;
    const styleId = 'racelens-korean-text-break';
    if (document.getElementById(styleId)) return undefined;
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = '* { word-break: keep-all; overflow-wrap: anywhere; }';
    document.head.appendChild(style);
    return () => {
      style.remove();
    };
  }, []);

  const params = useMemo(() => ({
    sport,
    date: analysisDate,
    meet,
    raceNo
  }), [analysisDate, meet, raceNo, sport]);

  function runAnalyze() {
    const scheduledDate = nearestRaceDate(sport, meet, analysisDate, raceSchedule);
    if (scheduledDate !== analysisDate) {
      userPinnedDateRef.current = false;
      setAnalysisDate(scheduledDate);
      setError('경기 일정이 있는 날짜로 자동 보정했습니다.');
      return;
    }
    const effectiveRemaining = currentDecision.dataLayer.ready
      ? currentDecision.appSession.freeAnalysisRemaining
      : Math.max(0, freeAnalysisLimit - freeAnalysisUsed);
    const rewardedCredits = currentDecision.dataLayer.ready ? currentDecision.appSession.rewardedAnalysisCredits : 0;
    if (currentDecision.appSession.entitlement !== 'pro' && effectiveRemaining <= 0 && rewardedCredits <= 0) {
      if (rewardAdsEnabled) {
        setError('');
        setAdGateVisible(true);
        return;
      }
      setError('오늘 무료 분석 한도를 모두 사용했습니다. Pro 준비 화면에서 한도 안내를 확인하세요.');
      setActiveTab('pro');
      return;
    }
    void executeAnalyze();
  }

  async function confirmAdGate() {
    if (rewardingAd) return;
    if (!rewardAdsEnabled) {
      setAdGateVisible(false);
      setError('오늘 무료 분석 한도를 모두 사용했습니다. Pro 준비 화면에서 한도 안내를 확인하세요.');
      setActiveTab('pro');
      return;
    }
    setRewardingAd(true);
    setError('');
    try {
      const adResult = await showRewardedAd();
      if (adResult.status === 'dismissed') {
        setError('광고를 끝까지 보면 분석 1회가 추가됩니다.');
        return;
      }
      if (adResult.status === 'unavailable') {
        setError(adResult.message);
        return;
      }
      let verified = false;
      for (let attempt = 0; attempt < 15; attempt += 1) {
        const session = await fetchAppSession();
        setAppSession(session.appSession);
        setSessionDataLayer(session.dataLayer);
        if (session.appSession.rewardedAnalysisCredits > 0) {
          verified = true;
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 1_000));
      }
      if (!verified) {
        setError('광고 시청은 확인됐지만 서버 보상 확인이 지연되고 있습니다. 잠시 후 다시 분석해 주세요.');
        return;
      }
      setAdGateVisible(false);
      trackUxEvent('rewarded_ad_credit', { raceNo, sport, tab: activeTab });
      void executeAnalyze();
    } catch (err) {
      recordFirebaseError(err, 'rewarded_ad_claim');
      setError(err instanceof Error ? err.message : '광고 보상 확인에 실패했습니다.');
    } finally {
      setRewardingAd(false);
    }
  }

  async function executeAnalyze() {
    if (analyzeInFlightRef.current) return;
    analyzeInFlightRef.current = true;
    const startedAt = Date.now();
    trackUxEvent('analysis_request', { raceNo, sport, tab: activeTab });
    setLoading(true);
    setError('');
    try {
      const next = await fetchRaceDecision(params);
      setDecision(next);
      setAppSession(next.appSession);
      setSessionDataLayer(next.dataLayer);
      if (next.dataLayer.ready) {
        setFreeAnalysisUsed(next.appSession.freeAnalysisUsed);
      } else if (next.appSession.entitlement !== 'pro' && next.status !== 'blocked') {
        setFreeAnalysisUsed((used) => Math.min(freeAnalysisLimit, used + 1));
      }
      setActiveTab('analyze');
      trackUxEvent('analysis_result', {
        latencyMs: Date.now() - startedAt,
        marketRiskLevel: next.marketRisk.level,
        marketUsed: next.marketUsed,
        raceNo: next.raceNo,
        sport: next.sport,
        tab: 'analyze',
        top1Pct: next.confidence.top1 * 100,
        trifectaPct: next.confidence.trifecta * 100
      });
    } catch (err) {
      recordFirebaseError(err, 'analysis_request');
      setError(err instanceof Error ? err.message : 'RaceLens API error');
      trackUxEvent('analysis_error', {
        errorKind: err instanceof Error ? 'api_error' : 'unknown',
        latencyMs: Date.now() - startedAt,
        raceNo,
        sport,
        tab: activeTab
      });
    } finally {
      analyzeInFlightRef.current = false;
      setLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab !== 'analyze' || !activeDecision || activeDecision.appSession.entitlement !== 'pro') {
      return undefined;
    }
    let cancelled = false;
    const refreshMs = Math.max(3000, Math.min(60000, activeDecision.pollDelayMs || 15000));
    const timerId = setTimeout(() => {
      void fetchRaceDecision(params).then((next) => {
        if (cancelled) return;
        setDecision(next);
        setAppSession(next.appSession);
        setSessionDataLayer(next.dataLayer);
        trackUxEvent('live_odds_refresh', {
          marketRiskLevel: next.marketRisk.level,
          marketUsed: next.marketUsed,
          pollDelayMs: refreshMs,
          raceNo: next.raceNo,
          sport: next.sport,
          tab: 'analyze'
        });
      }).catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'RaceLens live refresh error');
      });
    }, refreshMs);
    return () => {
      cancelled = true;
      clearTimeout(timerId);
    };
  }, [activeDecision?.pollDelayMs, activeDecision?.updatedAt, activeDecision?.appSession.entitlement, activeTab, params]);

  useEffect(() => {
    void fetchAppSession().then((session) => {
      setAppSession(session.appSession);
      setSessionDataLayer(session.dataLayer);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void fetchRaceDates({ sport, meet }).then((calendar) => {
      if (cancelled) return;
      const days = calendar.days;
      setRaceCount(calendar.raceCount);
      if (!userPinnedRaceRef.current) {
        setRaceNo(calendar.defaultRaceNo);
      }
      if (days.length === 0) return;
      const refreshedToday = todayInKorea();
      const selectionSchedule: RaceDateSchedule = {
        [sport]: {
          [meet]: days
        }
      };
      setRaceSchedule((current) => {
        return {
          ...current,
          [sport]: {
            ...(current[sport] ?? {}),
            [meet]: days
          }
        };
      });
      setTodayKey(refreshedToday);
      setAnalysisDate((currentDate) => nearestRaceDate(
        sport,
        meet,
        userPinnedDateRef.current ? currentDate : refreshedToday,
        selectionSchedule
      ));
    });
    return () => {
      cancelled = true;
    };
  }, [calendarRefreshToken, meet, sport]);

  function refreshCalendarFromClock() {
    const refreshedToday = todayInKorea();
    setTodayKey(refreshedToday);
    setCalendarRefreshToken((token) => token + 1);
    setAnalysisDate((currentDate) => nearestRaceDate(
      sport,
      meet,
      userPinnedDateRef.current ? currentDate : refreshedToday,
      raceSchedule
    ));
  }

  useEffect(() => {
    const intervalId = setInterval(refreshCalendarFromClock, calendarRefreshMs);
    return () => clearInterval(intervalId);
  }, [meet, raceSchedule, sport]);

  useEffect(() => {
    const appStateSubscription = AppState.addEventListener('change', (state) => {
      if (state === 'active') {
        refreshCalendarFromClock();
      }
    });
    const onFocus = () => refreshCalendarFromClock();
    const onVisibilityChange = () => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        refreshCalendarFromClock();
      }
    };
    if (typeof window !== 'undefined') {
      window.addEventListener('focus', onFocus);
    }
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibilityChange);
    }
    return () => {
      appStateSubscription.remove();
      if (typeof window !== 'undefined') {
        window.removeEventListener('focus', onFocus);
      }
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisibilityChange);
      }
    };
  }, [meet, raceSchedule, sport]);

  useEffect(() => {
    trackUxEvent('app_open', { raceNo, sport, tab: activeTab });
  }, []);

  useEffect(() => {
    trackUxEvent('screen_view', { raceNo, sport, tab: activeTab });
  }, [activeTab, raceNo, sport]);

  function changeTab(tab: TabKey) {
    trackUxEvent('tab_select', { previousTab: activeTab, raceNo, sport, tab });
    setActiveTab(tab);
  }

  function changeSport(nextSport: Sport) {
    const nextMeet = defaultRaceVenue(nextSport);
    const preferredDate = userPinnedDateRef.current ? analysisDate : todayInKorea();
    const nextDate = nearestRaceDate(nextSport, nextMeet, preferredDate, raceSchedule);
    trackUxEvent('race_context_change', { raceNo, sport: nextSport, tab: activeTab });
    userPinnedRaceRef.current = false;
    setSport(nextSport);
    setMeet(nextMeet);
    setAnalysisDate(nextDate);
    setRaceCount(defaultRaceCount(nextSport, nextMeet));
  }

  function changeMeet(nextMeet: string) {
    const preferredDate = userPinnedDateRef.current ? analysisDate : todayInKorea();
    const nextDate = nearestRaceDate(sport, nextMeet, preferredDate, raceSchedule);
    trackUxEvent('race_context_change', { raceNo, sport, tab: activeTab });
    userPinnedRaceRef.current = false;
    setMeet(nextMeet);
    setAnalysisDate(nextDate);
    setRaceCount(defaultRaceCount(sport, nextMeet));
  }

  function changeRaceNo(nextRaceNo: number) {
    trackUxEvent('race_context_change', { raceNo: nextRaceNo, sport, tab: activeTab });
    userPinnedRaceRef.current = true;
    setRaceNo(nextRaceNo);
  }

  function changeDate(nextDate: string) {
    userPinnedDateRef.current = true;
    userPinnedRaceRef.current = false;
    trackUxEvent('race_context_change', { raceNo, sport, tab: activeTab });
    setAnalysisDate(nextDate);
  }

  const currentDecision = activeDecision ?? {
    status: 'hold',
    sport,
    date: params.date,
    meet,
    raceNo,
    headline: '분석 조건 선택 대기',
    marketUsed: false,
    marketSource: 'unavailable' as const,
    marketRisk: {
      level: 'neutral' as const,
      title: '분석 전',
      message: '모델 신호 보기를 누르면 선택한 경기의 공식 출전표와 배당을 조회합니다.'
    },
    confidence: {
      label: '대기',
      top1: 0,
      trifecta: 0,
      sample: 0
    },
    picks: [],
    participants: [],
    marketOdds: [],
    rosterVerification: {
      state: 'unverified' as const,
      message: '공식 대조 미완료'
    },
    dataLayer: {
      ...sessionDataLayer
    },
    appSession: sessionDataLayer.ready ? appSession : {
      ...appSession,
      freeAnalysisUsed,
      freeAnalysisRemaining: Math.max(0, freeAnalysisLimit - freeAnalysisUsed),
      rewardedAnalysisCredits: 0
    },
    analysisError: false,
    officialDataPending: false,
    pollDelayMs: 60000,
    updatedAt: new Date().toISOString(),
    oddsAgeSec: null
  };
  const displayedLimit = currentDecision.appSession.freeAnalysisLimit || freeAnalysisLimit;
  const displayedUsed = currentDecision.dataLayer.ready
    ? currentDecision.appSession.freeAnalysisUsed
    : freeAnalysisUsed;
  const displayedRewardedCredits = currentDecision.dataLayer.ready
    ? currentDecision.appSession.rewardedAnalysisCredits
    : 0;

  return (
    <SafeAreaProvider>
      <StatusBar style={mode === 'dark' ? 'light' : 'dark'} />
      <SafeAreaView style={[styles.safe, { backgroundColor: colors.surfaceBase }]}>
        <LinearGradient
          colors={gradients.app}
          style={StyleSheet.absoluteFill}
        />
        {activeTab === 'home' && (
          <FadeInUp distance={0} duration={180} style={styles.tabContent}>
            <HomeScreen
              decision={currentDecision}
              freeAnalysisLimit={displayedLimit}
              freeAnalysisUsed={displayedUsed}
              mode={mode}
              raceDates={raceDates}
              raceCount={raceCount}
              raceNo={raceNo}
              rewardedAnalysisCredits={displayedRewardedCredits}
              rewardAdsEnabled={rewardAdsEnabled}
              sport={sport}
              onAnalyze={runAnalyze}
              onDateChange={changeDate}
              onMeetChange={changeMeet}
              onRaceChange={changeRaceNo}
              onSportChange={changeSport}
            />
          </FadeInUp>
        )}
        {activeTab === 'analyze' && (
          <FadeInUp distance={0} duration={180} style={styles.tabContent}>
            <AnalyzeScreen
              decision={currentDecision}
              freeAnalysisLimit={displayedLimit}
              freeAnalysisUsed={displayedUsed}
              mode={mode}
              onChooseRace={() => changeTab('home')}
              rewardAdsEnabled={rewardAdsEnabled}
              rewardedAnalysisCredits={displayedRewardedCredits}
              onRetry={() => {
                void executeAnalyze();
              }}
              onViewPro={() => changeTab('pro')}
            />
          </FadeInUp>
        )}
        {activeTab === 'pro' && (
          <FadeInUp distance={0} duration={180} style={styles.tabContent}>
            <ProScreen
              decision={currentDecision}
              freeAnalysisLimit={displayedLimit}
              freeAnalysisUsed={displayedUsed}
              mode={mode}
            />
          </FadeInUp>
        )}
        {error ? (
          <View style={[styles.toast, { backgroundColor: colors.surfaceRaised, borderColor: colors.accentRose }]}>
            <Text style={[styles.toastText, { color: colors.textPrimary }]}>{error}</Text>
          </View>
        ) : null}
        {loading ? (
          <View
            accessibilityLabel="분석 요청 처리 중"
            accessibilityRole="progressbar"
            testID="analysis-loading"
            style={[styles.loading, { backgroundColor: colors.surfaceGlass }]}
          >
            <ActivityIndicator color={colors.accentPrimary} />
          </View>
        ) : null}
        {adGateVisible && rewardAdsEnabled ? (
          <FreeAdGate
            mode={mode}
            confirming={rewardingAd}
            preview={rewardAdPreview}
            onCancel={() => setAdGateVisible(false)}
            onConfirm={confirmAdGate}
          />
        ) : null}
        <BottomTabs active={activeTab} meet={meet} mode={mode} raceNo={raceNo} sport={sport} onChange={changeTab} />
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

function FreeAdGate({
  mode,
  confirming,
  preview,
  onCancel,
  onConfirm
}: {
  mode: ThemeMode;
  confirming: boolean;
  preview: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const colors = palette(mode);
  return (
    <View
      accessibilityRole="summary"
      style={[styles.adOverlay, { backgroundColor: colors.overlayScrim }]}
    >
      <View style={[styles.adPanel, { backgroundColor: colors.surfaceRaised, borderColor: colors.borderSubtle }]}>
        <Text style={[styles.adEyebrow, { color: colors.accentGold }]}>
          {preview ? 'PREVIEW TEST AD' : '무료 플랜 광고'}
        </Text>
        <Text style={[styles.adTitle, { color: colors.textPrimary }]}>광고 보고 분석 1회 추가</Text>
        <Text style={[styles.adBody, { color: colors.textSecondary }]}>
          무료 3회 이후에는 광고를 볼 때마다 분석 1회 이용권을 충전합니다. 베팅 이동과 도박성 광고는 차단합니다.
        </Text>
        <View style={[styles.adCreative, { backgroundColor: colors.surfaceInset, borderColor: colors.borderSubtle }]}>
          <Text style={[styles.adCreativeLabel, { color: colors.textMuted }]}>AD</Text>
          <Text style={[styles.adCreativeTitle, { color: colors.textPrimary }]}>스포츠 데이터 리포트</Text>
          <Text style={[styles.adCreativeCopy, { color: colors.textSecondary }]}>보상 확인 후 분석 화면이 바로 열립니다.</Text>
        </View>
        <PressableScale
          accessibilityRole="button"
          accessibilityLabel="광고 보고 분석 1회 추가"
          testID="rewarded-ad-confirm"
          disabled={confirming}
          onPress={onConfirm}
          style={[
            styles.adConfirm,
            {
              backgroundColor: colors.accentPrimary,
              opacity: confirming ? 0.88 : 1
            }
          ]}
        >
          <Text style={[styles.adConfirmText, { color: colors.textOnBoard }]}>
            {confirming ? '광고 확인 중...' : preview ? '테스트 광고 보고 1회 추가' : '광고 보고 1회 추가'}
          </Text>
        </PressableScale>
        <PressableScale
          accessibilityRole="button"
          accessibilityLabel="광고 닫기"
          onPress={onCancel}
          style={styles.adCancel}
        >
          <Text style={[styles.adCancelText, { color: colors.textMuted }]}>닫기</Text>
        </PressableScale>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1
  },
  tabContent: {
    flex: 1
  },
  toast: {
    borderRadius: 16,
    borderWidth: 1,
    bottom: 108,
    left: space.space5,
    padding: space.space4,
    position: 'absolute',
    right: space.space5
  },
  toastText: {
    ...typography.bodySm
  },
  loading: {
    alignItems: 'center',
    borderRadius: 999,
    bottom: 112,
    height: 48,
    justifyContent: 'center',
    position: 'absolute',
    right: space.space5,
    width: 48
  },
  adOverlay: {
    alignItems: 'center',
    bottom: 0,
    justifyContent: 'center',
    left: 0,
    padding: space.space5,
    position: 'absolute',
    right: 0,
    top: 0,
    zIndex: 20
  },
  adPanel: {
    borderRadius: 22,
    borderWidth: 1,
    gap: space.space3,
    maxWidth: 420,
    padding: space.space5,
    width: '100%'
  },
  adEyebrow: {
    ...typography.caption
  },
  adTitle: {
    ...typography.h2
  },
  adBody: {
    ...typography.bodySm
  },
  adCreative: {
    borderRadius: 16,
    borderWidth: 1,
    gap: space.space1,
    padding: space.space4
  },
  adCreativeLabel: {
    ...typography.mono
  },
  adCreativeTitle: {
    ...typography.bodyStrong
  },
  adCreativeCopy: {
    ...typography.bodySm
  },
  adConfirm: {
    alignItems: 'center',
    borderRadius: 999,
    justifyContent: 'center',
    minHeight: 48,
    paddingHorizontal: space.space4,
    paddingVertical: space.space3
  },
  adConfirmText: {
    ...typography.bodyStrong
  },
  adCancel: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44
  },
  adCancelText: {
    ...typography.bodyStrong
  }
});
