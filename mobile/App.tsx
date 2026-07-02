import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, useColorScheme, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { StatusBar } from 'expo-status-bar';

import { BottomTabs } from './src/components/BottomTabs';
import { fetchRaceDecision } from './src/services/raceApi';
import { AnalyzeScreen } from './src/screens/AnalyzeScreen';
import { HomeScreen } from './src/screens/HomeScreen';
import { LabScreen } from './src/screens/LabScreen';
import { ProScreen } from './src/screens/ProScreen';
import { gradient, palette, space, typography } from './src/theme/tokens';
import type { RaceDecision, Sport, TabKey } from './src/types/race';

export default function App() {
  const scheme = useColorScheme();
  const mode = scheme === 'light' ? 'light' : 'dark';
  const colors = palette(mode);
  const gradients = gradient(mode);
  const [activeTab, setActiveTab] = useState<TabKey>('home');
  const [sport, setSport] = useState<Sport>('keirin');
  const [raceNo, setRaceNo] = useState(5);
  const [decision, setDecision] = useState<RaceDecision | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const params = useMemo(() => ({
    sport,
    date: '2026-06-28',
    meet: sport === 'keirin' ? '광명' : '서울',
    raceNo
  }), [raceNo, sport]);

  async function runAnalyze() {
    setLoading(true);
    setError('');
    try {
      const next = await fetchRaceDecision(params);
      setDecision(next);
      setActiveTab('analyze');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'RaceLens API error');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void fetchRaceDecision(params).then(setDecision).catch(() => undefined);
  }, [params]);

  const currentDecision = decision ?? {
    status: 'hold',
    sport,
    date: params.date,
    meet: params.meet,
    raceNo,
    headline: '초기 모델 신호 준비',
    marketUsed: false,
    marketRisk: {
      level: 'neutral' as const,
      title: '데모 모드',
      message: 'API URL이 설정되기 전에는 앱 내 데모 데이터로 화면을 확인합니다.'
    },
    confidence: {
      label: '데모',
      top1: 0.62,
      trifecta: 0.18,
      sample: 10886
    },
    picks: [],
    updatedAt: new Date().toISOString()
  };

  return (
    <SafeAreaProvider>
      <StatusBar style={mode === 'dark' ? 'light' : 'dark'} />
      <SafeAreaView style={[styles.safe, { backgroundColor: colors.surfaceBase }]}>
        <LinearGradient
          colors={gradients.app}
          style={StyleSheet.absoluteFill}
        />
        {activeTab === 'home' && (
          <HomeScreen
            decision={currentDecision}
            mode={mode}
            raceNo={raceNo}
            sport={sport}
            onAnalyze={runAnalyze}
            onRaceChange={setRaceNo}
            onSportChange={setSport}
          />
        )}
        {activeTab === 'analyze' && <AnalyzeScreen decision={currentDecision} mode={mode} />}
        {activeTab === 'lab' && <LabScreen mode={mode} />}
        {activeTab === 'pro' && <ProScreen mode={mode} />}
        {error ? (
          <View style={[styles.toast, { backgroundColor: colors.surfaceRaised, borderColor: colors.accentRose }]}>
            <Text style={[styles.toastText, { color: colors.textPrimary }]}>{error}</Text>
          </View>
        ) : null}
        {loading ? (
          <View style={[styles.loading, { backgroundColor: colors.surfaceGlass }]}>
            <ActivityIndicator color={colors.accentPrimary} />
          </View>
        ) : null}
        <BottomTabs active={activeTab} mode={mode} onChange={setActiveTab} />
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: {
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
  }
});
