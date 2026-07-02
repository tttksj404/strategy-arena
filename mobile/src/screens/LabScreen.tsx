import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { LensCard } from '../components/LensCard';
import { ProbabilityRail } from '../components/ProbabilityRail';
import { StatusPill } from '../components/StatusPill';
import { palette, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';

type LabScreenProps = {
  mode: ThemeMode;
};

const experiments = [
  { label: '시장 배당 반영', value: 0.63, status: '조건부 상승' },
  { label: '공식 예상지 교차', value: 0.61, status: '검증 유지' },
  { label: '삼쌍 탐색 후보', value: 0.22, status: '표본 부족' }
];

export function LabScreen({ mode }: LabScreenProps) {
  const colors = palette(mode);
  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <StatusPill mode={mode} level="pro" label="MODEL LAB" />
        <Text style={[styles.title, { color: colors.textPrimary }]}>검증 랩</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          새 알고리즘은 여기서 표본, 연도별 홀드아웃, 리스크 플래그를 통과해야만 본 화면에 올라옵니다.
        </Text>
      </View>

      {experiments.map((item) => (
        <LensCard key={item.label} mode={mode} variant={item.value >= 0.5 ? 'verified' : 'caution'}>
          <View style={styles.experimentTop}>
            <Text style={[styles.experimentTitle, { color: colors.textPrimary }]}>{item.label}</Text>
            <Text style={[styles.experimentStatus, { color: colors.textMuted }]}>{item.status}</Text>
          </View>
          <ProbabilityRail mode={mode} label="최근 검증값" value={item.value} tone={item.value >= 0.5 ? 'teal' : 'amber'} />
        </LensCard>
      ))}

      <LensCard mode={mode} variant="pro">
        <Text style={[styles.experimentTitle, { color: colors.textPrimary }]}>스토어 안전 원칙</Text>
        <Text style={[styles.body, { color: colors.textSecondary }]}>
          이 앱은 베팅 금액, 구매 링크, 수익 보장 문구를 넣지 않습니다. 분석 결과는 정보이며, 고확신 상태도 검증 수치와 함께만 표시됩니다.
        </Text>
      </LensCard>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    gap: space.space5,
    padding: space.space5,
    paddingBottom: 120
  },
  header: {
    gap: space.space3,
    paddingTop: space.space4
  },
  title: {
    ...typography.h1
  },
  subtitle: {
    ...typography.body
  },
  experimentTop: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: space.space4
  },
  experimentTitle: {
    ...typography.h3
  },
  experimentStatus: {
    ...typography.caption
  },
  body: {
    ...typography.bodySm,
    marginTop: space.space3
  }
});
