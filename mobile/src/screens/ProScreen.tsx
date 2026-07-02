import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { LensCard } from '../components/LensCard';
import { StatusPill } from '../components/StatusPill';
import { StoreSafeNotice } from '../components/StoreSafeNotice';
import { palette, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';

type ProScreenProps = {
  mode: ThemeMode;
};

const benefits = [
  '광고 제거 및 무제한 조회',
  '고확신 필터와 마감 직전 갱신',
  '과거 유사 경주와 복기 리포트',
  '모델 랩 후보의 검증 로그'
];

export function ProScreen({ mode }: ProScreenProps) {
  const colors = palette(mode);
  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <StatusPill mode={mode} level="pro" label="SUBSCRIPTION READY" />
        <Text style={[styles.title, { color: colors.textPrimary }]}>RaceLens Pro</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          결제는 스토어 심사 전까지 비활성 상태입니다. 기능 구조만 준비하고, 실제 과금은 앱 심사 정책에 맞춰 켭니다.
        </Text>
      </View>

      <LensCard mode={mode} variant="pro">
        <Text style={[styles.price, { color: colors.textPrimary }]}>월 4,900원</Text>
        <Text style={[styles.caption, { color: colors.textSecondary }]}>예정 가격. 실제 출시 전 A/B 테스트 가능.</Text>
        <View style={styles.benefits}>
          {benefits.map((benefit) => (
            <View key={benefit} style={styles.benefitRow}>
              <Ionicons name="checkmark-circle-outline" size={20} color={colors.accentTeal} />
              <Text style={[styles.benefit, { color: colors.textPrimary }]}>{benefit}</Text>
            </View>
          ))}
        </View>
      </LensCard>

      <LensCard mode={mode}>
        <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>광고 슬롯</Text>
        <Text style={[styles.caption, { color: colors.textSecondary }]}>
          무료 플랜에는 정책 안전 광고만 배치합니다. 도박성 광고, 베팅 사이트 광고, 구매 유도 광고는 차단 대상으로 둡니다.
        </Text>
      </LensCard>

      <StoreSafeNotice mode={mode} />
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
  price: {
    ...typography.display
  },
  caption: {
    ...typography.bodySm,
    marginTop: space.space2
  },
  benefits: {
    gap: space.space3,
    marginTop: space.space5
  },
  benefitRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space3
  },
  benefit: {
    ...typography.body
  },
  sectionTitle: {
    ...typography.h3
  }
});
