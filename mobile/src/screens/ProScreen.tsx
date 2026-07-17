import Constants from 'expo-constants';
import { Ionicons } from '@expo/vector-icons';
import { Linking, ScrollView, StyleSheet, Text, View } from 'react-native';

import { LensCard } from '../components/LensCard';
import { PressableScale } from '../components/PressableScale';
import { StatusPill } from '../components/StatusPill';
import { StoreSafeNotice } from '../components/StoreSafeNotice';
import { isMonetizationEnabled, purchasePro, restorePurchases } from '../services/monetization';
import { palette, radius, space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import type { RaceDecision } from '../types/race';

type ProScreenProps = {
  mode: ThemeMode;
  decision: RaceDecision;
  freeAnalysisLimit: number;
  freeAnalysisUsed: number;
};

type LegalLink = {
  label: string;
  value?: string;
};

const freeFeatures = [
  '기본 분석 순서와 확률 요약',
  '선수 및 출전 정보 확인',
  '무료 분석 횟수 제공',
  '고급 지표는 출시 후 단계적으로 제공'
];

const proPreviewFeatures = [
  '분석 횟수 확대',
  '마감 전 데이터 갱신',
  '과거 유사 경주 비교 리포트',
  '모델 검증 로그 열람'
];

const enabledProFeatures = [
  '분석 횟수 확대',
  '고급 데이터 갱신',
  '과거 유사 경주 리포트',
  '모델 검증 로그 열람'
];

const proExperience = [
  ['분석 횟수', '확대 제공'],
  ['화면 흐름', '간결한 분석 중심'],
  ['고급 지표', '출시 후 순차 제공'],
  ['검증 로그', '전체 열람']
];

export function ProScreen({ mode, decision, freeAnalysisLimit, freeAnalysisUsed }: ProScreenProps) {
  const colors = palette(mode);
  const proActive = decision.appSession.entitlement === 'pro';
  const monetizationEnabled = isMonetizationEnabled();
  const purchaseDisabled = !monetizationEnabled || proActive;
  const purchaseBackground = purchaseDisabled ? colors.surfaceInset : colors.accentViolet;
  const purchaseForeground = purchaseDisabled ? colors.textMuted : mode === 'light' ? colors.textOnBoard : colors.surfaceBase;
  const safeUsed = Math.min(freeAnalysisLimit, freeAnalysisUsed);
  const freeRemaining = Math.max(0, freeAnalysisLimit - safeUsed);
  const proFeatures = monetizationEnabled ? enabledProFeatures : proPreviewFeatures;
  const legalLinks = legalLinkItems();
  const freeExperience = [
    ['분석 횟수', `오늘 ${safeUsed}/${freeAnalysisLimit} 사용`],
    ['남은 횟수', `${freeRemaining}회`],
    ['데이터', '공식 출전표 확인 후 표시'],
    ['고급 지표', '출시 후 제공 예정']
  ];

  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false} style={styles.scroll}>
      <View style={styles.header}>
        <StatusPill mode={mode} level={proActive ? 'verified' : 'neutral'} label={proActive ? 'PRO ACTIVE' : 'FREE PLAN'} />
        <Text style={[styles.title, { color: colors.textPrimary }]}>RaceLens Pro</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          Pro 기능은 출시 준비 중입니다. 지금은 무료 분석을 그대로 이용할 수 있습니다.
        </Text>
      </View>

      <LensCard mode={mode} variant={proActive ? 'pro' : 'caution'}>
        <View style={styles.planHeader}>
          <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>현재 이용 상태</Text>
          <StatusPill mode={mode} level={proActive ? 'pro' : 'neutral'} label={proActive ? 'Pro 이용 중' : '무료 이용 중'} />
        </View>
        <View style={styles.experienceGrid}>
          {(proActive ? proExperience : freeExperience).map(([label, value]) => (
            <View key={label} style={[styles.experienceCell, { borderColor: colors.borderSubtle, backgroundColor: colors.surfaceInset }]}>
              <Text style={[styles.experienceLabel, { color: colors.textMuted }]}>{label}</Text>
              <Text style={[styles.experienceValue, { color: colors.textPrimary }]}>{value}</Text>
            </View>
          ))}
        </View>
      </LensCard>

      <View style={styles.planGrid}>
        <LensCard mode={mode} variant={proActive ? 'neutral' : 'verified'}>
          <View style={styles.planHeader}>
            <Text style={[styles.planName, { color: colors.textPrimary }]}>무료 플랜</Text>
            {!proActive ? <StatusPill mode={mode} level="verified" label="현재 이용 중" /> : null}
          </View>
          <Text style={[styles.price, { color: colors.textSecondary }]}>0원</Text>
          <Text style={[styles.caption, { color: colors.textSecondary }]}>공식 데이터 확인 후 기본 분석을 제공합니다.</Text>
          <View style={styles.benefits}>
            {freeFeatures.map((feature) => (
              <FeatureRow key={feature} mode={mode} label={feature} locked={feature.includes('출시 후')} />
            ))}
          </View>
        </LensCard>

        <LensCard mode={mode} variant="pro">
          <View style={styles.planHeader}>
            <Text style={[styles.planName, { color: colors.textPrimary }]}>Pro 플랜</Text>
            {proActive ? <StatusPill mode={mode} level="pro" label="활성" /> : <StatusPill mode={mode} level="pro" label={monetizationEnabled ? '구독 가능' : '준비 중'} />}
          </View>
          {monetizationEnabled ? (
            <>
              <Text style={[styles.price, { color: colors.textSecondary }]}>월 5,000원</Text>
              <Text style={[styles.caption, { color: colors.textSecondary }]}>고급 분석 기능을 순차적으로 제공합니다.</Text>
            </>
          ) : (
            <Text style={[styles.caption, { color: colors.textSecondary }]}>Pro 기능은 출시 준비 중입니다. 지금은 무료 분석을 그대로 이용할 수 있습니다.</Text>
          )}
          <View style={styles.benefits}>
            {proFeatures.map((feature) => (
              <FeatureRow key={feature} mode={mode} label={feature} locked={!monetizationEnabled && !proActive} />
            ))}
          </View>
          {monetizationEnabled ? (
            <View style={styles.ctaGroup}>
              <PressableScale
                accessibilityLabel={proActive ? 'Pro 이용 중' : 'Pro 구독 시작'}
                accessibilityRole="button"
                accessibilityState={{ disabled: purchaseDisabled }}
                disabled={purchaseDisabled}
                onPress={() => {
                  if (!purchaseDisabled) void purchasePro();
                }}
                style={[
                  styles.ctaButton,
                  {
                    backgroundColor: purchaseBackground,
                    borderColor: purchaseDisabled ? colors.borderSubtle : colors.accentViolet,
                    opacity: purchaseDisabled ? 0.82 : 1
                  }
                ]}
                testID="pro-purchase-cta"
              >
                <Text style={[styles.ctaText, { color: purchaseForeground }]}>
                  {proActive ? 'Pro 이용 중' : 'Pro 구독 시작'}
                </Text>
              </PressableScale>
              <PressableScale
                accessibilityLabel="구매 복원"
                accessibilityRole="button"
                onPress={() => {
                  void restorePurchases();
                }}
                style={styles.restoreButton}
                testID="pro-restore-cta"
              >
                <Text style={[styles.restoreText, { color: colors.textSecondary }]}>구매 복원</Text>
              </PressableScale>
            </View>
          ) : null}
        </LensCard>
      </View>

      <LensCard mode={mode} variant={decision.appSession.entitlement === 'pro' ? 'verified' : 'neutral'}>
        <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>계정 상태</Text>
        <Text style={[styles.caption, { color: colors.textSecondary }]}>
          현재 기기의 이용 권한과 데이터 준비 상태를 확인합니다. Pro 구독이 제공되면 스토어 검증 결과를 기준으로 권한을 갱신합니다.
        </Text>
        <Text style={[styles.mono, { color: colors.textMuted }]}>
          {proActive ? 'Pro 이용 중' : '무료 이용 중'} · {decision.dataLayer.ready ? '공식 데이터 확인 가능' : '예시 데이터 대기'}
        </Text>
      </LensCard>

      {legalLinks.length > 0 ? (
        <LensCard mode={mode}>
          <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>정보 및 문의</Text>
          <View style={styles.legalLinks}>
            {legalLinks.map((item) => (
              <PressableScale
                accessibilityLabel={`${item.label} 열기`}
                accessibilityRole="link"
                key={item.label}
                onPress={() => {
                  if (item.value) void Linking.openURL(item.value);
                }}
                style={[styles.legalLink, { borderColor: colors.borderSubtle }]}
              >
                <Text style={[styles.legalLinkText, { color: colors.textPrimary }]}>{item.label}</Text>
                <Ionicons
                  accessibilityElementsHidden
                  accessible={false}
                  importantForAccessibility="no-hide-descendants"
                  name="open-outline"
                  size={18}
                  color={colors.textSecondary}
                />
              </PressableScale>
            ))}
          </View>
        </LensCard>
      ) : null}

      <StoreSafeNotice mode={mode} />
    </ScrollView>
  );
}

function legalLinkItems(): LegalLink[] {
  const extra = Constants.expoConfig?.extra ?? {};
  const supportEmail = safeExtra(extra.supportEmail);
  const supportUrl = safeExtra(extra.supportUrl);
  return [
    { label: '개인정보처리방침', value: safeExtra(extra.privacyPolicyUrl) },
    { label: '이용약관', value: safeExtra(extra.termsUrl) },
    { label: '계정 삭제 안내', value: safeExtra(extra.accountDeletionUrl) },
    { label: '지원 문의', value: supportUrl ?? (supportEmail ? `mailto:${supportEmail}` : undefined) }
  ].filter((item): item is LegalLink & { value: string } => Boolean(item.value));
}

function safeExtra(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function FeatureRow({ mode, label, locked }: { mode: ThemeMode; label: string; locked: boolean }) {
  const colors = palette(mode);
  return (
    <View style={styles.benefitRow}>
      <Ionicons
        accessibilityElementsHidden
        accessible={false}
        importantForAccessibility="no-hide-descendants"
        name={locked ? 'lock-closed-outline' : 'checkmark-circle-outline'}
        size={20}
        color={locked ? colors.textMuted : colors.accentTeal}
      />
      <Text style={[styles.benefit, { color: locked ? colors.textSecondary : colors.textPrimary }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flex: 1
  },
  content: {
    gap: space.space4,
    padding: space.space5,
    paddingBottom: 120
  },
  header: {
    gap: space.space2,
    paddingTop: space.space2
  },
  title: {
    ...typography.h1
  },
  subtitle: {
    ...typography.body
  },
  planGrid: {
    gap: space.space4
  },
  planHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: space.space3
  },
  planName: {
    ...typography.h3
  },
  price: {
    ...typography.h2,
    marginTop: space.space2
  },
  caption: {
    ...typography.bodySm,
    marginTop: space.space2
  },
  benefits: {
    gap: space.space2,
    marginTop: space.space3
  },
  ctaGroup: {
    gap: space.space2,
    marginTop: space.space3
  },
  ctaButton: {
    alignItems: 'center',
    borderRadius: radius.pill,
    borderWidth: 1,
    justifyContent: 'center',
    minHeight: 48,
    paddingHorizontal: space.space4
  },
  ctaText: {
    ...typography.bodyStrong
  },
  restoreButton: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44
  },
  restoreText: {
    ...typography.bodyStrong
  },
  benefitRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: space.space3
  },
  benefit: {
    ...typography.body
  },
  experienceGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: space.space2,
    marginTop: space.space3
  },
  experienceCell: {
    borderRadius: radius.small,
    borderWidth: 1,
    flexBasis: '48%',
    flexGrow: 1,
    minHeight: 64,
    padding: space.space2
  },
  experienceLabel: {
    ...typography.caption,
    marginBottom: space.space1
  },
  experienceValue: {
    ...typography.bodyStrong
  },
  sectionTitle: {
    ...typography.h3
  },
  mono: {
    ...typography.mono,
    marginTop: space.space3
  },
  legalLinks: {
    gap: space.space2,
    marginTop: space.space4
  },
  legalLink: {
    alignItems: 'center',
    borderRadius: radius.small,
    borderWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    minHeight: 48,
    paddingHorizontal: space.space3
  },
  legalLinkText: {
    ...typography.bodyStrong
  }
});
