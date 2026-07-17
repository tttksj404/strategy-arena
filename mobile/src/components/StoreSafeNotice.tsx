import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { space, typography } from '../theme/tokens';
import type { ThemeMode } from '../theme/tokens';
import { palette } from '../theme/tokens';
import { LensCard } from './LensCard';

type StoreSafeNoticeProps = {
  mode: ThemeMode;
  compact?: boolean;
};

export function StoreSafeNotice({ mode, compact = false }: StoreSafeNoticeProps) {
  const colors = palette(mode);
  return (
    <LensCard mode={mode} variant="caution">
      <View style={styles.row}>
        <Ionicons
          accessibilityElementsHidden
          accessible={false}
          importantForAccessibility="no-hide-descendants"
          name="shield-checkmark-outline"
          size={22}
          color={colors.accentAmber}
        />
        <View style={styles.copy}>
          <Text style={[styles.title, { color: colors.textPrimary }]}>정보 분석 도구</Text>
          <Text style={[styles.body, { color: colors.textSecondary }]}>
            {compact
              ? '적중률은 수익을 뜻하지 않습니다 · 정보 분석 전용입니다.'
              : '경륜·경마는 공제(약 20~28%) 때문에 아무리 정확해도 장기적으로 평균 손실입니다. RaceLens는 수익 도구가 아니라 경주 데이터를 이해하기 위한 정보 분석 도구이며, 구매·베팅 연결이나 수익 보장을 제공하지 않습니다. 만 19세 이상, 도박 중독에 유의해 책임 있게 이용하세요. 도박문제 상담은 1336에서 안내받을 수 있습니다.'}
          </Text>
        </View>
      </View>
    </LensCard>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    gap: space.space3
  },
  copy: {
    flex: 1,
    gap: space.space1
  },
  title: {
    ...typography.h3
  },
  body: {
    ...typography.bodySm
  }
});
