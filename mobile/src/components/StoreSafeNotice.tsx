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
        <Ionicons name="shield-checkmark-outline" size={22} color={colors.accentAmber} />
        <View style={styles.copy}>
          <Text style={[styles.title, { color: colors.textPrimary }]}>정보 분석 도구</Text>
          <Text style={[styles.body, { color: colors.textSecondary }]}>
            {compact
              ? '구매·베팅 연결 없이 모델 신호와 검증 상태만 제공합니다.'
              : 'RaceLens는 경주 데이터 분석과 복기용 정보 앱입니다. 구매 금액, 수익 보장, 베팅 연결을 제공하지 않으며 만 19세 이상 책임 이용을 전제로 합니다.'}
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
