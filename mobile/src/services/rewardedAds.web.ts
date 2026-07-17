import Constants from 'expo-constants';

import { claimRewardedAdCredit } from './raceApi';
import type { RewardedAdResult } from './rewardedAds.types';

function enabled(value: unknown) {
  return ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase());
}

export function isRewardedAdPreview() {
  return enabled(Constants.expoConfig?.extra?.rewardedAdsPreview);
}

export async function showRewardedAd(): Promise<RewardedAdResult> {
  if (!isRewardedAdPreview()) {
    return { status: 'unavailable', message: '실제 보상형 광고는 Android 앱에서만 실행됩니다.' };
  }
  await new Promise((resolve) => setTimeout(resolve, 900));
  const reward = await claimRewardedAdCredit();
  if (!reward.rewardGranted && reward.appSession.rewardedAnalysisCredits <= 0) {
    return { status: 'unavailable', message: '프리뷰 테스트 보상을 확인하지 못했습니다.' };
  }
  return { status: 'earned' };
}
