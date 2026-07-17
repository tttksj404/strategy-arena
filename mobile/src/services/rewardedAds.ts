import type { RewardedAdResult } from './rewardedAds.types';

export type { RewardedAdResult } from './rewardedAds.types';

export function isRewardedAdPreview() {
  return false;
}

export async function showRewardedAd(): Promise<RewardedAdResult> {
  return { status: 'unavailable', message: '이 환경에서는 보상형 광고를 실행할 수 없습니다.' };
}
