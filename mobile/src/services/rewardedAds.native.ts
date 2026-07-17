import Constants from 'expo-constants';
import mobileAds, {
  AdEventType,
  RewardedAd,
  RewardedAdEventType,
  TestIds
} from 'react-native-google-mobile-ads';

import { getClientDeviceId } from './deviceIdentity';
import type { RewardedAdResult } from './rewardedAds.types';

let initializePromise: Promise<unknown> | null = null;

function enabled(value: unknown) {
  return ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase());
}

function configuredAdUnitId() {
  return String(Constants.expoConfig?.extra?.rewardedAdUnitId ?? '').trim();
}

function testModeEnabled() {
  return enabled(Constants.expoConfig?.extra?.rewardedAdTestMode);
}

async function initializeAds() {
  initializePromise ??= mobileAds().initialize();
  await initializePromise;
}

export function isRewardedAdPreview() {
  return testModeEnabled();
}

export async function showRewardedAd(): Promise<RewardedAdResult> {
  const testMode = testModeEnabled();
  const adUnitId = testMode ? TestIds.REWARDED : configuredAdUnitId();
  if (!adUnitId) {
    return { status: 'unavailable', message: '보상형 광고 단위가 아직 설정되지 않았습니다.' };
  }

  try {
    await initializeAds();
    const deviceId = await getClientDeviceId();
    const rewarded = RewardedAd.createForAdRequest(adUnitId, {
      requestNonPersonalizedAdsOnly: true,
      serverSideVerificationOptions: {
        userId: deviceId,
        customData: 'quota_gate'
      }
    });

    return await new Promise<RewardedAdResult>((resolve) => {
      let earned = false;
      let settled = false;
      const subscriptions: Array<() => void> = [];
      const finish = (result: RewardedAdResult) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        subscriptions.forEach((unsubscribe) => unsubscribe());
        resolve(result);
      };
      const timeout = setTimeout(
        () => finish({ status: 'unavailable', message: '광고를 불러오는 데 시간이 너무 오래 걸렸습니다.' }),
        90_000
      );
      subscriptions.push(
        rewarded.addAdEventListener(RewardedAdEventType.LOADED, () => {
          void rewarded.show().catch(() => {
            finish({ status: 'unavailable', message: '광고 화면을 열지 못했습니다.' });
          });
        }),
        rewarded.addAdEventListener(RewardedAdEventType.EARNED_REWARD, () => {
          earned = true;
        }),
        rewarded.addAdEventListener(AdEventType.CLOSED, () => {
          finish(earned ? { status: 'earned' } : { status: 'dismissed' });
        }),
        rewarded.addAdEventListener(AdEventType.ERROR, () => {
          finish({ status: 'unavailable', message: '현재 표시할 수 있는 광고가 없습니다. 잠시 후 다시 시도하세요.' });
        })
      );
      rewarded.load();
    });
  } catch {
    return { status: 'unavailable', message: '광고 서비스를 초기화하지 못했습니다.' };
  }
}
