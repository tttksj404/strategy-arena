import Constants from 'expo-constants';

declare const process: { env?: Record<string, string | undefined> } | undefined;

export type PurchaseResult = {
  readonly ok: boolean;
  readonly reason?: 'not_configured' | 'store_unavailable' | 'cancelled' | 'failed';
};

export type AdSlot = {
  readonly placement: 'free_analysis_gate';
  readonly enabled: boolean;
};

function monetizationFlagValue() {
  const env = typeof process === 'undefined' ? undefined : process.env;
  const billingMode = env?.EXPO_PUBLIC_RACELENS_BILLING_MODE ?? env?.RACELENS_BILLING_MODE;
  if (billingMode && billingMode.trim().toLowerCase() === 'disabled') return 'disabled';
  const envValue = env?.EXPO_PUBLIC_MONETIZATION;
  const extraValue = Constants.expoConfig?.extra?.billingMode ?? Constants.expoConfig?.extra?.monetization;
  return String(envValue ?? extraValue ?? '').trim().toLowerCase();
}

export function isMonetizationEnabled() {
  return ['1', 'true', 'yes', 'on'].includes(monetizationFlagValue());
}

function rewardedAdsFlagValue() {
  const env = typeof process === 'undefined' ? undefined : process.env;
  const envValue = env?.EXPO_PUBLIC_RACELENS_REWARDED_ADS;
  const extraValue = Constants.expoConfig?.extra?.rewardedAds;
  return String(envValue ?? extraValue ?? '').trim().toLowerCase();
}

export function isRewardedAdsEnabled() {
  return ['1', 'true', 'yes', 'on'].includes(rewardedAdsFlagValue());
}

export async function purchasePro(): Promise<PurchaseResult> {
  if (!isMonetizationEnabled()) {
    return { ok: false, reason: 'not_configured' };
  }
  return { ok: false, reason: 'store_unavailable' };
}

export async function restorePurchases(): Promise<PurchaseResult> {
  if (!isMonetizationEnabled()) {
    return { ok: false, reason: 'not_configured' };
  }
  return { ok: false, reason: 'store_unavailable' };
}

export function adSlot(placement: AdSlot['placement']): AdSlot | null {
  if (!isRewardedAdsEnabled()) return null;
  return { placement, enabled: true };
}
