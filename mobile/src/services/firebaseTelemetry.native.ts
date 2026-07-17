import Constants from 'expo-constants';
import { Platform } from 'react-native';

import type { AnalyticsEventName, AnalyticsPayload } from './uxAnalytics';

type FirebaseParamValue = string | number | boolean;
type FirebaseParams = Record<string, FirebaseParamValue>;

const enabledValues = ['1', 'true', 'yes', 'on'] as const;
const firebaseEnabledValue = String(Constants.expoConfig?.extra?.firebaseEnabled ?? '').trim().toLowerCase();
const firebaseEnabled = enabledValues.some((value) => value === firebaseEnabledValue);
const nativeFirebaseEnabled = firebaseEnabled && (Platform.OS === 'android' || Platform.OS === 'ios');

function eventName(name: AnalyticsEventName): string {
  return name.replace(/[^a-zA-Z0-9_]/g, '_').slice(0, 40);
}

function firebaseParams(payload: AnalyticsPayload = {}): FirebaseParams {
  const params: FirebaseParams = {};
  if (payload.tab) params.tab = payload.tab;
  if (payload.previousTab) params.previous_tab = payload.previousTab;
  if (payload.sport) params.sport = payload.sport;
  if (typeof payload.raceNo === 'number') params.race_no = payload.raceNo;
  if (typeof payload.marketUsed === 'boolean') params.market_used = payload.marketUsed;
  if (payload.marketRiskLevel) params.market_risk_level = payload.marketRiskLevel;
  if (typeof payload.top1Pct === 'number') params.top1_pct = Math.round(payload.top1Pct);
  if (typeof payload.trifectaPct === 'number') params.trifecta_pct = Math.round(payload.trifectaPct);
  if (typeof payload.latencyMs === 'number') params.latency_ms = Math.max(0, Math.round(payload.latencyMs));
  if (typeof payload.pollDelayMs === 'number') params.poll_delay_ms = Math.max(0, Math.round(payload.pollDelayMs));
  if (payload.errorKind) params.error_kind = payload.errorKind;
  return params;
}

function captureTelemetryFailure(error: unknown): void {
  if (error instanceof Error && typeof __DEV__ !== 'undefined' && __DEV__) {
    console.warn('Firebase telemetry unavailable:', error.message);
  }
}

export function trackFirebaseUxEvent(name: AnalyticsEventName, payload?: AnalyticsPayload): void {
  if (!nativeFirebaseEnabled) return;
  void sendFirebaseUxEvent(name, payload).catch(captureTelemetryFailure);
}

async function sendFirebaseUxEvent(name: AnalyticsEventName, payload?: AnalyticsPayload): Promise<void> {
  const analyticsModule = await import('@react-native-firebase/analytics');
  const crashlyticsModule = await import('@react-native-firebase/crashlytics');
  const analytics = analyticsModule.getAnalytics();
  const crashlytics = crashlyticsModule.getCrashlytics();
  const safeName = eventName(name);
  await analyticsModule.logEvent(analytics, safeName, firebaseParams(payload));
  crashlyticsModule.log(crashlytics, `ux:${safeName}`);
  if (payload?.tab) {
    await crashlyticsModule.setAttribute(crashlytics, 'last_tab', payload.tab);
  }
  if (payload?.sport) {
    await crashlyticsModule.setAttribute(crashlytics, 'last_sport', payload.sport);
  }
}

export function recordFirebaseError(error: unknown, context: string): void {
  if (!nativeFirebaseEnabled) return;
  void sendFirebaseError(error, context).catch(captureTelemetryFailure);
}

async function sendFirebaseError(error: unknown, context: string): Promise<void> {
  const crashlyticsModule = await import('@react-native-firebase/crashlytics');
  const crashlytics = crashlyticsModule.getCrashlytics();
  await crashlyticsModule.setAttribute(crashlytics, 'last_error_context', context.slice(0, 64));
  crashlyticsModule.recordError(crashlytics, error instanceof Error ? error : new Error(String(error || 'Unknown error')));
}
