import Constants from 'expo-constants';

import type { Sport, TabKey } from '../types/race';
import { trackFirebaseUxEvent } from './firebaseTelemetry';

export type AnalyticsEventName =
  | 'app_open'
  | 'screen_view'
  | 'tab_select'
  | 'race_context_change'
  | 'analysis_request'
  | 'analysis_result'
  | 'analysis_error'
  | 'live_odds_refresh'
  | 'rewarded_ad_credit';

export type AnalyticsPayload = {
  tab?: TabKey;
  previousTab?: TabKey;
  sport?: Sport;
  raceNo?: number;
  marketUsed?: boolean;
  marketRiskLevel?: string;
  top1Pct?: number;
  trifectaPct?: number;
  latencyMs?: number;
  pollDelayMs?: number;
  errorKind?: 'api_error' | 'unknown';
};

type AnalyticsEvent = {
  app: 'racelens';
  version: '0.1.0';
  name: AnalyticsEventName;
  sessionId: string;
  anonymousId: string;
  platform: 'web' | 'native';
  timestamp: string;
  payload: AnalyticsPayload;
};

const extraAnalyticsUrl = (Constants.expoConfig?.extra?.analyticsUrl as string | undefined) ?? '';
const analyticsUrl = normalizeAnalyticsUrl(extraAnalyticsUrl);
const enabled = Boolean(analyticsUrl);
const sessionId = `sess_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
const anonymousId = `anon_${Math.random().toString(36).slice(2, 12)}`;
const platform = typeof document === 'undefined' ? 'native' : 'web';

function normalizeAnalyticsUrl(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return '';
  try {
    const parsed = new URL(trimmed);
    const local = parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1';
    if (parsed.protocol === 'https:' || (parsed.protocol === 'http:' && local)) {
      return parsed.toString();
    }
  } catch {
    return '';
  }
  return '';
}

function sanitizePayload(payload: AnalyticsPayload = {}): AnalyticsPayload {
  return {
    tab: payload.tab,
    previousTab: payload.previousTab,
    sport: payload.sport,
    raceNo: typeof payload.raceNo === 'number' ? payload.raceNo : undefined,
    marketUsed: payload.marketUsed,
    marketRiskLevel: payload.marketRiskLevel,
    top1Pct: typeof payload.top1Pct === 'number' ? Math.round(payload.top1Pct) : undefined,
    trifectaPct: typeof payload.trifectaPct === 'number' ? Math.round(payload.trifectaPct) : undefined,
    latencyMs: typeof payload.latencyMs === 'number' ? Math.max(0, Math.round(payload.latencyMs)) : undefined,
    pollDelayMs: typeof payload.pollDelayMs === 'number' ? Math.max(0, Math.round(payload.pollDelayMs)) : undefined,
    errorKind: payload.errorKind
  };
}

export function trackUxEvent(name: AnalyticsEventName, payload?: AnalyticsPayload) {
  trackFirebaseUxEvent(name, payload);
  if (!enabled || !analyticsUrl) return;

  const event: AnalyticsEvent = {
    app: 'racelens',
    version: '0.1.0',
    name,
    sessionId,
    anonymousId,
    platform,
    timestamp: new Date().toISOString(),
    payload: sanitizePayload(payload)
  };

  void fetch(analyticsUrl, {
    body: JSON.stringify(event),
    headers: {
      'content-type': 'application/json',
      'x-racelens-analytics': 'ux-v1'
    },
    keepalive: true,
    method: 'POST'
  }).catch(() => undefined);
}
