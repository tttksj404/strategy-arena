import type { AnalyticsEventName, AnalyticsPayload } from './uxAnalytics';

export function trackFirebaseUxEvent(name: AnalyticsEventName, payload?: AnalyticsPayload): void {
  void name;
  void payload;
}

export function recordFirebaseError(error: unknown, context: string): void {
  void error;
  void context;
}
