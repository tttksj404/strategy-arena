import assert from 'node:assert/strict';
import test from 'node:test';

import { shouldKeepVisibleDecision } from '../src/services/decisionRefresh.ts';
import type { RaceDecision } from '../src/types/race';


function decision(overrides: Partial<RaceDecision> = {}): RaceDecision {
  return {
    analysisError: false,
    appSession: {
      deviceId: 'device',
      entitlement: 'pro',
      freeAnalysisLimit: 3,
      freeAnalysisRemaining: 3,
      freeAnalysisUsed: 0,
      rewardedAnalysisCredits: 0,
      userId: 'user'
    },
    confidence: { label: '검증됨', sample: 100, top1: 0.42, trifecta: 0.08 },
    dataLayer: { ready: true, schemas: [], storage: 'test' },
    date: '2026-07-17',
    headline: '광명 1R 분석',
    marketOdds: [],
    marketRisk: { level: 'odds_live', message: '정상', title: '실시간' },
    marketSource: 'live',
    marketUsed: true,
    meet: '광명',
    oddsAgeSec: 4,
    officialDataPending: false,
    participants: [],
    picks: [{ code: 'TRI', grade: '중', label: '삼쌍', probability: 0.08, selection: '1-2-3' }],
    pollDelayMs: 3000,
    raceNo: 1,
    rosterVerification: { message: '확인됨', state: 'verified' },
    sport: 'keirin',
    status: 'ready',
    updatedAt: '2026-07-17T10:00:00+09:00',
    ...overrides
  };
}


test('keeps the prior complete analysis when the same race refresh is pending', () => {
  const previous = decision();
  const pending = decision({
    confidence: { label: '대기', sample: 0, top1: 0, trifecta: 0 },
    officialDataPending: true,
    participants: [],
    picks: [],
    status: 'hold'
  });

  assert.equal(shouldKeepVisibleDecision(previous, pending), true);
});


test('replaces the prior analysis when a complete refresh for the same race arrives', () => {
  const previous = decision();
  const refreshed = decision({ updatedAt: '2026-07-17T10:01:00+09:00' });

  assert.equal(shouldKeepVisibleDecision(previous, refreshed), false);
});
