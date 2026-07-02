import Constants from 'expo-constants';

import type { RaceDecision, Sport } from '../types/race';

const extraApiBaseUrl = (Constants.expoConfig?.extra?.apiBaseUrl as string | undefined) ?? '';
const apiBaseUrl = extraApiBaseUrl;

const demoDecision: RaceDecision = {
  status: 'hold',
  sport: 'keirin',
  date: '2026-06-28',
  meet: '광명',
  raceNo: 5,
  headline: '시장 배당 미사용, 모델 신호만 표시',
  marketUsed: false,
  marketRisk: {
    level: 'caution',
    title: '실시간 배당 대기',
    message: '앱은 데이터 분석 결과만 제공합니다. 배당 수집 여부와 표본 수를 확인한 뒤 해석하세요.'
  },
  confidence: {
    label: '검증 대기',
    top1: 0.62,
    trifecta: 0.18,
    sample: 10886
  },
  picks: [
    { code: 'TOP1', label: '1순위 후보', selection: '5', probability: 0.62, grade: '중' },
    { code: 'QNL', label: '복승 조합', selection: '5-1', probability: 0.31, grade: '중' },
    { code: 'TRI', label: '삼쌍 순서', selection: '5-1-7', probability: 0.18, grade: '약' },
    { code: 'TRB', label: '삼복 조합', selection: '1-5-7', probability: 0.26, grade: '중' }
  ],
  updatedAt: new Date().toISOString()
};

type LiveDecisionPayload = {
  status?: string;
  market_used?: boolean;
  market_risk?: { level?: string; message?: string };
  poll_delay_ms?: number;
};

function riskLevel(level: string | undefined): RaceDecision['marketRisk']['level'] {
  if (level === 'blocked' || level === 'live_blocked') return 'blocked';
  if (level === 'odds_live') return 'verified';
  if (level === 'odds_unavailable' || level === 'pre_race') return 'caution';
  return 'neutral';
}

export async function fetchRaceDecision(params: {
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
}): Promise<RaceDecision> {
  if (!apiBaseUrl) {
    return {
      ...demoDecision,
      sport: params.sport,
      date: params.date,
      meet: params.meet,
      raceNo: params.raceNo,
      updatedAt: new Date().toISOString()
    };
  }

  const query = new URLSearchParams({
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    race_no: String(params.raceNo)
  });
  const response = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/api/live-decision?${query.toString()}`);
  if (!response.ok) {
    throw new Error(`RaceLens API ${response.status}`);
  }
  const payload = (await response.json()) as LiveDecisionPayload;
  const level = riskLevel(payload.market_risk?.level);
  return {
    ...demoDecision,
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    raceNo: params.raceNo,
    status: payload.status === 'blocked' ? 'blocked' : payload.status === 'ready' ? 'ready' : 'hold',
    headline: payload.market_used ? '실시간 시장 신호 반영' : '모델 신호 중심 분석',
    marketUsed: Boolean(payload.market_used),
    marketRisk: {
      level,
      title: level === 'verified' ? '실시간 배당 반영' : level === 'blocked' ? '실시간 접근 제한' : '배당 미사용',
      message: payload.market_risk?.message ?? demoDecision.marketRisk.message
    },
    updatedAt: new Date().toISOString()
  };
}
