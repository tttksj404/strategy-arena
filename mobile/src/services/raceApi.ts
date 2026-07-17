import Constants from 'expo-constants';
import { Platform } from 'react-native';

import {
  recordFromJson,
  riskLevel,
  safeText,
  sanitizeAppSession,
  sanitizeDataLayer,
  sanitizeMarketOdds,
  sanitizeParticipants,
  sanitizePicks,
  sanitizeTrifectaEnsemble,
  sanitizeRosterVerification
} from './racePayload';
import type { AppSession, DataLayerStatus, RaceActualResult, RaceDecision, RacePick, Sport } from '../types/race';
import { getClientDeviceId } from './deviceIdentity';
import { defaultRaceCount } from './raceSchedule';

export type RaceDatesPayload = {
  readonly days: readonly string[];
  readonly defaultRaceNo: number;
  readonly raceCount: number;
};

const extraApiBaseUrl = (Constants.expoConfig?.extra?.apiBaseUrl as string | undefined) ?? '';
const apiBaseUrl = normalizeApiBaseUrl(extraApiBaseUrl) || defaultSameOriginApiBaseUrl();
const offlineExampleEnabled = ['1', 'true', 'yes', 'on'].includes(
  String(Constants.expoConfig?.extra?.offlineExampleEnabled ?? '').trim().toLowerCase()
);
const apiTimeoutMs = 10000;
const apiClientPlatform = Platform.OS === 'web' ? 'web' : 'mobile';
export const hostedPublicPro = apiBaseUrl === 'https://strategy-arena.onrender.com';
const inFlightRequests = new Map<string, Promise<RaceDecision>>();
const inFlightPreloads = new Map<string, Promise<void>>();
const kstDateFormatter = new Intl.DateTimeFormat('en-CA', {
  day: '2-digit',
  month: '2-digit',
  timeZone: 'Asia/Seoul',
  year: 'numeric'
});

const demoDecision: RaceDecision = {
  status: 'hold',
  sport: 'keirin',
  date: '2026-06-28',
  meet: '광명',
  raceNo: 1,
  headline: '원자료 연결 대기',
  marketUsed: false,
  marketSource: 'unavailable',
  marketRisk: {
    level: 'neutral',
    title: '분석 전',
    message: '공식 출전표 API가 연결되기 전에는 선수명과 배당을 표시하지 않습니다.'
  },
  confidence: {
    label: '검증 대기',
    top1: 0,
    trifecta: 0,
    sample: 0
  },
  picks: [],
  participants: [],
  marketOdds: [],
  actualResult: undefined,
  rosterVerification: {
    state: 'unverified',
    message: '공식 대조 미완료'
  },
  dataLayer: {
    ready: false,
    storage: 'demo',
    schemas: []
  },
  appSession: {
    userId: 'demo-user',
    deviceId: 'demo-device',
    entitlement: hostedPublicPro ? 'pro' : 'free',
    freeAnalysisLimit: 3,
    freeAnalysisUsed: 0,
    freeAnalysisRemaining: 3,
    rewardedAnalysisCredits: 0
  },
  analysisError: false,
  officialDataPending: false,
  pollDelayMs: 60000,
  updatedAt: new Date().toISOString(),
  oddsAgeSec: null
};

let latestAppSession: AppSession | null = null;

function fallbackAppSession(deviceId = demoDecision.appSession.deviceId): AppSession {
  return {
    ...(latestAppSession ?? demoDecision.appSession),
    deviceId
  };
}

function normalizeApiBaseUrl(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return '';
  try {
    const parsed = new URL(trimmed);
    const local = parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1';
    if (parsed.protocol === 'https:' || (parsed.protocol === 'http:' && local)) {
      return parsed.toString().replace(/\/$/, '');
    }
  } catch {
    return '';
  }
  return '';
}

function defaultSameOriginApiBaseUrl() {
  if (typeof window === 'undefined' || !window.location?.origin) return '';
  const hostname = window.location.hostname;
  const protocol = window.location.protocol;
  const port = window.location.port;
  const rewardedPreview = ['1', 'true', 'yes', 'on'].includes(
    String(Constants.expoConfig?.extra?.rewardedAdsPreview ?? '').trim().toLowerCase()
  );
  const localPreview = (hostname === 'localhost' || hostname === '127.0.0.1') && (port === '4173' || rewardedPreview);
  if (!localPreview && (hostname === 'localhost' || hostname === '127.0.0.1')) return '';
  if (protocol !== 'https:' && protocol !== 'http:') return '';
  return window.location.origin.replace(/\/$/, '');
}

function unavailableDecision(params: {
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
}, message: string): RaceDecision {
  return {
    ...demoDecision,
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    raceNo: params.raceNo,
    headline: '원자료 연결 대기',
    marketUsed: false,
    marketSource: 'unavailable',
    marketRisk: {
      level: 'caution',
      title: '원자료 연결 대기',
      message
    },
    confidence: {
      label: '대기',
      top1: 0,
      trifecta: 0,
      sample: 0
    },
    picks: [],
    participants: [],
    marketOdds: [],
    actualResult: undefined,
    rosterVerification: demoDecision.rosterVerification,
    dataLayer: {
      ...demoDecision.dataLayer,
      error: message
    },
    appSession: {
      ...fallbackAppSession()
    },
    pollDelayMs: 60000,
    updatedAt: new Date().toISOString(),
    oddsAgeSec: null
  };
}

async function offlineExampleDecision(params: {
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
}, message: string): Promise<RaceDecision> {
  if (!offlineExampleEnabled) return unavailableDecision(params, message);
  const demoData = await import('./demoRaceData');
  const participants = demoData.demoParticipants(params.sport);
  const picks = demoData.demoPicks(params.sport);
  const topPick = picks[0];
  const trifectaPick = picks.find((pick) => pick.code === 'TRI');
  return {
    ...demoDecision,
    status: 'ready',
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    raceNo: params.raceNo,
    headline: '과거 데이터 예시',
    marketUsed: false,
    marketSource: 'sample',
    marketRisk: {
      level: 'caution',
      title: '과거 데이터 예시',
      message: `과거 데이터 예시(${params.date} 기준)입니다. ${message}`
    },
    confidence: {
      label: '예시',
      top1: topPick?.probability ?? 0,
      trifecta: trifectaPick?.probability ?? 0,
      sample: participants.length
    },
    picks,
    participants,
    marketOdds: demoData.demoMarketOdds(params.sport),
    rosterVerification: {
      state: 'unverified',
      message: '과거 데이터 예시: 공식 출전표 검증 전 실데이터가 아닙니다.',
      source: 'offline-example'
    },
    dataLayer: {
      ready: false,
      storage: 'offline-example',
      schemas: [],
      error: message
    },
    appSession: {
      ...fallbackAppSession()
    },
    updatedAt: new Date().toISOString()
  };
}

function isApiRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function rowNumber(row: Record<string, unknown>) {
  const numeric = Number(row.bno ?? row.number);
  return Number.isInteger(numeric) && numeric > 0 ? numeric : null;
}

function rowProbability(row: Record<string, unknown>) {
  const numeric = Number(row.pwin_blended ?? row.pwin);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(1, numeric));
}

function pickGrade(probability: number): RacePick['grade'] {
  if (probability >= 0.45) return '강';
  if (probability >= 0.25) return '중';
  return '약';
}

function picksFromRows(value: unknown): RacePick[] {
  if (!Array.isArray(value)) return [];
  const rows = value
    .filter(isApiRecord)
    .map((row) => ({ number: rowNumber(row), probability: rowProbability(row) }))
    .filter((row): row is { number: number; probability: number } => row.number !== null)
    .slice(0, 3);
  const firstRow = rows[0];
  if (!firstRow) return [];
  const first = firstRow.probability;
  const picks: RacePick[] = [{
    code: 'TOP1',
    label: '1착 후보',
    selection: String(firstRow.number),
    probability: first,
    grade: pickGrade(first)
  }];
  const secondRow = rows[1];
  if (secondRow) {
    const pairProbability = Math.max(0, Math.min(1, (firstRow.probability + secondRow.probability) / 2));
    picks.push({
      code: 'QNL',
      label: '복승 조합',
      selection: `${firstRow.number}-${secondRow.number}`,
      probability: pairProbability,
      grade: pickGrade(pairProbability)
    });
  }
  const thirdRow = rows[2];
  if (secondRow && thirdRow) {
    const ordered = `${firstRow.number}-${secondRow.number}-${thirdRow.number}`;
    const unordered = [firstRow.number, secondRow.number, thirdRow.number].sort((left, right) => left - right).join('-');
    const trifecta = Math.max(
      0,
      Math.min(1, firstRow.probability * Math.max(secondRow.probability, 0.01) * Math.max(thirdRow.probability, 0.01))
    );
    const trio = Math.max(0, Math.min(1, (firstRow.probability + secondRow.probability + thirdRow.probability) / 3));
    picks.push(
      {
        code: 'TRI',
        label: '1-2-3 순서',
        selection: ordered,
        probability: trifecta,
        grade: '약'
      },
      {
        code: 'TRB',
        label: '삼복 조합',
        selection: unordered,
        probability: trio,
        grade: pickGrade(trio)
      }
    );
  }
  return picks;
}

function sanitizePollDelayMs(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 15000;
  return Math.max(3000, Math.min(60000, Math.round(numeric)));
}

function sanitizeOddsAgeSec(value: unknown) {
  if (value === null || value === undefined) return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) return null;
  return Math.floor(numeric);
}

function liveDecisionErrorMessage(status: number) {
  if (status === 401) return '인증이 필요해 분석을 불러오지 못했습니다. 로그인 상태를 확인하고 다시 시도하세요.';
  if (status === 429) return '요청이 너무 많아 잠시 제한되었습니다. 잠시 후 다시 시도하세요.';
  if (status >= 500) return `서버 오류 ${status}로 분석을 불러오지 못했습니다. 잠시 후 다시 시도하세요.`;
  return `분석 서버 응답 오류 ${status}로 결과를 표시하지 않습니다. 잠시 후 다시 시도하세요.`;
}

function confidenceFromPayload(payload: Record<string, unknown>, rowsValue: unknown): RaceDecision['confidence'] {
  const rows = Array.isArray(rowsValue) ? rowsValue.filter(isApiRecord) : [];
  const first = rows[0] ? rowProbability(rows[0]) : 0;
  const second = rows[1] ? rowProbability(rows[1]) : 0;
  const third = rows[2] ? rowProbability(rows[2]) : 0;
  const confidence = isApiRecord(payload.top_conf) ? payload.top_conf : {};
  const rawTop1 = Number(confidence.pwin ?? first);
  const top1 = Number.isFinite(rawTop1) ? Math.max(0, Math.min(1, rawTop1)) : first;
  const trifecta = first && second && third
    ? Math.max(0, Math.min(1, first * Math.max(second, 0.01) * Math.max(third, 0.01)))
    : 0;
  return {
    label: safeText(confidence.grade, top1 >= 0.45 ? '강' : top1 >= 0.25 ? '중' : '약', 8),
    top1,
    trifecta,
    sample: rows.length
  };
}

function sanitizeActualResult(value: unknown): RaceActualResult | undefined {
  const result = recordFromJson(value);
  const actualOrder = Array.isArray(result.actual_order)
    ? result.actual_order
      .map((item) => Number(item))
      .filter((item) => Number.isInteger(item) && item > 0)
      .slice(0, 3)
    : [];
  if (actualOrder.length === 0) return undefined;

  const racers = Array.isArray(result.racers)
    ? result.racers
      .filter(isApiRecord)
      .map((racer) => ({
        number: Number(racer.bno ?? racer.number),
        name: safeText(racer.name, '', 24),
        rank: Number(racer.rank)
      }))
      .filter((racer) => (
        Number.isInteger(racer.number) &&
        racer.number > 0 &&
        racer.name.length > 0 &&
        Number.isInteger(racer.rank) &&
        racer.rank > 0
      ))
      .slice(0, 16)
    : [];

  const payoutRecord = recordFromJson(result.payouts);
  const payouts = Object.entries(payoutRecord)
    .map(([label, raw]) => {
      const payout = recordFromJson(raw);
      const odds = Number(payout.odds);
      return {
        label: safeText(label, '', 16),
        winner: safeText(payout.winner, '', 32),
        odds: Number.isFinite(odds) && odds >= 1 ? Math.round(odds * 100) / 100 : 0
      };
    })
    .filter((payout) => payout.label.length > 0 && payout.winner.length > 0 && payout.odds > 0)
    .slice(0, 10);

  return {
    actualOrder,
    racers,
    payouts,
    source: safeText(result.source, '', 64) || undefined,
    sourceUrl: safeText(result.source_url ?? result.sourceUrl, '', 180) || undefined
  };
}

export async function fetchRaceDecision(params: {
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
}): Promise<RaceDecision> {
  const requestKey = `${params.sport}|${params.date}|${params.meet}|${params.raceNo}`;
  const activeRequest = inFlightRequests.get(requestKey);
  if (activeRequest) return activeRequest;
  const request = fetchRaceDecisionUncached(params).finally(() => {
    inFlightRequests.delete(requestKey);
  });
  inFlightRequests.set(requestKey, request);
  return request;
}

export async function preloadRaceDecisions(params: {
  sport: Sport;
  date: string;
  meet: string;
  raceCount: number;
  priorityRaceNo: number;
}): Promise<void> {
  if (!apiBaseUrl) return;
  const requestKey = `${params.sport}|${params.date}|${params.meet}|${params.raceCount}|${params.priorityRaceNo}`;
  const activeRequest = inFlightPreloads.get(requestKey);
  if (activeRequest) return activeRequest;
  const request = preloadRaceDecisionsUncached(params).finally(() => {
    inFlightPreloads.delete(requestKey);
  });
  inFlightPreloads.set(requestKey, request);
  return request;
}

export async function fetchAppSession(): Promise<{ appSession: AppSession; dataLayer: DataLayerStatus }> {
  const clientDeviceId = await getClientDeviceId();
  if (!apiBaseUrl) {
    return {
      appSession: {
        ...demoDecision.appSession,
        deviceId: clientDeviceId
      },
      dataLayer: demoDecision.dataLayer
    };
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), apiTimeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl}/api/app-session`, {
      headers: {
        'X-RaceLens-Device-Id': clientDeviceId,
        'X-RaceLens-Platform': apiClientPlatform
      },
      signal: controller.signal
    });
    if (!response.ok) {
      return {
        appSession: fallbackAppSession(clientDeviceId),
        dataLayer: {
          ...demoDecision.dataLayer,
          error: `세션 서버 응답 오류 ${response.status}`
        }
      };
    }
    const payload = recordFromJson(await response.json());
    const appSession = payload.app_session
      ? sanitizeAppSession(payload.app_session, clientDeviceId)
      : fallbackAppSession(clientDeviceId);
    latestAppSession = appSession;
    return {
      appSession,
      dataLayer: sanitizeDataLayer(payload.data_layer, demoDecision.dataLayer)
    };
  } catch (error) {
    return {
      appSession: fallbackAppSession(clientDeviceId),
      dataLayer: {
        ...demoDecision.dataLayer,
        error: error instanceof Error ? error.message : '세션 서버 연결 실패'
      }
    };
  } finally {
    clearTimeout(timeout);
  }
}

export async function claimRewardedAdCredit(): Promise<{
  appSession: AppSession;
  dataLayer: DataLayerStatus;
  rewardGranted: boolean;
}> {
  if (!apiBaseUrl) {
    return {
      appSession: demoDecision.appSession,
      dataLayer: {
        ...demoDecision.dataLayer,
        error: '광고 보상 서버 URL이 설정되지 않았습니다'
      },
      rewardGranted: false
    };
  }
  const clientDeviceId = await getClientDeviceId();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), apiTimeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl}/api/rewarded-ad/claim`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-RaceLens-Device-Id': clientDeviceId,
        'X-RaceLens-Platform': apiClientPlatform
      },
      body: JSON.stringify({ placement: 'quota_gate' }),
      signal: controller.signal
    });
    if (!response.ok) {
      return {
        appSession: demoDecision.appSession,
        dataLayer: {
          ...demoDecision.dataLayer,
          error: `광고 보상 서버 응답 오류 ${response.status}`
        },
        rewardGranted: false
      };
    }
    const payload = recordFromJson(await response.json());
    return {
      appSession: sanitizeAppSession(payload.app_session, clientDeviceId),
      dataLayer: sanitizeDataLayer(payload.data_layer, demoDecision.dataLayer),
      rewardGranted: payload.reward_granted === true || payload.rewardGranted === true
    };
  } catch (error) {
    return {
      appSession: demoDecision.appSession,
      dataLayer: {
        ...demoDecision.dataLayer,
        error: error instanceof Error ? error.message : '광고 보상 서버 연결 실패'
      },
      rewardGranted: false
    };
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchRaceDates(params: {
  sport: Sport;
  meet: string;
}): Promise<RaceDatesPayload> {
  const fallbackRaceCount = defaultRaceCount(params.sport, params.meet);
  const fallbackPayload: RaceDatesPayload = {
    days: [],
    defaultRaceNo: 1,
    raceCount: fallbackRaceCount
  };
  if (!apiBaseUrl) return fallbackPayload;
  const clientDeviceId = await getClientDeviceId();
  const query = new URLSearchParams({
    sport: params.sport,
    meet: params.meet
  });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), apiTimeoutMs);
  try {
    const response = await fetch(`${apiBaseUrl}/recent?${query.toString()}`, {
      headers: {
        'X-RaceLens-Device-Id': clientDeviceId,
        'X-RaceLens-Platform': apiClientPlatform
      },
      signal: controller.signal
    });
    if (!response.ok) return fallbackPayload;
    const payload = recordFromJson(await response.json());
    if (!Array.isArray(payload.days)) return fallbackPayload;
    const raceCount = sanitizeRaceCount(payload.race_count, fallbackRaceCount);
    return {
      days: payload.days
      .filter((item): item is string => typeof item === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(item))
      .sort(),
      defaultRaceNo: sanitizeRaceNo(payload.default_race_no, raceCount),
      raceCount
    };
  } catch {
    return fallbackPayload;
  } finally {
    clearTimeout(timeout);
  }
}

function sanitizeRaceCount(value: unknown, fallback: number) {
  const numeric = Number(value);
  if (!Number.isInteger(numeric)) return fallback;
  return Math.max(1, Math.min(24, numeric));
}

function sanitizeRaceNo(value: unknown, raceCount: number) {
  const numeric = Number(value);
  if (!Number.isInteger(numeric)) return 1;
  return Math.max(1, Math.min(raceCount, numeric));
}

async function preloadRaceDecisionsUncached(params: {
  sport: Sport;
  date: string;
  meet: string;
  raceCount: number;
  priorityRaceNo: number;
}): Promise<void> {
  const clientDeviceId = await getClientDeviceId();
  const query = new URLSearchParams({
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    race_count: String(params.raceCount),
    priority_race_no: String(params.priorityRaceNo)
  });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), apiTimeoutMs);
  try {
    await fetch(`${apiBaseUrl}/api/live-decisions/preload?${query.toString()}`, {
      method: 'POST',
      headers: {
        'X-RaceLens-Device-Id': clientDeviceId,
        'X-RaceLens-Platform': apiClientPlatform
      },
      signal: controller.signal
    });
  } catch {
    return;
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchRaceDecisionUncached(params: {
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
}): Promise<RaceDecision> {
  if (!apiBaseUrl) {
    return offlineExampleDecision(params, 'API URL이 설정되지 않아 공식 출전표를 표시하지 않습니다.');
  }

  const clientDeviceId = await getClientDeviceId();
  const query = new URLSearchParams({
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    race_no: String(params.raceNo)
  });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), apiTimeoutMs);
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/api/live-decision?${query.toString()}`, {
      headers: {
        'X-RaceLens-Device-Id': clientDeviceId,
        'X-RaceLens-Platform': apiClientPlatform
      },
      signal: controller.signal
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return unavailableDecision(params, '공식 출전표를 확인하고 있습니다. 잠시 후 자동으로 갱신됩니다.');
    }
    return unavailableDecision(params, '네트워크 연결이 끊겨 분석을 불러오지 못했습니다. 다시 시도하세요.');
  } finally {
    clearTimeout(timeout);
  }
  if (!response.ok) {
    return unavailableDecision(params, liveDecisionErrorMessage(response.status));
  }
  let rawPayload: unknown;
  try {
    rawPayload = await response.json();
  } catch {
    return unavailableDecision(params, '분석 서버 응답 형식이 올바르지 않아 결과를 표시하지 않습니다. 다시 시도하세요.');
  }
  const payload = recordFromJson(rawPayload);
  const fallbackPicks = picksFromRows(payload.rows);
  const picksPayload = Array.isArray(payload.picks) && payload.picks.length > 0 ? payload.picks : fallbackPicks;
  const marketRisk = recordFromJson(payload.market_risk);
  const rawMarketOdds = payload.market_odds ?? payload.odds;
  const marketOdds = sanitizeMarketOdds(rawMarketOdds, params.sport);
  const rawRiskLevel = typeof marketRisk?.level === 'string' ? marketRisk.level : undefined;
  const level = riskLevel(rawRiskLevel);
  const status = typeof payload.status === 'string' ? payload.status : '';
  const officialDataPending = payload.snapshot_phase === 'pending' || payload.snapshot_phase === 'in_progress';
  const analysisError = payload.snapshot_phase === 'failed';
  const noRace = status === 'no_race' || rawRiskLevel === 'no_race';
  const actualResult = sanitizeActualResult(payload.actual_result);
  const settled = status === 'settled' || actualResult !== undefined;
  const todayKst = kstDateFormatter.format(new Date());
  const marketSource = settled
    ? 'settled'
    : payload.market_used === true
    ? (params.date < todayKst ? 'historical' : 'live')
    : marketOdds.length > 0 ? 'sample' : 'unavailable';
  const displayMarketOdds = marketSource === 'historical'
    ? marketOdds.map((entry) => ({
      ...entry,
      change: entry.change === '실시간' ? '과거 배당' : entry.change,
      source: 'historical' as const
    }))
    : marketSource === 'live'
      ? marketOdds.map((entry) => ({
        ...entry,
        source: 'live' as const
      }))
      : marketOdds;
  const appSession = payload.app_session
    ? sanitizeAppSession(payload.app_session, clientDeviceId)
    : fallbackAppSession(clientDeviceId);
  latestAppSession = appSession;
  return {
    ...demoDecision,
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    raceNo: params.raceNo,
    status: settled ? 'settled' : status === 'roster_mismatch' ? 'blocked' : status === 'blocked' ? 'blocked' : status === 'ready' ? 'ready' : 'hold',
    headline: settled && actualResult
      ? `경주 종료 · 실제 착순 ${actualResult.actualOrder.join('-')}`
      : payload.market_used === true
      ? marketSource === 'historical' ? '과거 배당과 모델 신호 반영' : '실시간 시장 신호 반영'
      : marketOdds.length > 0 ? '샘플 배당과 모델 신호 표시' : '모델 신호 중심 분석',
    marketUsed: payload.market_used === true,
    marketSource,
    marketRisk: {
      level,
      title: officialDataPending || analysisError ? safeText(marketRisk?.title, officialDataPending ? '공식 출전표 확인 중' : '분석 데이터 오류', 40) :
        marketSource === 'settled' ? '확정 결과 반영' :
        noRace ? '해당 날짜에는 경주가 없습니다' :
        marketSource === 'historical' ? '과거 배당 반영' :
        marketSource === 'live' ? '실시간 배당 반영' :
        marketSource === 'sample' ? '샘플 배당' :
        rawRiskLevel === 'oracle_api_not_configured' ? 'Oracle API 미설정' :
        rawRiskLevel === 'kcycle_disabled' ? '배당 비활성' :
        level === 'blocked' ? '실시간 접근 제한' : '배당 미사용',
      message: noRace
        ? '해당 날짜에는 경주가 없습니다'
        : safeText(marketRisk?.message, demoDecision.marketRisk.message, 160)
    },
    confidence: confidenceFromPayload(payload, payload.rows),
    picks: sanitizePicks(picksPayload, params.sport),
    trifectaEnsemble: sanitizeTrifectaEnsemble(payload.trifecta_ensemble),
    participants: sanitizeParticipants(payload.participants, params.sport, params.date),
    marketOdds: displayMarketOdds,
    actualResult,
    rosterVerification: sanitizeRosterVerification(payload.roster_verification ?? payload.rosterVerification),
    dataLayer: sanitizeDataLayer(payload.data_layer, demoDecision.dataLayer),
    appSession,
    analysisError,
    officialDataPending,
    pollDelayMs: sanitizePollDelayMs(payload.poll_delay_ms),
    updatedAt: safeText(payload.updated_at ?? payload.updatedAt, new Date().toISOString(), 40),
    oddsAgeSec: sanitizeOddsAgeSec(payload.odds_age_sec ?? payload.oddsAgeSec)
  };
}
