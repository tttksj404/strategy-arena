import Constants from 'expo-constants';

import type { MarketOddsEntry, RaceDecision, RaceParticipant, Sport } from '../types/race';

const extraApiBaseUrl = (Constants.expoConfig?.extra?.apiBaseUrl as string | undefined) ?? '';
const apiBaseUrl = normalizeApiBaseUrl(extraApiBaseUrl);
const apiTimeoutMs = 6000;
const maxParticipants = 16;
const maxMarketOdds = 12;

type JsonRecord = Record<string, unknown>;

const keirinParticipants: RaceParticipant[] = [
  { number: 1, name: '김태훈', subtitle: '우수급 / 87.4점', stats: '최근 3주 2-1-0', trait: '선행', note: '초반 주도력 강점, 후반 유지력은 보통', signal: 'teal' },
  { number: 2, name: '박민재', subtitle: '우수급 / 84.8점', stats: '최근 3주 1-1-1', trait: '마크', note: '강자 뒤 추주 안정, 단독 승부는 제한적', signal: 'primary' },
  { number: 3, name: '이현수', subtitle: '선발급 / 81.9점', stats: '최근 3주 0-2-1', trait: '추입', note: '직선 반응 좋지만 위치 선정 변동 큼', signal: 'amber' },
  { number: 4, name: '정도윤', subtitle: '선발급 / 80.7점', stats: '최근 3주 0-1-2', trait: '자력', note: '몸싸움 회피형, 전개 이득 필요', signal: 'amber' },
  { number: 5, name: '최강우', subtitle: '특선급 / 91.2점', stats: '최근 3주 3-0-0', trait: '젖히기', note: '득점·컨디션 우위, 중심 후보로 분류', signal: 'teal' },
  { number: 6, name: '윤성민', subtitle: '우수급 / 83.6점', stats: '최근 3주 1-0-1', trait: '마크', note: '내선 운영 안정, 외선 전환은 느림', signal: 'primary' },
  { number: 7, name: '서지환', subtitle: '우수급 / 85.1점', stats: '최근 3주 1-2-0', trait: '추입', note: '마지막 반 바퀴 탄력, 초반 위치 중요', signal: 'primary' },
  { number: 8, name: '오민석', subtitle: '선발급 / 79.8점', stats: '최근 3주 0-1-0', trait: '선행', note: '초반 속도는 있으나 종반 저하 리스크', signal: 'rose' }
];

const horseParticipants: RaceParticipant[] = [
  { number: 1, name: '새벽질주', subtitle: '기수 김도윤 / 55kg', stats: '최근 4전 1-1-1', trait: '선입', note: '출발 안정, 직선 탄력은 평균 이상', signal: 'primary' },
  { number: 2, name: '블랙윈드', subtitle: '기수 박하린 / 56kg', stats: '최근 4전 0-2-1', trait: '추입', note: '막판 추입 좋지만 전개 의존도 높음', signal: 'amber' },
  { number: 3, name: '스톰레이크', subtitle: '기수 이준 / 54kg', stats: '최근 4전 2-0-0', trait: '선행', note: '게이트 반응 빠름, 페이스 과속 주의', signal: 'teal' },
  { number: 4, name: '라이트닝문', subtitle: '기수 최서윤 / 55kg', stats: '최근 4전 1-1-0', trait: '자유', note: '거리 적응 양호, 모래 맞으면 집중력 저하', signal: 'primary' },
  { number: 5, name: '골든포커스', subtitle: '기수 문태오 / 57kg', stats: '최근 4전 3-0-0', trait: '선입', note: '지구력·기록 우위, 부담중량 확인 필요', signal: 'teal' },
  { number: 6, name: '청운대로', subtitle: '기수 안유진 / 54kg', stats: '최근 4전 0-1-2', trait: '추입', note: '후반 반응은 안정, 초반 자리 손실 잦음', signal: 'amber' },
  { number: 7, name: '오션프라임', subtitle: '기수 정민호 / 56kg', stats: '최근 4전 1-0-1', trait: '선행', note: '단거리 페이스 적합, 긴 직선은 부담', signal: 'primary' },
  { number: 8, name: '레드노바', subtitle: '기수 한서우 / 53kg', stats: '최근 4전 0-0-1', trait: '자유', note: '부담중량 이점, 최근 컨디션 확인 필요', signal: 'rose' }
];

const keirinMarketOdds: MarketOddsEntry[] = [
  { code: 'WIN', label: '단승', selection: '5', odds: 2.1, change: '대기', signal: 'teal' },
  { code: 'QNL', label: '복승', selection: '5-1', odds: 4.8, change: '대기', signal: 'primary' },
  { code: 'EXA', label: '쌍승', selection: '5-1', odds: 7.6, change: '대기', signal: 'amber' },
  { code: 'TRI', label: '삼쌍', selection: '5-1-7', odds: 21.4, change: '대기', signal: 'violet' }
];

const horseMarketOdds: MarketOddsEntry[] = [
  { code: 'WIN', label: '단승', selection: '5', odds: 2.4, change: '대기', signal: 'teal' },
  { code: 'QNL', label: '복승', selection: '5-3', odds: 5.2, change: '대기', signal: 'primary' },
  { code: 'EXA', label: '쌍승', selection: '5-3', odds: 8.1, change: '대기', signal: 'amber' },
  { code: 'TRI', label: '삼쌍', selection: '5-3-1', odds: 24.6, change: '대기', signal: 'violet' }
];

function demoParticipants(sport: Sport) {
  return sport === 'horse' ? horseParticipants : keirinParticipants;
}

function demoMarketOdds(sport: Sport) {
  return sport === 'horse' ? horseMarketOdds : keirinMarketOdds;
}

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
  participants: keirinParticipants,
  marketOdds: keirinMarketOdds,
  updatedAt: new Date().toISOString()
};

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
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

function safeText(value: unknown, fallback: string, maxLength: number) {
  if (typeof value !== 'string') return fallback;
  const compact = value.replace(/\s+/g, ' ').trim();
  if (!compact) return fallback;
  return compact.length > maxLength ? `${compact.slice(0, maxLength - 1)}…` : compact;
}

function safeSignal(value: unknown): RaceParticipant['signal'] {
  if (value === 'teal' || value === 'amber' || value === 'rose' || value === 'violet' || value === 'primary') {
    return value;
  }
  return 'primary';
}

function safeOdds(value: unknown, fallback: number) {
  const numeric = Number(value);
  if (Number.isFinite(numeric) && numeric >= 1 && numeric <= 9999) {
    return Math.round(numeric * 100) / 100;
  }
  return fallback;
}

function fallbackParticipant(sport: Sport, index: number): RaceParticipant {
  const list = demoParticipants(sport);
  return list[index % list.length] ?? {
    number: index + 1,
    name: sport === 'horse' ? '출전마' : '출전 선수',
    subtitle: '기본 정보 대기',
    stats: '최근 흐름 대기',
    trait: '확인',
    note: '상세 자료가 도착하면 자동 갱신됩니다.',
    signal: 'primary'
  };
}

function sanitizeParticipants(value: unknown, sport: Sport) {
  if (!Array.isArray(value)) return demoParticipants(sport);
  const parsed = value
    .slice(0, maxParticipants)
    .map((item, index): RaceParticipant | null => {
      if (!isRecord(item)) return null;
      const fallback = fallbackParticipant(sport, index);
      const rawNumber = Number(item.number);
      return {
        number: Number.isInteger(rawNumber) && rawNumber > 0 && rawNumber <= 99 ? rawNumber : index + 1,
        name: safeText(item.name, fallback.name, 24),
        subtitle: safeText(item.subtitle, fallback.subtitle, 44),
        stats: safeText(item.stats, fallback.stats, 36),
        trait: safeText(item.trait, fallback.trait, 12),
        note: safeText(item.note, fallback.note, 72),
        signal: safeSignal(item.signal)
      };
    })
    .filter((item): item is RaceParticipant => item !== null);
  return parsed.length ? parsed : demoParticipants(sport);
}

function fallbackMarketOdds(sport: Sport, index: number): MarketOddsEntry {
  const list = demoMarketOdds(sport);
  return list[index % list.length] ?? {
    code: 'ODDS',
    label: '배당',
    selection: '-',
    odds: 1,
    change: '대기',
    signal: 'primary'
  };
}

function sanitizeMarketOdds(value: unknown, sport: Sport) {
  if (!Array.isArray(value)) return demoMarketOdds(sport);
  const parsed = value
    .slice(0, maxMarketOdds)
    .map((item, index): MarketOddsEntry | null => {
      if (!isRecord(item)) return null;
      const fallback = fallbackMarketOdds(sport, index);
      return {
        code: safeText(item.code, fallback.code, 8).toUpperCase(),
        label: safeText(item.label, fallback.label, 18),
        selection: safeText(item.selection, fallback.selection, 18),
        odds: safeOdds(item.odds, fallback.odds),
        change: safeText(item.change, fallback.change, 16),
        signal: safeSignal(item.signal)
      };
    })
    .filter((item): item is MarketOddsEntry => item !== null);
  return parsed.length ? parsed : demoMarketOdds(sport);
}

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
      participants: demoParticipants(params.sport),
      marketOdds: demoMarketOdds(params.sport),
      updatedAt: new Date().toISOString()
    };
  }

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
    response = await fetch(`${apiBaseUrl}/api/live-decision?${query.toString()}`, { signal: controller.signal });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('RaceLens API 응답 시간이 초과됐습니다.');
    }
    throw new Error('RaceLens API에 연결할 수 없습니다.');
  } finally {
    clearTimeout(timeout);
  }
  if (!response.ok) {
    throw new Error(`RaceLens API 응답 오류 ${response.status}`);
  }
  let rawPayload: unknown;
  try {
    rawPayload = await response.json();
  } catch {
    throw new Error('RaceLens API 응답 형식이 올바르지 않습니다.');
  }
  const payload = isRecord(rawPayload) ? rawPayload : {};
  const marketRisk = isRecord(payload.market_risk) ? payload.market_risk : undefined;
  const rawMarketOdds = payload.market_odds ?? payload.odds;
  const level = riskLevel(typeof marketRisk?.level === 'string' ? marketRisk.level : undefined);
  const status = typeof payload.status === 'string' ? payload.status : '';
  return {
    ...demoDecision,
    sport: params.sport,
    date: params.date,
    meet: params.meet,
    raceNo: params.raceNo,
    status: status === 'blocked' ? 'blocked' : status === 'ready' ? 'ready' : 'hold',
    headline: payload.market_used === true ? '실시간 시장 신호 반영' : '모델 신호 중심 분석',
    marketUsed: payload.market_used === true,
    marketRisk: {
      level,
      title: level === 'verified' ? '실시간 배당 반영' : level === 'blocked' ? '실시간 접근 제한' : '배당 미사용',
      message: safeText(marketRisk?.message, demoDecision.marketRisk.message, 160)
    },
    participants: sanitizeParticipants(payload.participants, params.sport),
    marketOdds: sanitizeMarketOdds(rawMarketOdds, params.sport),
    updatedAt: new Date().toISOString()
  };
}
