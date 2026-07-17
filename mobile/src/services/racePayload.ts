import { buildParticipantInsight } from './participantInsight';
import type {
  AppSession,
  DataLayerSchema,
  DataLayerStatus,
  MarketOddsEntry,
  ParticipantAlgorithmReason,
  ParticipantMetric,
  RaceDecision,
  RaceParticipant,
  RacePick,
  RosterVerification,
  Sport,
  TrifectaEnsemble,
  TrifectaEnsembleTier
} from '../types/race';

const maxParticipants = 16;
const maxMarketOdds = 12;
const maxPicks = 8;

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function safeText(value: unknown, fallback: string, maxLength: number) {
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

function safeSource(value: unknown): MarketOddsEntry['source'] {
  if (value === 'live' || value === 'historical' || value === 'settled' || value === 'sample' || value === 'unavailable') return value;
  return undefined;
}

function safeOdds(value: unknown, fallback: number) {
  const numeric = Number(value);
  if (Number.isFinite(numeric) && numeric >= 1 && numeric <= 9999) {
    return Math.round(numeric * 100) / 100;
  }
  return fallback;
}

function safeProbability(value: unknown, fallback: number) {
  const numeric = Number(value);
  if (Number.isFinite(numeric) && numeric >= 0 && numeric <= 1) {
    return Math.round(numeric * 1000) / 1000;
  }
  return fallback;
}

function safeGrade(value: unknown, fallback: RacePick['grade']) {
  if (value === '강' || value === '중' || value === '약') return value;
  return fallback;
}

function sanitizeMetrics(value: unknown, fallback: ParticipantMetric[], limit = 8) {
  if (!Array.isArray(value)) return fallback;
  const parsed = value
    .slice(0, limit)
    .map((item): ParticipantMetric | null => {
      if (!isRecord(item)) return null;
      const label = safeText(item.label, '', 12);
      const metricValue = safeText(item.value, '', 24);
      if (!label || !metricValue) return null;
      return {
        label,
        value: metricValue,
        tone: safeSignal(item.tone)
      };
    })
    .filter((item): item is ParticipantMetric => item !== null);
  return parsed.length ? parsed : fallback;
}

function sanitizeAlgorithmReasons(value: unknown, fallback: ParticipantAlgorithmReason[], limit = 8) {
  return sanitizeMetrics(value, fallback, limit);
}

function fallbackParticipant(sport: Sport, index: number): RaceParticipant {
  return {
    number: index + 1,
    name: sport === 'horse' ? '출전마' : '출전 선수',
    subtitle: '기본 정보 대기',
    stats: '최근 흐름 대기',
    trait: '확인',
    note: '상세 자료가 도착하면 자동 갱신됩니다.',
    signal: 'primary',
    profile: [],
    form: [],
    tactics: []
  };
}

export function sanitizeParticipants(value: unknown, sport: Sport, raceDate?: string) {
  if (!Array.isArray(value)) return [];
  const parsed = value
    .slice(0, maxParticipants)
    .map((item, index): RaceParticipant | null => {
      if (!isRecord(item)) return null;
      const fallback = fallbackParticipant(sport, index);
      const rawNumber = Number(item.number);
      const algorithmReasons = sanitizeAlgorithmReasons(item.algorithm_reasons ?? item.algorithmReasons, []);
      const algorithmNote = safeText(item.algorithm_note ?? item.algorithmNote, '', 180);
      const participant: RaceParticipant = {
        number: Number.isInteger(rawNumber) && rawNumber > 0 && rawNumber <= 99 ? rawNumber : index + 1,
        name: safeText(item.name, fallback.name, 24),
        subtitle: safeText(item.subtitle, fallback.subtitle, 44),
        stats: safeText(item.stats, fallback.stats, 36),
        trait: safeText(item.trait, fallback.trait, 12),
        note: safeText(item.note, fallback.note, 140),
        signal: safeSignal(item.signal),
        profile: sanitizeMetrics(item.profile, fallback.profile),
        form: sanitizeMetrics(item.form, fallback.form, 6),
        tactics: sanitizeMetrics(item.tactics, fallback.tactics, 6),
        algorithmLocked: item.algorithm_locked === true || item.algorithmLocked === true,
        algorithmNote: algorithmNote || undefined,
        algorithmReasons
      };
      return {
        ...participant,
        note: participant.algorithmNote ?? fallback.note
      };
    })
    .filter((item): item is RaceParticipant => item !== null);
  return parsed.map((participant) => ({
    ...participant,
    note: participant.algorithmNote || buildParticipantInsight(participant, sport, raceDate, parsed)
  }));
}

function fallbackMarketOdds(sport: Sport, index: number): MarketOddsEntry {
  return {
    code: 'ODDS',
    label: '배당',
    selection: '-',
    odds: 1,
    change: '대기',
    signal: 'primary'
  };
}

export function sanitizeMarketOdds(value: unknown, sport: Sport) {
  if (isRecord(value)) {
    return Object.entries(value)
      .map(([selection, odds]) => ({ selection, odds: Number(odds) }))
      .filter((entry) => Number.isFinite(entry.odds) && entry.odds >= 1 && entry.odds <= 9999)
      .sort((left, right) => left.odds - right.odds || left.selection.localeCompare(right.selection))
      .slice(0, maxMarketOdds)
      .map((entry, index): MarketOddsEntry => {
        const fallback = fallbackMarketOdds(sport, index);
        return {
          code: 'WIN',
          label: '단승',
          selection: safeText(entry.selection, fallback.selection, 18),
          odds: safeOdds(entry.odds, fallback.odds),
          change: '실시간',
          signal: index === 0 ? 'teal' : 'primary',
          source: 'live'
        };
      });
  }
  if (!Array.isArray(value)) return [];
  return value
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
        signal: safeSignal(item.signal),
        source: safeSource(item.source) ?? fallback.source
      };
    })
    .filter((item): item is MarketOddsEntry => item !== null);
}

export function sanitizePicks(value: unknown, sport: Sport) {
  if (!Array.isArray(value)) return [];
  const defaultPick: RacePick = {
    code: 'TRI',
    label: '1-2-3 순서',
    selection: '1-2-3',
    probability: 0,
    grade: '약'
  };
  const parsed = value
    .slice(0, maxPicks)
    .map((item, index): RacePick | null => {
      if (!isRecord(item)) return null;
      const fallback = defaultPick;
      const code = safeText(item.code, '', 8).toUpperCase();
      const selection = safeText(item.selection, '', 18);
      if (!code || !selection) return null;
      return {
        code,
        label: safeText(item.label, fallback.label, 20),
        selection,
        probability: safeProbability(item.probability, fallback.probability),
        grade: safeGrade(item.grade, fallback.grade)
      };
    })
    .filter((item): item is RacePick => item !== null);
  return parsed;
}

function isTrifectaEnsembleTier(value: unknown): value is TrifectaEnsembleTier {
  return value === 'T0_base' || value === 'T1_strong' || value === 'T2_top16';
}

function clampUnit(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(0, Math.min(1, numeric)) : null;
}

function safeTrifectaCombo(value: unknown) {
  const text = safeText(value, '', 18);
  const parts = text.split('-').map((part) => Number(part));
  if (parts.length !== 3 || parts.some((part) => !Number.isInteger(part) || part < 1 || part > 7) || new Set(parts).size !== 3) {
    return null;
  }
  return text;
}

export function sanitizeTrifectaEnsemble(value: unknown): TrifectaEnsemble | undefined {
  if (!isRecord(value)) return undefined;
  const pick = safeTrifectaCombo(value.pick);
  if (!pick) return undefined;
  const top5 = Array.isArray(value.top5)
    ? value.top5.map(safeTrifectaCombo).filter((item): item is string => item !== null).slice(0, 5)
    : [];
  const tier = isTrifectaEnsembleTier(value.tier)
    ? value.tier
    : 'T0_base';
  const historicalExact = clampUnit(value.tier_historical_exact ?? value.tierHistoricalExact) ?? 0;
  const selection = safeText(value.selection, '', 32);
  const coverage = clampUnit(value.coverage);
  const rawSignalStrength = value.signal_strength ?? value.signalStrength;
  const signalStrength = rawSignalStrength === null ? null : clampUnit(rawSignalStrength);
  const base: TrifectaEnsemble = {
    pick,
    top5,
    tier,
    tierHistoricalExact: historicalExact,
    source: safeText(value.source, 'ensemble_v1', 32)
  };
  return {
    ...base,
    ...(selection ? { selection } : {}),
    ...(typeof value.board_complete === 'boolean' || typeof value.boardComplete === 'boolean'
      ? { boardComplete: value.board_complete === true || value.boardComplete === true }
      : {}),
    ...(coverage === null ? {} : { coverage }),
    ...(rawSignalStrength === null || signalStrength !== null ? { signalStrength } : {})
  };
}

function sanitizeDataLayerSchema(value: unknown): DataLayerSchema | null {
  if (!isRecord(value)) return null;
  const name = safeText(value.name, '', 32);
  if (!name) return null;
  const tables = Array.isArray(value.tables)
    ? value.tables.map((table) => safeText(table, '', 48)).filter((table) => table.length > 0).slice(0, 12)
    : [];
  const rowCount = Number(value.row_count ?? value.rowCount);
  return {
    name,
    tables,
    rowCount: Number.isFinite(rowCount) && rowCount >= 0 ? Math.floor(rowCount) : 0
  };
}

export function sanitizeDataLayer(value: unknown, fallback: DataLayerStatus): DataLayerStatus {
  if (!isRecord(value)) return fallback;
  const schemas = Array.isArray(value.schemas)
    ? value.schemas.map(sanitizeDataLayerSchema).filter((schema): schema is DataLayerSchema => schema !== null)
    : [];
  return {
    ready: value.ready === true,
    storage: safeText(value.storage, 'unknown', 24),
    schemas,
    error: typeof value.error === 'string' ? safeText(value.error, '', 120) : undefined
  };
}

export function sanitizeAppSession(value: unknown, clientDeviceId: string): AppSession {
  if (!isRecord(value)) {
    return {
      userId: 'anonymous',
      deviceId: clientDeviceId,
      entitlement: 'free',
      freeAnalysisLimit: 3,
      freeAnalysisUsed: 0,
      freeAnalysisRemaining: 3,
      rewardedAnalysisCredits: 0
    };
  }
  const rawLimit = Math.floor(Number(value.free_analysis_limit ?? value.freeAnalysisLimit ?? 3));
  const limit = Number.isFinite(rawLimit) ? Math.max(0, Math.min(99, rawLimit)) : 3;
  const rawUsed = Math.floor(Number(value.free_analysis_used ?? value.freeAnalysisUsed ?? 0));
  const used = Number.isFinite(rawUsed) ? Math.max(0, Math.min(limit, rawUsed)) : 0;
  const remainingFallback = Math.max(0, limit - used);
  const rawRemaining = Math.floor(Number(value.free_analysis_remaining ?? value.freeAnalysisRemaining ?? remainingFallback));
  const remaining = Number.isFinite(rawRemaining) ? Math.max(0, Math.min(limit, rawRemaining)) : remainingFallback;
  const rawRewardedCredits = Math.floor(Number(value.rewarded_analysis_credits ?? value.rewardedAnalysisCredits ?? 0));
  const rewardedAnalysisCredits = Number.isFinite(rawRewardedCredits) ? Math.max(0, Math.min(99, rawRewardedCredits)) : 0;
  return {
    userId: safeText(value.user_id ?? value.userId, 'anonymous', 64),
    deviceId: safeText(value.device_id ?? value.deviceId, clientDeviceId, 96),
    entitlement: value.entitlement === 'pro' ? 'pro' : 'free',
    freeAnalysisLimit: limit,
    freeAnalysisUsed: used,
    freeAnalysisRemaining: remaining,
    rewardedAnalysisCredits
  };
}

export function sanitizeRosterVerification(value: unknown): RosterVerification {
  if (!isRecord(value)) {
    return {
      state: 'unknown'
    };
  }
  const rawState = safeText(value.state, 'unverified', 24);
  const state = rawState === 'verified' || rawState === 'mismatch' ? rawState : 'unverified';
  return {
    state,
    message: safeText(value.message, '', 120) || undefined,
    checkedAt: safeText(value.checked_at ?? value.checkedAt, '', 40) || undefined,
    source: safeText(value.source, '', 80) || undefined
  };
}

export function riskLevel(level: string | undefined): RaceDecision['marketRisk']['level'] {
  if (level === 'blocked' || level === 'live_blocked') return 'blocked';
  if (level === 'odds_live' || level === 'settled_result_available') return 'verified';
  if (
    level === 'odds_unavailable' ||
    level === 'pre_race' ||
    level === 'kcycle_disabled' ||
    level === 'oracle_api_not_configured'
  ) return 'caution';
  return 'neutral';
}

export function recordFromJson(value: unknown): JsonRecord {
  return isRecord(value) ? value : {};
}
