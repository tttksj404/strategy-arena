export type Sport = 'keirin' | 'horse';

export type RiskLevel = 'verified' | 'caution' | 'blocked' | 'neutral';

export type MarketSource = 'live' | 'historical' | 'settled' | 'sample' | 'unavailable';

export type RosterVerificationState = 'verified' | 'unverified' | 'mismatch' | 'unknown';

export type RosterVerification = {
  state: RosterVerificationState;
  message?: string;
  checkedAt?: string;
  source?: string;
};

export type RacePick = {
  code: string;
  label: string;
  selection: string;
  probability: number;
  grade: '강' | '중' | '약';
};

export type TrifectaEnsembleTier = 'T0_base' | 'T1_strong' | 'T2_top16';

export type TrifectaEnsemble = {
  pick: string;
  top5: readonly string[];
  tier: TrifectaEnsembleTier;
  tierHistoricalExact: number;
  source: string;
  selection?: string;
  boardComplete?: boolean;
  coverage?: number;
  signalStrength?: number | null;
};

export type ParticipantMetric = {
  label: string;
  value: string;
  tone?: 'primary' | 'teal' | 'amber' | 'rose' | 'violet';
};

export type ParticipantAlgorithmReason = ParticipantMetric;

export type RaceParticipant = {
  number: number;
  name: string;
  subtitle: string;
  stats: string;
  trait: string;
  note: string;
  signal: 'primary' | 'teal' | 'amber' | 'rose' | 'violet';
  profile: ParticipantMetric[];
  form: ParticipantMetric[];
  tactics: ParticipantMetric[];
  algorithmLocked?: boolean;
  algorithmNote?: string;
  algorithmReasons?: ParticipantAlgorithmReason[];
};

export type MarketOddsEntry = {
  code: string;
  label: string;
  selection: string;
  odds: number;
  change: string;
  signal: 'primary' | 'teal' | 'amber' | 'rose' | 'violet';
  source?: MarketSource;
};

export type RaceActualRacer = {
  number: number;
  name: string;
  rank: number;
};

export type RaceActualPayout = {
  label: string;
  winner: string;
  odds: number;
};

export type RaceActualResult = {
  actualOrder: number[];
  racers: RaceActualRacer[];
  payouts: RaceActualPayout[];
  source?: string;
  sourceUrl?: string;
};

export type DataLayerSchema = {
  name: string;
  tables: string[];
  rowCount: number;
};

export type DataLayerStatus = {
  ready: boolean;
  storage: string;
  schemas: DataLayerSchema[];
  error?: string;
};

export type AppSession = {
  userId: string;
  deviceId: string;
  entitlement: 'free' | 'pro';
  freeAnalysisLimit: number;
  freeAnalysisUsed: number;
  freeAnalysisRemaining: number;
  rewardedAnalysisCredits: number;
};

export type RaceDecision = {
  status: 'ready' | 'hold' | 'blocked' | 'settled';
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
  headline: string;
  marketUsed: boolean;
  marketSource: MarketSource;
  marketRisk: {
    level: RiskLevel;
    title: string;
    message: string;
  };
  confidence: {
    label: string;
    top1: number;
    trifecta: number;
    sample: number;
  };
  picks: RacePick[];
  trifectaEnsemble?: TrifectaEnsemble;
  participants: RaceParticipant[];
  marketOdds: MarketOddsEntry[];
  actualResult?: RaceActualResult;
  rosterVerification: RosterVerification;
  dataLayer: DataLayerStatus;
  appSession: AppSession;
  analysisError: boolean;
  officialDataPending: boolean;
  pollDelayMs: number;
  updatedAt: string;
  oddsAgeSec: number | null;
};

export type TabKey = 'home' | 'analyze' | 'pro';
