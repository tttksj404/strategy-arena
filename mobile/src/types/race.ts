export type Sport = 'keirin' | 'horse';

export type RiskLevel = 'verified' | 'caution' | 'blocked' | 'neutral';

export type RacePick = {
  code: string;
  label: string;
  selection: string;
  probability: number;
  grade: '강' | '중' | '약';
};

export type RaceDecision = {
  status: 'ready' | 'hold' | 'blocked';
  sport: Sport;
  date: string;
  meet: string;
  raceNo: number;
  headline: string;
  marketUsed: boolean;
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
  updatedAt: string;
};

export type TabKey = 'home' | 'analyze' | 'lab' | 'pro';
