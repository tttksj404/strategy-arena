import type { RaceDecision } from '../types/race';


function isSameRace(previous: RaceDecision, next: RaceDecision): boolean {
  return previous.sport === next.sport &&
    previous.date === next.date &&
    previous.meet === next.meet &&
    previous.raceNo === next.raceNo;
}


function hasVisibleAnalysis(decision: RaceDecision): boolean {
  return decision.picks.length > 0 && !decision.officialDataPending && !decision.analysisError;
}


export function shouldKeepVisibleDecision(previous: RaceDecision | null, next: RaceDecision): boolean {
  return previous !== null &&
    isSameRace(previous, next) &&
    hasVisibleAnalysis(previous) &&
    (next.officialDataPending || next.analysisError);
}
