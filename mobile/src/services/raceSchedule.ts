import type { Sport } from '../types/race';

export const raceVenues: Record<Sport, readonly string[]> = {
  keirin: ['광명'],
  horse: ['서울', '부경', '제주']
};

export type RaceDateSchedule = Partial<Record<Sport, Record<string, readonly string[]>>>;

export function defaultRaceCount(sport: Sport, meet?: string): number {
  if (sport === 'keirin') return 16;
  return meet === '제주' ? 8 : 11;
}

export function defaultRaceVenue(sport: Sport): string {
  return sport === 'keirin' ? '광명' : '서울';
}

export function todayInKorea(now = new Date()): string {
  const parts = new Intl.DateTimeFormat('en-CA', {
    day: '2-digit',
    month: '2-digit',
    timeZone: 'Asia/Seoul',
    year: 'numeric'
  }).formatToParts(now);
  const year = parts.find((part) => part.type === 'year')?.value ?? String(now.getUTCFullYear());
  const month = parts.find((part) => part.type === 'month')?.value ?? '01';
  const day = parts.find((part) => part.type === 'day')?.value ?? '01';
  return `${year}-${month}-${day}`;
}

export function availableRaceDates(sport: Sport, meet: string, schedule?: RaceDateSchedule): string[] {
  const serverDates = schedule?.[sport]?.[meet] ?? schedule?.[sport]?.[defaultRaceVenue(sport)];
  const dates = serverDates && serverDates.length > 0 ? [...serverDates] : [];
  return [...new Set(dates.filter((date) => /^\d{4}-\d{2}-\d{2}$/.test(date)))].sort();
}

export function nearestRaceDate(
  sport: Sport,
  meet: string,
  preferredDate: string,
  schedule?: RaceDateSchedule
): string {
  const dates = availableRaceDates(sport, meet, schedule);
  if (dates.length === 0) return preferredDate;
  if (dates.includes(preferredDate)) return preferredDate;
  const preferredTime = dateTime(preferredDate);
  if (preferredTime === null) return dates[dates.length - 1] ?? preferredDate;
  return dates
    .map((date: string) => ({ date, distance: Math.abs((dateTime(date) ?? preferredTime) - preferredTime) }))
    .sort((left, right) => left.distance - right.distance || left.date.localeCompare(right.date))[0]?.date ?? preferredDate;
}

export function adjacentRaceDate(
  sport: Sport,
  meet: string,
  currentDate: string,
  step: -1 | 1,
  schedule?: RaceDateSchedule
): string {
  const dates = availableRaceDates(sport, meet, schedule);
  const normalized = nearestRaceDate(sport, meet, currentDate, schedule);
  const index = dates.indexOf(normalized);
  const nextIndex = Math.max(0, Math.min(dates.length - 1, index + step));
  return dates[nextIndex] ?? normalized;
}

export function formatRaceDateLabel(date: string): string {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return date;
  const weekday = ['일', '월', '화', '수', '목', '금', '토'][parsed.getUTCDay()];
  return `${date.slice(5).replace('-', '.')} ${weekday}`;
}

function dateTime(date: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  const parsed = new Date(`${date}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed.getTime();
}
