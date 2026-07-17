import type { ParticipantMetric, RaceParticipant, Sport } from '../types/race';

type MetricGroup = Pick<RaceParticipant, 'form' | 'profile' | 'tactics'>;
type UnusualMetric = {
  readonly label: string;
  readonly value: string;
  readonly zscore: number;
};

function metricValue(items: ParticipantMetric[], label: string): string {
  return items.find((item) => item.label === label)?.value ?? '-';
}

function metricNumber(items: ParticipantMetric[], label: string): number | null {
  const value = metricValue(items, label);
  const match = value.match(/-?\d+(?:\.\d+)?/);
  if (!match) return null;
  const numeric = Number(match[0]);
  return Number.isFinite(numeric) ? numeric : null;
}

function allMetrics(participant: RaceParticipant): ParticipantMetric[] {
  return [...participant.profile, ...participant.form, ...participant.tactics];
}

function unusualMetric(participant: RaceParticipant, peers: RaceParticipant[]): UnusualMetric {
  const metrics = allMetrics(participant);
  let best: (UnusualMetric & { readonly magnitude: number }) | null = null;
  for (const metric of metrics) {
    const current = metricNumber([metric], metric.label);
    if (current === null) continue;
    const values = peers
      .map((peer) => metricNumber(allMetrics(peer), metric.label))
      .filter((value): value is number => value !== null && Number.isFinite(value));
    if (values.length < 2) continue;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
    if (variance <= 0) continue;
    const zscore = (current - mean) / Math.sqrt(variance);
    const candidate = {
      label: metric.label,
      value: metric.value,
      zscore,
      magnitude: Math.abs(zscore)
    };
    if (best === null || candidate.magnitude > best.magnitude) {
      best = candidate;
    }
  }
  return best ?? {
    label: participant.profile[0]?.label ?? '기본 지표',
    value: participant.profile[0]?.value ?? '-',
    zscore: 0
  };
}

function frameIndex(participant: RaceParticipant, raceDate?: string) {
  const dateSeed = [...(raceDate ?? '')].reduce((sum, char) => {
    const numeric = Number(char);
    return Number.isInteger(numeric) ? sum + numeric : sum;
  }, 0);
  return (participant.number + dateSeed) % 3;
}

function topicParticle(value: string) {
  const last = value.trim().at(-1);
  if (!last) return '는';
  const code = last.charCodeAt(0);
  if (code >= 0xac00 && code <= 0xd7a3 && (code - 0xac00) % 28 > 0) return '은';
  return '는';
}

function leadSentence(participant: RaceParticipant, peers: RaceParticipant[], raceDate?: string) {
  const metric = unusualMetric(participant, peers);
  const zText = `${metric.zscore >= 0 ? '+' : ''}${metric.zscore.toFixed(1)}σ`;
  const direction = metric.zscore >= 0 ? '높습니다' : '낮습니다';
  const frames = [
    `${participant.name}의 가장 특이한 지표는 ${metric.label} ${metric.value}로, 경주 평균보다 ${zText} ${direction}.`,
    `경주 내 편차가 가장 큰 항목은 ${participant.name}의 ${metric.label}이며 ${metric.value}(${zText})입니다.`,
    `${participant.name}${topicParticle(participant.name)} ${metric.label} ${metric.value}에서 가장 두드러져 평균 대비 ${zText} 차이를 보입니다.`
  ] as const;
  return frames[frameIndex(participant, raceDate)] ?? frames[0];
}

function topTactic(tactics: ParticipantMetric[]): { label: string; value: string; percent: number } {
  return tactics
    .map((item) => ({
      label: item.label,
      value: item.value,
      percent: metricNumber([item], item.label) ?? -1
    }))
    .sort((left, right) => right.percent - left.percent)[0] ?? { label: '전법', value: '-', percent: -1 };
}

function recentStrength(value: string): string {
  const counts = value.match(/\d+/g)?.map(Number) ?? [];
  const [wins = 0, seconds = 0, thirds = 0] = counts;
  const podiums = wins + seconds + thirds;
  if (wins >= 3) return '최근 승수 흐름이 강합니다';
  if (wins >= 1 && podiums >= 3) return '최근 3착권 유지가 안정적입니다';
  if (podiums >= 2) return '최근 입상권에는 걸치지만 1착 결정력은 더 확인해야 합니다';
  return '최근 흐름은 보수적으로 봐야 합니다';
}

function keirinInsight(participant: RaceParticipant, metrics: MetricGroup, peers: RaceParticipant[], raceDate?: string): string {
  const score = metricNumber(metrics.profile, '평균득점');
  const sprint = metricValue(metrics.profile, '200m');
  const sprintNumber = metricNumber(metrics.profile, '200m');
  const gear = metricValue(metrics.profile, '기어');
  const podium = metricNumber(metrics.form, '입상률');
  const recent = metricValue(metrics.form, '최근 3주');
  const condition = metricValue(metrics.form, '컨디션');
  const tactic = topTactic(metrics.tactics);

  const scoreRead = score === null
    ? '득점 자료는 보수적으로 봅니다'
    : score >= 90
      ? '축 후보권'
      : score >= 85
        ? '상위권 추격권'
        : '전개 도움 필요';

  const sprintRead = sprintNumber === null
    ? '순발력 추가 확인'
    : sprintNumber <= 11.35
      ? '순간 가속 우위'
      : sprintNumber <= 11.55
        ? '초반 자리 경쟁 가능'
        : '빠른 앞선 압박 부담';

  const podiumRead = podium === null
    ? '입상 안정성 자료 대기'
    : podium >= 60
      ? '3착권 안정성'
      : podium >= 35
        ? '조합권 검토'
        : '순위권 고정 위험';

  const tacticRead = tactic.percent >= 40
    ? `${tactic.label} 전개 강점`
    : `${tactic.label} 흐름 의존`;

  const scoreText = score === null ? '-' : score.toFixed(1);
  const podiumText = podium === null ? '-' : `${podium}%`;
  return `${leadSentence(participant, peers, raceDate)} ${participant.name}: 평균득점 ${scoreText}점은 ${scoreRead}, 200m ${sprint}는 ${sprintRead}, 입상률 ${podiumText}는 ${podiumRead}, ${tactic.label} ${tactic.value}는 ${tacticRead}입니다. 최근 ${recent}, 컨디션 ${condition}, 기어 ${gear}까지 같이 봅니다.`;
}

function horseInsight(participant: RaceParticipant, metrics: MetricGroup, peers: RaceParticipant[], raceDate?: string): string {
  const quinella = metricNumber(metrics.form, '복승률');
  const recent = metricValue(metrics.form, '최근 4전');
  const gate = metricValue(metrics.form, '게이트');
  const weight = metricValue(metrics.profile, '부담중량');
  const weightNumber = metricNumber(metrics.profile, '부담중량');
  const body = metricValue(metrics.profile, '마체');
  const distance = metricValue(metrics.profile, '거리');
  const training = metricValue(metrics.profile, '조교');
  const tactic = topTactic(metrics.tactics);

  const quinellaRead = quinella === null
    ? '복승률 자료 대기'
    : quinella >= 60
      ? '2착권 안정성'
      : quinella >= 40
        ? '상대 조합권'
        : '중심 고정 주의';

  const weightRead = weightNumber === null
    ? '부담중량 추가 확인'
    : weightNumber >= 57
      ? '막판 반응 리스크'
      : weightNumber <= 54
        ? '체력 소모 이점'
        : '평균권 변수';

  const gateRead = gate === '1번' || gate === '2번' || gate === '3번'
    ? '초반 자리 선점 유리'
    : '초반 외곽 손실 확인';

  const bodyRead = body.startsWith('+') && Math.abs(metricNumber(metrics.profile, '마체') ?? 0) >= 4
    ? '컨디션 리스크'
    : '기본 상태 양호';

  const quinellaText = quinella === null ? '-' : `${quinella}%`;
  return `${leadSentence(participant, peers, raceDate)} ${participant.name}: 복승률 ${quinellaText}는 ${quinellaRead}, 최근 4전 ${recent}은 ${recentStrength(recent)}, 게이트 ${gate}은 ${gateRead}, 부담중량 ${weight}은 ${weightRead}, 마체 ${body}은 ${bodyRead}입니다. 주행성향은 ${tactic.label} ${tactic.value}, 거리 ${distance}, 조교 ${training}까지 같이 봅니다.`;
}

export function buildParticipantInsight(
  participant: RaceParticipant,
  sport: Sport,
  raceDate?: string,
  peers: RaceParticipant[] = [participant]
): string {
  const metrics = {
    form: participant.form,
    profile: participant.profile,
    tactics: participant.tactics
  };
  return sport === 'horse'
    ? horseInsight(participant, metrics, peers, raceDate)
    : keirinInsight(participant, metrics, peers, raceDate);
}

export function withParticipantInsights(participants: RaceParticipant[], sport: Sport, raceDate?: string): RaceParticipant[] {
  return participants.map((participant) => ({
    ...participant,
    note: buildParticipantInsight(participant, sport, raceDate, participants)
  }));
}
