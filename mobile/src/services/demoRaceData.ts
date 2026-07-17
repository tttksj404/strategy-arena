import type { MarketOddsEntry, RaceParticipant, RacePick, Sport } from '../types/race';
import { withParticipantInsights } from './participantInsight';

const keirinParticipants: RaceParticipant[] = [
  {
    number: 1,
    name: '김태훈',
    subtitle: '우수급 · 34세 · 광명 훈련',
    stats: '평균득점 87.4 · 200m 11.42초 · 기어 3.92',
    trait: '선행',
    note: '초반 주도권을 잡으면 버티는 힘이 좋지만, 외선 압박이 강하면 종반 유지력이 흔들립니다.',
    signal: 'teal',
    profile: [
      { label: '등급', value: '우수' },
      { label: '나이', value: '34세' },
      { label: '200m', value: '11.42초', tone: 'teal' },
      { label: '기어', value: '3.92' },
      { label: '훈련지', value: '광명' },
      { label: '평균득점', value: '87.4', tone: 'teal' }
    ],
    form: [
      { label: '입상률', value: '42%' },
      { label: '최근 3주', value: '2-1-0' },
      { label: '컨디션', value: '상승', tone: 'teal' }
    ],
    tactics: [
      { label: '선행', value: '46%', tone: 'teal' },
      { label: '젖히기', value: '22%' },
      { label: '마크', value: '18%' }
    ]
  },
  {
    number: 2,
    name: '박민재',
    subtitle: '우수급 · 31세 · 창원 훈련',
    stats: '평균득점 84.8 · 200m 11.59초 · 기어 3.86',
    trait: '마크',
    note: '강자 뒤 추주 안정감은 좋지만 단독으로 판을 여는 힘은 제한적입니다.',
    signal: 'primary',
    profile: [
      { label: '등급', value: '우수' },
      { label: '나이', value: '31세' },
      { label: '200m', value: '11.59초' },
      { label: '기어', value: '3.86' },
      { label: '훈련지', value: '창원' },
      { label: '평균득점', value: '84.8' }
    ],
    form: [
      { label: '입상률', value: '36%' },
      { label: '최근 3주', value: '1-1-1' },
      { label: '컨디션', value: '유지' }
    ],
    tactics: [
      { label: '마크', value: '52%', tone: 'teal' },
      { label: '추입', value: '24%' },
      { label: '선행', value: '8%' }
    ]
  },
  {
    number: 3,
    name: '이현수',
    subtitle: '선발급 · 29세 · 부산 훈련',
    stats: '평균득점 81.9 · 200m 11.66초 · 기어 3.79',
    trait: '추입',
    note: '직선 반응은 좋지만 위치 선정 변동이 커서 앞선이 무너지면 기회가 커집니다.',
    signal: 'amber',
    profile: [
      { label: '등급', value: '선발' },
      { label: '나이', value: '29세' },
      { label: '200m', value: '11.66초' },
      { label: '기어', value: '3.79' },
      { label: '훈련지', value: '부산' },
      { label: '평균득점', value: '81.9' }
    ],
    form: [
      { label: '입상률', value: '28%' },
      { label: '최근 3주', value: '0-2-1' },
      { label: '컨디션', value: '변동', tone: 'amber' }
    ],
    tactics: [
      { label: '추입', value: '49%', tone: 'teal' },
      { label: '마크', value: '26%' },
      { label: '젖히기', value: '12%' }
    ]
  },
  {
    number: 4,
    name: '정도윤',
    subtitle: '선발급 · 36세 · 광명 훈련',
    stats: '평균득점 80.7 · 200m 11.73초 · 기어 3.85',
    trait: '자력',
    note: '몸싸움 회피 성향이 있어 내선이 막히면 전개 이득이 필요합니다.',
    signal: 'amber',
    profile: [
      { label: '등급', value: '선발' },
      { label: '나이', value: '36세' },
      { label: '200m', value: '11.73초' },
      { label: '기어', value: '3.85' },
      { label: '훈련지', value: '광명' },
      { label: '평균득점', value: '80.7' }
    ],
    form: [
      { label: '입상률', value: '24%' },
      { label: '최근 3주', value: '0-1-2' },
      { label: '컨디션', value: '보통' }
    ],
    tactics: [
      { label: '젖히기', value: '31%' },
      { label: '마크', value: '28%' },
      { label: '선행', value: '20%' }
    ]
  },
  {
    number: 5,
    name: '최강우',
    subtitle: '특선급 · 32세 · 광명 훈련',
    stats: '평균득점 91.2 · 200m 11.21초 · 기어 4.00',
    trait: '젖히기',
    note: '득점, 200m 기록, 최근 흐름이 모두 앞서 중심축으로 분류됩니다.',
    signal: 'teal',
    profile: [
      { label: '등급', value: '특선', tone: 'teal' },
      { label: '나이', value: '32세' },
      { label: '200m', value: '11.21초', tone: 'teal' },
      { label: '기어', value: '4.00' },
      { label: '훈련지', value: '광명' },
      { label: '평균득점', value: '91.2', tone: 'teal' }
    ],
    form: [
      { label: '입상률', value: '68%', tone: 'teal' },
      { label: '최근 3주', value: '3-0-0', tone: 'teal' },
      { label: '컨디션', value: '강' }
    ],
    tactics: [
      { label: '젖히기', value: '43%', tone: 'teal' },
      { label: '추입', value: '27%' },
      { label: '선행', value: '18%' }
    ]
  },
  {
    number: 6,
    name: '윤성민',
    subtitle: '우수급 · 35세 · 창원 훈련',
    stats: '평균득점 83.6 · 200m 11.61초 · 기어 3.86',
    trait: '마크',
    note: '내선 운영은 안정적이나 외선 전환 타이밍이 늦어지는 편입니다.',
    signal: 'primary',
    profile: [
      { label: '등급', value: '우수' },
      { label: '나이', value: '35세' },
      { label: '200m', value: '11.61초' },
      { label: '기어', value: '3.86' },
      { label: '훈련지', value: '창원' },
      { label: '평균득점', value: '83.6' }
    ],
    form: [
      { label: '입상률', value: '31%' },
      { label: '최근 3주', value: '1-0-1' },
      { label: '컨디션', value: '유지' }
    ],
    tactics: [
      { label: '마크', value: '55%', tone: 'teal' },
      { label: '추입', value: '20%' },
      { label: '선행', value: '9%' }
    ]
  },
  {
    number: 7,
    name: '서지환',
    subtitle: '우수급 · 28세 · 부산 훈련',
    stats: '평균득점 85.1 · 200m 11.48초 · 기어 3.92',
    trait: '추입',
    note: '마지막 반 바퀴 탄력은 좋지만 초반 위치가 결과를 크게 좌우합니다.',
    signal: 'primary',
    profile: [
      { label: '등급', value: '우수' },
      { label: '나이', value: '28세' },
      { label: '200m', value: '11.48초', tone: 'teal' },
      { label: '기어', value: '3.92' },
      { label: '훈련지', value: '부산' },
      { label: '평균득점', value: '85.1' }
    ],
    form: [
      { label: '입상률', value: '39%' },
      { label: '최근 3주', value: '1-2-0' },
      { label: '컨디션', value: '상승', tone: 'teal' }
    ],
    tactics: [
      { label: '추입', value: '44%', tone: 'teal' },
      { label: '마크', value: '32%' },
      { label: '젖히기', value: '14%' }
    ]
  }
];

const horseParticipants: RaceParticipant[] = [
  {
    number: 1,
    name: '새벽질주',
    subtitle: '기수 김도윤 · 55kg · 4세 암',
    stats: '최근 4전 1-1-1 · 1200m 적성 보통',
    trait: '선입',
    note: '출발 안정감이 있고 직선 탄력은 평균 이상입니다. 안쪽 게이트 이점이 있습니다.',
    signal: 'primary',
    profile: [
      { label: '말', value: '4세 암' },
      { label: '기수', value: '김도윤' },
      { label: '부담중량', value: '55kg' },
      { label: '조교', value: '중상' },
      { label: '마체', value: '+2kg' },
      { label: '거리', value: '보통' }
    ],
    form: [
      { label: '복승률', value: '44%' },
      { label: '최근 4전', value: '1-1-1' },
      { label: '게이트', value: '1번' }
    ],
    tactics: [
      { label: '선입', value: '48%', tone: 'teal' },
      { label: '선행', value: '26%' },
      { label: '추입', value: '12%' }
    ]
  },
  {
    number: 3,
    name: '스톰레이크',
    subtitle: '기수 이준 · 54kg · 3세 수',
    stats: '최근 4전 2-0-0 · 출발 빠름',
    trait: '선행',
    note: '게이트 반응이 빠른 편입니다. 페이스가 빨라지면 종반 버티기가 관건입니다.',
    signal: 'teal',
    profile: [
      { label: '말', value: '3세 수' },
      { label: '기수', value: '이준' },
      { label: '부담중량', value: '54kg', tone: 'teal' },
      { label: '조교', value: '상' },
      { label: '마체', value: '+1kg' },
      { label: '거리', value: '양호' }
    ],
    form: [
      { label: '복승률', value: '50%', tone: 'teal' },
      { label: '최근 4전', value: '2-0-0' },
      { label: '게이트', value: '3번' }
    ],
    tactics: [
      { label: '선행', value: '61%', tone: 'teal' },
      { label: '선입', value: '24%' },
      { label: '추입', value: '5%' }
    ]
  },
  {
    number: 5,
    name: '골든포커스',
    subtitle: '기수 문태오 · 57kg · 5세 거',
    stats: '최근 4전 3-0-0 · 지구력 우위',
    trait: '선입',
    note: '기록과 지구력이 앞서지만 부담중량 57kg이 막판 반응에 영향을 줄 수 있습니다.',
    signal: 'teal',
    profile: [
      { label: '말', value: '5세 거', tone: 'teal' },
      { label: '기수', value: '문태오' },
      { label: '부담중량', value: '57kg', tone: 'amber' },
      { label: '조교', value: '상' },
      { label: '마체', value: '-1kg' },
      { label: '거리', value: '우수', tone: 'teal' }
    ],
    form: [
      { label: '복승률', value: '67%', tone: 'teal' },
      { label: '최근 4전', value: '3-0-0', tone: 'teal' },
      { label: '게이트', value: '5번' }
    ],
    tactics: [
      { label: '선입', value: '54%', tone: 'teal' },
      { label: '추입', value: '25%' },
      { label: '선행', value: '15%' }
    ]
  },
  {
    number: 6,
    name: '청운대로',
    subtitle: '기수 안유진 · 54kg · 4세 수',
    stats: '최근 4전 0-1-2 · 후반 반응 안정',
    trait: '추입',
    note: '후반 반응은 안정적이지만 초반 자리 손실이 잦아 전개 의존도가 있습니다.',
    signal: 'amber',
    profile: [
      { label: '말', value: '4세 수' },
      { label: '기수', value: '안유진' },
      { label: '부담중량', value: '54kg', tone: 'teal' },
      { label: '조교', value: '중' },
      { label: '마체', value: '0kg' },
      { label: '거리', value: '보통' }
    ],
    form: [
      { label: '복승률', value: '31%' },
      { label: '최근 4전', value: '0-1-2' },
      { label: '게이트', value: '6번' }
    ],
    tactics: [
      { label: '추입', value: '58%', tone: 'teal' },
      { label: '선입', value: '26%' },
      { label: '선행', value: '4%' }
    ]
  },
  {
    number: 8,
    name: '레드노바',
    subtitle: '기수 한서우 · 53kg · 3세 암',
    stats: '최근 4전 0-0-1 · 부담중량 이점',
    trait: '자유',
    note: '부담중량은 좋지만 최근 성적과 모래 반응은 확인이 필요합니다.',
    signal: 'rose',
    profile: [
      { label: '말', value: '3세 암' },
      { label: '기수', value: '한서우' },
      { label: '부담중량', value: '53kg', tone: 'teal' },
      { label: '조교', value: '중하' },
      { label: '마체', value: '+4kg', tone: 'amber' },
      { label: '거리', value: '확인' }
    ],
    form: [
      { label: '복승률', value: '18%' },
      { label: '최근 4전', value: '0-0-1' },
      { label: '게이트', value: '8번' }
    ],
    tactics: [
      { label: '자유', value: '38%' },
      { label: '선입', value: '28%' },
      { label: '추입', value: '20%' }
    ]
  }
];

const keirinMarketOdds: MarketOddsEntry[] = [
  { code: 'WIN', label: '단승', selection: '5', odds: 2.1, change: '과거 배당 예시', signal: 'teal', source: 'sample' },
  { code: 'QNL', label: '복승', selection: '5-1', odds: 4.8, change: '과거 배당 예시', signal: 'primary', source: 'sample' },
  { code: 'EXA', label: '쌍승', selection: '5-1', odds: 7.6, change: '과거 배당 예시', signal: 'amber', source: 'sample' },
  { code: 'TRI', label: '삼쌍', selection: '5-1-7', odds: 21.4, change: '과거 배당 예시', signal: 'violet', source: 'sample' }
];

const horseMarketOdds: MarketOddsEntry[] = [
  { code: 'WIN', label: '단승', selection: '5', odds: 2.4, change: '과거 배당 예시', signal: 'teal', source: 'sample' },
  { code: 'QNL', label: '복승', selection: '5-3', odds: 5.2, change: '과거 배당 예시', signal: 'primary', source: 'sample' },
  { code: 'EXA', label: '쌍승', selection: '5-3', odds: 8.1, change: '과거 배당 예시', signal: 'amber', source: 'sample' },
  { code: 'TRI', label: '삼쌍', selection: '5-3-1', odds: 24.6, change: '과거 배당 예시', signal: 'violet', source: 'sample' }
];

const keirinPicks: RacePick[] = [
  { code: 'TOP1', label: '1착 후보', selection: '5', probability: 0.62, grade: '중' },
  { code: 'QNL', label: '복승 조합', selection: '5-1', probability: 0.31, grade: '중' },
  { code: 'TRI', label: '1-2-3 순서', selection: '5-1-7', probability: 0.18, grade: '약' },
  { code: 'TRB', label: '삼복 조합', selection: '1-5-7', probability: 0.26, grade: '중' }
];

const horsePicks: RacePick[] = [
  { code: 'TOP1', label: '1착 후보', selection: '5', probability: 0.62, grade: '중' },
  { code: 'QNL', label: '복승 조합', selection: '5-3', probability: 0.31, grade: '중' },
  { code: 'TRI', label: '1-2-3 순서', selection: '5-3-1', probability: 0.18, grade: '약' },
  { code: 'TRB', label: '삼복 조합', selection: '1-3-5', probability: 0.26, grade: '중' }
];

export function demoParticipants(sport: Sport): RaceParticipant[] {
  return withParticipantInsights(sport === 'horse' ? horseParticipants : keirinParticipants, sport);
}

export function demoMarketOdds(sport: Sport): MarketOddsEntry[] {
  return sport === 'horse' ? horseMarketOdds : keirinMarketOdds;
}

export function demoPicks(sport: Sport): RacePick[] {
  return sport === 'horse' ? horsePicks : keirinPicks;
}
