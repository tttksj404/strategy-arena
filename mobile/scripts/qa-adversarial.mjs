import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createServer as createNetServer } from 'node:net';
import { tmpdir } from 'node:os';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';
import { exitIfFailed, spawnQa } from './qa-utils.mjs';

const artifactRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-adversarial-'));
const reportPath = process.env.RACELENS_QA_CASE
  ? join(artifactRoot, 'qa_adversarial_report.md')
  : resolve('..', 'runs', 'qa_adversarial_report.md');
const distRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-adversarial-dist-'));

async function freePort() {
  const server = createNetServer();
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  await new Promise((resolveClose) => server.close(resolveClose));
  if (!address || typeof address === 'string') throw new Error('Unable to allocate QA port');
  return address.port;
}

const appPort = await freePort();
const apiPort = await freePort();
const apiBaseUrl = `http://127.0.0.1:${apiPort}`;
let activeCase = null;

const exportResult = spawnQa('npx', ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
  env: {
    ...process.env,
    EXPO_PUBLIC_RACELENS_API_BASE_URL: apiBaseUrl,
    EXPO_PUBLIC_RACELENS_ANALYTICS_URL: `${apiBaseUrl}/api/ux-events`,
    EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '0',
    EXPO_PUBLIC_RACELENS_BILLING_MODE: 'disabled'
  }
});
exitIfFailed(exportResult);

function bundleName() {
  const metadata = JSON.parse(readFileSync(join(distRoot, 'metadata.json'), 'utf8'));
  const bundle = metadata?.bundler === 'metro'
    ? readFileSync(join(distRoot, 'index.html'), 'utf8').match(/_expo\/static\/js\/web\/[^"]+\.js/)?.[0]
    : undefined;
  if (!bundle) throw new Error('Unable to locate Expo web bundle');
  return bundle.replace('_expo/static/js/web/', '');
}

const bundleText = readFileSync(join(distRoot, '_expo/static/js/web', bundleName()), 'utf8');
if (!bundleText.includes(apiBaseUrl)) {
  throw new Error(`Adversarial QA API base URL was not inlined into the web bundle: ${apiBaseUrl}`);
}

function headers(type = 'application/json') {
  return {
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'GET,POST,OPTIONS',
    'access-control-allow-headers': 'content-type,x-racelens-device-id,x-racelens-platform,x-racelens-analytics',
    'content-type': type
  };
}

function writeJson(response, status, body) {
  response.writeHead(status, headers());
  response.end(JSON.stringify(body));
}

function defaultAppSession(remaining = 3) {
  const safeRemaining = Math.max(0, remaining);
  return {
    app_session: {
      user_id: 'usr_qa',
      device_id: 'dev_qa',
      entitlement: 'free',
      free_analysis_limit: 3,
      free_analysis_used: 3 - safeRemaining,
      free_analysis_remaining: remaining
    },
    data_layer: {
      ready: true,
      schemas: [{ name: 'qa', tables: ['adversarial'], row_count: 1 }],
      storage: 'qa'
    }
  };
}

function readyDecision(overrides = {}) {
  return {
    status: 'ready',
    sport: 'keirin',
    date: '2026-07-03',
    meet: '광명',
    race_no: 1,
    market_used: true,
    roster_verification: {
      state: 'verified',
      message: 'QA mock roster verified',
      source: 'qa-adversarial'
    },
    market_risk: {
      level: 'odds_live',
      message: '실시간 배당과 공식 출전표가 연결되었습니다.'
    },
    participants: [
      participant(1, '김현우'),
      participant(2, '박정민'),
      participant(3, '이로운')
    ],
    rows: [
      { bno: 1, pwin: 0.43 },
      { bno: 2, pwin: 0.31 },
      { bno: 3, pwin: 0.19 }
    ],
    picks: [
      { code: 'TOP1', label: '1착 후보', selection: '1', probability: 0.43, grade: '강' },
      { code: 'QNL', label: '복승 후보', selection: '1-2', probability: 0.31, grade: '중' },
      { code: 'TRI', label: '1-2-3 순서', selection: '1-2-3', probability: 0.057, grade: '약' }
    ],
    market_odds: [
      { code: 'WIN', label: '단승', selection: '1', odds: 2.4, change: '실시간', signal: 'teal' },
      { code: 'TRI', label: '삼쌍', selection: '1-2-3', odds: 7.8, change: '실시간', signal: 'primary' }
    ],
    data_layer: {
      ready: true,
      storage: 'qa',
      schemas: [{ name: 'qa', tables: ['live_decision'], row_count: 3 }]
    },
    app_session: {
      user_id: 'usr_qa',
      device_id: 'dev_qa',
      entitlement: 'free',
      free_analysis_limit: 3,
      free_analysis_used: 0,
      free_analysis_remaining: 3
    },
    poll_delay_ms: 15000,
    updated_at: '2026-07-03T05:30:00.000Z',
    ...overrides
  };
}

function pendingDecision(appSession) {
  return readyDecision({
    app_session: appSession,
    market_odds: [],
    market_risk: {
      level: 'caution',
      message: '공식 출전표를 다시 확인하고 있습니다.',
      title: '공식 출전표 확인 중'
    },
    market_used: false,
    participants: [],
    picks: [],
    rows: [],
    snapshot_phase: 'pending',
    status: 'hold'
  });
}

function participant(number, name) {
  return {
    number,
    name,
    subtitle: '광명팀 / 선행',
    stats: '평균득점 92.4',
    trait: '선행',
    note: '최근 흐름과 전법 지표를 함께 확인합니다.',
    signal: 'teal',
    profile: [{ label: '나이', value: '31세', tone: 'primary' }],
    form: [{ label: '최근 3주', value: '상승', tone: 'teal' }],
    tactics: [{ label: '선행', value: '44%', tone: 'teal' }]
  };
}

function longPayload() {
  const longName = '가나다라마바사아자차카타파하초장문출전자명테스트🙂🙂';
  return readyDecision({
    meet: '광명<>특수&문자/검증',
    market_risk: {
      level: 'odds_live',
      message: '특수문자와 초장문 이름이 들어와도 레이아웃은 겹치지 않아야 합니다.'
    },
    participants: [
      participant(1, longName),
      participant(2, `${longName}B`),
      participant(3, `${longName}C`)
    ],
    picks: [
      { code: 'TOP1', label: '1착 후보', selection: '1', probability: 0.51, grade: '강' },
      { code: 'QNL', label: '복승 후보', selection: '1-2', probability: 0.34, grade: '중' },
      { code: 'TRI', label: '1-2-3 순서', selection: '1-2-3', probability: 0.08, grade: '중' }
    ]
  });
}

function handleApi(request, response) {
  if (request.method === 'OPTIONS') {
    response.writeHead(204, headers());
    response.end();
    return;
  }
  if (request.url?.startsWith('/api/ux-events')) {
    if (activeCase?.analyticsDown) {
      writeJson(response, 503, { error: 'analytics_down' });
      return;
    }
    response.writeHead(204, headers());
    response.end();
    return;
  }
  if (request.url?.startsWith('/recent')) {
    writeJson(response, activeCase?.recentStatus ?? 200, {
      sport: 'keirin',
      meet: '광명',
      days: ['2026-07-03', '2026-07-04', '2026-07-05'],
      default_race_no: 1,
      race_count: 12
    });
    return;
  }
  if (request.url?.startsWith('/api/app-session')) {
    if (activeCase?.appSessionFailure) {
      writeJson(response, 503, { error: 'session_down' });
      return;
    }
    writeJson(response, 200, activeCase?.appSessionPayload ?? defaultAppSession(3));
    return;
  }
  if (request.url?.startsWith('/api/live-decisions/preload')) {
    response.writeHead(204, headers());
    response.end();
    return;
  }
  if (!request.url?.startsWith('/api/live-decision')) {
    writeJson(response, 404, { error: 'not_found' });
    return;
  }
  activeCase.liveRequests += 1;
  if (activeCase.kind === 'html') {
    response.writeHead(200, headers('text/html'));
    response.end('<!doctype html><title>not json</title><h1>proxy error</h1>');
    return;
  }
  if (activeCase.kind === 'status') {
    writeJson(response, activeCase.status, { error: 'forced_status' });
    return;
  }
  if (activeCase.kind === 'timeout') {
    setTimeout(() => writeJson(response, 200, readyDecision()), 15000);
    return;
  }
  if (activeCase.kind === 'pending-after-ready' && activeCase.liveRequests > 1) {
    writeJson(response, 200, pendingDecision(activeCase.payload.app_session));
    return;
  }
  if (activeCase.kind === 'flap' && activeCase.liveRequests === 1) {
    writeJson(response, 503, { error: 'temporary_network_flap' });
    return;
  }
  writeJson(response, 200, activeCase.payload ?? readyDecision());
}

function handleStatic(request, response) {
  const cleanPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
  const candidate = cleanPath === '/' ? join(distRoot, 'index.html') : join(distRoot, cleanPath);
  const file = existsSync(candidate) && statSync(candidate).isFile() ? candidate : join(distRoot, 'index.html');
  const type = extname(file) === '.js' ? 'application/javascript' :
    extname(file) === '.css' ? 'text/css' :
    extname(file) === '.png' ? 'image/png' :
    extname(file) === '.ico' ? 'image/x-icon' :
    'text/html';
  response.writeHead(200, { 'content-type': type });
  createReadStream(file).pipe(response);
}

const apiServer = createServer(handleApi);
const staticServer = createServer(handleStatic);
await Promise.all([
  new Promise((resolveListen) => apiServer.listen(apiPort, '127.0.0.1', resolveListen)),
  new Promise((resolveListen) => staticServer.listen(appPort, '127.0.0.1', resolveListen))
]);

function collectErrors(page) {
  const errors = [];
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() === 'error' && !text.includes('Failed to load resource')) errors.push(text);
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

async function inspectPage(page) {
  return page.evaluate(() => {
    const bodyText = document.body.innerText;
    return {
      badText: bodyText.includes('�'),
      bodyText,
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      visibleSpinner: Boolean(document.querySelector('[aria-label="분석 요청 처리 중"]'))
    };
  });
}

async function overlappingText(page) {
  return page.evaluate(() => {
    const elements = [...document.querySelectorAll('div, span, p, h1, h2, h3, h4, button, [role="button"]')]
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const text = (element.textContent ?? '').replace(/\s+/g, ' ').trim();
        const hasTextChild = [...element.children].some((child) => (child.textContent ?? '').trim().length > 0);
        const inTab = Boolean(element.closest('[role="tab"]'));
        return {
          bottom: rect.bottom,
          hasTextChild,
          height: rect.height,
          inTab,
          left: rect.left,
          right: rect.right,
          text,
          top: rect.top,
          width: rect.width
        };
      })
      .filter((item) => !item.inTab && !item.hasTextChild && item.text.length > 0 && item.width > 6 && item.height > 6 && item.bottom > 0 && item.top < window.innerHeight);
    const overlaps = [];
    for (let leftIndex = 0; leftIndex < elements.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < elements.length; rightIndex += 1) {
        const left = elements[leftIndex];
        const right = elements[rightIndex];
        const contains = left.left <= right.left && left.right >= right.right && left.top <= right.top && left.bottom >= right.bottom;
        const contained = right.left <= left.left && right.right >= left.right && right.top <= left.top && right.bottom >= left.bottom;
        if (contains || contained) continue;
        const horizontal = Math.min(left.right, right.right) - Math.max(left.left, right.left);
        const vertical = Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top);
        if (horizontal > 2 && vertical > 2) {
          overlaps.push(`${left.text.slice(0, 32)} <> ${right.text.slice(0, 32)}`);
        }
      }
    }
    return overlaps.slice(0, 5);
  });
}

async function requestAnalysis(page) {
  await page.getByRole('button', { name: /모델 신호 보기|한도 안내 보기/ }).click();
}

async function assertBase(page, errors, caseName) {
  const metrics = await inspectPage(page);
  if (errors.length) throw new Error(`${caseName}: console/page errors:\n${errors.join('\n')}`);
  if (metrics.badText) throw new Error(`${caseName}: broken replacement character rendered`);
  if (metrics.overflow > 1) throw new Error(`${caseName}: horizontal overflow ${metrics.overflow}px`);
  return metrics;
}

async function runCase(browser, item) {
  activeCase = {
    analyticsDown: false,
    appSessionFailure: false,
    appSessionPayload: defaultAppSession(3),
    kind: 'ready',
    liveRequests: 0,
    payload: readyDecision(),
    ...item.mock
  };
  const context = await browser.newContext({
    colorScheme: 'dark',
    deviceScaleFactor: 2,
    isMobile: item.viewport.width < 700,
    viewport: item.viewport
  });
  const page = await context.newPage();
  const errors = collectErrors(page);
  try {
    await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
    if (item.fontScale) {
      await page.evaluate(() => {
        document.documentElement.style.fontSize = '130%';
      });
    }
    await item.flow(page, activeCase);
    const metrics = await assertBase(page, errors, item.name);
    const screenshot = join(artifactRoot, `${item.name}.png`);
    await page.screenshot({ path: screenshot, fullPage: false });
    return {
      name: item.name,
      pass: true,
      finding: item.finding,
      fix: item.fix,
      screenshot,
      liveRequests: activeCase.liveRequests,
      overflow: metrics.overflow
    };
  } finally {
    await context.close();
  }
}

const cases = [
  {
    name: '01-html-live-decision',
    viewport: { width: 390, height: 844 },
    mock: { kind: 'html' },
    finding: 'HTML 응답이 JSON 파서에서 오류 상태로 전환되어 후보/가짜 데이터가 숨겨짐',
    fix: 'live-decision JSON 파싱 실패를 unavailableDecision으로 변환',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByText(/응답 형식이 올바르지 않아 결과를 표시하지 않습니다/).waitFor({ timeout: 8000 });
      await page.getByTestId('analysis-empty-state').waitFor();
    }
  },
  {
    name: '02-missing-required-fields',
    viewport: { width: 390, height: 844 },
    mock: {
      payload: {
        status: 'ready',
        picks: null,
        participants: [],
        market_risk: { level: 'odds_live', message: '필수 필드 누락 QA' },
        data_layer: { ready: true, schemas: [], storage: 'qa' },
        roster_verification: {
          state: 'unverified',
          message: '공식 출주표를 아직 확인하지 못했습니다',
          checked_at: null
        },
        app_session: defaultAppSession(3).app_session
      }
    },
    finding: '필수 예측 필드 누락 시 빈 분석 상태로 전환',
    fix: 'predictionAvailable가 1·2·3착 pick과 top1 확률을 동시에 요구',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByText(/공식 출주표를 아직 확인하지 못했습니다/).waitFor({ timeout: 8000 });
      await page.getByTestId('roster-waiting-state').waitFor();
    }
  },
  {
    name: '03-out-of-range-values',
    viewport: { width: 390, height: 844 },
    mock: {
      payload: readyDecision({
        race_no: 0,
        participants: [participant(0, ''), participant(2, '정상참가자'), participant(3, '확인참가자')],
        rows: [
          { bno: 1, pwin: -0.2 },
          { bno: 2, pwin: 1.7 },
          { bno: 3, pwin: 'NaN' }
        ],
        picks: [
          { code: 'TOP1', label: '1착 후보', selection: '1', probability: -0.2, grade: '강' },
          { code: 'QNL', label: '복승 후보', selection: '1-2', probability: 1.7, grade: '중' },
          { code: 'TRI', label: '1-2-3 순서', selection: '1-2-3', probability: 'NaN', grade: '약' }
        ]
      })
    },
    finding: '확률/이름/번호 이상값이 sanitizer에서 거부 또는 클램프됨',
    fix: 'racePayload 확률·텍스트 sanitizer와 app-session NaN 보정 확인',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByText('광명 1R 분석').waitFor({ timeout: 8000 });
      await page.getByTestId('analysis-empty-state').waitFor();
      const text = (await page.locator('body').innerText()).replace(/\s+/g, ' ');
      if (/(170%|-20%|NaN)/.test(text)) throw new Error('Out-of-range probability leaked to UI');
    }
  },
  {
    name: '04-long-special-text',
    viewport: { width: 320, height: 720 },
    mock: { payload: longPayload() },
    finding: '초장문 한글+이모지는 안전하게 줄이고, 서버 meet 특수문자는 선택 조건 밖이라 표시하지 않음',
    fix: 'safeText truncation, keep-all/anywhere wrapping, overlap gate',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByText('광명 1R 분석').waitFor({ timeout: 8000 });
      const overlaps = await overlappingText(page);
      if (overlaps.length) throw new Error(`Text bounding boxes overlap: ${overlaps.join(' | ')}`);
    }
  },
  {
    name: '05-timeout-live-decision',
    viewport: { width: 390, height: 844 },
    mock: { kind: 'timeout' },
    finding: '15초 지연 응답은 10초 AbortController timeout 후 안내로 종료',
    fix: 'live-decision timeout을 unavailableDecision으로 변환하고 스피너 종료',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByLabel('분석 요청 처리 중').waitFor({ timeout: 2500 });
      await page.getByText(/공식 출전표를 확인하고 있습니다/).waitFor({ timeout: 13000 });
      const metrics = await inspectPage(page);
      if (metrics.visibleSpinner) throw new Error('Timeout left an infinite spinner visible');
    }
  },
  {
    name: '06-status-5xx-429-401',
    viewport: { width: 390, height: 844 },
    mock: { kind: 'status', status: 500 },
    finding: '5xx/429/401 상태별 한국어 안내를 구분',
    fix: 'HTTP status mapper 추가',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByText(/서버 오류 500/).waitFor({ timeout: 8000 });
      activeCase.status = 429;
      await page.getByRole('button', { name: '다시 시도' }).click();
      await page.getByText(/요청이 너무 많아 잠시 제한/).waitFor({ timeout: 8000 });
      activeCase.status = 401;
      await page.getByRole('button', { name: '다시 시도' }).click();
      await page.getByText(/인증이 필요해 분석을 불러오지 못했습니다/).waitFor({ timeout: 8000 });
    }
  },
  {
    name: '07-network-flap-retry',
    viewport: { width: 390, height: 844 },
    mock: { kind: 'flap', payload: readyDecision() },
    finding: '첫 네트워크 실패 후 다시 시도로 정상 렌더 복구',
    fix: '빈 분석 상태에 재시도 액션 추가',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByText(/서버 오류 503/).waitFor({ timeout: 8000 });
      await page.getByRole('button', { name: '다시 시도' }).click();
      await page.getByText('예측 순서 1-2-3').waitFor({ timeout: 8000 });
    }
  },
  {
    name: '08-quota-boundary',
    viewport: { width: 390, height: 844 },
    mock: {
      appSessionPayload: defaultAppSession(0),
      payload: readyDecision()
    },
    finding: 'remaining 0은 분석 시도 시 한도 안내+Pro 화면으로 이동',
    fix: '한도 소진 CTA를 비활성 대신 안내 액션으로 유지',
    flow: async (page) => {
      await page.getByText(/오늘 0회 남음/).waitFor({ timeout: 8000 });
      await page.getByRole('button', { name: '무료 분석 한도 안내 보기' }).click();
      await page.getByText(/오늘 무료 분석 한도를 모두 사용했습니다/).waitFor({ timeout: 8000 });
      await page.getByText('계정 상태').waitFor({ timeout: 8000 });
      const bodyText = await page.locator('body').innerText();
      if (/Pro 구독 시작|월 5,000원|구매 복원|결제/.test(bodyText)) {
        throw new Error('Quota guidance rendered purchase-inducing Pro wording');
      }
    }
  },
  {
    name: '09-app-session-failure',
    viewport: { width: 390, height: 844 },
    mock: { appSessionFailure: true, payload: readyDecision() },
    finding: 'app-session 실패에도 앱 시작과 기능 제한 안내 유지',
    fix: 'HomeScreen에 session dataLayer error 안내 추가',
    flow: async (page) => {
      await page.getByText('세션 정보를 확인하지 못했습니다').waitFor({ timeout: 8000 });
      await requestAnalysis(page);
      await page.getByText('예측 순서 1-2-3').waitFor({ timeout: 8000 });
    }
  },
  {
    name: '10-font-scale-320',
    viewport: { width: 320, height: 720 },
    fontScale: true,
    mock: { payload: readyDecision() },
    finding: '320px + 130% 폰트 스케일에서 홈과 분석 화면 오버플로 없음',
    fix: '기존 토큰과 wrap 규칙을 실제 브라우저에서 검증',
    flow: async (page) => {
      await page.getByRole('button', { name: '모델 신호 보기' }).scrollIntoViewIfNeeded();
      await requestAnalysis(page);
      await page.getByText('예측 순서 1-2-3').waitFor({ timeout: 8000 });
      const overlaps = await overlappingText(page);
      if (overlaps.length) throw new Error(`Font-scale text overlap: ${overlaps.join(' | ')}`);
    }
  },
  {
    name: '11-analytics-down',
    viewport: { width: 390, height: 844 },
    mock: { analyticsDown: true, payload: readyDecision() },
    finding: 'ux-events 전송 실패는 기능에 영향 없이 조용히 드랍',
    fix: 'trackUxEvent catch-and-drop 경로 Playwright 검증',
    flow: async (page) => {
      await requestAnalysis(page);
      await page.getByText('예측 순서 1-2-3').waitFor({ timeout: 8000 });
    }
  },
  {
    name: '12-rapid-tap-inflight',
    viewport: { width: 390, height: 844 },
    mock: { payload: readyDecision() },
    finding: '분석 버튼 5연타에도 live-decision 중복 요청 폭주 없음',
    fix: 'App executeAnalyze ref guard와 raceApi request-key dedupe',
    flow: async (page, mock) => {
      await page.getByTestId('analyze-cta').waitFor({ timeout: 8000 });
      await page.evaluate(() => {
        const button = document.querySelector('[data-testid="analyze-cta"]');
        if (!(button instanceof HTMLElement)) throw new Error('Analyze CTA not found');
        for (let index = 0; index < 5; index += 1) button.click();
      });
      await page.getByText('예측 순서 1-2-3').waitFor({ timeout: 8000 });
      if (mock.liveRequests !== 1) throw new Error(`Expected one live-decision request, got ${mock.liveRequests}`);
    }
  },
  {
    name: '13-keep-analysis-during-pending-refresh',
    viewport: { width: 390, height: 844 },
    mock: {
      kind: 'pending-after-ready',
      payload: readyDecision({
        app_session: {
          user_id: 'usr_pro',
          device_id: 'dev_pro',
          entitlement: 'pro',
          free_analysis_limit: 3,
          free_analysis_used: 0,
          free_analysis_remaining: 3,
          rewarded_analysis_credits: 0
        },
        poll_delay_ms: 3000
      })
    },
    finding: '자동 갱신의 보류 응답이 기존 분석 카드와 후보 순서를 지우지 않음',
    fix: '동일 경주의 pending/failed 응답은 이전 완성 분석을 유지',
    flow: async (page, mock) => {
      await requestAnalysis(page);
      await page.getByTestId('prediction-podium').waitFor({ timeout: 8000 }).catch(async () => {
        throw new Error(`Initial analysis did not render: ${(await page.locator('body').innerText()).slice(0, 800)}`);
      });
      await page.waitForTimeout(3600);
      await page.getByTestId('prediction-podium').waitFor({ timeout: 2000 });
      if (await page.getByTestId('analysis-pending-state').count() > 0) {
        throw new Error('Pending refresh replaced the visible analysis');
      }
      if (mock.liveRequests < 2) throw new Error('Expected an automatic refresh request');
    }
  }
];

const browser = await chromium.launch();
const results = [];
const selectedCases = process.env.RACELENS_QA_CASE
  ? cases.filter((item) => item.name.startsWith(process.env.RACELENS_QA_CASE))
  : cases;
if (selectedCases.length === 0) throw new Error('No adversarial QA case matched RACELENS_QA_CASE');
try {
  for (const item of selectedCases) {
    results.push(await runCase(browser, item));
  }
} finally {
  await browser.close();
  await Promise.all([
    new Promise((resolveClose) => staticServer.close(resolveClose)),
    new Promise((resolveClose) => apiServer.close(resolveClose))
  ]);
}

mkdirSync(resolve('..', 'runs'), { recursive: true });
const reportRows = results.map((result, index) => (
  `| ${index + 1} | ${result.name} | PASS | ${result.finding} | ${result.fix} |`
));
writeFileSync(reportPath, [
  '# RaceLens Adversarial Runtime QA',
  '',
  `- 실행 시각: ${new Date().toISOString()}`,
  `- Web export: ${distRoot}`,
  `- Evidence screenshots: ${artifactRoot}`,
  '- 기준: crash 0, console/pageerror 0, 깨진문자 0, 가로 오버플로 <=1px, 정직한 한국어 상태 안내',
  '',
  '| # | 케이스 | 결과 | 발견결함/검증내용 | 수정내역 |',
  '|---:|---|---|---|---|',
  ...reportRows,
  ''
].join('\n'));

console.log(`ADVERSARIAL=${results.length}/${results.length} PASS report=${reportPath}`);
