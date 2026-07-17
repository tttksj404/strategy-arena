import { createReadStream, existsSync, mkdtempSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { createServer as createNetServer } from 'node:net';
import { tmpdir } from 'node:os';
import { extname, join } from 'node:path';
import { chromium } from 'playwright';
import { exitIfFailed, spawnQa } from './qa-utils.mjs';

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
const distRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-api-live-'));

const exportResult = spawnQa('npx', ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
  env: {
    ...process.env,
    EXPO_PUBLIC_RACELENS_API_BASE_URL: apiBaseUrl,
    EXPO_PUBLIC_RACELENS_ANALYTICS_URL: ''
  }
});
exitIfFailed(exportResult);

const bundleText = readFileSync(join(distRoot, '_expo/static/js/web', readdirBundleName()), 'utf8');
if (!bundleText.includes(apiBaseUrl)) {
  console.error(`API base URL was not inlined into the web bundle: ${apiBaseUrl}`);
  process.exit(1);
}

function readdirBundleName() {
  const metadata = JSON.parse(readFileSync(join(distRoot, 'metadata.json'), 'utf8'));
  const bundle = metadata?.bundler === 'metro'
    ? readFileSync(join(distRoot, 'index.html'), 'utf8').match(/_expo\/static\/js\/web\/[^"]+\.js/)?.[0]
    : undefined;
  if (!bundle) throw new Error('Unable to locate Expo web bundle');
  return bundle.replace('_expo/static/js/web/', '');
}

function headers(type = 'application/json') {
  return {
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'GET,OPTIONS',
    'access-control-allow-headers': 'content-type,x-racelens-device-id,x-racelens-platform',
    'content-type': type
  };
}

let liveDecisionRequestCount = 0;

const apiServer = createServer((request, response) => {
  if (request.method === 'OPTIONS') {
    response.writeHead(204, headers());
    response.end();
    return;
  }
  if (request.url?.startsWith('/recent')) {
    response.writeHead(200, headers());
    response.end(JSON.stringify({
      sport: 'keirin',
      meet: '광명',
      days: ['2026-07-03', '2026-07-04', '2026-07-05']
    }));
    return;
  }
  if (request.url?.startsWith('/api/app-session')) {
    response.writeHead(200, headers());
    response.end(JSON.stringify({
      user_id: 'usr_qa',
      device_id: request.headers['x-racelens-device-id'] ?? 'dev_qa',
      entitlement: 'free',
      free_analysis_limit: 3,
      free_analysis_used: 0,
      free_analysis_remaining: 3
    }));
    return;
  }
  if (!request.url?.startsWith('/api/live-decision')) {
    response.writeHead(404, headers());
    response.end(JSON.stringify({ error: 'not_found' }));
    return;
  }
  liveDecisionRequestCount += 1;
  if (liveDecisionRequestCount === 1) {
    response.writeHead(200, headers());
    response.end(JSON.stringify({
      status: 'hold',
      snapshot_phase: 'pending',
      market_used: false,
      market_risk: {
        level: 'official_data_pending',
        title: '공식 출전표 확인 중',
        message: '공식 출전표를 확인하고 있습니다. 잠시 후 자동으로 갱신됩니다.'
      },
      app_session: {
        user_id: 'usr_qa',
        device_id: request.headers['x-racelens-device-id'] ?? 'dev_qa',
        entitlement: 'pro'
      },
      data_layer: {
        ready: true,
        storage: 'postgresql',
        schemas: []
      },
      poll_delay_ms: 3000
    }));
    return;
  }
  response.writeHead(200, headers());
  response.end(JSON.stringify({
    status: 'ready',
    market_used: true,
    roster_verification: {
      state: 'verified',
      message: 'QA mock roster verified',
      source: 'qa-api-live'
    },
    market_risk: {
      level: 'odds_live',
      message: 'QA mock: live market path is reachable and rendered. '.repeat(8)
    },
    participants: [
      {
        number: 999,
        name: '<script>alert("x")</script>검증용초장문출전마이름'.repeat(2),
        subtitle: '기수 QA / 55kg '.repeat(6),
        stats: '최근 4전 1-1-1 '.repeat(5),
        trait: '선입'.repeat(8),
        note: '외부 API가 길거나 깨진 텍스트를 내려도 카드가 터지지 않아야 합니다. '.repeat(5),
        signal: 'unknown'
      },
      {
        number: 2,
        name: '',
        subtitle: '',
        stats: '',
        trait: '',
        note: '',
        signal: 'teal'
      }
    ],
    rows: [
      { bno: 2, name: '이정민', pwin: 0.37, pplc: 0.78 },
      { bno: 4, name: '김로운', pwin: 0.24, pplc: 0.64 },
      { bno: 5, name: '한기봉', pwin: 0.18, pplc: 0.52 }
    ],
    picks: [
      { code: 'TOP1', label: '1착 후보', selection: '2', probability: 0.37, grade: '강' },
      { code: 'QNL', label: '복승 후보', selection: '2-4', probability: 0.24, grade: '중' },
      { code: 'TRI', label: '삼쌍 순서', selection: '2-4-5', probability: 0.09, grade: '중' }
    ],
    trifecta_ensemble: {
      pick: '2-4-5',
      top5: ['2-4-5', '2-5-4', '4-2-5', '4-5-2', '5-2-4'],
      tier: 'T2_top16',
      tier_historical_exact: 0.434,
      selection: 'ensemble_v1_top1',
      board_complete: true,
      coverage: 0.026,
      signal_strength: 0.82,
      source: 'ensemble_v1'
    },
    market_odds: [
      {
        code: 'win',
        label: '단승',
        selection: '2',
        odds: 2.13,
        change: '-0.2',
        signal: 'teal'
      },
      {
        code: 'tri',
        label: '삼쌍 순서'.repeat(8),
        selection: '2-4-5'.repeat(6),
        odds: 4.8,
        change: '급등락 확인 필요'.repeat(5),
        signal: 'unknown'
      }
    ],
    data_layer: {
      ready: true,
      storage: 'postgresql',
      schemas: [
        { name: 'race_data', tables: ['market_odds_snapshots'], row_count: 13 },
        { name: 'prediction', tables: ['predictions'], row_count: 8 },
        { name: 'user_account', tables: ['users', 'devices'], row_count: 2 },
        { name: 'billing', tables: ['subscriptions'], row_count: 0 },
        { name: 'analytics', tables: ['user_view_events'], row_count: 9 }
      ]
    },
    app_session: {
      user_id: 'usr_qa',
      device_id: request.headers['x-racelens-device-id'] ?? 'dev_qa',
      entitlement: 'pro'
    },
    poll_delay_ms: 15000
  }));
});

const staticServer = createServer((request, response) => {
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
});

await Promise.all([
  new Promise((resolveListen) => apiServer.listen(apiPort, '127.0.0.1', resolveListen)),
  new Promise((resolveListen) => staticServer.listen(appPort, '127.0.0.1', resolveListen))
]);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, colorScheme: 'dark' });
const errors = [];
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text());
});
page.on('pageerror', (error) => errors.push(error.message));

try {
  await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: '모델 신호 보기' }).click();
  const adConfirm = page.getByRole('button', { name: '광고 확인 후 분석 보기' });
  await adConfirm.waitFor({ state: 'visible', timeout: 1500 }).then(() => adConfirm.click()).catch(() => {});
  await page.getByText('광명 1R 분석').waitFor();
  await page.getByTestId('analysis-pending-state').waitFor();
  await page.getByText('공식 출전표 확인 중', { exact: true }).waitFor();
  if (await page.getByTestId('analysis-empty-state').count()) {
    throw new Error('Official roster pending state was rendered as a fetch failure');
  }
  await page.getByText('예측 순서 2-4-5').waitFor();
  await page.getByTestId('trifecta-ensemble-card').waitFor();
  await page.getByText('강신호 경주').waitFor();
  await page.getByText('과거 적중 43%').waitFor();
  await page.getByText('과거 배당 반영').waitFor();
  await page.getByText('배당 자료', { exact: true }).waitFor();
  await page.getByText('2.13배').waitFor();
  await page.getByText('4.8배').waitFor();
  await page.getByText(/QA mock: live market path is reachable and rendered/).waitFor();
  await page.getByText(/<script>alert/).first().waitFor();
  if (await page.getByRole('tab', { name: /랩/ }).count()) {
    throw new Error('Lab tab should not be rendered in the production app shell');
  }
  await page.getByRole('tab', { name: /Pro|프로/ }).click();
  await page.getByText('계정 상태').waitFor();
  await page.getByText(/(?:Pro 이용 중|무료 이용 중) · 공식 데이터 확인 가능/).waitFor();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 1) throw new Error(`Horizontal overflow after hostile API payload: ${overflow}px`);
  if (errors.length) throw new Error(`Console/page errors:\n${errors.join('\n')}`);
  console.log('live API QA passed');
} finally {
  await browser.close();
  await new Promise((resolveClose) => staticServer.close(resolveClose));
  await new Promise((resolveClose) => apiServer.close(resolveClose));
}
