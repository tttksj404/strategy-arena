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
const expectedApiBaseUrl = `http://127.0.0.1:${await freePort()}`;
const distRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-api-failure-'));
let apiServer = null;

const exportResult = spawnQa('npx', ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
  env: {
    ...process.env,
    EXPO_PUBLIC_RACELENS_API_BASE_URL: expectedApiBaseUrl,
    EXPO_PUBLIC_RACELENS_ANALYTICS_URL: '',
    EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '0'
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
if (!bundleText.includes(expectedApiBaseUrl)) {
  console.error(`Failure QA API base URL was not inlined into the web bundle: ${expectedApiBaseUrl}`);
  process.exit(1);
}

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

await new Promise((resolveListen) => staticServer.listen(appPort, '127.0.0.1', resolveListen));

function jsonHeaders() {
  return {
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'GET,POST,OPTIONS',
    'access-control-allow-headers': 'content-type,x-racelens-device-id,x-racelens-platform',
    'content-type': 'application/json'
  };
}

function writeJson(response, status, body) {
  response.writeHead(status, jsonHeaders());
  response.end(JSON.stringify(body));
}

function freeQuotaBlockedPayload() {
  return {
    status: 'blocked',
    decision: 'blocked',
    sport: 'keirin',
    date: '2026-07-03',
    meet: '광명',
    race_no: 1,
    market_used: false,
    market_risk: {
      level: 'blocked',
      message: '오늘 무료 분석 3회를 모두 사용했습니다. Pro 권한이 확인되면 무제한 분석이 열립니다.'
    },
    data_layer: {
      ready: true,
      schemas: [{ name: 'qa', tables: ['free_quota'], row_count: 1 }],
      storage: 'qa'
    },
    app_session: {
      user_id: 'usr_quota',
      device_id: 'dev_quota',
      entitlement: 'free',
      free_analysis_limit: 3,
      free_analysis_used: 3,
      free_analysis_remaining: 0,
      rewarded_analysis_credits: 0
    },
    rows: [],
    picks: [],
    participants: [],
    market_odds: [],
    poll_delay_ms: 15000,
    updated_at: '2026-07-03T05:30:00.000Z'
  };
}

async function startBlockedQuotaApi() {
  const apiPort = Number(new URL(expectedApiBaseUrl).port);
  apiServer = createServer((request, response) => {
    if (request.method === 'OPTIONS') {
      response.writeHead(204, jsonHeaders());
      response.end();
      return;
    }
    if (request.url?.startsWith('/recent')) {
      writeJson(response, 200, {
        sport: 'keirin',
        meet: '광명',
        days: ['2026-07-03'],
        default_race_no: 1,
        race_count: 12
      });
      return;
    }
    if (request.url?.startsWith('/api/app-session')) {
      const payload = freeQuotaBlockedPayload();
      writeJson(response, 200, {
        app_session: payload.app_session,
        data_layer: payload.data_layer
      });
      return;
    }
    if (request.url?.startsWith('/api/live-decision')) {
      writeJson(response, 200, freeQuotaBlockedPayload());
      return;
    }
    writeJson(response, 404, { error: 'not_found' });
  });
  await new Promise((resolveListen) => apiServer.listen(apiPort, '127.0.0.1', resolveListen));
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, colorScheme: 'dark' });
const errors = [];
page.on('console', (message) => {
  const text = message.text();
  if (message.type() === 'error' && !text.includes('Failed to load resource') && !text.includes('net::ERR_CONNECTION_REFUSED')) {
    errors.push(text);
  }
});
page.on('pageerror', (error) => errors.push(error.message));

try {
  await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: '모델 신호 보기' }).click();
  const adConfirm = page.getByRole('button', { name: '광고 확인 후 분석 보기' });
  await adConfirm.waitFor({ state: 'visible', timeout: 1500 }).then(() => adConfirm.click()).catch(() => {});
  const emptyState = page.getByTestId('analysis-empty-state');
  await emptyState.waitFor({ timeout: 12000 });
  const emptyStateBox = await emptyState.boundingBox();
  const viewport = page.viewportSize();
  if (viewport === null) throw new Error('API failure QA requires a fixed viewport');
  const emptyStateVisibleAboveFold = emptyStateBox !== null && emptyStateBox.y >= 0 && emptyStateBox.y < viewport.height;
  if (!emptyStateVisibleAboveFold) {
    throw new Error('API failure state is not visible above the fold at scroll position 0');
  }
  await page.getByText('네트워크 연결이 끊겨 분석을 불러오지 못했습니다. 다시 시도하세요.').waitFor();
  await page.getByRole('button', { name: '다시 시도' }).waitFor();
  const fakeDataCount = await Promise.all([
    page.getByText('배당 자료', { exact: true }).count(),
    page.getByText('과거 데이터 예시').count(),
    page.getByText(/예시 기준일 \d{4}-\d{2}-\d{2}/).count(),
    page.getByTestId('prediction-podium').count()
  ]);
  if (fakeDataCount.some((count) => count > 0)) {
    throw new Error(`API failure rendered fake analysis data: ${fakeDataCount.join(',')}`);
  }
  await startBlockedQuotaApi();
  await emptyState.getByRole('button').click();
  await page.getByTestId('free-quota-exhausted-state').waitFor({ timeout: 12000 });
  const freeQuotaText = await page.locator('body').innerText();
  if (!freeQuotaText.includes('오늘 무료 분석 3회를 모두 사용했습니다. Pro 권한이 확인되면 무제한 분석이 열립니다.')) {
    throw new Error('Free quota blocked response did not render the server message');
  }
  if (/공식 출주표.*확인 못|출주표.*대기|roster-waiting-state/.test(freeQuotaText)) {
    throw new Error('Free quota blocked response was mislabeled as a roster verification failure');
  }
  await page.getByRole('link', { name: 'Pro 안내 보기' }).waitFor();
  const metrics = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    badText: document.body.innerText.includes('�') ? 1 : 0
  }));
  if (metrics.overflow > 1) throw new Error(`Horizontal overflow after API failure: ${metrics.overflow}px`);
  if (metrics.badText) throw new Error('Broken replacement character rendered after API failure');
  if (errors.length) throw new Error(`Console/page errors:\n${errors.join('\n')}`);
  console.log('API failure QA passed');
} finally {
  await browser.close();
  if (apiServer) {
    await new Promise((resolveClose) => apiServer.close(resolveClose));
  }
  await new Promise((resolveClose) => staticServer.close(resolveClose));
}
