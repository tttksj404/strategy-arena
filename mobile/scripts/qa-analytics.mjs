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
const analyticsPort = await freePort();
const analyticsUrl = `http://127.0.0.1:${analyticsPort}/events`;
const distRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-analytics-'));
const events = [];

const exportResult = spawnQa('npx', ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
  env: {
    ...process.env,
    EXPO_PUBLIC_RACELENS_API_BASE_URL: '',
    EXPO_PUBLIC_RACELENS_ANALYTICS_URL: analyticsUrl,
    EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '1'
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
if (!bundleText.includes(analyticsUrl)) {
  throw new Error(`Analytics URL was not inlined into the web bundle: ${analyticsUrl}`);
}

const analyticsServer = createServer((request, response) => {
  if (request.method === 'OPTIONS') {
    response.writeHead(204, {
      'access-control-allow-origin': '*',
      'access-control-allow-methods': 'POST,OPTIONS',
      'access-control-allow-headers': 'content-type,x-racelens-analytics'
    });
    response.end();
    return;
  }
  if (request.method !== 'POST' || request.url !== '/events') {
    response.writeHead(404, { 'access-control-allow-origin': '*', 'content-type': 'application/json' });
    response.end(JSON.stringify({ error: 'not_found' }));
    return;
  }
  let body = '';
  request.on('data', (chunk) => {
    body += chunk;
  });
  request.on('end', () => {
    events.push(JSON.parse(body));
    response.writeHead(204, { 'access-control-allow-origin': '*' });
    response.end();
  });
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
  new Promise((resolveListen) => analyticsServer.listen(analyticsPort, '127.0.0.1', resolveListen)),
  new Promise((resolveListen) => staticServer.listen(appPort, '127.0.0.1', resolveListen))
]);

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, colorScheme: 'dark' });
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: '경마' }).click();
  await page.getByRole('button', { name: '8R' }).click();
  await page.getByRole('button', { name: '모델 신호 보기' }).click();
  const adConfirm = page.getByRole('button', { name: '광고 확인 후 분석 보기' });
  await adConfirm.waitFor({ state: 'visible', timeout: 1500 }).then(() => adConfirm.click()).catch(() => {});
  await page.getByText('서울 8R 분석').waitFor();
  await page.getByText('예측 순서 5-3-1').waitFor();
  if (await page.getByRole('tab', { name: /랩/ }).count()) {
    throw new Error('Lab tab should not be rendered');
  }
  await page.getByRole('tab', { name: /Pro|프로/ }).click();
  await page.waitForTimeout(400);
  if (errors.length) throw new Error(`Console/page errors:\n${errors.join('\n')}`);
} finally {
  await browser.close();
  await new Promise((resolveClose) => staticServer.close(resolveClose));
  await new Promise((resolveClose) => analyticsServer.close(resolveClose));
}

const eventNames = new Set(events.map((event) => event.name));
for (const required of ['app_open', 'screen_view', 'race_context_change', 'analysis_request', 'analysis_result', 'tab_select']) {
  if (!eventNames.has(required)) throw new Error(`Missing analytics event: ${required}`);
}

const forbiddenKeys = ['name', 'participant', 'selection', 'deviceId', 'userId', 'meet'];
for (const event of events) {
  const serialized = JSON.stringify(event.payload ?? {});
  if (event.app !== 'racelens') throw new Error(`Invalid analytics app field: ${serialized}`);
  if (!event.sessionId?.startsWith('sess_')) throw new Error(`Missing anonymous session id: ${serialized}`);
  if (!event.anonymousId?.startsWith('anon_')) throw new Error(`Missing anonymous id: ${serialized}`);
  if (event.platform !== 'web') throw new Error(`Unexpected analytics platform: ${serialized}`);
  for (const key of forbiddenKeys) {
    if (serialized.includes(key)) throw new Error(`Analytics payload leaks forbidden field "${key}": ${serialized}`);
  }
}

console.log(`analytics QA passed: ${events.length} events`);
