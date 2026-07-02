import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';

const appPort = 8064;
const apiPort = 8065;
const apiBaseUrl = `http://127.0.0.1:${apiPort}`;
const distRoot = resolve('dist');

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
    'access-control-allow-headers': 'content-type',
    'content-type': type
  };
}

const apiServer = createServer((request, response) => {
  if (request.method === 'OPTIONS') {
    response.writeHead(204, headers());
    response.end();
    return;
  }
  if (!request.url?.startsWith('/api/live-decision')) {
    response.writeHead(404, headers());
    response.end(JSON.stringify({ error: 'not_found' }));
    return;
  }
  response.writeHead(200, headers());
  response.end(JSON.stringify({
    status: 'ready',
    market_used: true,
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
    market_odds: [
      {
        code: 'win',
        label: '단승',
        selection: '5',
        odds: 2.13,
        change: '-0.2',
        signal: 'teal'
      },
      {
        code: 'tri',
        label: '삼쌍 순서'.repeat(8),
        selection: '5-1-7'.repeat(6),
        odds: 999999,
        change: '급등락 확인 필요'.repeat(5),
        signal: 'unknown'
      }
    ],
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
  await page.getByText('실시간 시장 신호 반영').waitFor();
  await page.getByText('실시간 배당 반영').waitFor();
  await page.getByText('실시간 배당', { exact: true }).waitFor();
  await page.getByText('2.13배').waitFor();
  await page.getByText('4.8배').waitFor();
  await page.getByText(/QA mock: live market path is reachable and rendered/).waitFor();
  await page.getByText(/<script>alert/).waitFor();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 1) throw new Error(`Horizontal overflow after hostile API payload: ${overflow}px`);
  if (errors.length) throw new Error(`Console/page errors:\n${errors.join('\n')}`);
  console.log('live API QA passed');
} finally {
  await browser.close();
  await new Promise((resolveClose) => staticServer.close(resolveClose));
  await new Promise((resolveClose) => apiServer.close(resolveClose));
}
