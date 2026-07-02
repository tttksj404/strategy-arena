import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';

const appPort = 8064;
const expectedApiBaseUrl = 'http://127.0.0.1:8066';
const distRoot = resolve('dist');

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
  await page.getByText('RaceLens API에 연결할 수 없습니다.').waitFor({ timeout: 8000 });
  await page.getByText('RaceLens', { exact: true }).waitFor();
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
  await new Promise((resolveClose) => staticServer.close(resolveClose));
}
