import { createReadStream, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright';

const appPort = 8064;
const distRoot = resolve('dist');
const artifactRoot = '/tmp/racelens-release-readiness';

mkdirSync(artifactRoot, { recursive: true });

function bundleName() {
  const metadata = JSON.parse(readFileSync(join(distRoot, 'metadata.json'), 'utf8'));
  const bundle = metadata?.bundler === 'metro'
    ? readFileSync(join(distRoot, 'index.html'), 'utf8').match(/_expo\/static\/js\/web\/[^"]+\.js/)?.[0]
    : undefined;
  if (!bundle) throw new Error('Unable to locate Expo web bundle');
  return bundle.replace('_expo/static/js/web/', '');
}

const bundleText = readFileSync(join(distRoot, '_expo/static/js/web', bundleName()), 'utf8');
for (const forbidden of ['http://127.0.0.1:8065', 'http://127.0.0.1:8066', 'EXPO_PUBLIC_RACELENS_API_BASE_URL']) {
  if (bundleText.includes(forbidden)) throw new Error(`Default release bundle contains QA-only string: ${forbidden}`);
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

function collectErrors(page) {
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

async function inspect(page) {
  return page.evaluate(() => {
    const overflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
    const bodyText = document.body.innerText;
    const smallTargets = [...document.querySelectorAll('[role="button"], [role="tab"], button')]
      .filter((element) => {
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      })
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const label = (element.textContent || element.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
        return { label, width: Math.round(rect.width), height: Math.round(rect.height) };
      })
      .filter((target) => target.width < 44 || target.height < 44);
    const forbiddenActionButtons = [...document.querySelectorAll('[role="button"], button')]
      .map((element) => (element.textContent || element.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim())
      .filter((label) => /구매|베팅|결제/.test(label));
    return {
      badText: bodyText.includes('�') ? 1 : 0,
      forbiddenActionButtons,
      overflow,
      smallTargets
    };
  });
}

async function runCase(browser, item) {
  const page = await browser.newPage({
    viewport: { width: item.width, height: item.height },
    deviceScaleFactor: 2,
    isMobile: item.width < 700,
    colorScheme: item.colorScheme
  });
  const errors = collectErrors(page);
  await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
  await item.flow(page);
  const metrics = await inspect(page);
  const screenshot = join(artifactRoot, `${item.name}.png`);
  await page.screenshot({ path: screenshot, fullPage: false });
  await page.close();
  return { name: item.name, screenshot, metrics, errors };
}

await new Promise((resolveListen) => staticServer.listen(appPort, '127.0.0.1', resolveListen));

const browser = await chromium.launch();
try {
  const cases = [
    {
      name: '320-home-keirin-light',
      width: 320,
      height: 720,
      colorScheme: 'light',
      flow: async (page) => {
        await page.getByText('실시간 배당', { exact: true }).scrollIntoViewIfNeeded();
      }
    },
    {
      name: '375-home-horse-light',
      width: 375,
      height: 812,
      colorScheme: 'light',
      flow: async (page) => {
        await page.getByRole('button', { name: '경마' }).click();
        await page.getByText('출전마').scrollIntoViewIfNeeded();
      }
    },
    {
      name: '390-analyze-horse-dark',
      width: 390,
      height: 844,
      colorScheme: 'dark',
      flow: async (page) => {
        await page.getByRole('button', { name: '경마' }).click();
        await page.getByRole('button', { name: '8R' }).click();
        await page.getByRole('button', { name: '모델 신호 보기' }).click();
        await page.getByText('출전마').scrollIntoViewIfNeeded();
      }
    },
    {
      name: '320-pro-dark-safety',
      width: 320,
      height: 720,
      colorScheme: 'dark',
      flow: async (page) => {
        await page.getByRole('tab', { name: /Pro/ }).click();
        await page.getByText('광고 슬롯').scrollIntoViewIfNeeded();
      }
    },
    {
      name: '768-lab-light',
      width: 768,
      height: 900,
      colorScheme: 'light',
      flow: async (page) => {
        await page.getByRole('tab', { name: /랩/ }).click();
        await page.getByText('검증 랩').waitFor();
      }
    }
  ];

  const results = [];
  for (const item of cases) results.push(await runCase(browser, item));

  const failures = results.flatMap((result) => {
    const found = [];
    if (result.errors.length) found.push(`${result.name}: console/page errors: ${result.errors.join(' | ')}`);
    if (result.metrics.overflow > 1) found.push(`${result.name}: horizontal overflow ${result.metrics.overflow}px`);
    if (result.metrics.badText) found.push(`${result.name}: broken replacement character rendered`);
    if (result.metrics.smallTargets.length) found.push(`${result.name}: small tap targets ${JSON.stringify(result.metrics.smallTargets)}`);
    if (result.metrics.forbiddenActionButtons.length) found.push(`${result.name}: forbidden action buttons ${result.metrics.forbiddenActionButtons.join(', ')}`);
    return found;
  });

  if (failures.length) throw new Error(`Release readiness QA failed:\n${failures.join('\n')}`);
  console.log(JSON.stringify(results, null, 2));
  console.log('release readiness QA passed');
} finally {
  await browser.close();
  await new Promise((resolveClose) => staticServer.close(resolveClose));
}
