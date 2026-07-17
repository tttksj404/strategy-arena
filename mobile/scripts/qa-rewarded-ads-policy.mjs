import { createReadStream, existsSync, mkdtempSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { exitIfFailed, spawnQa } from './qa-utils.mjs';

const adTextPattern = /광고/g;

function serveDist(distRoot) {
  let rewardedCredit = 0;
  let freeUsed = 0;
  const appSession = () => ({
    user_id: 'qa-user',
    device_id: 'qa-device',
    entitlement: 'free',
    free_analysis_limit: 3,
    free_analysis_used: freeUsed,
    free_analysis_remaining: Math.max(0, 3 - freeUsed),
    rewarded_analysis_credits: rewardedCredit
  });
  return createServer((request, response) => {
    const cleanPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
    if (cleanPath === '/api/live-decision' && request.method === 'GET') {
      if (freeUsed < 3) {
        freeUsed += 1;
      } else if (rewardedCredit > 0) {
        rewardedCredit -= 1;
      } else {
        response.writeHead(429, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ ok: false, error: 'free_analysis_limit_reached' }));
        return;
      }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        status: 'ready',
        market_used: false,
        rows: [],
        app_session: appSession(),
        data_layer: { ready: true, storage: 'qa', schemas: [] }
      }));
      return;
    }
    if (cleanPath === '/api/rewarded-ad/claim' && request.method === 'POST') {
      rewardedCredit = 1;
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        ok: true,
        reward_granted: true,
        app_session: appSession(),
        data_layer: { ready: true, storage: 'qa', schemas: [] }
      }));
      return;
    }
    if (cleanPath === '/api/app-session') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        ok: true,
        app_session: appSession(),
        data_layer: { ready: true, storage: 'qa', schemas: [] }
      }));
      return;
    }
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
}

async function listen(server) {
  await new Promise((resolveListen) => server.listen(0, '127.0.0.1', resolveListen));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Unable to allocate rewarded ads QA port');
  return address.port;
}

async function driveQuotaExhaustion(page) {
  for (let index = 0; index < 3; index += 1) {
    await page.getByTestId('analyze-cta').click();
    const progress = page.getByTestId('analysis-loading');
    await progress.waitFor({ state: 'visible', timeout: 2000 }).catch(() => {});
    await progress.waitFor({ state: 'hidden', timeout: 10000 });
    await page.locator('section[role="region"]').waitFor({ state: 'hidden', timeout: 10000 });
    await page.locator('[role="tab"]').first().click();
    await page.getByTestId('analyze-cta').waitFor({ timeout: 10000 });
  }
  await page.getByTestId('analyze-cta').click();
  await page.waitForTimeout(500);
}

async function runCase({ firebaseEnabled, rewardedAds, slug }) {
  const distRoot = mkdtempSync(join(tmpdir(), `racelens-qa-${slug}-`));
  const expoCli = fileURLToPath(new URL('../node_modules/expo/bin/cli', import.meta.url));
  const exportResult = spawnQa(process.execPath, [expoCli, 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
    env: {
      ...process.env,
      EXPO_PUBLIC_RACELENS_API_BASE_URL: '',
      EXPO_PUBLIC_RACELENS_ANALYTICS_URL: '',
      EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED: firebaseEnabled ? '1' : '0',
      EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '1',
      EXPO_PUBLIC_RACELENS_REWARDED_ADS: rewardedAds ? '1' : '0',
      EXPO_PUBLIC_RACELENS_REWARDED_ADS_PREVIEW: rewardedAds ? '1' : '0'
    }
  });
  exitIfFailed(exportResult);

  const staticServer = serveDist(distRoot);
  const appPort = await listen(staticServer);
  const browser = await chromium.launch();
  const rewardRequests = [];
  const errors = [];

  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 }, colorScheme: 'dark' });
    page.on('request', (request) => {
      if (request.url().includes('/api/rewarded-ad/claim')) rewardRequests.push(request.url());
    });
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
    await driveQuotaExhaustion(page);
    const bodyText = await page.evaluate(() => document.body.innerText);
    const adTextCount = bodyText.match(adTextPattern)?.length ?? 0;
    const screenshot = join(distRoot, `${slug}.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    let rewardedScreenshot = null;
    if (rewardedAds) {
      await page.getByTestId('rewarded-ad-confirm').click();
      await page.getByText('PREVIEW TEST AD', { exact: true }).waitFor({ state: 'hidden', timeout: 10000 });
      rewardedScreenshot = join(distRoot, `${slug}-rewarded.png`);
      await page.screenshot({ path: rewardedScreenshot, fullPage: true });
    }
    await page.close();

    if (errors.length > 0) throw new Error(`${slug}: console/page errors:\n${errors.join('\n')}`);
    if (!rewardedAds && adTextCount !== 0) {
      throw new Error(`${slug}: rewarded ads off rendered ${adTextCount} ad text occurrence(s)`);
    }
    if (!rewardedAds && rewardRequests.length > 0) {
      throw new Error(`${slug}: rewarded ads off called reward API: ${rewardRequests.join(', ')}`);
    }
    if (rewardedAds && adTextCount === 0) {
      throw new Error(`${slug}: rewarded ads on did not render ad disclosure text`);
    }
    if (rewardedAds && !bodyText.includes('PREVIEW TEST AD')) {
      throw new Error(`${slug}: rewarded ads preview did not identify itself as a test ad`);
    }
    if (rewardedAds && rewardRequests.length !== 1) {
      throw new Error(`${slug}: preview reward CTA made ${rewardRequests.length} reward request(s); expected 1`);
    }

    return { adTextCount, firebaseEnabled, rewardRequestCount: rewardRequests.length, rewardedAds, rewardedScreenshot, screenshot };
  } finally {
    await browser.close();
    await new Promise((resolveClose) => staticServer.close(resolveClose));
  }
}

const results = [
  await runCase({ firebaseEnabled: false, rewardedAds: false, slug: 'rewarded-ads-off-firebase-off' }),
  await runCase({ firebaseEnabled: true, rewardedAds: true, slug: 'rewarded-ads-on-firebase-on' })
];

console.log(JSON.stringify(results, null, 2));
console.log('rewarded ads policy QA passed');
