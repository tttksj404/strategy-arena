import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoDir = resolve(scriptDir, '..', '..');
const stateDir = process.env.RACELENS_PREVIEW_STATE_DIR || join(repoDir, '.runtime', 'racelens-preview');
const screenshotDir = join(stateDir, 'screenshots');
const publicUrlFile = join(stateDir, 'public_url');
const configuredUrl = process.env.RACELENS_PREVIEW_URL?.trim();
const previewUrl = configuredUrl || (existsSync(publicUrlFile) ? readFileSync(publicUrlFile, 'utf8').trim() : '');
const expectHistorical = process.env.RACELENS_EXPECT_HISTORICAL !== '0';

if (!previewUrl) {
  console.error('RACELENS_PREVIEW_URL is missing and .runtime/racelens-preview/public_url does not exist.');
  process.exit(1);
}

mkdirSync(screenshotDir, { recursive: true });

function requireCondition(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function riskCode(payload) {
  const marketRisk = payload && typeof payload === 'object' ? payload.market_risk : undefined;
  if (marketRisk && typeof marketRisk === 'object' && 'level' in marketRisk) {
    return marketRisk.level;
  }
  return undefined;
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({
  viewport: { width: 390, height: 844 },
  isMobile: true,
  colorScheme: 'light'
});
const failedRequests = [];
const consoleErrors = [];
const apiPayloads = [];

page.on('requestfailed', (request) => {
  failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ''}`);
});
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text());
});
page.on('pageerror', (error) => {
  consoleErrors.push(error.message);
});
page.on('response', async (response) => {
  if (!response.url().includes('/api/live-decision')) return;
  try {
    apiPayloads.push({ status: response.status(), body: await response.json() });
  } catch (error) {
    apiPayloads.push({ status: response.status(), body: null, error: error instanceof Error ? error.message : String(error) });
  }
});

try {
  const started = Date.now();
  await page.goto(previewUrl, { waitUntil: 'networkidle', timeout: 30000 });
  const liveDecisionResponse = page.waitForResponse(
    (response) => response.url().includes('/api/live-decision') && response.status() === 200,
    { timeout: 15000 }
  );
  await page.getByRole('button', { name: /모델 신호 보기/ }).click({ timeout: 10000 });
  const adConfirm = page.getByRole('button', { name: '광고 확인 후 분석 보기' });
  await adConfirm.waitFor({ state: 'visible', timeout: 3000 }).then(() => adConfirm.click()).catch(() => {});
  await liveDecisionResponse;
  await page.getByText('배당 자료', { exact: true }).waitFor({ timeout: 15000 });
  const screenshotPath = join(screenshotDir, `live-preview-${Date.now()}.png`);
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const screen = await page.evaluate(() => {
    const bodyText = document.body.innerText;
    const oddsStart = bodyText.indexOf('배당 자료');
    const evidenceStart = bodyText.indexOf('기본 제공 자료', oddsStart + 1);
    const oddsSection = oddsStart >= 0 ? bodyText.slice(oddsStart, evidenceStart > oddsStart ? evidenceStart : oddsStart + 900) : '';
    const elements = [...document.querySelectorAll('*')];
    const overflowCount = elements.filter((element) => element.scrollWidth > element.clientWidth + 1 && getComputedStyle(element).overflowX !== 'visible').length;
    return {
      bodyText,
      oddsSection,
      hasOddsBoard: bodyText.includes('배당 자료'),
      hasParticipantBoard: bodyText.includes('기본 제공 자료') && bodyText.includes('출전 선수'),
      hasWaitingCopy: bodyText.includes('실시간 배당 대기'),
      mentionsRender: bodyText.toLowerCase().includes('render'),
      hasPastBadge: oddsSection.includes('과거'),
      hasPastOddsRows: oddsSection.includes('과거 배당'),
      hasLiveRowLeak: oddsSection.includes('실시간'),
      overflowCount
    };
  });
  const firstApi = apiPayloads[0];
  const firstBody = firstApi?.body;
  const marketOdds = firstBody && typeof firstBody === 'object' && Array.isArray(firstBody.market_odds) ? firstBody.market_odds : [];
  const settled = firstBody?.status === 'settled';
  requireCondition(firstApi?.status === 200, `live-decision did not return 200: ${firstApi?.status ?? 'missing'}`);
  if (settled) {
    requireCondition(Array.isArray(firstBody?.actual_result?.actual_order) && firstBody.actual_result.actual_order.length === 3, 'settled actual order is missing');
    requireCondition(firstBody?.market_used === false, `settled result reused market signal: ${firstBody?.market_used}`);
  } else {
    requireCondition(firstBody?.market_used === true, `market_used is not true: ${firstBody?.market_used}`);
    requireCondition(marketOdds.length >= 7, `market odds are too sparse: ${marketOdds.length}`);
  }
  requireCondition(screen.hasOddsBoard, 'odds board was not rendered');
  requireCondition(screen.hasParticipantBoard, 'participant evidence board was not rendered');
  requireCondition(!screen.hasWaitingCopy, 'stale waiting copy is still visible');
  requireCondition(!screen.mentionsRender, 'Render copy is still visible');
  requireCondition(screen.overflowCount === 0, `horizontal overflow count: ${screen.overflowCount}`);
  if (expectHistorical) {
    requireCondition(screen.hasPastBadge, 'historical source badge is missing');
    requireCondition(screen.hasPastOddsRows, 'historical odds row label is missing');
    requireCondition(!screen.hasLiveRowLeak, 'live row label leaked into historical odds section');
  }
  requireCondition(failedRequests.length === 0, `request failures:\n${failedRequests.join('\n')}`);
  requireCondition(consoleErrors.length === 0, `console/page errors:\n${consoleErrors.join('\n')}`);
  console.log(JSON.stringify({
    previewUrl,
    elapsedMs: Date.now() - started,
    apiCalls: apiPayloads.length,
    status: firstBody.status,
    marketUsed: firstBody.market_used,
    marketOddsCount: marketOdds.length,
    actualOrder: firstBody.actual_result?.actual_order,
    risk: riskCode(firstBody),
    pollDelayMs: firstBody.poll_delay_ms,
    screenshotPath
  }, null, 2));
} finally {
  await browser.close();
}
