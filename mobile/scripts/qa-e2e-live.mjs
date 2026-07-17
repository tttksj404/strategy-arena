import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createServer as createNetServer } from 'node:net';
import { tmpdir } from 'node:os';
import { extname, join } from 'node:path';
import { chromium } from 'playwright';
import { exitIfFailed, spawnQa } from './qa-utils.mjs';

const liveBaseUrl = (process.env.EXPO_PUBLIC_RACELENS_API_BASE_URL ?? 'https://168-107-2-218.sslip.io').replace(/\/+$/, '');
const runsDir = join(process.cwd(), '..', 'runs');
const reportPath = join(runsDir, 'e2e_live_report.md');
const legalPaths = {
  개인정보처리방침: '/legal/privacy',
  이용약관: '/legal/terms',
  '계정 삭제 안내': '/legal/account-deletion',
  '지원 문의': '/legal/support'
};

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

function bundleName(distRoot) {
  const metadata = JSON.parse(readFileSync(join(distRoot, 'metadata.json'), 'utf8'));
  const bundle = metadata?.bundler === 'metro'
    ? readFileSync(join(distRoot, 'index.html'), 'utf8').match(/_expo\/static\/js\/web\/[^"]+\.js/)?.[0]
    : undefined;
  if (!bundle) throw new Error('Unable to locate Expo web bundle');
  return bundle.replace('_expo/static/js/web/', '');
}

function staticServer(distRoot) {
  return createServer((request, response) => {
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
}

async function fetchJson(path) {
  const response = await fetch(`${liveBaseUrl}${path}`);
  const text = await response.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = { raw: text.slice(0, 200) };
  }
  return { status: response.status, body };
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function raceDateLabel(date) {
  const parsed = new Date(`${date}T00:00:00Z`);
  const weekday = ['일', '월', '화', '수', '목', '금', '토'][parsed.getUTCDay()];
  return `${date.slice(5).replace('-', '.')} ${weekday}`;
}

function sortedDates(days) {
  return [...new Set((days ?? []).filter((date) => /^\d{4}-\d{2}-\d{2}$/.test(date)))].sort();
}

function nearestRaceDate(days, preferredDate = todayKst()) {
  if (days.length === 0) return '';
  const preferred = Date.parse(`${preferredDate}T00:00:00Z`);
  if (Number.isNaN(preferred)) return days[days.length - 1] ?? '';
  return [...days]
    .map((date) => ({ date, distance: Math.abs(Date.parse(`${date}T00:00:00Z`) - preferred) }))
    .sort((left, right) => left.distance - right.distance || left.date.localeCompare(right.date))[0]?.date ?? '';
}

function todayKst() {
  return new Intl.DateTimeFormat('en-CA', {
    day: '2-digit',
    month: '2-digit',
    timeZone: 'Asia/Seoul',
    year: 'numeric'
  }).format(new Date());
}

async function clickAnalysis(page) {
  await page.getByRole('button', { name: '모델 신호 보기' }).click();
  const adConfirm = page.getByRole('button', { name: '광고 확인 후 분석 보기' });
  await adConfirm.waitFor({ state: 'visible', timeout: 1500 }).then(() => adConfirm.click()).catch(() => {});
}

async function assertNoOverflow(page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert(overflow <= 1, `horizontal overflow ${overflow}px`);
}

async function assertPageTextIncludes(page, expected) {
  const bodyText = await page.locator('body').textContent({ timeout: 10000 });
  assert(bodyText?.includes(expected), `page text missing ${expected}`);
}

async function waitForAnalysisState(page) {
  try {
    return await Promise.any([
      page.getByText(/경주 종료|실제 착순/).first().waitFor({ timeout: 20000 }).then(() => 'settled'),
      page.getByTestId('roster-waiting-state').waitFor({ timeout: 20000 }).then(() => 'roster_waiting'),
      page.getByTestId('analysis-empty-state').waitFor({ timeout: 20000 }).then(() => 'honest_unavailable'),
      page.getByTestId('prediction-podium').waitFor({ timeout: 20000 }).then(() => 'prediction_visible')
    ]);
  } catch (error) {
    if (error instanceof AggregateError) {
      throw new Error(error.errors.map((item) => item instanceof Error ? item.message : String(item)).join('\n'));
    }
    throw error;
  }
}

function stepRecorder() {
  const rows = [];
  return {
    rows,
    async run(name, fn, options = {}) {
      try {
        const evidence = await fn();
        rows.push({ name, result: 'PASS', evidence: String(evidence ?? 'ok') });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (options.expectedUntilDeploy) {
          rows.push({ name, result: 'EXPECTED_FAIL_UNTIL_DEPLOY', evidence: message });
          return;
        }
        rows.push({ name, result: 'FAIL', evidence: message });
        throw error;
      }
    }
  };
}

const appPort = await freePort();
const distRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-e2e-live-'));
const exportResult = spawnQa('npx', ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
  env: {
    ...process.env,
    EXPO_PUBLIC_RACELENS_ACCOUNT_DELETION_URL: `${liveBaseUrl}${legalPaths['계정 삭제 안내']}`,
    EXPO_PUBLIC_RACELENS_ANALYTICS_URL: `${liveBaseUrl}/api/ux-events`,
    EXPO_PUBLIC_RACELENS_API_BASE_URL: liveBaseUrl,
    EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '0',
    EXPO_PUBLIC_RACELENS_PRIVACY_URL: `${liveBaseUrl}${legalPaths['개인정보처리방침']}`,
    EXPO_PUBLIC_RACELENS_SUPPORT_EMAIL: 'tttksj@gmail.com',
    EXPO_PUBLIC_RACELENS_SUPPORT_URL: `${liveBaseUrl}${legalPaths['지원 문의']}`,
    EXPO_PUBLIC_RACELENS_TERMS_URL: `${liveBaseUrl}${legalPaths['이용약관']}`
  }
});
exitIfFailed(exportResult);

const bundleText = readFileSync(join(distRoot, '_expo/static/js/web', bundleName(distRoot)), 'utf8');
for (const expected of [liveBaseUrl, `${liveBaseUrl}/api/ux-events`, `${liveBaseUrl}/legal/privacy`]) {
  assert(bundleText.includes(expected), `export bundle missing ${expected}`);
}

const server = staticServer(distRoot);
await new Promise((resolveListen) => server.listen(appPort, '127.0.0.1', resolveListen));

const browser = await chromium.launch({ args: ['--disable-web-security'] });
const page = await browser.newPage({ ignoreHTTPSErrors: true, viewport: { width: 390, height: 844 }, colorScheme: 'dark' });
const consoleErrors = [];
const liveDecisionResponses = [];
const uxResponses = [];

page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text());
});
page.on('pageerror', (error) => consoleErrors.push(error.message));
page.on('response', async (response) => {
  if (response.url().includes('/api/live-decision')) {
    liveDecisionResponses.push({ status: response.status(), url: response.url() });
  }
  if (response.url().includes('/api/ux-events')) {
    uxResponses.push(response.status());
  }
});

const steps = stepRecorder();
let keirinDays = [];
let horseDays = [];
let selectedPastDate = '';

try {
  await steps.run('01 /recent keirin source and home chip match', async () => {
    const recent = await fetchJson('/recent?sport=keirin&meet=%EA%B4%91%EB%AA%85');
    assert(recent.status === 200, `/recent keirin status ${recent.status}`);
    keirinDays = sortedDates(recent.body?.days);
    assert(keirinDays.length > 0, 'keirin days empty');
    await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
    await page.getByText('RaceLens', { exact: true }).waitFor();
    const selectedDate = nearestRaceDate(keirinDays);
    await page.getByText(raceDateLabel(selectedDate), { exact: true }).waitFor();
    await assertNoOverflow(page);
    return `${keirinDays.length} days, selected ${selectedDate}: ${keirinDays.join(', ')}`;
  }, { expectedUntilDeploy: true });

  await steps.run('02 direct mobile alias /recent?sport=kra resolves horse Seoul', async () => {
    const recent = await fetchJson('/recent?sport=kra&meet=%EC%84%9C%EC%9A%B8');
    assert(recent.status === 200, `/recent kra status ${recent.status}`);
    assert(recent.body?.sport === 'horse', `sport ${recent.body?.sport}`);
    assert(recent.body?.meet === '서울', `meet ${recent.body?.meet}`);
    return JSON.stringify(recent.body);
  }, { expectedUntilDeploy: true });

  await steps.run('03 horse Seoul schedule loads in UI', async () => {
    await page.getByRole('button', { name: /경마/ }).click();
    await page.getByRole('button', { name: '서울' }).waitFor();
    const recent = await fetchJson('/recent?sport=horse&meet=%EC%84%9C%EC%9A%B8');
    assert(recent.status === 200, `/recent horse status ${recent.status}`);
    horseDays = sortedDates(recent.body?.days);
    assert(horseDays.length > 0, 'horse Seoul days empty');
    const selectedDate = nearestRaceDate(horseDays);
    await page.getByText(raceDateLabel(selectedDate), { exact: true }).waitFor();
    return `${horseDays.length} horse days, selected ${selectedDate}`;
  }, { expectedUntilDeploy: true });

  await steps.run('04 analysis journey renders honest result state', async () => {
    await page.getByRole('button', { name: /경륜/ }).click();
    await page.getByRole('button', { name: '광명' }).waitFor();
    await page.getByRole('button', { name: '1R', exact: true }).click();
    await clickAnalysis(page);
    await page.getByText(/광명 1R 분석/).waitFor({ timeout: 20000 });
    const state = await waitForAnalysisState(page);
    assert(liveDecisionResponses.length === 1, `live-decision count ${liveDecisionResponses.length}`);
    await assertNoOverflow(page);
    return `live status ${liveDecisionResponses[0]?.status}, state ${state}`;
  });

  await steps.run('05 tab roundtrip preserves selected race context', async () => {
    await page.getByRole('tab', { name: /홈/ }).click();
    await page.getByRole('tab', { name: /Pro/ }).click();
    await page.getByText('RaceLens Pro').waitFor();
    await page.getByRole('tab', { name: /분석/ }).click();
    await page.getByText(/광명 1R 분석/).waitFor();
    return 'home -> pro -> analyze kept 광명 1R';
  });

  await steps.run('06 legal links open live server URLs', async () => {
    await page.getByRole('tab', { name: /Pro/ }).click();
    const statuses = [];
    for (const [label, path] of Object.entries(legalPaths)) {
      const url = `${liveBaseUrl}${path}`;
      const response = await fetch(url);
      statuses.push(`${label}:${response.status}`);
      assert(response.status === 200, `${label} ${url} returned ${response.status}`);
      await page.getByRole('link', { name: label }).waitFor();
    }
    return statuses.join(', ');
  }, { expectedUntilDeploy: true });

  await steps.run('07 analysis refresh stays within quota', async () => {
    await page.getByRole('tab', { name: /홈/ }).click();
    await clickAnalysis(page);
    await page.getByText(/광명 1R 분석/).waitFor({ timeout: 20000 });
    assert(liveDecisionResponses.length === 2, `live-decision count ${liveDecisionResponses.length}`);
    return 'second analysis request completed';
  });

  await steps.run('08 ux-events posts to live server with 202', async () => {
    await page.waitForTimeout(1200);
    assert(uxResponses.includes(202), `ux statuses ${uxResponses.join(',') || 'none'}`);
    return `ux statuses ${uxResponses.join(',')}`;
  });

  await steps.run('09 past race day lookup renders settled or honest unavailable state', async () => {
    const today = todayKst();
    selectedPastDate = nearestRaceDate(keirinDays);
    if (selectedPastDate >= today) selectedPastDate = [...keirinDays].reverse().find((day) => day < today) ?? keirinDays[0] ?? '';
    assert(selectedPastDate, 'no race day available');
    await page.getByRole('tab', { name: /홈/ }).click();
    await page.getByText(raceDateLabel(selectedPastDate), { exact: true }).click();
    await clickAnalysis(page);
    await page.getByText(new RegExp(`분석일 ${selectedPastDate}`)).waitFor({ timeout: 20000 });
    const state = await waitForAnalysisState(page);
    assert(liveDecisionResponses.length === 3, `live-decision count ${liveDecisionResponses.length}`);
    await assertNoOverflow(page);
    return `${selectedPastDate}: ${state}`;
  }, { expectedUntilDeploy: true });

  await steps.run('10 browser stability gates', async () => {
    assert(consoleErrors.length === 0, `console/page errors: ${consoleErrors.join('\n')}`);
    await assertNoOverflow(page);
    return 'console error 0, overflow <= 1px';
  });
} finally {
  await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}

const unexpectedFailures = steps.rows.filter((row) => row.result === 'FAIL');
const passed = steps.rows.filter((row) => row.result === 'PASS').length;
const total = steps.rows.length;
mkdirSync(runsDir, { recursive: true });
writeFileSync(reportPath, [
  '# RaceLens Live E2E Report',
  '',
  `- Target: ${liveBaseUrl}`,
  `- Export API: ${liveBaseUrl}`,
  '- Browser harness: Chromium web security disabled to emulate native mobile networking against the live API',
  `- Analysis calls: ${liveDecisionResponses.length}/3`,
  `- Past date: ${selectedPastDate || 'n/a'}`,
  '',
  '| Journey | Result | Evidence |',
  '|---|---|---|',
  ...steps.rows.map((row) => `| ${row.name} | ${row.result} | ${row.evidence.replaceAll('|', '\\|')} |`),
  '',
  '## DEPLOY_REQUIRED',
  '',
  '- `/recent` CORS and sport aliases `kra -> horse`, `kcycle -> keirin` must be deployed to the live server.',
  '- `/legal/support` and `EXPO_PUBLIC_RACELENS_SUPPORT_URL` must be deployed before the fourth Pro information link can return HTTP 200 on the live server.',
  '- Past keirin result settlement fix must be deployed for old races to return `status: settled` whenever official results are available.',
  ''
].join('\n'), 'utf8');

if (unexpectedFailures.length > 0) process.exit(1);
console.log(`live E2E QA passed ${passed}/${total}; report=${reportPath}`);
