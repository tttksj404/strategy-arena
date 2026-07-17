import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoDir = resolve(scriptDir, '..', '..');
const publicUrlFile = join(repoDir, '.runtime', 'racelens-preview', 'public_url');
const baseUrl = (process.env.RACELENS_QA_URL || (existsSync(publicUrlFile) ? readFileSync(publicUrlFile, 'utf8') : '')).trim();

if (!baseUrl) {
  console.error('RACELENS_QA_URL is missing and .runtime/racelens-preview/public_url does not exist.');
  process.exit(1);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchJson(path, options = {}) {
  const response = await fetch(new URL(path, baseUrl), options);
  const text = await response.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: response.status, body };
}

async function confirmAdIfPresent(page) {
  const adConfirm = page.getByRole('button', { name: '광고 확인 후 분석 보기' });
  await adConfirm.waitFor({ state: 'visible', timeout: 2500 }).then(() => adConfirm.click()).catch(() => {});
}

async function waitForMarketPayload(page, apiPayloads) {
  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    if (apiPayloads.some((payload) => payload?.market_used === true || payload?.status === 'settled')) return;
    await page.waitForTimeout(250);
  }
}

async function assertScreen(page, label) {
  const metrics = await page.evaluate(() => {
    const bodyText = document.body.innerText;
    const smallTargets = [...document.querySelectorAll('[role="button"], [role="tab"], button')]
      .filter((element) => {
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      })
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const text = (element.textContent || element.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
        return { text, width: Math.round(rect.width), height: Math.round(rect.height) };
      })
      .filter((target) => target.width < 44 || target.height < 44);
    const forbiddenActionButtons = [...document.querySelectorAll('[role="button"], button')]
      .map((element) => (element.textContent || element.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim())
      .filter((text) => /구매|베팅|결제/.test(text));
    return {
      badText: bodyText.includes('�'),
      forbiddenActionButtons,
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      smallTargets
    };
  });
  assert(metrics.overflow <= 1, `${label}: horizontal overflow ${metrics.overflow}px`);
  assert(!metrics.badText, `${label}: broken replacement character rendered`);
  assert(metrics.smallTargets.length === 0, `${label}: small tap targets ${JSON.stringify(metrics.smallTargets)}`);
  assert(metrics.forbiddenActionButtons.length === 0, `${label}: forbidden betting/purchase buttons ${metrics.forbiddenActionButtons.join(', ')}`);
  return metrics;
}

async function runCase(browser, name, flow, viewport = { width: 390, height: 844 }) {
  const context = await browser.newContext({ viewport, isMobile: viewport.width < 700, colorScheme: name.includes('dark') ? 'dark' : 'light' });
  const page = await context.newPage();
  const consoleErrors = [];
  const failedRequests = [];
  const badResponses = [];
  const apiPayloads = [];

  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => consoleErrors.push(error.message));
  page.on('requestfailed', (request) => failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ''}`));
  page.on('response', async (response) => {
    if (response.status() >= 400) badResponses.push(`${response.status()} ${response.url()}`);
    if (response.url().includes('/api/live-decision')) {
      try {
        apiPayloads.push(await response.json());
      } catch {
        apiPayloads.push(null);
      }
    }
  });

  try {
    await page.goto(baseUrl, { waitUntil: 'networkidle', timeout: 30000 });
    await flow(page, apiPayloads);
    await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {});
    const metrics = await assertScreen(page, name);
    assert(consoleErrors.length === 0, `${name}: console/page errors\n${consoleErrors.join('\n')}`);
    const actionableFailedRequests = failedRequests.filter((failure) => {
      return !(failure.includes('/api/live-decision') && failure.includes('net::ERR_ABORTED'));
    });
    assert(actionableFailedRequests.length === 0, `${name}: request failures\n${actionableFailedRequests.join('\n')}`);
    assert(badResponses.length === 0, `${name}: bad HTTP responses\n${badResponses.join('\n')}`);
    return { name, apiCalls: apiPayloads.length, metrics };
  } finally {
    await context.close();
  }
}

const health = await fetchJson('/health');
assert(health.status === 200 && health.body?.ok === true, `/health failed: ${health.status}`);

const recentKeirin = await fetchJson('/recent?sport=keirin&meet=%EA%B4%91%EB%AA%85');
assert(recentKeirin.status === 200, `/recent keirin failed: ${recentKeirin.status}`);
assert(Array.isArray(recentKeirin.body?.days) && recentKeirin.body.days.includes('2026-07-03'), 'keirin recent days do not include 2026-07-03');

const recentHorse = await fetchJson('/recent?sport=horse&meet=%EB%B6%80%EA%B2%BD');
assert(recentHorse.status === 200, `/recent horse failed: ${recentHorse.status}`);
assert(Array.isArray(recentHorse.body?.days) && recentHorse.body.days.length > 0, 'horse recent days are empty');

const liveDecision = await fetchJson('/api/live-decision?sport=keirin&date=2026-07-03&meet=%EA%B4%91%EB%AA%85&race_no=1', {
  headers: {
    'X-RaceLens-Device-Id': `audit-${Date.now()}`,
    'X-RaceLens-Platform': 'web-audit'
  }
});
assert(liveDecision.status === 200, `/api/live-decision failed: ${liveDecision.status}`);
const settledKeirin = liveDecision.body?.status === 'settled';
if (settledKeirin) {
  assert(JSON.stringify(liveDecision.body?.actual_result?.actual_order) === JSON.stringify([4, 1, 5]), `settled actual order mismatch: ${JSON.stringify(liveDecision.body?.actual_result?.actual_order)}`);
  assert(liveDecision.body?.market_used === false, 'settled result should not reuse final odds as prediction signal');
} else {
  assert(liveDecision.body?.market_used === true, 'live decision did not use market odds');
  assert((liveDecision.body?.market_odds || []).length >= 7, 'live decision market odds are too sparse');
}
assert((liveDecision.body?.participants || []).length >= 7, 'live decision participants are too sparse');
const officialKeirinNames = ['황종대', '이흥주', '박진홍', '최건묵', '이승주', '박유찬', '김성진'];
const liveParticipantNames = [...(liveDecision.body?.participants || [])]
  .sort((left, right) => Number(left.number) - Number(right.number))
  .map((participant) => participant.name);
assert(JSON.stringify(liveParticipantNames) === JSON.stringify(officialKeirinNames), `live participants mismatch: ${liveParticipantNames.join(', ')}`);

const browser = await chromium.launch();
try {
  const results = [];
  results.push(await runCase(browser, 'home-keirin-320-light', async (page) => {
    await page.getByText('RaceLens', { exact: true }).waitFor();
    assert(await page.getByRole('tab', { name: /랩/ }).count() === 0, 'Lab tab should not render');
    await page.getByText('조건 선택').first().waitFor();
    await page.getByText('2026-07-03', { exact: true }).waitFor();
    await page.getByRole('button', { name: '07.03 금' }).waitFor();
    assert(await page.getByRole('button', { name: '07.03 금' }).getAttribute('aria-selected') === 'true', 'today chip is not selected');
    assert(await page.getByRole('button', { name: '창원' }).count() === 0, 'unsupported Changwon venue rendered');
    assert(await page.getByRole('button', { name: '부산' }).count() === 0, 'unsupported Busan venue rendered');
    await page.getByRole('button', { name: '광명' }).click();
    await page.getByRole('button', { name: '16R' }).click();
    assert(await page.getByRole('button', { name: '16R' }).getAttribute('aria-selected') === 'true', '16R was not selected');
  }, { width: 320, height: 720 }));

  results.push(await runCase(browser, 'keirin-analysis-390-dark', async (page, apiPayloads) => {
    await page.getByRole('button', { name: '모델 신호 보기' }).click();
    await confirmAdIfPresent(page);
    await page.getByText('광명 1R 분석').waitFor({ timeout: 15000 });
    await page.getByText(/^(예측 순서|실제 착순) \d/).waitFor();
    await page.getByText('경륜 판단 근거').waitFor();
    await page.getByText('배당 자료', { exact: true }).waitFor();
    await page.getByText('출전 선수', { exact: true }).waitFor();
    await page.getByText('정보 분석 도구', { exact: true }).waitFor();
    await waitForMarketPayload(page, apiPayloads);
    assert(apiPayloads.some((payload) => payload?.market_used === true || payload?.status === 'settled'), 'keirin browser flow did not receive live or settled source');
    if (apiPayloads.some((payload) => payload?.status === 'settled')) {
      await page.getByText(/^실제 착순 \d/).waitFor();
    }
  }));

  results.push(await runCase(browser, 'horse-analysis-390-dark', async (page) => {
    await page.getByRole('button', { name: '경마' }).click();
    await page.getByRole('button', { name: '부경' }).click();
    await page.getByRole('button', { name: '8R' }).click();
    await page.getByRole('button', { name: '모델 신호 보기' }).click();
    await confirmAdIfPresent(page);
    await page.getByText(/부경 8R 분석|서울 8R 분석/).waitFor({ timeout: 15000 });
    await page.getByText(/^예측 순서 \d/).waitFor();
    await page.getByText('경마 판단 근거').waitFor();
    await page.getByText('출전마', { exact: true }).waitFor();
    await page.getByText(/말·기수·부담중량|기수/).first().waitFor();
    await page.getByText('배당 자료', { exact: true }).waitFor();
  }));

  if (liveDecision.body.app_session?.entitlement === 'pro') {
    results.push(await runCase(browser, 'pro-unlimited-390-light', async (page) => {
      for (let count = 0; count < 4; count += 1) {
        await page.getByRole('button', { name: '모델 신호 보기' }).click();
        await page.getByText('Pro 고급 분석 열림', { exact: true }).waitFor({ timeout: 15000 });
        await page.getByRole('tab', { name: /홈/ }).click();
        await page.getByText('Pro 무제한 분석 활성', { exact: true }).waitFor({ timeout: 15000 });
      }
      assert(await page.getByRole('button', { name: 'Pro에서 무제한 사용' }).count() === 0, 'pro flow rendered exhausted free quota button');
    }));
  } else {
    results.push(await runCase(browser, 'free-quota-390-light', async (page) => {
      for (let count = 1; count <= 3; count += 1) {
        await page.getByRole('button', { name: '모델 신호 보기' }).click();
        await confirmAdIfPresent(page);
        await page.getByText(`무료 분석 ${count}/3 사용`, { exact: true }).waitFor({ timeout: 15000 });
        await page.getByRole('tab', { name: /홈/ }).click();
      }
      await page.getByRole('button', { name: 'Pro에서 무제한 사용' }).waitFor();
      assert(await page.getByRole('button', { name: 'Pro에서 무제한 사용' }).isDisabled(), 'free quota exhausted button is not disabled');
      await page.getByText('오늘 0회 남음').waitFor();
    }));
  }

  results.push(await runCase(browser, 'pro-surface-768-light', async (page) => {
    await page.getByRole('tab', { name: /Pro/ }).click();
    await page.getByText('RaceLens Pro', { exact: true }).waitFor();
    await page.getByText('무료 플랜', { exact: true }).waitFor();
    await page.getByText('Pro 플랜', { exact: true }).waitFor();
    await page.getByText('Pro 기능은 출시 준비 중입니다. 지금은 무료 분석을 그대로 이용할 수 있습니다.', { exact: true }).first().waitFor();
    await page.getByText('계정 상태', { exact: true }).waitFor();
    await page.getByText('정보 분석 도구', { exact: true }).waitFor();
  }, { width: 768, height: 900 }));

  console.log(JSON.stringify({
    baseUrl,
    api: {
      health: health.body.ok,
      recentKeirin: recentKeirin.body.days.slice(0, 4),
      recentHorse: recentHorse.body.days.slice(0, 4),
      liveMarketUsed: liveDecision.body.market_used,
      liveMarketOdds: liveDecision.body.market_odds.length,
      liveStatus: liveDecision.body.status,
      actualOrder: liveDecision.body.actual_result?.actual_order,
      liveParticipants: liveDecision.body.participants.length
    },
    cases: results
  }, null, 2));
} finally {
  await browser.close();
}
