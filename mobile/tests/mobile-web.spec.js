import { existsSync, readFileSync, statSync } from 'node:fs';
import { extname, join, normalize, sep } from 'node:path';
import { expect, test } from '@playwright/test';

const qaDist = process.env.RACELENS_QA_DIST ?? 'dist';
const qaNoServer = process.env.RACELENS_QA_NO_SERVER === '1';
const mimeTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ttf': 'font/ttf',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2'
};

function staticFilePath(pathname) {
  const relativePath = pathname === '/' ? 'index.html' : decodeURIComponent(pathname.replace(/^\/+/, ''));
  const root = normalize(qaDist);
  const candidate = normalize(join(root, relativePath));
  if (candidate !== root && !candidate.startsWith(`${root}${sep}`)) return null;
  if (!existsSync(candidate)) return null;
  if (statSync(candidate).isDirectory()) return join(candidate, 'index.html');
  return candidate;
}

test.beforeEach(async ({ page }) => {
  if (qaNoServer) {
    await page.route('**/*', async (route) => {
      const filePath = staticFilePath(new URL(route.request().url()).pathname);
      if (!filePath || !existsSync(filePath)) {
        await route.fulfill({ status: 404, body: 'Not found' });
        return;
      }
      await route.fulfill({
        body: readFileSync(filePath),
        contentType: mimeTypes[extname(filePath)] ?? 'application/octet-stream',
        status: 200
      });
    });
  }
  await page.addInitScript(() => {
    let fixedNow = new Date('2026-07-03T03:00:00.000Z').valueOf();
    const RealDate = Date;
    class FixedDate extends RealDate {
      constructor(...args) {
        if (args.length === 0) {
          super(fixedNow);
          return;
        }
        super(...args);
      }
      static now() {
        return fixedNow;
      }
    }
    FixedDate.UTC = RealDate.UTC;
    FixedDate.parse = RealDate.parse;
    FixedDate.prototype = RealDate.prototype;
    globalThis.__setRaceLensNow = (iso) => {
      fixedNow = new RealDate(iso).valueOf();
    };
    globalThis.Date = FixedDate;
  });
});

const tabChecks = [
  { label: '홈', text: 'RaceLens' },
  { label: '분석', text: '광명 1R 분석' },
  { label: 'Pro', text: 'RaceLens Pro' }
];

function collectErrors(page) {
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

function rgbChannels(value) {
  return value.match(/\d+/g).slice(0, 3).map(Number);
}

async function requestFreeAnalysis(page) {
  await page.getByRole('button', { name: '모델 신호 보기' }).click();
  await page.getByRole('button', { name: '광고 확인 후 분석 보기' })
    .waitFor({ state: 'visible', timeout: 500 })
    .then(() => page.getByRole('button', { name: '광고 확인 후 분석 보기' }).click())
    .catch(() => undefined);
}

test('home screen exposes the expected analysis controls', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('RaceLens', { exact: true })).toBeVisible();
  await expect(page.getByRole('tab', { name: /랩/ })).toHaveCount(0);
  await expect(page.getByText('조건 선택').first()).toBeVisible();
  await expect(page.getByText('모델 신호 보기').first()).toBeVisible();
  await expect(page.getByText('분석 확인')).toHaveCount(0);
  await expect(page.getByRole('textbox', { name: '분석일' })).toHaveCount(0);
  await expect(page.getByText('2026-07-03', { exact: true })).toBeVisible();
  await expect(page.getByText('공식 일정을 불러오지 못했습니다', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '이전 경기일' })).toBeVisible();
  await expect(page.getByRole('button', { name: '이전 경기일' })).toBeDisabled();
  await expect(page.getByRole('button', { name: '다음 경기일' })).toBeDisabled();
  await expect(page.getByRole('button', { name: /경륜/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: '창원' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '부산' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '1R', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: '모델 신호 보기' })).toBeVisible();
  await expect(page.getByText('무료 분석 0/3 사용 · 오늘 3회 남음', { exact: true })).toBeVisible();
  await expect(page.getByText('경륜 판단 근거')).toHaveCount(0);
  await expect(page.getByText('배당 자료', { exact: true })).toHaveCount(0);
  await expect(page.getByText('출전 선수')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('calendar auto-refreshes to the current race day when the app resumes', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('2026-07-03', { exact: true })).toBeVisible();
  await expect(page.getByText('공식 일정을 불러오지 못했습니다', { exact: true })).toBeVisible();

  await page.evaluate(() => {
    globalThis.__setRaceLensNow('2026-07-04T03:00:00.000Z');
    window.dispatchEvent(new Event('focus'));
  });

  await expect(page.getByText('2026-07-04', { exact: true })).toBeVisible();
  await expect(page.getByText('공식 일정을 불러오지 못했습니다', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('calendar refresh keeps a manually selected historical race day stable', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('공식 일정을 불러오지 못했습니다', { exact: true })).toBeVisible();

  await page.evaluate(() => {
    globalThis.__setRaceLensNow('2026-07-04T03:00:00.000Z');
    window.dispatchEvent(new Event('focus'));
  });

  await expect(page.getByText('2026-07-04', { exact: true })).toBeVisible();
  await expect(page.getByText('공식 일정을 불러오지 못했습니다', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('analysis does not invent demo racers when the official API is unavailable', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('예측 순서 5-1-7')).toHaveCount(0);
  await requestFreeAnalysis(page);
  await expect(page.getByText('광명 1R 분석')).toBeVisible();
  await expect(page.getByTestId('analysis-empty-state')).toBeVisible();
  await expect(page.getByText(/API URL이 설정되지 않아 공식 출전표를 표시하지 않습니다/)).toBeVisible();
  await expect(page.getByTestId('prediction-podium')).toHaveCount(0);
  await expect(page.getByText(/^예측 순서 \d/)).toHaveCount(0);
  await expect(page.getByText(/모델 추정\s*0%/)).toHaveCount(0);
  await expect(page.locator('body')).not.toContainText(/0번/);
  await expect(page.getByText('5번 최강우')).toHaveCount(0);
  await expect(page.getByText('1번 김태훈')).toHaveCount(0);
  await expect(page.getByText('7번 서지환')).toHaveCount(0);
  await expect(page.getByText('출전 선수')).toHaveCount(0);
  await expect(page.getByText('출전 정보 대기 중')).toHaveCount(0);
  await expect(page.getByText('초반 주도권을 잡으면 버티는 힘이 좋지만')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('layout keeps the sport action identity without horizontal overflow', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  const ctaBackground = await page.getByTestId('analyze-cta').evaluate((element) =>
    getComputedStyle(element).backgroundColor
  );
  const [red, green, blue] = rgbChannels(ctaBackground);
  expect(green).toBeGreaterThan(red + 25);
  expect(green).toBeGreaterThan(blue + 100);

  await page.getByTestId('sport-horse').click();
  const horseCtaBackground = await page.getByTestId('analyze-cta').evaluate((element) =>
    getComputedStyle(element).backgroundColor
  );
  const [horseRed, horseGreen, horseBlue] = rgbChannels(horseCtaBackground);
  expect(horseRed).toBeGreaterThan(horseGreen + 45);
  expect(horseGreen).toBeGreaterThan(horseBlue + 60);

  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth
  );
  expect(overflow).toBeLessThanOrEqual(1);
  expect(errors).toEqual([]);
});

test('interactive controls keep mobile-friendly tap targets', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  const smallTargets = await page.evaluate(() =>
    [...document.querySelectorAll('[role="button"], [role="tab"], button')]
      .map((element) => {
        const rect = element.getBoundingClientRect();
        const label = (element.textContent || element.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
        return { label, width: Math.round(rect.width), height: Math.round(rect.height) };
      })
      .filter((target) => target.width < 44 || target.height < 44)
  );

  expect(smallTargets).toEqual([]);
  expect(errors).toEqual([]);
});

test('free analysis limit opens an honest Pro guidance state', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  for (let count = 1; count <= 3; count += 1) {
    await requestFreeAnalysis(page);
    await expect(page.getByTestId('analysis-empty-state')).toBeVisible();
    await page.getByRole('tab', { name: /홈/ }).click();
    await expect(page.getByText(`무료 분석 ${count}/3 사용 · 오늘 ${3 - count}회 남음`, { exact: true })).toBeVisible();
  }

  await page.getByRole('button', { name: '무료 분석 한도 안내 보기' }).click();
  await expect(page.getByText(/오늘 무료 분석 한도를 모두 사용했습니다/)).toBeVisible();
  await expect(page.getByText('계정 상태')).toBeVisible();
  await expect(page.getByText('오늘 3/3 사용', { exact: true })).toBeVisible();
  await expect(page.getByText('0회', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /구매|베팅|결제|구독 시작/ })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('bottom content remains reachable above the tab bar', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  const scrolled = await page.evaluate(() => {
    const scrollable = [...document.querySelectorAll('div')]
      .find((element) => element.scrollHeight > element.clientHeight + 20 && getComputedStyle(element).overflowY !== 'visible');
    if (!scrollable) return false;
    scrollable.scrollTop = scrollable.scrollHeight;
    return true;
  });
  if (scrolled) await page.waitForTimeout(150);

  const finalNotice = await page.getByText('적중률은 수익을 뜻하지 않습니다 · 정보 분석 전용입니다.').boundingBox();
  const tabBar = await page.getByRole('tab', { name: /홈/ }).locator('..').boundingBox();
  expect(finalNotice).not.toBeNull();
  expect(tabBar).not.toBeNull();
  expect(finalNotice.y + finalNotice.height).toBeLessThan(tabBar.y - 12);
  expect(errors).toEqual([]);
});

test('race selector updates sport, race, and opens analysis', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await page.getByRole('button', { name: /경마/ }).click();
  await expect(page.getByRole('button', { name: /경마/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: /경륜/ })).toHaveAttribute('aria-selected', 'false');
  await expect(page.getByText('공식 일정을 불러오지 못했습니다', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '서울' })).toBeVisible();
  await expect(page.getByRole('button', { name: '부경' })).toBeVisible();
  await expect(page.getByRole('button', { name: '제주' })).toBeVisible();
  await page.getByRole('button', { name: '부경' }).click();
  await expect(page.getByRole('button', { name: '부경' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('24.6배')).toHaveCount(0);
  await expect(page.getByText('출전마')).toHaveCount(0);
  await page.getByRole('button', { name: /8R/ }).click();
  await expect(page.getByRole('button', { name: '8R', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('button', { name: '모델 신호 보기' }).click();

  await expect(page.getByText('부경 8R 분석')).toBeVisible();
  await expect(page.getByText('분석일 2026-07-03')).toBeVisible();
  await expect(page.getByTestId('analysis-empty-state')).toBeVisible();
  await expect(page.getByTestId('prediction-podium')).toHaveCount(0);
  await expect(page.getByText('예측 순서 5-3-1')).toHaveCount(0);
  await expect(page.getByText('5번 골든포커스')).toHaveCount(0);
  await expect(page.getByText('3번 스톰레이크')).toHaveCount(0);
  await expect(page.getByText('1번 새벽질주')).toHaveCount(0);
  await expect(page.getByText('경마 판단 근거')).toHaveCount(0);
  await expect(page.getByText('배당 자료', { exact: true })).toHaveCount(0);
  await expect(page.getByText('출전 정보 대기 중')).toHaveCount(0);
  await expect(page.getByText('표본 0')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('all bottom tabs render their primary workflows', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByRole('tab')).toHaveCount(3);
  await expect(page.getByRole('tab', { name: /랩/ })).toHaveCount(0);
  for (const item of tabChecks) {
    await page.getByRole('tab', { name: new RegExp(item.label) }).click();
    await expect(page.getByText(item.text, { exact: item.text === 'RaceLens' }).first()).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('store safety and pro surfaces avoid betting actions', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('정보 분석 도구')).toBeVisible();
  await page.getByRole('tab', { name: /Pro/ }).click();
  await expect(page.getByText('무료 플랜', { exact: true })).toBeVisible();
  await expect(page.getByText('Pro 플랜', { exact: true })).toBeVisible();
  await expect(page.getByText('월 5,000원', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Pro 기능은 출시 준비 중입니다. 지금은 무료 분석을 그대로 이용할 수 있습니다.', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('현재 이용 상태', { exact: true })).toBeVisible();
  await expect(page.getByText('무료 이용 중', { exact: true })).toBeVisible();
  await expect(page.getByText('오늘 0/3 사용', { exact: true })).toBeVisible();
  await expect(page.getByText('3회', { exact: true })).toBeVisible();
  await expect(page.getByText('공식 출전표 확인 후 표시', { exact: true })).toBeVisible();
  await expect(page.getByText('계정 상태', { exact: true })).toBeVisible();
  await expect(page.getByText('광고 슬롯', { exact: true })).toHaveCount(0);
  await expect(page.getByText(/도박성 광고|베팅 사이트 광고/)).toHaveCount(0);
  await expect(page.getByRole('button', { name: /베팅|결제|Pro 구독 시작|구매 복원/ })).toHaveCount(0);
  await expect(page.getByTestId('pro-purchase-cta')).toHaveCount(0);
  await expect(page.getByTestId('pro-restore-cta')).toHaveCount(0);
  expect(errors).toEqual([]);
});
