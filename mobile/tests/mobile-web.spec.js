import { expect, test } from '@playwright/test';

const tabChecks = [
  { label: '홈', text: 'RaceLens' },
  { label: '분석', text: '분석 상세' },
  { label: '랩', text: '검증 랩' },
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

test('home screen exposes the expected analysis controls', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('RaceLens', { exact: true })).toBeVisible();
  await expect(page.getByText('MODEL ONLY')).toBeVisible();
  await expect(page.getByText('광명 5R')).toBeVisible();
  await expect(page.getByText('1순위 신호')).toBeVisible();
  await expect(page.getByText('삼쌍 순서 신호')).toBeVisible();
  await expect(page.getByText('실시간 배당', { exact: true })).toBeVisible();
  await expect(page.getByText('21.4배')).toBeVisible();
  await expect(page.getByText('출전 선수')).toBeVisible();
  await expect(page.getByText('최강우')).toBeVisible();
  await expect(page.getByRole('button', { name: /경륜/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: /5R/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: '모델 신호 보기' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('layout keeps the copper identity without horizontal overflow', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  const ctaBackground = await page.getByRole('button', { name: '모델 신호 보기' }).evaluate((element) =>
    getComputedStyle(element).backgroundColor
  );
  const [red, green, blue] = rgbChannels(ctaBackground);
  expect(red).toBeGreaterThan(green + 35);
  expect(red).toBeGreaterThan(blue + 55);

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

test('race selector updates sport, race, and opens analysis', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await page.getByRole('button', { name: /경마/ }).click();
  await expect(page.getByText('서울', { exact: true })).toBeVisible();
  await expect(page.getByText('24.6배')).toBeVisible();
  await expect(page.getByText('출전마')).toBeVisible();
  await expect(page.getByText('골든포커스')).toBeVisible();
  await page.getByRole('button', { name: /8R/ }).click();
  await expect(page.getByText('서울 8R')).toBeVisible();
  await page.getByRole('button', { name: '모델 신호 보기' }).click();

  await expect(page.getByText('분석 상세')).toBeVisible();
  await expect(page.getByText('실시간 배당', { exact: true })).toBeVisible();
  await expect(page.getByText('기수 문태오 / 57kg')).toBeVisible();
  await expect(page.getByText('표본 10,886')).toBeVisible();
  await expect(page.getByText('삼쌍 순서')).toBeVisible();
  expect(errors).toEqual([]);
});

test('all bottom tabs render their primary workflows', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  for (const item of tabChecks) {
    await page.getByRole('tab', { name: new RegExp(item.label) }).click();
    await expect(page.getByText(item.text, { exact: item.text === 'RaceLens' })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('store safety and pro surfaces avoid betting actions', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('정보 분석 도구')).toBeVisible();
  await page.getByRole('tab', { name: /Pro/ }).click();
  await expect(page.getByText('결제는 스토어 심사 전까지 비활성 상태입니다.')).toBeVisible();
  await expect(page.getByText('광고 슬롯')).toBeVisible();
  await expect(page.getByText(/도박성 광고/)).toBeVisible();
  await expect(page.getByRole('button', { name: /구매|베팅|결제/ })).toHaveCount(0);
  expect(errors).toEqual([]);
});
