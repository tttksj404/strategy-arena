import { expect, test } from '@playwright/test';

function collectErrors(page) {
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

async function requestFreeAnalysis(page) {
  await page.getByRole('button', { name: '모델 신호 보기' }).click();
  await page.getByRole('button', { name: '광고 확인 후 분석 보기' })
    .waitFor({ state: 'visible', timeout: 500 })
    .then(() => page.getByRole('button', { name: '광고 확인 후 분석 보기' }).click())
    .catch(() => undefined);
}

test('store notice explains negative expected value near analysis output', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');

  await expect(page.getByText('정보 분석 도구')).toBeVisible();
  await expect(page.getByText('적중률은 수익을 뜻하지 않습니다 · 정보 분석 전용입니다.')).toBeVisible();
  await requestFreeAnalysis(page);
  await expect(page.getByText('광명 1R 분석')).toBeVisible();
  await expect(page.getByText(/장기적으로 평균 손실입니다/)).toBeVisible();
  await expect(page.getByText(/RaceLens는 수익 도구가 아니라/)).toBeVisible();
  await expect(page.getByText(/도박문제 상담은 1336/)).toBeVisible();
  expect(errors).toEqual([]);
});
