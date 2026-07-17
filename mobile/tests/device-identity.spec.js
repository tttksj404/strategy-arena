import { expect, test } from '@playwright/test';

test('install identity remains stable across a browser reload', async ({ page }) => {
  await page.goto('/');

  await expect.poll(() => page.evaluate(() => localStorage.getItem('racelens.install-id.v1'))).toBeTruthy();
  const initialDeviceId = await page.evaluate(() => localStorage.getItem('racelens.install-id.v1'));

  await page.reload({ waitUntil: 'networkidle' });

  await expect.poll(() => page.evaluate(() => localStorage.getItem('racelens.install-id.v1'))).toBe(initialDeviceId);
  expect(initialDeviceId).toMatch(/^dev_[A-Za-z0-9_-]{12,96}$/);
});

test('owner Pro link activates the allowlisted device identity without retaining the link fragment', async ({ page }) => {
  const ownerDeviceId = 'dev_owner_4w9nF1pQ2rT3uV5xY7zA9bC0dE6fG8hJ';

  await page.goto(`/#pro=${ownerDeviceId}`);

  await expect.poll(() => page.evaluate(() => localStorage.getItem('racelens.install-id.v1'))).toBe(ownerDeviceId);
  await expect.poll(() => page.evaluate(() => window.location.hash)).toBe('');
});
