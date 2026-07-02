import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: {
    timeout: 5_000
  },
  fullyParallel: false,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:8064',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure'
  },
  webServer: {
    command: 'python3 -m http.server 8064 --directory dist',
    reuseExistingServer: !process.env.CI,
    timeout: 10_000,
    url: 'http://127.0.0.1:8064'
  },
  projects: [
    {
      name: 'mobile-390',
      use: {
        ...devices['iPhone 15'],
        browserName: 'chromium',
        viewport: { width: 390, height: 844 }
      }
    },
    {
      name: 'tablet-768',
      use: {
        ...devices['iPad Pro 11'],
        browserName: 'chromium',
        viewport: { width: 768, height: 1024 }
      }
    }
  ]
});
