import { defineConfig, devices } from '@playwright/test';

const qaPort = process.env.RACELENS_QA_PORT ?? '8064';
const qaBaseUrl = `http://127.0.0.1:${qaPort}`;
const qaDist = process.env.RACELENS_QA_DIST ?? 'dist';
const noServer = process.env.RACELENS_QA_NO_SERVER === '1';

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: {
    timeout: 5_000
  },
  fullyParallel: false,
  reporter: [['list']],
  use: {
    baseURL: qaBaseUrl,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure'
  },
  ...(noServer ? {} : {
    webServer: {
      command: `node scripts/qa-static-server.mjs`,
      env: {
        RACELENS_QA_PORT: qaPort,
        RACELENS_QA_DIST: qaDist
      },
      reuseExistingServer: !process.env.CI,
      timeout: 10_000,
      url: qaBaseUrl
    }
  }),
  projects: [
    {
      name: 'compact-light-320',
      use: {
        ...devices['iPhone SE'],
        browserName: 'chromium',
        colorScheme: 'light',
        viewport: { width: 320, height: 740 }
      }
    },
    {
      name: 'compact-dark-320',
      use: {
        ...devices['iPhone SE'],
        browserName: 'chromium',
        colorScheme: 'dark',
        viewport: { width: 320, height: 740 }
      }
    },
    {
      name: 'mobile-light-390',
      use: {
        ...devices['iPhone 15'],
        browserName: 'chromium',
        colorScheme: 'light',
        viewport: { width: 390, height: 844 }
      }
    },
    {
      name: 'mobile-dark-390',
      use: {
        ...devices['iPhone 15'],
        browserName: 'chromium',
        colorScheme: 'dark',
        viewport: { width: 390, height: 844 }
      }
    },
    {
      name: 'tablet-light-768',
      use: {
        ...devices['iPad Pro 11'],
        browserName: 'chromium',
        colorScheme: 'light',
        viewport: { width: 768, height: 1024 }
      }
    },
    {
      name: 'tablet-dark-768',
      use: {
        ...devices['iPad Pro 11'],
        browserName: 'chromium',
        colorScheme: 'dark',
        viewport: { width: 768, height: 1024 }
      }
    }
  ]
});
