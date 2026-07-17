import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const upstreamBase = 'https://168-107-2-218.sslip.io';

const env = {
  ...process.env,
  EXPO_PUBLIC_RACELENS_ACCOUNT_DELETION_URL: `${upstreamBase}/legal/account-deletion`,
  EXPO_PUBLIC_RACELENS_ANALYTICS_URL: '',
  EXPO_PUBLIC_RACELENS_API_BASE_URL: '',
  EXPO_PUBLIC_RACELENS_BILLING_MODE: 'disabled',
  EXPO_PUBLIC_RACELENS_FIREBASE_AUTH_ENABLED: '0',
  EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED: '0',
  EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '0',
  EXPO_PUBLIC_RACELENS_PRIVACY_URL: `${upstreamBase}/legal/privacy`,
  EXPO_PUBLIC_RACELENS_REWARDED_ADS: '1',
  EXPO_PUBLIC_RACELENS_REWARDED_ADS_PREVIEW: '1',
  EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE: '0',
  EXPO_PUBLIC_RACELENS_SUPPORT_EMAIL: 'tttksj@gmail.com',
  EXPO_PUBLIC_RACELENS_SUPPORT_URL: `${upstreamBase}/legal/support`,
  EXPO_PUBLIC_RACELENS_TERMS_URL: `${upstreamBase}/legal/terms`,
  RACELENS_ACCOUNT_DELETION_URL: `${upstreamBase}/legal/account-deletion`,
  RACELENS_BILLING_MODE: 'disabled',
  RACELENS_PRIVACY_URL: `${upstreamBase}/legal/privacy`,
  RACELENS_SUPPORT_EMAIL: 'tttksj@gmail.com',
  RACELENS_SUPPORT_URL: `${upstreamBase}/legal/support`,
  RACELENS_TERMS_URL: `${upstreamBase}/legal/terms`
};

const result = spawnSync(
  process.execPath,
  [fileURLToPath(new URL('../node_modules/expo/bin/cli', import.meta.url)), 'export', '--platform', 'web', '--clear', '--output-dir', 'dist-preview'],
  {
    env,
    shell: false,
    stdio: 'inherit'
  }
);

if (result.error) {
  console.error(`Failed to run expo export: ${result.error.message}`);
}

process.exit(result.status ?? 1);
