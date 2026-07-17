import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnQa } from './qa-utils.mjs';

const distRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-release-visual-'));

const exportResult = spawnQa('npx', ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
  env: {
    ...process.env,
    EXPO_PUBLIC_RACELENS_API_BASE_URL: '',
    EXPO_PUBLIC_RACELENS_ANALYTICS_URL: '',
    EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '1',
    EXPO_PUBLIC_RACELENS_REWARDED_ADS: '0'
  }
});

if (exportResult.status !== 0) process.exit(exportResult.status ?? 1);

const qaResult = spawnQa('node', ['scripts/qa-release-readiness.mjs'], {
  env: {
    ...process.env,
    RACELENS_QA_DIST: distRoot
  }
});

if (qaResult.status !== 0) process.exit(qaResult.status ?? 1);

const adsPolicyResult = spawnQa('node', ['scripts/qa-rewarded-ads-policy.mjs']);

process.exit(adsPolicyResult.status ?? 1);
