import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptPath = fileURLToPath(import.meta.url);
const mobileRoot = dirname(dirname(scriptPath));
const repoRoot = dirname(mobileRoot);
const releaseEnvPath = join(mobileRoot, 'release.env');
const reportPath = join(repoRoot, 'runs', 'release_finalize_report.md');
const timeoutMs = 8000;
const staleServerWarning = '서버 코드 stale — Mac에서 deploy/oracle/deploy.sh 재배포 필요';

const results = [];

function usage() {
  console.log(`Usage: node scripts/finalize-release.mjs --domain api.example.com --support-email you@x.com

Creates mobile/release.env, runs production smoke checks, and verifies scripts/check-store-release-env.mjs.

Options:
  --domain <domain>           Production API domain without protocol.
  --support-email <email>     Reviewer/support email address.
  --help                      Show this help.`);
}

function addResult(status, check, detail) {
  results.push({ status, check, detail });
}

function parseArgs(argv) {
  const parsed = { help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      parsed.help = true;
      continue;
    }
    if (arg === '--domain' || arg === '--support-email') {
      const value = argv[index + 1];
      if (!value || value.startsWith('--')) {
        parsed.error = `${arg} requires a value`;
        return parsed;
      }
      parsed[arg.slice(2)] = value;
      index += 1;
      continue;
    }
    parsed.error = `Unknown argument: ${arg}`;
    return parsed;
  }
  return parsed;
}

function isPlaceholderHost(hostname) {
  const normalized = hostname.toLowerCase();
  return normalized === 'localhost' ||
    normalized === '127.0.0.1' ||
    normalized === '::1' ||
    normalized.endsWith('.localhost') ||
    normalized.endsWith('.local') ||
    normalized.endsWith('.test') ||
    normalized.endsWith('.invalid') ||
    normalized.endsWith('.example') ||
    normalized === 'example.com' ||
    normalized.endsWith('.example.com') ||
    normalized.includes('placeholder') ||
    normalized.includes('your-') ||
    normalized.includes('replace-me');
}

function validateDomain(value) {
  const domain = value?.trim().toLowerCase() ?? '';
  if (!domain) return { ok: false, error: 'domain is required' };
  if (/^https?:\/\//i.test(domain)) return { ok: false, error: 'domain must not include a protocol' };
  if (/[/?#:@\s]/.test(domain)) return { ok: false, error: 'domain must be a hostname only' };
  if (domain.length > 253) return { ok: false, error: 'domain is too long' };
  if (isPlaceholderHost(domain)) return { ok: false, error: `domain must be a real production hostname, not ${domain}` };

  const labels = domain.split('.');
  if (labels.length < 2) return { ok: false, error: 'domain must include a public suffix' };
  for (const label of labels) {
    if (!/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(label)) {
      return { ok: false, error: `domain contains an invalid label: ${label || '(empty)'}` };
    }
  }
  if (!/^[a-z]{2,63}$/.test(labels.at(-1))) return { ok: false, error: 'domain suffix must be alphabetic' };
  return { ok: true, value: domain };
}

function validateEmail(value) {
  const email = value?.trim() ?? '';
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email)) {
    return { ok: false, error: 'support email must be a valid email address' };
  }
  const domain = email.split('@').at(-1);
  const domainResult = validateDomain(domain);
  if (!domainResult.ok) return { ok: false, error: `support email domain is invalid: ${domainResult.error}` };
  return { ok: true, value: email };
}

function releaseEnvText(domain, email) {
  return [
    `EXPO_PUBLIC_RACELENS_API_BASE_URL=https://${domain}`,
    `EXPO_PUBLIC_RACELENS_ANALYTICS_URL=https://${domain}/api/ux-events`,
    `RACELENS_PRIVACY_URL=https://${domain}/legal/privacy`,
    `RACELENS_TERMS_URL=https://${domain}/legal/terms`,
    `RACELENS_ACCOUNT_DELETION_URL=https://${domain}/legal/account-deletion`,
    `RACELENS_SUPPORT_URL=https://${domain}/legal/support`,
    `RACELENS_SUPPORT_EMAIL=${email}`,
    'RACELENS_BILLING_MODE=disabled',
    'EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED=1',
    'EXPO_PUBLIC_RACELENS_FIREBASE_AUTH_ENABLED=0',
    'RACELENS_FIREBASE_PROJECT_ID=racelens-tttksj',
    'GOOGLE_SERVICES_JSON=google-services.json',
    'RACELENS_FIREBASE_ANDROID_SERVICES_FILE=google-services.json',
    'EXPO_PUBLIC_RACELENS_REWARDED_ADS=1',
    'EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE=0',
    'EXPO_PUBLIC_RACELENS_ADMOB_ANDROID_APP_ID=',
    'EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID=',
    'RACELENS_ADMOB_REWARDED_AD_UNIT_ID=',
    ''
  ].join('\n');
}

function parseEnvFile(text) {
  const env = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const separator = line.indexOf('=');
    if (separator <= 0) continue;
    env[line.slice(0, separator)] = line.slice(separator + 1);
  }
  return env;
}

function todayKst() {
  const parts = new Intl.DateTimeFormat('en', {
    day: '2-digit',
    month: '2-digit',
    timeZone: 'Asia/Seoul',
    year: 'numeric'
  }).formatToParts(new Date());
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...options,
      headers: {
        'user-agent': 'RaceLensReleaseFinalize/1.0',
        ...(options.headers ?? {})
      },
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeout);
  }
}

function errorMessage(error) {
  return error?.name === 'AbortError' ? `request timed out after ${timeoutMs}ms` : error?.message ?? String(error);
}

async function checkHttp200(name, url) {
  try {
    const response = await fetchWithTimeout(url);
    if (response.status !== 200) {
      addResult('FAIL', name, `${url} returned HTTP ${response.status}`);
      return;
    }
    addResult('PASS', name, `${url} returned HTTP 200`);
  } catch (error) {
    addResult('FAIL', name, `${url} failed: ${errorMessage(error)}`);
  }
}

async function checkLegalPage(name, url) {
  try {
    const response = await fetchWithTimeout(url);
    const text = await response.text();
    if (response.status !== 200) {
      addResult('FAIL', name, `${url} returned HTTP ${response.status}`);
      return;
    }
    if (!/[\uac00-\ud7a3]/.test(text)) {
      addResult('FAIL', name, `${url} returned HTTP 200 but no Korean body text`);
      return;
    }
    addResult('PASS', name, `${url} returned HTTP 200 with Korean body text`);
  } catch (error) {
    addResult('FAIL', name, `${url} failed: ${errorMessage(error)}`);
  }
}

async function checkLiveDecision(baseUrl) {
  const url = new URL('/api/live-decision', baseUrl);
  url.searchParams.set('sport', 'kcycle');
  url.searchParams.set('date', todayKst());
  url.searchParams.set('meet', '광명');
  url.searchParams.set('race_no', '1');

  try {
    const response = await fetchWithTimeout(url.toString());
    const text = await response.text();
    if (/^\s*</.test(text)) {
      addResult('FAIL', 'live decision JSON', `${url} returned HTML`);
      return;
    }

    let json;
    try {
      json = JSON.parse(text);
    } catch {
      addResult('FAIL', 'live decision JSON', `${url} did not return JSON`);
      return;
    }

    if (!json || typeof json !== 'object' || (!Object.hasOwn(json, 'ok') && !Object.hasOwn(json, 'error'))) {
      addResult('FAIL', 'live decision JSON', `${url} JSON did not include ok/error`);
      return;
    }
    addResult('PASS', 'live decision JSON', `${url} returned ${response.status} JSON with ok/error`);
  } catch (error) {
    addResult('FAIL', 'live decision JSON', `${url} failed: ${errorMessage(error)}`);
  }
}

async function checkUxEvents(baseUrl) {
  const url = new URL('/api/ux-events', baseUrl);
  const event = {
    anonymousId: 'anon_release_finalize',
    app: 'racelens',
    name: 'live_odds_refresh',
    payload: {
      marketRiskLevel: 'smoke',
      marketUsed: false,
      pollDelayMs: 0,
      raceNo: 1,
      sport: 'keirin'
    },
    platform: 'web',
    sessionId: 'sess_release_finalize',
    timestamp: new Date().toISOString(),
    version: '0.1.0'
  };

  try {
    const response = await fetchWithTimeout(url.toString(), {
      body: JSON.stringify(event),
      headers: {
        'content-type': 'application/json',
        'x-racelens-analytics': 'ux-v1'
      },
      method: 'POST'
    });
    const text = await response.text();
    let json = null;
    if (text.trim()) {
      try {
        json = JSON.parse(text);
      } catch {
        addResult('FAIL', 'ux events live_odds_refresh', `${url} returned non-JSON HTTP ${response.status}`);
        return;
      }
    }

    if (json?.error === 'unsupported_event') {
      addResult('WARN', 'ux events live_odds_refresh', staleServerWarning);
      return;
    }
    if (response.ok) {
      addResult('PASS', 'ux events live_odds_refresh', `${url} accepted event with HTTP ${response.status}`);
      return;
    }
    addResult('FAIL', 'ux events live_odds_refresh', `${url} returned HTTP ${response.status}${json?.error ? ` (${json.error})` : ''}`);
  } catch (error) {
    addResult('FAIL', 'ux events live_odds_refresh', `${url} failed: ${errorMessage(error)}`);
  }
}

function runStoreReadiness(env) {
  const result = spawnSync(process.execPath, ['scripts/check-store-release-env.mjs'], {
    cwd: mobileRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      ...env
    },
    windowsHide: true
  });

  if (result.status === 0) {
    addResult('PASS', 'check-store-release-env', (result.stdout || '').trim() || 'store release environment gate passed');
    return;
  }
  const output = `${result.stdout ?? ''}${result.stderr ?? ''}`.trim().replace(/\s+/g, ' ');
  addResult('FAIL', 'check-store-release-env', output || `exited with ${result.status ?? 'unknown status'}`);
}

function renderTable(rows) {
  const headers = ['Status', 'Check', 'Detail'];
  const widths = headers.map((header, index) => Math.max(
    header.length,
    ...rows.map((row) => [row.status, row.check, row.detail][index].length)
  ));
  const line = (values) => `| ${values.map((value, index) => value.padEnd(widths[index])).join(' | ')} |`;
  return [
    line(headers),
    `| ${widths.map((width) => '-'.repeat(width)).join(' | ')} |`,
    ...rows.map((row) => line([row.status, row.check, row.detail]))
  ].join('\n');
}

function writeReport(overall) {
  mkdirSync(dirname(reportPath), { recursive: true });
  const table = renderTable(results);
  writeFileSync(reportPath, [
    '# RaceLens Release Finalize Report',
    '',
    `- Result: ${overall}`,
    `- Generated: ${new Date().toISOString()}`,
    `- release.env: ${existsSync(releaseEnvPath) ? releaseEnvPath : 'not written'}`,
    '',
    table,
    ''
  ].join('\n'));
  return table;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return 0;
  }

  if (args.error) addResult('FAIL', 'arguments', args.error);
  const domain = validateDomain(args.domain);
  const email = validateEmail(args['support-email']);
  if (!domain.ok) addResult('FAIL', 'domain validation', domain.error);
  if (!email.ok) addResult('FAIL', 'support email validation', email.error);

  if (results.some((result) => result.status === 'FAIL')) {
    addResult('WARN', 'release.env', 'not written because argument validation failed');
  } else {
    writeFileSync(releaseEnvPath, releaseEnvText(domain.value, email.value));
    addResult('PASS', 'release.env', `wrote ${releaseEnvPath}`);

    const env = parseEnvFile(readFileSync(releaseEnvPath, 'utf8'));
    Object.assign(process.env, env);
    const baseUrl = env.EXPO_PUBLIC_RACELENS_API_BASE_URL;

    await checkHttp200('healthz', new URL('/healthz', baseUrl).toString());
    await checkLegalPage('privacy page', env.RACELENS_PRIVACY_URL);
    await checkLegalPage('terms page', env.RACELENS_TERMS_URL);
    await checkLegalPage('account deletion page', env.RACELENS_ACCOUNT_DELETION_URL);
    await checkLiveDecision(baseUrl);
    await checkUxEvents(baseUrl);
    runStoreReadiness(env);
  }

  const overall = results.some((result) => result.status === 'FAIL')
    ? 'FAIL'
    : results.some((result) => result.status === 'WARN')
      ? 'WARN'
      : 'PASS';
  const table = writeReport(overall);
  console.log(table);
  console.log(`FINALIZE_RELEASE_RESULT=${overall} report=${reportPath}`);
  return overall === 'FAIL' ? 1 : 0;
}

process.exitCode = await main();
