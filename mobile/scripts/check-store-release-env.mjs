import { existsSync, readFileSync } from 'node:fs';

if (existsSync('release.env')) {
  process.loadEnvFile('release.env');
}

const requiredHttpsVars = [
  'EXPO_PUBLIC_RACELENS_API_BASE_URL',
  'EXPO_PUBLIC_RACELENS_ANALYTICS_URL',
  'RACELENS_PRIVACY_URL',
  'RACELENS_TERMS_URL',
  'RACELENS_ACCOUNT_DELETION_URL',
  'RACELENS_SUPPORT_URL'
];

const failures = [];
const parsedUrls = new Map();
const appJson = JSON.parse(readFileSync('app.json', 'utf8'));
const expo = appJson.expo ?? {};

function isPlaceholderHost(hostname) {
  const normalized = hostname.toLowerCase();
  return normalized === 'localhost' ||
    normalized === '127.0.0.1' ||
    normalized.endsWith('.local') ||
    normalized.endsWith('.test') ||
    normalized.endsWith('.invalid') ||
    normalized.endsWith('.example') ||
    normalized.endsWith('.trycloudflare.com') ||
    normalized.endsWith('.loca.lt') ||
    normalized.endsWith('.tunnelmole.net') ||
    normalized.endsWith('.ngrok-free.app') ||
    normalized.endsWith('.ngrok.io') ||
    normalized.includes('example.com') ||
    normalized.includes('your-api') ||
    normalized.includes('replace-me');
}

function httpsUrl(name) {
  const value = process.env[name]?.trim() ?? '';
  if (!value) {
    failures.push(`${name} is required`);
    return null;
  }
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== 'https:') failures.push(`${name} must use https`);
    if (isPlaceholderHost(parsed.hostname)) failures.push(`${name} must point to a real production domain, not ${parsed.hostname}`);
    parsedUrls.set(name, parsed);
    return parsed;
  } catch {
    failures.push(`${name} must be a valid URL`);
    return null;
  }
}

for (const name of requiredHttpsVars) httpsUrl(name);

const supportEmail = process.env.RACELENS_SUPPORT_EMAIL?.trim() ?? '';
if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(supportEmail)) {
  failures.push('RACELENS_SUPPORT_EMAIL must be a valid reviewer/support email');
} else {
  const supportDomain = supportEmail.split('@')[1].toLowerCase();
  if (isPlaceholderHost(supportDomain) || supportDomain === 'example.com') {
    failures.push('RACELENS_SUPPORT_EMAIL must not use a placeholder domain');
  }
}

const billingMode = process.env.RACELENS_BILLING_MODE?.trim() ?? '';
if (billingMode !== 'disabled') {
  failures.push('RACELENS_BILLING_MODE must stay disabled until Google Play server verification is implemented');
}
const rewardedAds = process.env.EXPO_PUBLIC_RACELENS_REWARDED_ADS?.trim().toLowerCase() ?? '';
if (rewardedAds && !['0', '1', 'false', 'true', 'no', 'yes', 'off', 'on'].includes(rewardedAds)) {
  failures.push('EXPO_PUBLIC_RACELENS_REWARDED_ADS must be a boolean flag when set');
}
if (['1', 'true', 'yes', 'on'].includes(rewardedAds)) {
  const androidAppId = process.env.EXPO_PUBLIC_RACELENS_ADMOB_ANDROID_APP_ID?.trim() ?? '';
  const rewardedAdUnitId = process.env.EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID?.trim() ?? '';
  const serverRewardedAdUnitId = process.env.RACELENS_ADMOB_REWARDED_AD_UNIT_ID?.trim() ?? '';
  const testMode = process.env.EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE?.trim().toLowerCase() ?? '';
  if (!/^ca-app-pub-\d+~\d+$/.test(androidAppId)) {
    failures.push('EXPO_PUBLIC_RACELENS_ADMOB_ANDROID_APP_ID must be a valid AdMob Android app ID');
  }
  if (!/^ca-app-pub-\d+\/\d+$/.test(rewardedAdUnitId)) {
    failures.push('EXPO_PUBLIC_RACELENS_ADMOB_REWARDED_AD_UNIT_ID must be a valid AdMob rewarded ad unit ID');
  }
  if (androidAppId === 'ca-app-pub-3940256099942544~3347511713' || rewardedAdUnitId === 'ca-app-pub-3940256099942544/5224354917') {
    failures.push('Production rewarded ads must not use Google test IDs');
  }
  if (['1', 'true', 'yes', 'on'].includes(testMode)) {
    failures.push('EXPO_PUBLIC_RACELENS_ADMOB_TEST_MODE must be disabled for production');
  }
  if (!serverRewardedAdUnitId || serverRewardedAdUnitId !== rewardedAdUnitId) {
    failures.push('RACELENS_ADMOB_REWARDED_AD_UNIT_ID must match the mobile rewarded ad unit ID for SSV');
  }
} else {
  failures.push('EXPO_PUBLIC_RACELENS_REWARDED_ADS must be enabled for the 3-free-then-rewarded production flow');
}
const firebaseEnabled = process.env.EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED?.trim().toLowerCase() ?? '';
if (firebaseEnabled && !['0', '1', 'false', 'true', 'no', 'yes', 'off', 'on'].includes(firebaseEnabled)) {
  failures.push('EXPO_PUBLIC_RACELENS_FIREBASE_ENABLED must be a boolean flag when set');
}
if (!['1', 'true', 'yes', 'on'].includes(firebaseEnabled)) {
  failures.push('Firebase Analytics and Crashlytics must be enabled for the production release');
}
const firebaseAuthEnabled = process.env.EXPO_PUBLIC_RACELENS_FIREBASE_AUTH_ENABLED?.trim().toLowerCase() ?? '';
if (firebaseAuthEnabled && !['0', '1', 'false', 'true', 'no', 'yes', 'off', 'on'].includes(firebaseAuthEnabled)) {
  failures.push('EXPO_PUBLIC_RACELENS_FIREBASE_AUTH_ENABLED must be a boolean flag when set');
}
if (['1', 'true', 'yes', 'on'].includes(firebaseAuthEnabled)) {
  failures.push('Firebase Auth must stay disabled until account-based Pro restore is implemented');
}
if (!['0', 'false', 'no', 'off'].includes(firebaseAuthEnabled)) {
  failures.push('EXPO_PUBLIC_RACELENS_FIREBASE_AUTH_ENABLED must be explicitly disabled');
}
if (['1', 'true', 'yes', 'on'].includes(firebaseEnabled)) {
  const androidServicesFile =
    process.env.GOOGLE_SERVICES_JSON?.trim() ||
    process.env.RACELENS_FIREBASE_ANDROID_SERVICES_FILE?.trim() ||
    'google-services.json';
  const expectedProjectId = process.env.RACELENS_FIREBASE_PROJECT_ID?.trim() ?? '';
  if (!expectedProjectId) failures.push('RACELENS_FIREBASE_PROJECT_ID is required when Firebase is enabled');
  try {
    const services = JSON.parse(readFileSync(androidServicesFile, 'utf8'));
    const clients = Array.isArray(services?.client) ? services.client : [];
    const client = clients.find((candidate) =>
      candidate?.client_info?.android_client_info?.package_name === expo.android?.package
    );
    if (!services.project_info?.project_id) {
      failures.push(`${androidServicesFile} must include project_info.project_id`);
    }
    if (expectedProjectId && services.project_info?.project_id !== expectedProjectId) {
      failures.push(`${androidServicesFile} project_id must be ${expectedProjectId}`);
    }
    if (!client) {
      failures.push(`${androidServicesFile} must include Android client for ${expo.android?.package}`);
    } else if (!client.client_info?.mobilesdk_app_id) {
      failures.push(`${androidServicesFile} must include mobilesdk_app_id for ${expo.android?.package}`);
    }
  } catch (error) {
    if (error instanceof SyntaxError) {
      failures.push(`${androidServicesFile} must be valid JSON`);
    } else {
      failures.push(`Firebase enabled requires Android service file at ${androidServicesFile}`);
    }
  }
}

if (!expo.ios?.bundleIdentifier || !expo.android?.package) {
  failures.push('iOS bundleIdentifier and Android package must be configured');
}
if ((expo.android?.permissions ?? []).length !== 0) {
  failures.push('Android permissions must stay empty unless a store-reviewed feature requires them');
}
if (expo.extra?.apiBaseUrl) {
  failures.push('app.json must not hard-code the production API URL; inject it from release env');
}
if (expo.extra?.analyticsUrl) {
  failures.push('app.json must not hard-code the production analytics URL; inject it from release env');
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function requireHttpOk(name, url, requiredText = null) {
  try {
    const response = await fetchWithTimeout(url, { headers: { 'User-Agent': 'RaceLensStoreReadiness/1.0' } });
    if (!response.ok) {
      failures.push(`${name} must be reachable for reviewers; got HTTP ${response.status}`);
      return;
    }
    const text = await response.text();
    if (text.includes('�')) failures.push(`${name} response contains broken replacement characters`);
    if (requiredText && !text.includes(requiredText)) failures.push(`${name} response must contain ${requiredText}`);
  } catch (error) {
    failures.push(`${name} must be reachable for reviewers: ${error.name === 'AbortError' ? 'request timed out' : error.message}`);
  }
}

const apiBaseUrl = parsedUrls.get('EXPO_PUBLIC_RACELENS_API_BASE_URL');
if (apiBaseUrl) {
  await requireHttpOk('EXPO_PUBLIC_RACELENS_API_BASE_URL /health', new URL('/health', apiBaseUrl).toString(), '"ok"');
}
for (const [name, requiredText] of [
  ['RACELENS_PRIVACY_URL', '개인정보'],
  ['RACELENS_TERMS_URL', '이용약관'],
  ['RACELENS_ACCOUNT_DELETION_URL', '삭제'],
]) {
  const parsed = parsedUrls.get(name);
  if (parsed) await requireHttpOk(name, parsed.toString(), requiredText);
}
if (failures.length) {
  console.error(`Store readiness blocked:\n- ${failures.join('\n- ')}`);
  process.exit(1);
}

console.log('store release environment gate passed');
