import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { extname, join } from 'node:path';

const root = process.cwd();
const scanRoots = ['src', 'app.config.js'];
const allowedScriptLogs = new Set([
  'scripts/check-design-contrast.mjs',
  'scripts/check-design-resources.mjs',
  'scripts/check-design-tokens.mjs',
  'scripts/check-security-surface.mjs',
  'scripts/qa-api-live.mjs'
]);
const allowedPersistenceFiles = new Set(['src/services/deviceIdentity.web.ts']);

const riskyPatterns = [
  { pattern: /dangerouslySetInnerHTML/, reason: 'unsafe HTML injection surface' },
  { pattern: /\beval\s*\(/, reason: 'dynamic code execution' },
  { pattern: /\bnew\s+Function\s*\(/, reason: 'dynamic code execution' },
  { pattern: /\.innerHTML\b/, reason: 'raw HTML mutation' },
  { pattern: /\bAsyncStorage\b|\blocalStorage\b|\bsessionStorage\b/, reason: 'local persistence of sensitive data' },
  { pattern: /\bAuthorization\b/, reason: 'authorization header in client bundle' },
  { pattern: /\b(apiKey|secret|password)\b/i, reason: 'possible secret-bearing identifier' }
];

function collectFiles(entry) {
  const absolute = join(root, entry);
  const stats = statSync(absolute);
  if (stats.isFile()) return [entry];
  return readdirSync(absolute).flatMap((name) => {
    const relative = join(entry, name);
    const child = join(root, relative);
    if (statSync(child).isDirectory()) return collectFiles(relative);
    return ['.ts', '.tsx', '.js', '.mjs'].includes(extname(child)) ? [relative] : [];
  });
}

const files = scanRoots.flatMap(collectFiles);
const offenders = [];

for (const file of files) {
  const body = readFileSync(join(root, file), 'utf8');
  const normalizedFile = file.replaceAll('\\', '/');
  for (const { pattern, reason } of riskyPatterns) {
    if (reason === 'local persistence of sensitive data' && allowedPersistenceFiles.has(normalizedFile)) continue;
    if (pattern.test(body)) offenders.push(`${file}: ${reason}`);
  }
  if (body.includes('console.log') && !allowedScriptLogs.has(file)) {
    offenders.push(`${file}: console.log in app code`);
  }
}

const raceApi = readFileSync(join(root, 'src/services/raceApi.ts'), 'utf8');
const requiredApiGuards = [
  'normalizeApiBaseUrl',
  'AbortController',
  'safeText',
  'sanitizeParticipants',
  'sanitizeMarketOdds'
];
for (const guard of requiredApiGuards) {
  if (!raceApi.includes(guard)) offenders.push(`src/services/raceApi.ts: missing ${guard} guard`);
}
if (raceApi.includes('createClientDeviceId')) {
  offenders.push('src/services/raceApi.ts: install identity must not be regenerated at module load');
}
if (!raceApi.includes('getClientDeviceId')) {
  offenders.push('src/services/raceApi.ts: missing persisted install identity lookup');
}

const packageJson = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
if (!packageJson.dependencies?.['expo-secure-store']) {
  offenders.push('package.json: expo-secure-store is required for native install identity persistence');
}
for (const identityFile of ['src/services/deviceIdentity.native.ts', 'src/services/deviceIdentity.web.ts']) {
  if (!existsSync(join(root, identityFile))) offenders.push(`${identityFile}: missing persisted install identity implementation`);
}

if (offenders.length) {
  console.error(`security surface check failed:\n${offenders.join('\n')}`);
  process.exit(1);
}

console.log('security surface check passed');
