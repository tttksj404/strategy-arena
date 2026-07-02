import { readdirSync, readFileSync, statSync } from 'node:fs';
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
  for (const { pattern, reason } of riskyPatterns) {
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
  'sanitizeParticipants'
];
for (const guard of requiredApiGuards) {
  if (!raceApi.includes(guard)) offenders.push(`src/services/raceApi.ts: missing ${guard} guard`);
}

if (offenders.length) {
  console.error(`security surface check failed:\n${offenders.join('\n')}`);
  process.exit(1);
}

console.log('security surface check passed');
