import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const allowed = new Set([
  'src/theme/tokens.ts',
  'app.json',
  'DESIGN.md'
]);

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    if (name === 'node_modules' || name === '.expo' || name === 'dist') return [];
    const path = join(dir, name);
    const stat = statSync(path);
    if (stat.isDirectory()) return files(path);
    return path;
  });
}

const hexPattern = /#[0-9A-Fa-f]{3,8}/g;
const offenders = [];

for (const file of files(root)) {
  const rel = relative(root, file).replaceAll('\\', '/');
  if (!/\.(ts|tsx|json|md)$/.test(rel)) continue;
  if (allowed.has(rel)) continue;
  const text = readFileSync(file, 'utf8');
  const matches = text.match(hexPattern);
  if (matches) offenders.push(`${rel}: ${matches.join(', ')}`);
}

if (offenders.length) {
  console.error(`Raw color tokens outside DESIGN.md/theme:\n${offenders.join('\n')}`);
  process.exit(1);
}

console.log('design token check passed');
