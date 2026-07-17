import { existsSync, lstatSync, renameSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const commandArgs = process.argv.slice(2);
if (commandArgs.length === 0) {
  console.error('Usage: node scripts/with-workspace-junction-detached.mjs <command> [...args]');
  process.exit(2);
}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const mobileDir = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(mobileDir, '..');
const codegraphPath = path.join(repoRoot, '.codegraph');
const holdPath = path.join(path.dirname(repoRoot), `.${path.basename(repoRoot)}-codegraph-eas-hold`);

let detached = false;
try {
  if (process.platform === 'win32' && existsSync(codegraphPath)) {
    if (existsSync(holdPath)) {
      throw new Error(`CodeGraph hold path already exists: ${holdPath}`);
    }
    if (!lstatSync(codegraphPath).isSymbolicLink()) {
      throw new Error(`Expected a directory junction at ${codegraphPath}`);
    }
    renameSync(codegraphPath, holdPath);
    detached = true;
  }

  const [rawCommand, ...args] = commandArgs;
  const command = process.platform === 'win32' && /^(npm|npx|pnpm|yarn)$/.test(rawCommand)
    ? `${rawCommand}.cmd`
    : rawCommand;
  const result = spawnSync(command, args, {
    cwd: mobileDir,
    env: process.env,
    stdio: 'inherit'
  });
  if (result.error) {
    throw result.error;
  }
  process.exitCode = result.status ?? 1;
} finally {
  if (detached) {
    renameSync(holdPath, codegraphPath);
  }
}
