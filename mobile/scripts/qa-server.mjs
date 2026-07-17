import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { exitIfFailed, spawnQa } from './qa-utils.mjs';

const repoRoot = join(import.meta.dirname, '..', '..');
const venvPython = process.platform === 'win32'
  ? join(repoRoot, '.venv', 'Scripts', 'python.exe')
  : join(repoRoot, '.venv', 'bin', 'python');
const python = existsSync(venvPython) ? venvPython : 'python';

const result = spawnQa(python, ['-m', 'pytest', 'tests/test_app_data_layer.py', '-q'], {
  cwd: repoRoot,
  env: process.env
});
exitIfFailed(result);
