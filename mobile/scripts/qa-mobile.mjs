import { mkdtempSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { exitIfFailed, spawnQa } from './qa-utils.mjs';

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  await new Promise((resolveClose) => server.close(resolveClose));
  if (!address || typeof address === 'string') throw new Error('Unable to allocate QA port');
  return address.port;
}

async function qaTarget() {
  try {
    return {
      appPort: String(await freePort()),
      noServer: false
    };
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'EPERM') {
      return {
        appPort: process.env.RACELENS_QA_PORT ?? '8064',
        noServer: true
      };
    }
    throw error;
  }
}

function run(command, args, env = {}) {
  const result = spawnQa(command, args, {
    env: {
      ...process.env,
      EXPO_PUBLIC_RACELENS_API_BASE_URL: '',
      EXPO_PUBLIC_RACELENS_ANALYTICS_URL: '',
      ...env
    }
  });
  exitIfFailed(result);
}

const { appPort, noServer } = await qaTarget();
const distRoot = mkdtempSync(join(tmpdir(), 'racelens-qa-mobile-'));
run('npx', ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot]);
run('npx', ['playwright', 'test'], {
  RACELENS_QA_DIST: distRoot,
  RACELENS_QA_NO_SERVER: noServer ? '1' : '0',
  RACELENS_QA_PORT: appPort
});
