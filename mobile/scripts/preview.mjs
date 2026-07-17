import { spawnSync } from 'node:child_process';

function commandName(command) {
  if (process.platform !== 'win32') return command;
  const normalized = command.toLowerCase();
  if (
    command.includes('\\') ||
    command.includes('/') ||
    normalized === 'node' ||
    normalized.endsWith('.cmd') ||
    normalized.endsWith('.exe')
  ) {
    return command;
  }
  return `${command}.cmd`;
}

function runNpmScript(name) {
  return spawnSync(commandName('npm'), ['run', name], {
    shell: process.platform === 'win32',
    stdio: 'inherit'
  });
}

const build = runNpmScript('preview:build');
if (build.error) {
  console.error(`Failed to run preview:build: ${build.error.message}`);
}
if (build.status !== 0) {
  process.exit(build.status ?? 1);
}

const serve = runNpmScript('preview:serve');
if (serve.error) {
  console.error(`Failed to run preview:serve: ${serve.error.message}`);
}
process.exit(serve.status ?? 1);
