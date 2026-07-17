import { spawnSync } from 'node:child_process';

export function commandName(command) {
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

export function spawnQa(command, args, options = {}) {
  const resolvedCommand = commandName(command);
  const needsShell = process.platform === 'win32' && resolvedCommand.toLowerCase().endsWith('.cmd');
  const result = spawnSync(resolvedCommand, args, {
    ...options,
    shell: options.shell ?? needsShell,
    stdio: options.stdio ?? 'inherit'
  });
  if (result.error) {
    console.error(`Failed to run ${command}: ${result.error.message}`);
  }
  return result;
}

export function exitIfFailed(result) {
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
