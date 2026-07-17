import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const tokens = readFileSync(join(root, 'src/theme/tokens.ts'), 'utf8');

function readColor(token, mode) {
  const pattern = new RegExp(`${token}: \\{ light: '(#[0-9A-Fa-f]{6,8})', dark: '(#[0-9A-Fa-f]{6,8})' \\}`);
  const match = tokens.match(pattern);
  if (!match) throw new Error(`Missing color token: ${token}`);
  return mode === 'light' ? match[1] : match[2];
}

function luminance(hex) {
  const channels = hex
    .slice(1, 7)
    .match(/../g)
    .map((part) => {
      const value = Number.parseInt(part, 16) / 255;
      return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    });
  return (0.2126 * channels[0]) + (0.7152 * channels[1]) + (0.0722 * channels[2]);
}

function contrast(foreground, background) {
  const high = Math.max(luminance(foreground), luminance(background));
  const low = Math.min(luminance(foreground), luminance(background));
  return (high + 0.05) / (low + 0.05);
}

const checks = [
  ['textPrimary', 'surfaceBase'],
  ['textPrimary', 'surfaceRaised'],
  ['textPrimary', 'surfaceInset'],
  ['textSecondary', 'surfaceBase'],
  ['textSecondary', 'surfaceRaised'],
  ['textSecondary', 'surfaceInset'],
  ['textMuted', 'surfaceBase'],
  ['textMuted', 'surfaceRaised'],
  ['accentPrimary', 'surfaceBase'],
  ['accentPrimary', 'surfaceRaised'],
  ['accentPrimary', 'surfaceInset'],
  ['surfaceRaised', 'accentPrimary'],
  ['textPrimary', 'accentSignal', 'light'],
  ['surfaceBase', 'accentSignal', 'dark'],
  ['textPrimary', 'accentGoldSurface', 'light'],
  ['surfaceBoard', 'accentGoldSurface', 'dark'],
  ['textOnBoard', 'surfaceBoard'],
  ['textBoardMuted', 'surfaceBoard'],
  ['textBoardQuiet', 'surfaceBoard']
];

const failures = [];
for (const mode of ['light', 'dark']) {
  for (const [foregroundToken, backgroundToken, requiredMode] of checks) {
    if (requiredMode && requiredMode !== mode) continue;
    const ratio = contrast(readColor(foregroundToken, mode), readColor(backgroundToken, mode));
    if (ratio < 4.5) failures.push(`${mode} ${foregroundToken} on ${backgroundToken}: ${ratio.toFixed(2)}`);
  }
}

if (failures.length) {
  console.error(`Design contrast check failed:\n${failures.join('\n')}`);
  process.exit(1);
}

console.log('design contrast check passed');
