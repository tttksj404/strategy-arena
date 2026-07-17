import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const requiredHosts = [
  'curated.design',
  'godly.website',
  'awwwards.com',
  'landing.love',
  'saaspo.com',
  'onepagelove.com',
  'navbar.gallery',
  'cta.gallery',
  'collectui.com',
  'mobbin.com',
  '60fps.design',
  '21st.dev',
  'component.gallery',
  'rebrand.gallery',
  'logofolio.com',
  'svgl.app',
  'coolors.co',
  'fontpair.co',
  'hugeicons.com',
  'dezignheroes.framer.website'
];

const docs = [
  'DESIGN.md',
  'DESIGN_RESOURCES.md',
  'README.md'
];

const combined = docs
  .map((doc) => readFileSync(join(root, doc), 'utf8'))
  .join('\n');

const missing = requiredHosts.filter((host) => !combined.includes(host));
if (missing.length) {
  console.error(`Missing design resource references:\n${missing.join('\n')}`);
  process.exit(1);
}

const registry = readFileSync(join(root, 'DESIGN_RESOURCES.md'), 'utf8');
const registryMissing = requiredHosts.filter((host) => !registry.includes(host));
if (registryMissing.length) {
  console.error(`DESIGN_RESOURCES.md is missing:\n${registryMissing.join('\n')}`);
  process.exit(1);
}

console.log('design resource check passed');
