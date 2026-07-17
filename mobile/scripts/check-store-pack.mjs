import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const storePackDir = join(process.cwd(), '..', 'runs', 'store_pack');

const expectedPngs = new Map([
  ['feature_graphic.png', { width: 1024, height: 500 }],
  ['icon_512.png', { width: 512, height: 512 }],
  ['screenshot_01_home_keirin_dark.png', { width: 1080, height: 1920 }],
  ['screenshot_02_home_horse_light.png', { width: 1080, height: 1920 }],
  ['screenshot_03_analysis_podium.png', { width: 1080, height: 1920 }],
  ['screenshot_04_evidence_data.png', { width: 1080, height: 1920 }],
  ['screenshot_05_market_odds.png', { width: 1080, height: 1920 }],
  ['screenshot_06_pro.png', { width: 1080, height: 1920 }]
]);

const requiredDataSafetyTerms = [
  'Google Firebase Analytics',
  'Crashlytics',
  'Google as the Firebase service provider',
  'Google Mobile Ads SDK',
  'rewarded-ad interactions'
];

const forbiddenListingTerms = ['추천', '보장', '필승', '수익', '베팅CTA'];

function pngSize(path) {
  const bytes = readFileSync(path);
  const signature = bytes.subarray(0, 8).toString('hex');
  if (signature !== '89504e470d0a1a0a') {
    throw new Error(`${path} is not a PNG`);
  }
  return {
    width: bytes.readUInt32BE(16),
    height: bytes.readUInt32BE(20)
  };
}

function verifyPngs() {
  const actualFiles = new Set(readdirSync(storePackDir));
  for (const [file, expected] of expectedPngs.entries()) {
    if (!actualFiles.has(file)) throw new Error(`missing ${file}`);
    const actual = pngSize(join(storePackDir, file));
    if (actual.width !== expected.width || actual.height !== expected.height) {
      throw new Error(`${file} is ${actual.width}x${actual.height}; expected ${expected.width}x${expected.height}`);
    }
  }
}

function verifyListing() {
  const listing = readFileSync(join(storePackDir, 'listing.md'), 'utf8');
  const forbidden = forbiddenListingTerms.filter((term) => listing.includes(term));
  if (forbidden.length > 0) {
    throw new Error(`listing forbidden terms present: ${forbidden.join(', ')}`);
  }
  if (!listing.includes('만 19세 이상')) {
    throw new Error('listing missing 만 19세 이상 frame');
  }
  if (!listing.includes('정보 분석')) {
    throw new Error('listing missing 정보 분석 frame');
  }
}

function verifyDataSafetyDisclosure() {
  const listing = readFileSync(join(storePackDir, 'listing.md'), 'utf8');
  const missingDataSafetyTerms = requiredDataSafetyTerms.filter((term) => !listing.includes(term));
  if (missingDataSafetyTerms.length > 0) {
    throw new Error(`listing missing Firebase/Data safety disclosure: ${missingDataSafetyTerms.join(', ')}`);
  }
}

verifyPngs();
verifyListing();
verifyDataSafetyDisclosure();

console.log('store pack verified: feature 1024x500, icon 512x512, screenshots 6x1080x1920, listing policy terms clean');
