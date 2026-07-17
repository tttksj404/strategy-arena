import { spawnSync } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createServer as createNetServer } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const assetsDir = join(root, 'assets');
const runsDir = join(dirname(root), 'runs');
const storePackDir = join(runsDir, 'store_pack');

const palette = {
  background: '#0B0D0C',
  lensInner: '#10150F',
  lime: '#C9F24A',
  gold: '#F4B83F',
  white: '#F7FAF2'
};

const targets = {
  icon: { file: 'icon.png', width: 1024, height: 1024 },
  adaptive: { file: 'adaptive-icon.png', width: 1024, height: 1024 },
  splash: { file: 'splash.png', width: 1284, height: 2778 },
  favicon: { file: 'favicon.png', width: 192, height: 192 },
  preview48: { file: 'icon_preview_48.png', width: 48, height: 48 },
  featureGraphic: { file: 'feature_graphic.png', width: 1024, height: 500 },
  storeIcon: { file: 'icon_512.png', width: 512, height: 512 }
};

function polar(cx, cy, radius, degrees) {
  const radians = (degrees - 90) * Math.PI / 180;
  return {
    x: cx + radius * Math.cos(radians),
    y: cy + radius * Math.sin(radians)
  };
}

function lensMarkSvg(size, options = {}) {
  const scale = size / 1024;
  const safeScale = options.safeScale ?? 1;
  const center = 512 * scale;
  const scaled = (value) => value * scale * safeScale;
  const ringRadius = scaled(300);
  const ringWidth = scaled(88);
  const innerRadius = scaled(256);
  const arcRadius = scaled(424);
  const arcWidth = scaled(56);
  const apertureRadius = scaled(96);
  const pupilRadius = scaled(40);
  const arcStart = polar(center, center, arcRadius, -50);
  const arcEnd = polar(center, center, arcRadius, 50);

  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${center}" cy="${center}" r="${ringRadius}" fill="none" stroke="${palette.lime}" stroke-width="${ringWidth}"/>
      <circle cx="${center}" cy="${center}" r="${innerRadius}" fill="${palette.lensInner}"/>
      <circle cx="${center}" cy="${center}" r="${apertureRadius}" fill="${palette.white}"/>
      <circle cx="${center}" cy="${center}" r="${pupilRadius}" fill="${palette.background}"/>
      <path d="M ${arcStart.x} ${arcStart.y} A ${arcRadius} ${arcRadius} 0 0 1 ${arcEnd.x} ${arcEnd.y}" fill="none" stroke="${palette.gold}" stroke-linecap="round" stroke-width="${arcWidth}"/>
    </svg>`;
}

function htmlDocument(body, options = {}) {
  const background = options.transparent ? 'transparent' : palette.background;
  return `<!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            overflow: hidden;
            background: ${background};
          }
          body {
            display: grid;
            place-items: center;
            font-family: "Segoe UI", Arial, sans-serif;
          }
        </style>
      </head>
      <body>${body}</body>
    </html>`;
}

function iconHtml() {
  return htmlDocument(`
    <main style="width:1024px;height:1024px;background:${palette.background};display:grid;place-items:center;">
      ${lensMarkSvg(1024)}
    </main>`);
}

function adaptiveHtml() {
  return htmlDocument(`
    <main style="width:1024px;height:1024px;display:grid;place-items:center;">
      ${lensMarkSvg(1024, { safeScale: 0.66 })}
    </main>`, { transparent: true });
}

function faviconHtml() {
  return htmlDocument(`
    <main style="width:192px;height:192px;display:grid;place-items:center;">
      ${lensMarkSvg(192)}
    </main>`, { transparent: true });
}

function splashHtml() {
  return htmlDocument(`
    <main style="width:1284px;height:2778px;position:relative;background:${palette.background};">
      <div style="position:absolute;left:492px;top:961.2px;width:300px;height:300px;display:grid;place-items:center;">
        ${lensMarkSvg(300)}
      </div>
      <div style="position:absolute;left:0;right:0;top:1333.2px;text-align:center;color:${palette.white};font-size:96px;font-weight:700;letter-spacing:1.92px;line-height:1;">RaceLens</div>
      <div style="position:absolute;left:0;right:0;top:1469.2px;text-align:center;color:#98A590;font-size:34px;font-weight:700;letter-spacing:11.9px;line-height:1;">RACE DATA ANALYSIS</div>
    </main>`);
}

function featureGraphicHtml() {
  return htmlDocument(`
    <main style="width:1024px;height:500px;position:relative;background:${palette.background};overflow:hidden;font-family:'Malgun Gothic','Segoe UI',Arial,sans-serif;">
      <div style="position:absolute;left:64px;top:92px;width:156px;height:156px;display:grid;place-items:center;">
        ${lensMarkSvg(156)}
      </div>
      <div style="position:absolute;left:252px;top:108px;color:${palette.white};font-size:82px;font-weight:800;line-height:1;letter-spacing:0;">RaceLens</div>
      <div style="position:absolute;left:256px;top:210px;color:${palette.gold};font-size:38px;font-weight:800;line-height:1.24;letter-spacing:0;">경륜·경마 데이터 분석</div>
      <div style="position:absolute;left:256px;top:286px;color:#A9B4AA;font-size:24px;font-weight:700;line-height:1.45;letter-spacing:0;">출전 정보 · 모델 신호 · 검증 상태를 한 화면에서 확인</div>
      <div style="position:absolute;left:64px;right:64px;bottom:54px;height:2px;background:linear-gradient(90deg, ${palette.lime}, ${palette.gold}, transparent);"></div>
      <div style="position:absolute;right:72px;top:74px;width:230px;height:230px;border-radius:999px;border:1px solid rgba(201,242,74,.22);"></div>
      <div style="position:absolute;right:118px;top:120px;width:138px;height:138px;border-radius:999px;border:1px solid rgba(244,184,63,.28);"></div>
    </main>`);
}

async function render(page, target, html, options = {}) {
  await page.setViewportSize({ width: target.width, height: target.height });
  await page.setContent(html, { waitUntil: 'load' });
  const path = options.storePack ? join(storePackDir, target.file) : join(assetsDir, target.file);
  await page.screenshot({
    animations: 'disabled',
    fullPage: false,
    omitBackground: options.transparent === true,
    path,
    type: 'png'
  });
  return path;
}

async function readPngDetails(page, path) {
  const dataUrl = `data:image/png;base64,${readFileSync(path).toString('base64')}`;
  return page.evaluate(async (src) => {
    const image = new Image();
    image.src = src;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas context unavailable');
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let transparent = 0;
    let visible = 0;
    for (let index = 3; index < pixels.length; index += 4) {
      const alpha = pixels[index];
      if (alpha === 0) transparent += 1;
      if (alpha !== 0) visible += 1;
    }
    return {
      width: image.naturalWidth,
      height: image.naturalHeight,
      transparent,
      visible
    };
  }, dataUrl);
}

async function assertPng(page, name, target, checks = {}) {
  const path = checks.storePack ? join(storePackDir, target.file) : join(assetsDir, target.file);
  const details = await readPngDetails(page, path);
  if (details.width !== target.width || details.height !== target.height) {
    throw new Error(`${name} dimensions ${details.width}x${details.height}, expected ${target.width}x${target.height}`);
  }
  if (checks.transparent && details.transparent === 0) {
    throw new Error(`${name} must preserve transparent pixels`);
  }
  if (details.visible === 0) {
    throw new Error(`${name} has no visible pixels`);
  }
  if (checks.safeZone) {
    await assertAdaptiveSafeZone(page, path, target.width, target.height);
  }
  if (checks.centered) {
    await assertPrimaryMarkCentered(page, path, target.width, target.height);
  }
}

async function assertAdaptiveSafeZone(page, path, width, height) {
  const dataUrl = `data:image/png;base64,${readFileSync(path).toString('base64')}`;
  const result = await page.evaluate(async ({ src, width, height }) => {
    const image = new Image();
    image.src = src;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas context unavailable');
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, width, height).data;
    const centerX = width / 2;
    const centerY = height / 2;
    const safeRadius = width * 0.33;
    let outsideAlpha = 0;
    let insideAlpha = 0;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const alpha = pixels[(y * width + x) * 4 + 3];
        const distance = Math.hypot(x + 0.5 - centerX, y + 0.5 - centerY);
        if (distance > safeRadius && alpha !== 0) outsideAlpha += 1;
        if (distance <= safeRadius && alpha !== 0) insideAlpha += 1;
      }
    }
    return { outsideAlpha, insideAlpha };
  }, { src: dataUrl, width, height });

  if (result.outsideAlpha !== 0) {
    throw new Error(`adaptive-icon safe zone failed: ${result.outsideAlpha} outside pixels have alpha`);
  }
  if (result.insideAlpha === 0) {
    throw new Error('adaptive-icon safe zone failed: mark has no visible alpha inside safe zone');
  }
}

async function assertPrimaryMarkCentered(page, path, width, height) {
  const dataUrl = `data:image/png;base64,${readFileSync(path).toString('base64')}`;
  const result = await page.evaluate(async ({ src, width, height }) => {
    const image = new Image();
    image.src = src;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas context unavailable');
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, width, height).data;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const offset = (y * width + x) * 4;
        const alpha = pixels[offset + 3];
        if (alpha === 0) continue;
        const red = pixels[offset];
        const green = pixels[offset + 1];
        const blue = pixels[offset + 2];
        const isPrimaryMark = (
          (red > 180 && green > 220 && blue < 120) ||
          (red > 235 && green > 235 && blue > 225)
        );
        if (!isPrimaryMark) continue;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
    if (maxX < 0 || maxY < 0) return { found: false };
    return {
      found: true,
      centerX: (minX + maxX + 1) / 2,
      centerY: (minY + maxY + 1) / 2
    };
  }, { src: dataUrl, width, height });

  if (!result.found) throw new Error('mark center assert failed: no primary mark pixels found');
  const deltaX = Math.abs(result.centerX - width / 2);
  const deltaY = Math.abs(result.centerY - height / 2);
  if (deltaX > 8 || deltaY > 8) {
    throw new Error(`mark center assert failed: primary mark center ${result.centerX},${result.centerY}; expected within 8px of ${width / 2},${height / 2}`);
  }
}

async function renderPreview48(page) {
  const iconPath = join(assetsDir, targets.icon.file);
  const dataUrl = `data:image/png;base64,${readFileSync(iconPath).toString('base64')}`;
  await page.setViewportSize({ width: 48, height: 48 });
  await page.setContent(htmlDocument(`
    <main style="width:48px;height:48px;background:${palette.background};display:grid;place-items:center;">
      <img src="${dataUrl}" style="width:48px;height:48px;display:block;" />
    </main>`), { waitUntil: 'load' });
  await page.screenshot({
    animations: 'disabled',
    fullPage: false,
    path: join(runsDir, targets.preview48.file),
    type: 'png'
  });
}

async function renderStoreIcon(page) {
  const iconPath = join(assetsDir, targets.icon.file);
  const dataUrl = `data:image/png;base64,${readFileSync(iconPath).toString('base64')}`;
  await page.setViewportSize({ width: targets.storeIcon.width, height: targets.storeIcon.height });
  await page.setContent(htmlDocument(`
    <main style="width:512px;height:512px;background:${palette.background};display:grid;place-items:center;">
      <img src="${dataUrl}" style="width:512px;height:512px;display:block;" />
    </main>`), { waitUntil: 'load' });
  await page.screenshot({
    animations: 'disabled',
    fullPage: false,
    path: join(storePackDir, targets.storeIcon.file),
    type: 'png'
  });
}

function commandName(command) {
  if (process.platform !== 'win32') return command;
  const normalized = command.toLowerCase();
  if (command.includes('\\') || command.includes('/') || normalized === 'node' || normalized.endsWith('.cmd') || normalized.endsWith('.exe')) {
    return command;
  }
  return `${command}.cmd`;
}

async function freePort() {
  const server = createNetServer();
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  await new Promise((resolveClose) => server.close(resolveClose));
  if (!address || typeof address === 'string') throw new Error('Unable to allocate port');
  return address.port;
}

function bundleName(distRoot) {
  const metadata = JSON.parse(readFileSync(join(distRoot, 'metadata.json'), 'utf8'));
  const bundle = metadata?.bundler === 'metro'
    ? readFileSync(join(distRoot, 'index.html'), 'utf8').match(/_expo\/static\/js\/web\/[^"]+\.js/)?.[0]
    : undefined;
  if (!bundle) throw new Error('Unable to locate Expo web bundle');
  return bundle.replace('_expo/static/js/web/', '');
}

function staticServer(distRoot) {
  return createServer((request, response) => {
    const cleanPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
    const candidate = cleanPath === '/' ? join(distRoot, 'index.html') : join(distRoot, cleanPath);
    const file = existsSync(candidate) && statSync(candidate).isFile() ? candidate : join(distRoot, 'index.html');
    const type = extname(file) === '.js' ? 'application/javascript' :
      extname(file) === '.css' ? 'text/css' :
      extname(file) === '.png' ? 'image/png' :
      extname(file) === '.ico' ? 'image/x-icon' :
      'text/html';
    response.writeHead(200, { 'content-type': type });
    createReadStream(file).pipe(response);
  });
}

function mockApiServer() {
  return createServer((request, response) => {
    const headers = {
      'access-control-allow-headers': 'content-type,x-racelens-device-id,x-racelens-platform',
      'access-control-allow-methods': 'GET,POST,OPTIONS',
      'access-control-allow-origin': '*',
      'content-type': 'application/json'
    };
    if (request.method === 'OPTIONS') {
      response.writeHead(204, headers);
      response.end();
      return;
    }
    if (request.url?.startsWith('/recent')) {
      const isHorse = request.url.includes('sport=horse');
      response.writeHead(200, headers);
      response.end(JSON.stringify({
        sport: isHorse ? 'horse' : 'keirin',
        meet: isHorse ? '서울' : '광명',
        days: isHorse ? ['2026-07-04', '2026-07-05', '2026-07-11'] : ['2026-07-03', '2026-07-04', '2026-07-05'],
        default_race_no: isHorse ? 8 : 11,
        race_count: isHorse ? 11 : 16
      }));
      return;
    }
    if (request.url?.startsWith('/api/app-session')) {
      response.writeHead(200, headers);
      response.end(JSON.stringify({
        app_session: {
          user_id: 'store_pack_user',
          device_id: request.headers['x-racelens-device-id'] ?? 'store_pack_device',
          entitlement: 'free',
          free_analysis_limit: 3,
          free_analysis_used: 0,
          free_analysis_remaining: 3
        },
        data_layer: { ready: true, storage: 'mock', schemas: [] }
      }));
      return;
    }
    if (!request.url?.startsWith('/api/live-decision')) {
      response.writeHead(404, headers);
      response.end(JSON.stringify({ error: 'not_found' }));
      return;
    }
    const url = new URL(request.url, 'http://127.0.0.1');
    const sport = url.searchParams.get('sport') === 'horse' ? 'horse' : 'keirin';
    const horse = sport === 'horse';
    const participants = horse ? [
      storeParticipant(5, '골든포커스', '기수 김하늘 · 55kg', '선입', '마체 +4kg · 게이트 5'),
      storeParticipant(3, '스톰레이크', '기수 이도윤 · 54kg', '추입', '최근 4전 1-1-0'),
      storeParticipant(1, '새벽질주', '기수 박서준 · 56kg', '선행', '거리 적성 양호')
    ] : [
      storeParticipant(4, '김로운', '선발 · 92.4점', '젖히기', '200m 11.42 · 입상률 71%'),
      storeParticipant(2, '이정민', '선발 · 90.1점', '선행', '최근 3주 상승'),
      storeParticipant(7, '한기봉', '우수 · 88.7점', '추입', '후반 가속 안정')
    ];
    const order = horse ? [5, 3, 1] : [4, 2, 7];
    response.writeHead(200, headers);
    response.end(JSON.stringify({
      ok: true,
      status: 'ready',
      message: '공식 출전표 검증 완료 · 정보 분석 화면',
      updated_at: '2026-07-04T16:40:00',
      market_used: true,
      odds_age_sec: 42,
      top: { bno: order[0], name: participants[0].name, pwin: 0.46, pplc: 0.78 },
      rows: order.map((bno, index) => ({ bno, name: participants[index].name, pwin: [0.46, 0.28, 0.18][index], pplc: [0.78, 0.61, 0.49][index] })),
      picks: [
        { code: 'TOP1', label: '1착 후보', selection: String(order[0]), probability: 0.46, grade: '강' },
        { code: 'QNL', label: '복승 조합', selection: `${order[0]}-${order[1]}`, probability: 0.34, grade: '중' },
        { code: 'TRI', label: '1-2-3 순서', selection: order.join('-'), probability: 0.112, grade: '중' }
      ],
      participants,
      market_odds: [
        { code: 'WIN', label: '단승', selection: String(order[0]), odds: 2.1, change: '검증 자료', signal: 'teal' },
        { code: 'QNL', label: '복승', selection: `${order[0]}-${order[1]}`, odds: 5.4, change: '모델 비교', signal: 'primary' },
        { code: 'TRI', label: '삼쌍', selection: order.join('-'), odds: 18.6, change: '정보 분석', signal: 'amber' }
      ],
      market_risk: {
        level: 'verified',
        title: '공식 데이터 확인',
        message: '출전표, 배당 자료, 모델 신호를 분리해 표시합니다.'
      },
      roster_verification: {
        state: 'verified',
        message: '공식 출전표와 대조 완료',
        source: 'store-pack-mock'
      },
      data_layer: {
        ready: true,
        storage: 'mock',
        schemas: [
          { name: 'race_data', tables: ['race_cards', 'market_odds_snapshots'], row_count: 82 },
          { name: 'prediction', tables: ['predictions'], row_count: 31 },
          { name: 'analytics', tables: ['user_view_events'], row_count: 12 }
        ]
      },
      app_session: {
        user_id: 'store_pack_user',
        device_id: request.headers['x-racelens-device-id'] ?? 'store_pack_device',
        entitlement: 'free',
        free_analysis_limit: 3,
        free_analysis_used: 1,
        free_analysis_remaining: 2
      },
      poll_delay_ms: 15000
    }));
  });
}

function storeParticipant(number, name, subtitle, trait, note) {
  return {
    number,
    name,
    subtitle,
    stats: note,
    trait,
    note,
    signal: 'teal',
    profile: [
      { label: '평균득점', value: '92.4', tone: 'teal' },
      { label: '200m', value: '11.42', tone: 'primary' },
      { label: '기수', value: subtitle.split(' · ')[0] ?? subtitle, tone: 'primary' },
      { label: '부담중량', value: subtitle.split(' · ')[1] ?? '55kg', tone: 'teal' },
      { label: '마체', value: '+4kg', tone: 'amber' },
      { label: '거리', value: '1400m', tone: 'violet' }
    ],
    form: [
      { label: '입상률', value: '71%', tone: 'teal' },
      { label: '최근 3주', value: '상승', tone: 'primary' },
      { label: '복승률', value: '54%', tone: 'teal' },
      { label: '최근 4전', value: '1-1-0', tone: 'primary' },
      { label: '게이트', value: String(number), tone: 'violet' }
    ],
    tactics: [
      { label: trait, value: '38%', tone: 'primary' }
    ]
  };
}

async function exportStoreApp(apiBaseUrl, distRoot) {
  const result = spawnSync(commandName('npx'), ['expo', 'export', '--platform', 'web', '--clear', '--output-dir', distRoot], {
    cwd: root,
    env: {
      ...process.env,
      EXPO_PUBLIC_RACELENS_ACCOUNT_DELETION_URL: 'https://168-107-2-218.sslip.io/legal/account-deletion',
      EXPO_PUBLIC_RACELENS_ANALYTICS_URL: '',
      EXPO_PUBLIC_RACELENS_API_BASE_URL: apiBaseUrl,
      EXPO_PUBLIC_RACELENS_OFFLINE_EXAMPLE: '0',
      EXPO_PUBLIC_RACELENS_PRIVACY_URL: 'https://168-107-2-218.sslip.io/legal/privacy',
      EXPO_PUBLIC_RACELENS_SUPPORT_URL: 'https://168-107-2-218.sslip.io/legal/support',
      EXPO_PUBLIC_RACELENS_TERMS_URL: 'https://168-107-2-218.sslip.io/legal/terms'
    },
    shell: process.platform === 'win32',
    stdio: 'inherit'
  });
  if (result.status !== 0) throw new Error(`Expo export failed: ${result.status}`);
  const bundleText = readFileSync(join(distRoot, '_expo/static/js/web', bundleName(distRoot)), 'utf8');
  if (!bundleText.includes(apiBaseUrl)) throw new Error('Store app export did not inline mock API URL');
}

async function captureStoreScreenshots(browser) {
  const apiPort = await freePort();
  const appPort = await freePort();
  const apiBaseUrl = `http://127.0.0.1:${apiPort}`;
  const distRoot = join(tmpdir(), `racelens-store-pack-${Date.now()}`);
  const apiServer = mockApiServer();
  const appServer = staticServer(distRoot);
  await new Promise((resolveListen) => apiServer.listen(apiPort, '127.0.0.1', resolveListen));
  await exportStoreApp(apiBaseUrl, distRoot);
  await new Promise((resolveListen) => appServer.listen(appPort, '127.0.0.1', resolveListen));

  try {
    const cases = [
      {
        file: 'screenshot_01_home_keirin_dark.png',
        colorScheme: 'dark',
        flow: async (page) => {
          await page.getByRole('button', { name: '모델 신호 보기' }).scrollIntoViewIfNeeded();
        }
      },
      {
        file: 'screenshot_02_home_horse_light.png',
        colorScheme: 'light',
        flow: async (page) => {
          await page.getByRole('button', { name: '경마' }).click();
          await page.getByRole('button', { name: '서울' }).waitFor();
        }
      },
      {
        file: 'screenshot_03_analysis_podium.png',
        colorScheme: 'dark',
        flow: async (page) => {
          await requestAnalysis(page);
          await page.getByTestId('prediction-podium').waitFor();
        }
      },
      {
        file: 'screenshot_04_evidence_data.png',
        colorScheme: 'light',
        flow: async (page) => {
          await requestAnalysis(page);
          await page.getByText(/판단 근거/).first().scrollIntoViewIfNeeded();
        }
      },
      {
        file: 'screenshot_05_market_odds.png',
        colorScheme: 'dark',
        flow: async (page) => {
          await requestAnalysis(page);
          await page.getByText('배당 자료', { exact: true }).scrollIntoViewIfNeeded();
        }
      },
      {
        file: 'screenshot_06_pro.png',
        colorScheme: 'light',
        flow: async (page) => {
          await page.getByRole('tab', { name: /Pro/ }).click();
          await page.getByText('RaceLens Pro').waitFor();
        }
      }
    ];

    for (const item of cases) {
      const page = await browser.newPage({
        colorScheme: item.colorScheme,
        deviceScaleFactor: 3,
        isMobile: true,
        viewport: { width: 360, height: 640 }
      });
      const errors = [];
      page.on('console', (message) => {
        if (message.type() === 'error') errors.push(message.text());
      });
      page.on('pageerror', (error) => errors.push(error.message));
      await page.goto(`http://127.0.0.1:${appPort}/`, { waitUntil: 'networkidle' });
      await item.flow(page);
      if (errors.length) throw new Error(`${item.file} console/page errors: ${errors.join(' | ')}`);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      if (overflow > 1) throw new Error(`${item.file} overflow ${overflow}px`);
      const path = join(storePackDir, item.file);
      await page.screenshot({ animations: 'disabled', fullPage: false, path, type: 'png' });
      await assertPng(page, item.file, { file: item.file, width: 1080, height: 1920 }, { storePack: true });
      await page.close();
    }
  } finally {
    await new Promise((resolveClose) => appServer.close(resolveClose));
    await new Promise((resolveClose) => apiServer.close(resolveClose));
  }
}

async function requestAnalysis(page) {
  await page.getByRole('button', { name: '모델 신호 보기' }).click();
  const adConfirm = page.getByRole('button', { name: '광고 확인 후 분석 보기' });
  await adConfirm.waitFor({ state: 'visible', timeout: 1500 }).then(() => adConfirm.click()).catch(() => {});
}

function writeListing() {
  const listing = `# RaceLens Play Store Listing

## App title
RaceLens

## Short description
경륜·경마 출전 정보와 모델 신호를 한눈에 정리합니다.

## Full description
RaceLens는 경륜·경마 데이터를 정보 분석 관점에서 정리하는 모바일 앱입니다.

공식 출전 정보, 참가자 기록, 모델 신호, 배당 자료 상태, 검증 상태를 한 화면에서 확인할 수 있습니다. 화면은 홈, 분석, Pro 안내로 나뉘며 사용자는 종목, 개최장, 경기일, 경주 번호를 고른 뒤 데이터 상태를 확인합니다.

주요 기능
- 경륜·경마 개최일과 경주 번호 선택
- 공식 출전 정보 기반 참가자 카드
- 모델 신호와 근거 데이터 요약
- 배당 자료 상태와 갱신 시각 분리 표시
- 경주 종료 후 실제 착순이 확인된 경우 복기용 결과 표시
- 무료 이용 상태와 Pro 준비 상태 안내

RaceLens는 정보 분석 도구입니다. 앱 안에서 참여 연결, 금액 산정, 구매 대행, 외부 참여 사이트 이동 기능을 제공하지 않습니다. 만 19세 이상 사용자를 대상으로 하며, 사용자는 거주 지역의 법령과 스토어 정책을 준수해야 합니다.

## Reviewer notes
RaceLens is a non-betting data analysis app for Korean keirin and horse racing information. The app does not include betting links, stake sizing, purchase agency flows, or gambling-site ads. The Pro screen is configured as a preparation/entitlement surface unless store billing is enabled. Legal pages are served by the API server:
- Privacy: https://168-107-2-218.sslip.io/legal/privacy
- Terms: https://168-107-2-218.sslip.io/legal/terms
- Account deletion: https://168-107-2-218.sslip.io/legal/account-deletion
- Support: https://168-107-2-218.sslip.io/legal/support
- Third-party SDK disclosure: the Android release uses Google Firebase Analytics, Crashlytics, and Google Mobile Ads SDK. After 3 free analyses, a user may voluntarily view a rewarded ad for 1 additional analysis. Gambling-site ads and betting links are not part of the product flow.

## Data safety answers
Based on runs/audit_gpt55.md:
- Anonymous device ID: collected for free usage limits, session continuity, abuse prevention, and entitlement state.
- App instance ID / Firebase installation identifier: collected by Google Firebase Analytics and Crashlytics for analytics, app stability, and crash diagnostics.
- IP address and User-Agent derived data: collected server-side for rate limiting and security diagnostics.
- UX events: app_open, screen_view, tab_select, race context, analysis request/result/error; payload blocks participant names, selections, user IDs, and device IDs.
- App activity events: coarse Firebase Analytics events such as tab, sport, race number, latency, error kind, and rounded confidence percentages for analytics and app functionality/stability.
- Diagnostics: Crashlytics crash logs, device model, OS, app version, stack traces, and error context for crash analysis and app stability.
- Analysis context and result metadata: sport, date, venue, race number, model response, market snapshot status for service operation and diagnostics.
- Subscription verification result: product/status/expiry only when store verification is configured.
- Email: not collected in app; only received if the user contacts support or requests deletion.
- Third-party ad/analytics SDK identifiers: Google Firebase Analytics, Crashlytics, and Google Mobile Ads SDK are present. Google may process advertising identifiers, IP address, device information, and rewarded-ad interactions for ad delivery, measurement, and fraud prevention.
- Third-party sharing / processing: Firebase data can be transmitted to and processed by Google as the Firebase service provider; Google Mobile Ads data can be processed by Google as the advertising service provider.
- Transport encryption: yes. Production URLs use HTTPS, so data is encrypted in transit.
- Deletion path: https://168-107-2-218.sslip.io/legal/account-deletion and support email on legal pages.
`;
  writeFileSync(join(storePackDir, 'listing.md'), listing, 'utf8');
}

mkdirSync(assetsDir, { recursive: true });
mkdirSync(runsDir, { recursive: true });
mkdirSync(storePackDir, { recursive: true });

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  await render(page, targets.icon, iconHtml());
  await render(page, targets.adaptive, adaptiveHtml(), { transparent: true });
  await render(page, targets.splash, splashHtml());
  await render(page, targets.favicon, faviconHtml(), { transparent: true });
  await render(page, targets.featureGraphic, featureGraphicHtml(), { storePack: true });
  await renderPreview48(page);
  await renderStoreIcon(page);

  await assertPng(page, 'icon', targets.icon, { centered: true });
  await assertPng(page, 'adaptive-icon', targets.adaptive, { transparent: true, safeZone: true });
  await assertPng(page, 'splash', targets.splash);
  await assertPng(page, 'favicon', targets.favicon, { transparent: true });
  await assertPng(page, '48px preview', { ...targets.preview48, file: join('..', '..', 'runs', targets.preview48.file) });
  await assertPng(page, 'feature graphic', targets.featureGraphic, { storePack: true });
  await assertPng(page, 'store icon 512', targets.storeIcon, { storePack: true, centered: true });
  await captureStoreScreenshots(browser);
  writeListing();
} finally {
  await browser.close();
}

console.log('brand assets and store pack generated: icon 1024, adaptive 1024, splash 1284x2778, favicon 192, feature 1024x500, icon_512, screenshots 6x1080x1920, listing.md');
