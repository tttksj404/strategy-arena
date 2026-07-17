const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const root = path.resolve(__dirname, '..', 'dist');
const port = Number(process.env.RACELENS_PREVIEW_PORT || 4173);
const upstreamBase = (process.env.RACELENS_UPSTREAM_API || process.env.ORACLE_BASE_URL || '').replace(/\/$/, '');
const upstreamTimeoutMs = Number(process.env.RACELENS_UPSTREAM_TIMEOUT_MS || 12000);

const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.ttf': 'font/ttf'
};

function staticHeaders(contentType) {
  return {
    'cache-control': 'no-store',
    'content-type': contentType
  };
}

function sendJson(res, status, body) {
  res.writeHead(status, {
    'access-control-allow-origin': '*',
    'cache-control': 'no-store',
    'content-type': 'application/json; charset=utf-8'
  });
  res.end(JSON.stringify(body));
}

function fallback(reason) {
  return {
    ok: true,
    status: 'pre_race',
    decision: 'hold',
    market_used: false,
    market_odds: [],
    market_risk: {
      level: upstreamBase ? 'odds_unavailable' : 'oracle_api_not_configured',
      message: reason || '현재 배당 원자료를 가져오지 못했습니다.'
    },
    message: '배당 미반영 - Oracle/Korea API 프록시 폴백',
    poll_delay_ms: 15000,
    snapshot_phase: 'pre_race',
    rows: [],
    top: null,
    updated_at: new Date().toISOString()
  };
}

function fallbackSession() {
  return {
    ok: true,
    app_session: {
      user_id: 'anonymous',
      device_id: 'preview-fallback',
      entitlement: 'free',
      free_analysis_limit: 3,
      free_analysis_used: 3,
      free_analysis_remaining: 0
    },
    data_layer: {
      ready: false,
      storage: 'proxy_fallback',
      error: 'upstream_unavailable',
      schemas: []
    }
  };
}

function health(res) {
  sendJson(res, 200, {
    ok: true,
    dist_ready: fs.existsSync(path.join(root, 'index.html')),
    upstream_configured: Boolean(upstreamBase),
    upstream_base: upstreamBase || null,
    port
  });
}

function apiCorsHeaders() {
  return {
    'access-control-allow-headers': 'Content-Type, X-RaceLens-Device-Id, X-RaceLens-Platform, X-RaceLens-Analytics',
    'access-control-allow-methods': 'GET,POST,OPTIONS',
    'access-control-allow-origin': '*'
  };
}

function filteredApiHeaders(req) {
  const headers = {};
  for (const name of ['content-type', 'x-racelens-device-id', 'x-racelens-platform', 'x-racelens-analytics']) {
    const value = req.headers[name];
    if (value) {
      headers[name] = value;
    }
  }
  return headers;
}

async function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > 8192) {
        reject(new Error('payload_too_large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(chunks.length ? Buffer.concat(chunks) : undefined));
    req.on('error', reject);
  });
}

async function proxyApi(req, res, pathname, search) {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, apiCorsHeaders());
    res.end();
    return;
  }

  if (!upstreamBase) {
    sendJson(
      res,
      200,
      pathname === '/api/app-session'
        ? fallbackSession()
        : fallback('Oracle/Korea API 주소가 설정되지 않아 배당 없이 표시합니다.')
    );
    return;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), upstreamTimeoutMs);
  try {
    const init = {
      headers: filteredApiHeaders(req),
      method: req.method,
      signal: controller.signal
    };
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      init.body = await readBody(req);
    }
    const upstream = await fetch(`${upstreamBase}${pathname}${search}`, init);
    clearTimeout(timeout);
    if (!upstream.ok) {
      sendJson(
        res,
        200,
        pathname === '/api/app-session'
          ? fallbackSession()
          : fallback(`Oracle/Korea API ${upstream.status}`)
      );
      return;
    }
    const text = await upstream.text();
    res.writeHead(200, {
      ...apiCorsHeaders(),
      'cache-control': 'no-store',
      'content-type': 'application/json; charset=utf-8'
    });
    res.end(text);
  } catch (error) {
    clearTimeout(timeout);
    sendJson(
      res,
      200,
      pathname === '/api/app-session'
        ? fallbackSession()
        : fallback(error && error.name === 'AbortError'
          ? 'Oracle/Korea API 응답 시간이 길어 배당 없이 표시합니다.'
          : 'Oracle/Korea API 연결 실패로 배당 없이 표시합니다.')
    );
  }
}

function serveStatic(urlPath, res) {
  const requested = urlPath === '/' ? '/index.html' : decodeURIComponent(urlPath);
  const fullPath = path.normalize(path.join(root, requested));
  if (!fullPath.startsWith(root)) {
    res.writeHead(403);
    res.end('forbidden');
    return;
  }

  fs.readFile(fullPath, (error, data) => {
    if (error) {
      fs.readFile(path.join(root, 'index.html'), (indexError, indexData) => {
        if (indexError) {
          res.writeHead(404);
          res.end('not found');
          return;
        }
        res.writeHead(200, staticHeaders('text/html; charset=utf-8'));
        res.end(indexData);
      });
      return;
    }
    res.writeHead(200, staticHeaders(contentTypes[path.extname(fullPath)] || 'application/octet-stream'));
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  try {
    const parsed = new URL(req.url, `http://127.0.0.1:${port}`);
    if (parsed.pathname === '/health' || parsed.pathname === '/healthz') {
      health(res);
      return;
    }
    if (
      parsed.pathname === '/recent' ||
      parsed.pathname === '/api/live-decision' ||
      parsed.pathname === '/api/app-session' ||
      parsed.pathname === '/api/ux-events'
    ) {
      void proxyApi(req, res, parsed.pathname, parsed.search);
      return;
    }
    serveStatic(parsed.pathname, res);
  } catch {
    sendJson(res, 500, { error: 'proxy_server_error' });
  }
});

server.listen(port, '0.0.0.0', () => {
  console.log(`RaceLens preview proxy listening on ${port}, upstream=${upstreamBase || 'UNCONFIGURED_ORACLE_API'}`);
});
