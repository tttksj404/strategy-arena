import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { dirname, extname, join, normalize, resolve, sep } from 'node:path';
import { networkInterfaces } from 'node:os';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const port = Number.parseInt(process.env.PORT ?? '4173', 10);
const upstream = new URL(process.env.PREVIEW_UPSTREAM ?? 'https://168-107-2-218.sslip.io');
const distRoot = resolveDistRoot();
const apiPrefixes = ['/api', '/recent', '/healthz', '/legal', '/predict'];
const previewSessions = new Map();
const previewFreeLimit = 3;

const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ttf': 'font/ttf',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2'
};

function resolveDistRoot() {
  const configured = process.env.PREVIEW_DIST?.trim();
  if (!configured) return resolve(scriptDir, '..', 'dist-preview');
  return resolve(process.cwd(), configured);
}

function corsHeaders(request) {
  return {
    'access-control-allow-headers': request.headers['access-control-request-headers'] ??
      'Content-Type, Authorization, X-RaceLens-Device-Id, X-RaceLens-Platform, X-RaceLens-Analytics, X-RaceLens-Install-Id, X-RaceLens-*',
    'access-control-allow-methods': 'GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS',
    'access-control-allow-origin': '*',
    'access-control-expose-headers': '*'
  };
}

function writeError(response, status, message, request) {
  response.writeHead(status, {
    ...corsHeaders(request),
    'cache-control': 'no-store',
    'content-type': 'text/plain; charset=utf-8'
  });
  response.end(message);
}

function writeJson(response, status, payload, request) {
  const body = Buffer.from(JSON.stringify(payload));
  response.writeHead(status, {
    ...corsHeaders(request),
    'cache-control': 'no-store',
    'content-length': String(body.length),
    'content-type': 'application/json; charset=utf-8'
  });
  response.end(body);
}

function previewDeviceId(request) {
  const value = String(request.headers['x-racelens-device-id'] ?? '').trim();
  return value.slice(0, 96) || 'preview-browser';
}

function previewState(request) {
  const deviceId = previewDeviceId(request);
  let state = previewSessions.get(deviceId);
  if (!state) {
    state = { credits: 0, deviceId, used: 0 };
    previewSessions.set(deviceId, state);
  }
  return state;
}

function previewAppSession(state, upstreamSession = {}) {
  return {
    ...upstreamSession,
    user_id: upstreamSession.user_id ?? `preview-${state.deviceId}`,
    device_id: state.deviceId,
    entitlement: 'free',
    free_analysis_limit: previewFreeLimit,
    free_analysis_used: state.used,
    free_analysis_remaining: Math.max(0, previewFreeLimit - state.used),
    rewarded_analysis_credits: state.credits
  };
}

function previewDataLayer(payload) {
  return payload?.data_layer ?? {
    ready: true,
    storage: 'preview-memory',
    schemas: []
  };
}

function consumesPreviewQuota(payload, statusCode) {
  if (statusCode >= 400 || payload?.status === 'blocked' || payload?.status === 'rate_limited') return false;
  if (payload?.status === 'settled') return true;
  const releasedErrorKinds = new Set([
    'missing_api_key',
    'invalid_request',
    'invalid_date',
    'unsupported_meet',
    'no_race',
    'upstream_api_error',
    'base_prediction_error',
    'roster_mismatch'
  ]);
  if (releasedErrorKinds.has(String(payload?.error_kind ?? ''))) return false;
  return !(payload?.decision === 'hold' && !payload?.rows?.length && !payload?.top && !payload?.market_used);
}

function handlePreviewReward(request, response) {
  request.resume();
  const state = previewState(request);
  const granted = state.credits < 1;
  if (granted) state.credits += 1;
  writeJson(response, 200, {
    ok: true,
    preview_test_reward: true,
    reward_granted: granted,
    app_session: previewAppSession(state),
    data_layer: previewDataLayer()
  }, request);
}

function writePreviewQuotaBlocked(request, response) {
  const state = previewState(request);
  writeJson(response, 429, {
    ok: false,
    status: 'blocked',
    decision: 'hold',
    message: '오늘 무료 분석 3회를 모두 사용했습니다. 프리뷰 테스트 광고를 보면 1회 추가됩니다.',
    snapshot_phase: 'quota_exhausted',
    app_session: previewAppSession(state),
    data_layer: previewDataLayer()
  }, request);
}

function isProxyPath(pathname) {
  return apiPrefixes.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

function proxyRequest(clientRequest, clientResponse, parsedUrl, transformJson = null) {
  const headers = { ...clientRequest.headers };
  delete headers.connection;
  delete headers['keep-alive'];
  delete headers['proxy-authenticate'];
  delete headers['proxy-authorization'];
  delete headers.te;
  delete headers.trailer;
  delete headers['transfer-encoding'];
  delete headers.upgrade;
  if (transformJson) delete headers['accept-encoding'];
  headers.host = upstream.host;

  const upstreamRequest = httpsRequest({
    protocol: upstream.protocol,
    hostname: upstream.hostname,
    port: upstream.port || 443,
    method: clientRequest.method,
    path: `${parsedUrl.pathname}${parsedUrl.search}`,
    headers
  }, (upstreamResponse) => {
    const responseHeaders = {
      ...upstreamResponse.headers,
      ...corsHeaders(clientRequest)
    };
    if (!transformJson) {
      clientResponse.writeHead(upstreamResponse.statusCode ?? 502, responseHeaders);
      upstreamResponse.pipe(clientResponse);
      return;
    }
    const chunks = [];
    upstreamResponse.on('data', (chunk) => chunks.push(chunk));
    upstreamResponse.on('end', () => {
      const rawBody = Buffer.concat(chunks);
      try {
        const transformed = transformJson(JSON.parse(rawBody.toString('utf8')), upstreamResponse.statusCode ?? 502);
        const body = Buffer.from(JSON.stringify(transformed));
        delete responseHeaders['content-encoding'];
        delete responseHeaders['transfer-encoding'];
        responseHeaders['content-length'] = String(body.length);
        responseHeaders['content-type'] = 'application/json; charset=utf-8';
        responseHeaders['cache-control'] = 'no-store';
        clientResponse.writeHead(upstreamResponse.statusCode ?? 502, responseHeaders);
        clientResponse.end(body);
      } catch (error) {
        writeError(clientResponse, 502, `preview response transform error: ${error.message}`, clientRequest);
      }
    });
  });

  upstreamRequest.on('error', (error) => {
    if (clientResponse.headersSent) {
      clientResponse.destroy(error);
      return;
    }
    writeError(clientResponse, 502, `preview upstream error: ${error.message}`, clientRequest);
  });

  clientRequest.pipe(upstreamRequest);
}

function safeStaticPath(pathname) {
  let decodedPath;
  try {
    decodedPath = decodeURIComponent(pathname);
  } catch {
    return null;
  }
  const relativePath = decodedPath === '/' ? 'index.html' : decodedPath.replace(/^\/+/, '');
  const candidate = normalize(join(distRoot, relativePath));
  const rootWithSep = distRoot.endsWith(sep) ? distRoot : `${distRoot}${sep}`;
  if (candidate !== distRoot && !candidate.startsWith(rootWithSep)) return null;
  return candidate;
}

function serveStatic(clientRequest, clientResponse, pathname) {
  const candidate = safeStaticPath(pathname);
  if (!candidate) {
    writeError(clientResponse, 403, 'forbidden', clientRequest);
    return;
  }

  const filePath = existsSync(candidate) && statSync(candidate).isFile()
    ? candidate
    : join(distRoot, 'index.html');

  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    writeError(clientResponse, 404, 'dist-preview/index.html not found. Run npm run preview:build first.', clientRequest);
    return;
  }

  clientResponse.writeHead(200, {
    ...corsHeaders(clientRequest),
    'cache-control': 'no-store',
    'content-type': contentTypes[extname(filePath)] ?? 'application/octet-stream'
  });
  if (clientRequest.method === 'HEAD') {
    clientResponse.end();
    return;
  }
  createReadStream(filePath).pipe(clientResponse);
}

function lanUrls() {
  return Object.values(networkInterfaces())
    .flatMap((items) => items ?? [])
    .filter((item) => item.family === 'IPv4' && !item.internal)
    .map((item) => `http://${item.address}:${port}`);
}

const server = createServer((clientRequest, clientResponse) => {
  if (clientRequest.method === 'OPTIONS') {
    clientResponse.writeHead(204, corsHeaders(clientRequest));
    clientResponse.end();
    return;
  }

  const parsedUrl = new URL(clientRequest.url ?? '/', `http://127.0.0.1:${port}`);
  if (parsedUrl.pathname === '/api/rewarded-ad/claim' && clientRequest.method === 'POST') {
    handlePreviewReward(clientRequest, clientResponse);
    return;
  }
  if (parsedUrl.pathname === '/api/app-session') {
    const state = previewState(clientRequest);
    proxyRequest(clientRequest, clientResponse, parsedUrl, (payload) => ({
      ...payload,
      app_session: previewAppSession(state, payload.app_session),
      data_layer: previewDataLayer(payload)
    }));
    return;
  }
  if (parsedUrl.pathname === '/api/live-decision') {
    const state = previewState(clientRequest);
    if (state.used >= previewFreeLimit && state.credits <= 0) {
      writePreviewQuotaBlocked(clientRequest, clientResponse);
      return;
    }
    proxyRequest(clientRequest, clientResponse, parsedUrl, (payload, statusCode) => {
      const consumed = consumesPreviewQuota(payload, statusCode);
      const previewClaimSource = state.used < previewFreeLimit ? 'preview_free' : 'preview_rewarded_ad';
      if (consumed) {
        if (state.used < previewFreeLimit) state.used += 1;
        else state.credits = Math.max(0, state.credits - 1);
      }
      return {
        ...payload,
        app_session: {
          ...previewAppSession(state, payload.app_session),
          ...(consumed ? { analysis_claim_source: previewClaimSource } : {})
        },
        data_layer: previewDataLayer(payload)
      };
    });
    return;
  }
  if (isProxyPath(parsedUrl.pathname)) {
    proxyRequest(clientRequest, clientResponse, parsedUrl);
    return;
  }
  serveStatic(clientRequest, clientResponse, parsedUrl.pathname);
});

server.listen(port, '0.0.0.0', () => {
  console.log(`RaceLens preview server listening on http://localhost:${port}`);
  const urls = lanUrls();
  if (urls.length > 0) {
    console.log(`Same Wi-Fi phone URL: ${urls.join(', ')}`);
  } else {
    console.log('Same Wi-Fi phone URL: use this PC LAN IP with port 4173');
  }
  console.log(`Serving ${distRoot}`);
  console.log(`Proxying API requests to ${upstream.origin}`);
});
