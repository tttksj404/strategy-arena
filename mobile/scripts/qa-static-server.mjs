import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, resolve } from 'node:path';

const port = Number.parseInt(process.env.RACELENS_QA_PORT ?? '8064', 10);
const root = resolve(process.env.RACELENS_QA_DIST ?? 'dist');

function contentType(file) {
  const extension = extname(file);
  if (extension === '.js') return 'application/javascript';
  if (extension === '.css') return 'text/css';
  if (extension === '.png') return 'image/png';
  if (extension === '.ico') return 'image/x-icon';
  if (extension === '.json') return 'application/json';
  return 'text/html';
}

const server = createServer((request, response) => {
  const cleanPath = decodeURIComponent((request.url ?? '/').split('?')[0] ?? '/');
  const candidate = cleanPath === '/' ? join(root, 'index.html') : join(root, cleanPath);
  const file = existsSync(candidate) && statSync(candidate).isFile() ? candidate : join(root, 'index.html');
  response.writeHead(200, { 'content-type': contentType(file) });
  createReadStream(file).pipe(response);
});

server.listen(port, '127.0.0.1');
