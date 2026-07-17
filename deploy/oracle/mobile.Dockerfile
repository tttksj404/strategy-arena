FROM node:22-alpine AS build

WORKDIR /src/mobile

COPY mobile/package*.json ./
RUN npm ci

COPY mobile ./
ENV EXPO_PUBLIC_RACELENS_API_BASE_URL=""
RUN npm run export:web

FROM node:22-alpine

ENV NODE_ENV=production \
    RACELENS_PREVIEW_PORT=4173 \
    RACELENS_UPSTREAM_API=http://app:8000

WORKDIR /app

COPY --from=build /src/mobile/dist ./dist
COPY mobile/scripts/preview-proxy-server.cjs ./scripts/preview-proxy-server.cjs

EXPOSE 4173

CMD ["node", "scripts/preview-proxy-server.cjs"]
