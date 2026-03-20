FROM node:20-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --only=production 2>/dev/null || npm install
COPY . .
RUN npm run build

FROM node:20-alpine AS runtime

WORKDIR /app
RUN addgroup --gid 1001 appgroup && \
    adduser --uid 1001 --ingroup appgroup --shell /bin/sh --disabled-password appuser

COPY --from=builder --chown=appuser:appgroup /app/.next/standalone ./
COPY --from=builder --chown=appuser:appgroup /app/.next/static ./.next/static
COPY --from=builder --chown=appuser:appgroup /app/public ./public

USER appuser
EXPOSE 3000
ENV NODE_ENV=production PORT=3000 HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:3000 || exit 1

CMD ["node", "server.js"]
