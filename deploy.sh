#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# AuthentiGuard Production Deploy Script
# ═══════════════════════════════════════════════════════════════
# Usage: ./deploy.sh [--build] [--pull]
#   --build   Force rebuild of Docker images
#   --pull    Pull latest code from git before deploying

set -euo pipefail

COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ────────────────────────────────────────────────
BUILD_FLAG=""
PULL_FLAG=""
for arg in "$@"; do
  case $arg in
    --build) BUILD_FLAG="--build" ;;
    --pull)  PULL_FLAG="1" ;;
  esac
done

# ── Pre-flight checks ─────────────────────────────────────────
echo "==> Pre-flight checks..."

if [ ! -f .env ]; then
  if [ -f .env.production ]; then
    echo "    .env not found, using .env.production"
    cp .env.production .env
  else
    echo "ERROR: No .env or .env.production file found."
    echo "       Copy .env.production.example to .env and fill in all values."
    exit 1
  fi
fi

# Check for CHANGE_ME values
if grep -q "CHANGE_ME" .env 2>/dev/null; then
  echo "WARNING: .env still contains CHANGE_ME placeholder values."
  echo "         Please replace them with real secrets before production use."
fi

# ── Pull latest code ──────────────────────────────────────────
if [ -n "$PULL_FLAG" ]; then
  echo "==> Pulling latest code..."
  git pull --ff-only
fi

# ── Build and start services ──────────────────────────────────
echo "==> Building and starting services..."
docker compose $COMPOSE_FILES up -d $BUILD_FLAG

# ── Wait for health checks ────────────────────────────────────
echo "==> Waiting for services to become healthy..."

wait_for_service() {
  local service=$1
  local max_wait=${2:-60}
  local elapsed=0
  while [ $elapsed -lt $max_wait ]; do
    status=$(docker compose $COMPOSE_FILES ps --format json "$service" 2>/dev/null | grep -o '"Health":"[^"]*"' | head -1 || echo "")
    if echo "$status" | grep -q "healthy"; then
      echo "    $service: healthy"
      return 0
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done
  echo "    $service: NOT healthy after ${max_wait}s (continuing anyway)"
  return 0
}

wait_for_service postgres 30
wait_for_service redis 20
wait_for_service minio 30
wait_for_service backend 60

# ── Verify API health ─────────────────────────────────────────
echo "==> Checking API health..."
for i in 1 2 3 4 5; do
  if curl -sf http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "    API is responding!"
    break
  fi
  if [ $i -eq 5 ]; then
    echo "    WARNING: API health check failed after 5 attempts"
    echo "    Check logs: docker compose $COMPOSE_FILES logs backend"
  fi
  sleep 3
done

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "==> Deploy complete!"
echo "    Services: docker compose $COMPOSE_FILES ps"
echo "    Logs:     docker compose $COMPOSE_FILES logs -f"
echo "    Backend:  http://localhost:8000"
echo "    Flower:   http://localhost:5555"
echo ""
