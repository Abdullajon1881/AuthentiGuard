#!/usr/bin/env bash
# AuthentiGuard — One-command local setup
# Usage: ./setup.sh
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

info "AuthentiGuard local setup starting..."

# ── Prerequisites check ────────────────────────────────────────
command -v docker   >/dev/null 2>&1 || error "Docker not found. Install: https://docker.com"
command -v python3  >/dev/null 2>&1 || error "Python 3.11+ required"
command -v node     >/dev/null 2>&1 || warn "Node.js not found — frontend won't run locally without Docker"

# ── Generate secrets ───────────────────────────────────────────
if [ ! -f .env ]; then
    info "Generating .env from template..."
    cp .env.example .env

    # Generate secure secrets
    JWT_SECRET=$(openssl rand -hex 64 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(64))")
    APP_SECRET=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")
    ENC_KEY=$(python3 -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())")
    DB_PASS=$(openssl rand -hex 16 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(16))")
    REDIS_PASS=$(openssl rand -hex 16 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(16))")
    MINIO_PASS=$(openssl rand -hex 16 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(16))")

    # Patch .env
    sed -i.bak \
        -e "s|CHANGE_ME_generate_with_openssl_rand_hex_64|$JWT_SECRET|g" \
        -e "s|CHANGE_ME_generate_with_openssl_rand_hex_32|$APP_SECRET|g" \
        -e "s|CHANGE_ME_generate_with_fernet_key_generate|$ENC_KEY|g" \
        -e "s|CHANGE_ME_strong_password|$DB_PASS|g" \
        .env
    # Fix Redis URL to use generated password
    sed -i.bak "s|redis://:CHANGE_ME@|redis://:$REDIS_PASS@|g" .env
    sed -i.bak "s|REDIS_PASSWORD=.*|REDIS_PASSWORD=$REDIS_PASS|g" .env
    sed -i.bak "s|MINIO_ROOT_PASSWORD=.*|MINIO_ROOT_PASSWORD=$MINIO_PASS|g" .env
    sed -i.bak "s|AWS_SECRET_ACCESS_KEY=CHANGE_ME.*|AWS_SECRET_ACCESS_KEY=$MINIO_PASS|g" .env
    rm -f .env.bak
    info "✓ .env generated with secure secrets"
else
    info ".env already exists — skipping secret generation"
fi

# ── MinIO bucket setup helper ──────────────────────────────────
setup_minio() {
    info "Setting up MinIO buckets..."
    sleep 3  # wait for MinIO to be ready
    docker compose exec -T minio mc alias set local http://localhost:9000 minioadmin $(grep MINIO_ROOT_PASSWORD .env | cut -d= -f2) 2>/dev/null || true
    docker compose exec -T minio mc mb local/ag-uploads --ignore-existing 2>/dev/null || true
    docker compose exec -T minio mc mb local/ag-reports --ignore-existing 2>/dev/null || true
    info "✓ MinIO buckets ready"
}

# ── Start services ─────────────────────────────────────────────
info "Starting services (this pulls Docker images — may take a few minutes on first run)..."
docker compose pull --quiet 2>/dev/null || true
docker compose up -d --wait 2>/dev/null || docker compose up -d

# ── Wait for health ────────────────────────────────────────────
info "Waiting for services..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        info "✓ API is healthy"
        break
    fi
    sleep 2
    [ $i -eq 30 ] && warn "API health check timed out — check: docker compose logs backend"
done

# ── Database migrations ────────────────────────────────────────
info "Running database migrations..."
docker compose exec -T backend alembic upgrade head 2>/dev/null || \
    warn "Migration failed — run manually: docker compose exec backend alembic upgrade head"

setup_minio

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     AuthentiGuard is running!                     ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Frontend:  http://localhost:3000                  ║${NC}"
echo -e "${GREEN}║  API:       http://localhost:8000                  ║${NC}"
echo -e "${GREEN}║  API docs:  http://localhost:8000/docs             ║${NC}"
echo -e "${GREEN}║  Flower:    http://localhost:5555                  ║${NC}"
echo -e "${GREEN}║  MLflow:    http://localhost:5000                  ║${NC}"
echo -e "${GREEN}║  MinIO:     http://localhost:9001                  ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Quick test:                                       ║${NC}"
echo -e "${GREEN}║  curl http://localhost:8000/health                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo "Logs:  docker compose logs -f"
echo "Stop:  docker compose down"
echo "Reset: docker compose down -v && ./setup.sh"
