#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# AuthentiGuard VPS Deployment Script
# ══════════════════════════════════════════════════════════════
# Usage:
#   1. SSH into your Hetzner VPS: ssh root@YOUR_VPS_IP
#   2. Run: curl -fsSL https://raw.githubusercontent.com/Abdullajon1881/AuthentiGuard/main/scripts/deploy.sh | bash
#   OR clone and run:
#   3. git clone https://github.com/Abdullajon1881/AuthentiGuard.git
#   4. cd AuthentiGuard && bash scripts/deploy.sh
# ══════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Check we're on the VPS ───────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    err "Run as root: sudo bash scripts/deploy.sh"
fi

log "Starting AuthentiGuard deployment..."

# ── 1. System updates ────────────────────────────────────────
log "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# ── 2. Install Docker ────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    log "Docker installed: $(docker --version)"
else
    log "Docker already installed: $(docker --version)"
fi

# ── 3. Install Docker Compose (v2 plugin) ────────────────────
if ! docker compose version &>/dev/null; then
    log "Installing Docker Compose plugin..."
    apt-get install -y -qq docker-compose-plugin
fi
log "Docker Compose: $(docker compose version)"

# ── 4. Create app directory ──────────────────────────────────
APP_DIR="/opt/authentiguard"
if [ ! -d "$APP_DIR" ]; then
    log "Cloning repository..."
    git clone https://github.com/Abdullajon1881/AuthentiGuard.git "$APP_DIR"
else
    log "Updating existing installation..."
    cd "$APP_DIR"
    git pull origin main
fi
cd "$APP_DIR"

# ── 5. Generate production secrets ───────────────────────────
ENV_FILE="$APP_DIR/.env.production"
if [ ! -f "$ENV_FILE" ] || grep -q "CHANGE_ME" "$ENV_FILE" 2>/dev/null; then
    log "Generating production secrets..."
    cp "$APP_DIR/.env.production" "$ENV_FILE.bak" 2>/dev/null || true

    APP_SECRET=$(openssl rand -hex 32)
    PG_PASSWORD=$(openssl rand -hex 24)
    REDIS_PASSWORD=$(openssl rand -hex 24)
    JWT_SECRET=$(openssl rand -hex 64)
    MINIO_PASSWORD=$(openssl rand -hex 24)
    ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || openssl rand -base64 32)
    FLOWER_PASSWORD=$(openssl rand -hex 16)

    cat > "$ENV_FILE" << ENVEOF
# AuthentiGuard Production — generated $(date -u +%Y-%m-%dT%H:%M:%SZ)
APP_NAME=AuthentiGuard
APP_VERSION=0.2.0
APP_ENV=production
APP_SECRET_KEY=${APP_SECRET}

DOMAIN=authentiguard.io
CORS_ORIGINS=https://authentiguard.io,https://www.authentiguard.io,https://api.authentiguard.io

POSTGRES_USER=authentiguard
POSTGRES_PASSWORD=${PG_PASSWORD}
POSTGRES_DB=authentiguard
DATABASE_URL=postgresql+asyncpg://authentiguard:${PG_PASSWORD}@postgres:5432/authentiguard

REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/1

JWT_SECRET_KEY=${JWT_SECRET}
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

MINIO_ROOT_USER=authentiguard
MINIO_ROOT_PASSWORD=${MINIO_PASSWORD}
S3_BUCKET_UPLOADS=authentiguard-uploads
S3_BUCKET_REPORTS=authentiguard-reports
AWS_ACCESS_KEY_ID=authentiguard
AWS_SECRET_ACCESS_KEY=${MINIO_PASSWORD}
AWS_REGION=us-east-1
S3_ENDPOINT_URL=http://minio:9000

ENCRYPTION_KEY=${ENCRYPTION_KEY}

FLOWER_USER=admin
FLOWER_PASSWORD=${FLOWER_PASSWORD}

RATE_LIMIT_FREE_TIER=10
RATE_LIMIT_PRO_TIER=100
RATE_LIMIT_ENTERPRISE_TIER=1000

MAX_UPLOAD_SIZE_MB=500
UPLOAD_RETENTION_DAYS=30
REPORT_RETENTION_DAYS=365
ENVEOF

    chmod 600 "$ENV_FILE"
    log "Secrets generated and saved to $ENV_FILE"
    warn "IMPORTANT: Back up $ENV_FILE securely. If lost, all data encrypted with these keys is unrecoverable."
else
    log "Using existing $ENV_FILE"
fi

# ── 6. Create backup directory ───────────────────────────────
mkdir -p "$APP_DIR/backups"

# ── 7. Set up firewall ──────────────────────────────────────
log "Configuring firewall..."
if command -v ufw &>/dev/null; then
    ufw allow 22/tcp   # SSH
    ufw allow 80/tcp   # HTTP
    ufw allow 443/tcp  # HTTPS
    ufw --force enable
    log "Firewall configured (SSH + HTTP + HTTPS only)"
fi

# ── 8. Build and start services ──────────────────────────────
log "Building Docker images (this may take 5-10 minutes on first run)..."
docker compose -f docker-compose.prod.yml --env-file .env.production build

log "Starting services..."
docker compose -f docker-compose.prod.yml --env-file .env.production up -d

# ── 9. Wait for services to be healthy ───────────────────────
log "Waiting for services to start..."
sleep 10

for service in postgres redis minio backend worker frontend; do
    if docker compose -f docker-compose.prod.yml ps "$service" | grep -q "healthy\|running"; then
        log "  $service: running"
    else
        warn "  $service: may still be starting..."
    fi
done

# ── 10. Create MinIO buckets ────────────────────────────────
log "Creating MinIO buckets..."
docker compose -f docker-compose.prod.yml exec -T minio sh -c "
    mc alias set local http://localhost:9000 \$MINIO_ROOT_USER \$MINIO_ROOT_PASSWORD 2>/dev/null || true
    mc mb local/authentiguard-uploads 2>/dev/null || true
    mc mb local/authentiguard-reports 2>/dev/null || true
" 2>/dev/null || warn "MinIO bucket setup may need manual configuration"

# ── 11. Set up database backup cron ──────────────────────────
log "Setting up daily database backups..."
BACKUP_SCRIPT="$APP_DIR/scripts/backup.sh"
chmod +x "$BACKUP_SCRIPT" 2>/dev/null || true

# Daily backup at 3 AM UTC, keep 30 days
CRON_LINE="0 3 * * * $BACKUP_SCRIPT >> /var/log/authentiguard-backup.log 2>&1"
(crontab -l 2>/dev/null | grep -v "authentiguard" ; echo "$CRON_LINE") | crontab -
log "Daily backup cron installed (3 AM UTC)"

# ── 12. Print status ────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}AuthentiGuard deployment complete!${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Services running:"
docker compose -f docker-compose.prod.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Next steps:"
echo "  1. Point DNS records to this server's IP: $(curl -4s ifconfig.me 2>/dev/null || echo 'YOUR_IP')"
echo "     A    authentiguard.io       → YOUR_IP"
echo "     A    www.authentiguard.io   → YOUR_IP"
echo "     A    api.authentiguard.io   → YOUR_IP"
echo "     A    flower.authentiguard.io → YOUR_IP"
echo ""
echo "  2. Once DNS propagates, Caddy auto-provisions SSL certificates"
echo ""
echo "  3. Smoke test:"
echo "     curl https://api.authentiguard.io/api/v1/health"
echo ""
echo "  4. View logs:"
echo "     docker compose -f docker-compose.prod.yml logs -f"
echo ""
echo "  5. Flower dashboard password: (check .env.production)"
echo ""
echo "Backups: daily at 3 AM UTC → $APP_DIR/backups/"
echo "Logs:    docker compose -f docker-compose.prod.yml logs [service]"
echo "Stop:    docker compose -f docker-compose.prod.yml down"
echo "Update:  git pull && docker compose -f docker-compose.prod.yml up -d --build"
echo ""
