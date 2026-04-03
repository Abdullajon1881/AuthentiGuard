#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# AuthentiGuard Database Backup Script
# Runs daily via cron, keeps 30 days of backups
# ══════════════════════════════════════════════════════════════

set -euo pipefail

APP_DIR="/opt/authentiguard"
BACKUP_DIR="$APP_DIR/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/authentiguard_${TIMESTAMP}.sql.gz"

# Source env for credentials
set -a
source "$APP_DIR/.env.production"
set +a

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting database backup..."

# Dump database via docker exec, compress with gzip
docker exec ag_postgres pg_dump \
    -U "${POSTGRES_USER:-authentiguard}" \
    -d "${POSTGRES_DB:-authentiguard}" \
    --no-owner \
    --no-privileges \
    | gzip > "$BACKUP_FILE"

# Check backup was created and is non-empty
if [ -s "$BACKUP_FILE" ]; then
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backup complete: $BACKUP_FILE ($SIZE)"
else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: Backup file is empty!"
    rm -f "$BACKUP_FILE"
    exit 1
fi

# Clean up old backups (keep last 30 days)
DELETED=$(find "$BACKUP_DIR" -name "authentiguard_*.sql.gz" -mtime +${RETENTION_DAYS} -delete -print | wc -l)
if [ "$DELETED" -gt 0 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Cleaned up $DELETED old backup(s)"
fi

# Show backup stats
TOTAL=$(ls -1 "$BACKUP_DIR"/authentiguard_*.sql.gz 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Backups: $TOTAL files, $TOTAL_SIZE total"
