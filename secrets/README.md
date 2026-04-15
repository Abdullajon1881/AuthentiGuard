# secrets/

Runtime credentials consumed by `docker-compose.prod.yml` via Docker secrets.
**Never commit the credential files.** They are ignored by `.gitignore` —
only `README.md` and `.gitkeep` are tracked.

## Required files

| File | Consumed by | Purpose |
|------|-------------|---------|
| `minio_root_user` | `minio` service (`MINIO_ROOT_USER_FILE`) | MinIO root account — **bootstrap/admin only** |
| `minio_root_password` | `minio` service (`MINIO_ROOT_PASSWORD_FILE`) | MinIO root password — **bootstrap/admin only** |
| `s3_app_access_key` | `backend`, `worker` (`AWS_ACCESS_KEY_ID_FILE`) | App MinIO service-account access key |
| `s3_app_secret_key` | `backend`, `worker` (`AWS_SECRET_ACCESS_KEY_FILE`) | App MinIO service-account secret key |

The backend and worker processes never read the root credentials. They
authenticate with a dedicated MinIO service account that has access only
to the app buckets (`S3_BUCKET_UPLOADS`, `S3_BUCKET_REPORTS`).

## First-time setup

```bash
# 1. Root credentials (admin / bootstrap only)
printf '%s' "$ADMIN_USER"     > secrets/minio_root_user
printf '%s' "$ADMIN_PASSWORD" > secrets/minio_root_password

# 2. Boot MinIO so we can provision the service account
docker compose -f docker-compose.prod.yml up -d minio

# 3. Create a least-privilege service account via `mc`
docker run --rm --network authentiguard_authentiguard \
  -e MC_HOST_ag=http://"$ADMIN_USER":"$ADMIN_PASSWORD"@minio:9000 \
  minio/mc admin user svcacct add ag "$ADMIN_USER" \
  --access-key "$APP_ACCESS_KEY" \
  --secret-key "$APP_SECRET_KEY"

# 4. Persist the service-account keys as Docker secrets
printf '%s' "$APP_ACCESS_KEY" > secrets/s3_app_access_key
printf '%s' "$APP_SECRET_KEY" > secrets/s3_app_secret_key

chmod 600 secrets/minio_root_* secrets/s3_app_*
```

No trailing newline in any file. Contents are read verbatim by MinIO and
by `backend/app/core/config.py::_read_secret_file`.

## Rotation

- **Root credentials:** overwrite `minio_root_user` / `minio_root_password`,
  then `docker compose -f docker-compose.prod.yml up -d --force-recreate minio`.
- **App service account:** issue a new key pair via `mc admin user svcacct add`,
  overwrite `s3_app_access_key` / `s3_app_secret_key`, then
  `docker compose -f docker-compose.prod.yml up -d --force-recreate backend worker beat`.
  Revoke the old service account once no service references it.

## External secret managers

In staging/prod, swap `file:` references in `docker-compose.prod.yml`
with an external secret provider (AWS Secrets Manager, HashiCorp Vault
via the `vault-agent` sidecar, or Kubernetes `Secret` objects if
migrating to K8s). The application code reads files from
`/run/secrets/*`, so any mechanism that mounts those paths works
without further changes.

## Alerting webhook (required in production)

`ALERT_WEBHOOK_URL` is consumed by `backend/app/workers/alerting.py` to
post detector-fallback, high-failure-rate, and high-queue-depth alerts
to an external endpoint (Slack incoming-webhook or generic JSON).
It is NOT a Docker-secret file — it is a regular environment variable
stored alongside `JWT_SECRET_KEY` etc., because the payload is a URL
that the process uses for outbound HTTPS.

Where to set it:

| Deployment path | Location |
|-----------------|----------|
| `docker-compose.prod.yml` | `.env.production` → `ALERT_WEBHOOK_URL=...` (see `.env.production.example`) |
| Kubernetes (ESO)          | AWS Secrets Manager entry `authentiguard/prod::ALERT_WEBHOOK_URL`; pulled into the `authentiguard-secrets` ExternalSecret (see `infra/k8s/base/namespace.yaml`) |

Verifying it works end-to-end before launch:

```bash
# Trigger the alerting task manually inside a worker container
docker compose -f docker-compose.prod.yml exec worker \
  celery -A app.workers.celery_app call app.workers.alerting.check_health

# Or force a test payload from a Python shell:
docker compose -f docker-compose.prod.yml exec worker python -c "
from app.workers.alerting import _post_alert_webhook
_post_alert_webhook('detector_fallback', 'critical', 'launch-day smoke test')
"
```

You should see a message appear in the configured Slack channel (or
whatever JSON endpoint receives the POST). If nothing arrives, check
worker logs for `alert_webhook_failed` or `alert_webhook_non_2xx`
entries. A 15-minute per-alert-key cooldown prevents repeat delivery.
