# AuthentiGuard — Operational Runbooks

Runbooks for common production scenarios. Each runbook follows the format:
**Symptoms → Diagnosis → Resolution → Post-incident**.

---

## RB-001: API high latency (p95 > 2s)

**Severity:** P2 — degraded service

### Symptoms
- Grafana: `authentiguard_http_request_duration_p95` > 2s
- Users report slow analysis results
- PagerDuty alert: "API P95 latency threshold exceeded"

### Diagnosis

```bash
# 1. Check which endpoint is slow
kubectl top pods -n authentiguard

# 2. Check Celery queue depths (backed-up queues = slow async jobs)
kubectl exec -n authentiguard deploy/api -- \
  celery -A backend.app.workers.celery_app inspect active

# 3. Check DB query times
kubectl exec -n authentiguard deploy/api -- \
  psql "$DATABASE_URL" -c "
    SELECT query, mean_exec_time, calls
    FROM pg_stat_statements
    ORDER BY mean_exec_time DESC LIMIT 10;"

# 4. Check Redis memory
kubectl exec -n authentiguard deploy/redis-0 -- \
  redis-cli info memory | grep used_memory_human
```

### Resolution

**If Celery queues are backed up (> 50 pending tasks):**
```bash
# Scale up CPU workers
kubectl scale deployment worker-cpu -n authentiguard --replicas=10
# Monitor: watch kubectl get pods -n authentiguard -l app=worker-cpu
```

**If DB queries are slow:**
```bash
# Check for missing indexes
kubectl exec -n authentiguard deploy/api -- \
  psql "$DATABASE_URL" -c "
    SELECT relname, seq_scan, idx_scan
    FROM pg_stat_user_tables ORDER BY seq_scan DESC LIMIT 10;"
# If seq_scans are high on detection_jobs, run:
# CREATE INDEX CONCURRENTLY idx_jobs_user_status ON detection_jobs(user_id, status);
```

**If Redis is evicting keys (low hit rate):**
```bash
# Check eviction
kubectl exec -n authentiguard deploy/redis-0 -- redis-cli info stats | grep evicted
# If high, increase maxmemory in Redis ConfigMap or scale ElastiCache
```

### Post-incident
- Create a GitHub issue with latency timeline and root cause
- Update the relevant runbook if a new failure mode was discovered

---

## RB-002: Celery worker pod OOMKilled

**Severity:** P2

### Symptoms
- `kubectl get pods -n authentiguard` shows `OOMKilled` restart reason
- `kubectl describe pod <worker-pod>` shows memory limit exceeded
- Celery tasks are failing with worker restart errors

### Diagnosis

```bash
# Check which pod OOMKilled and how many times
kubectl get pods -n authentiguard -l app=worker-cpu \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\n"}{end}'

# Check the memory limits currently set
kubectl get deployment worker-cpu -n authentiguard \
  -o jsonpath='{.spec.template.spec.containers[0].resources}'
```

### Resolution

**Immediate (increase memory limit):**
```bash
kubectl patch deployment worker-cpu -n authentiguard \
  --type=json \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "12Gi"}]'
```

**Root cause investigation:**
```bash
# Check which task type was running when OOM occurred
kubectl logs <oom-pod> --previous | grep "Task" | tail -30

# Video/audio tasks use the most memory — if CPU workers are processing them,
# ensure they're only routed to GPU workers:
kubectl exec -n authentiguard deploy/api -- \
  celery -A backend.app.workers.celery_app inspect registered
```

**If video tasks are on CPU workers (misconfigured routing):**
```bash
# Verify queue assignments in celery_app.py — video_queue and audio_queue
# must only be consumed by worker-gpu, not worker-cpu
```

### Post-incident
- Update resource limits in `infra/k8s/helm/authentiguard/values.yaml`
- Add memory-based HPA metric if not already present
- File issue to investigate which task type caused the spike

---

## RB-003: Database connection pool exhausted

**Severity:** P1 — service degraded / unavailable

### Symptoms
- API returns 503 errors: "could not connect to server"
- Logs: `asyncpg.exceptions.TooManyConnectionsError`
- Grafana: `pg_stat_activity.count` approaching `max_connections`

### Diagnosis

```bash
# Check active connections
kubectl exec -n authentiguard deploy/api -- \
  psql "$DATABASE_URL" -c "
    SELECT count(*), state, wait_event_type, wait_event
    FROM pg_stat_activity
    GROUP BY state, wait_event_type, wait_event
    ORDER BY count DESC;"

# Check idle connections (connection leaks)
kubectl exec -n authentiguard deploy/api -- \
  psql "$DATABASE_URL" -c "
    SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';"
```

### Resolution

**Immediate (terminate idle connections):**
```bash
kubectl exec -n authentiguard deploy/api -- \
  psql "$DATABASE_URL" -c "
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE state = 'idle'
    AND query_start < now() - interval '5 minutes';"
```

**If connection leaks detected (idle > 10 minutes):**
```bash
# Check SQLAlchemy pool settings
# Expected: pool_size=10, max_overflow=20 per API pod
# If 5 API pods × 30 connections = 150 → approaching max_connections=500
# Scale down API replicas temporarily or increase RDS max_connections
```

**Scale RDS max_connections (requires parameter group change):**
```bash
# In AWS Console: RDS → Parameter Groups → max_connections = 1000
# Restart RDS (brief downtime) or use a dynamic parameter group
```

### Post-incident
- Ensure `pool_pre_ping=True` is set in SQLAlchemy engine config
- Consider PgBouncer connection pooler if connections remain high

---

## RB-004: ML model loading failure at worker startup

**Severity:** P2

### Symptoms
- New worker pods failing to start
- Logs: `ModelLoadError: checkpoint not found at /app/model_cache/...`
- Jobs stuck in `pending` state with no workers picking them up

### Diagnosis

```bash
# Check worker pod logs
kubectl logs -n authentiguard -l app=worker-cpu --tail=50

# Check model cache PVC
kubectl get pvc model-cache-pvc -n authentiguard
kubectl describe pvc model-cache-pvc -n authentiguard

# Check if EFS mount is working
kubectl exec -n authentiguard deploy/worker-cpu -- ls /app/model_cache/
```

### Resolution

**If PVC not mounted (EFS CSI driver issue):**
```bash
# Check EFS CSI driver pods
kubectl get pods -n kube-system -l app=efs-csi-node

# If pods are not running, reinstall the driver:
helm upgrade -i aws-efs-csi-driver aws-efs-csi-driver/aws-efs-csi-driver \
  -n kube-system
```

**If model files are missing from the cache:**
```bash
# Manually trigger model download job
kubectl apply -f infra/k8s/base/jobs/download-models-job.yaml

# Or exec into a worker pod and run the download script:
kubectl exec -it -n authentiguard deploy/worker-cpu -- \
  python -m ai.text_detector.scripts.download_models --all
```

**Emergency fallback (use heuristic-only detection):**
```bash
# Set env var to skip transformer models and use heuristics only
kubectl set env deployment/worker-cpu -n authentiguard \
  USE_TRANSFORMER=false USE_HEURISTIC_FALLBACK=true
```

### Post-incident
- Add model pre-download as an init container to worker pods
- Store model checksums in ConfigMap; verify on startup before accepting tasks

---

## RB-005: JWT secret rotation

**Severity:** Planned maintenance (no incident)

### When to rotate
- Every 90 days (scheduled) or immediately if a secret leak is suspected

### Steps

```bash
# 1. Generate new JWT secret
NEW_SECRET=$(openssl rand -hex 64)

# 2. Update AWS Secrets Manager
aws secretsmanager update-secret \
  --secret-id "authentiguard/prod" \
  --secret-string "{\"JWT_SECRET_KEY\": \"$NEW_SECRET\"}"

# 3. Wait for External Secrets Operator to sync (5 min max, or force)
kubectl annotate externalsecret authentiguard-secrets -n authentiguard \
  force-sync=$(date +%s) --overwrite

# 4. Verify new secret propagated
kubectl get secret authentiguard-secrets -n authentiguard \
  -o jsonpath='{.data.JWT_SECRET_KEY}' | base64 -d | wc -c
# Should print 128 (64 bytes hex = 128 chars)

# 5. Rolling restart to pick up new secret
kubectl rollout restart deployment/api -n authentiguard
kubectl rollout status deployment/api -n authentiguard

# 6. All existing refresh tokens are now invalid — users must re-login
# This is expected and acceptable for a planned rotation.
# Notify users 24h in advance via status page if possible.
```

### Post-rotation
- Log the rotation date in the security runbook audit trail
- Verify login still works: `curl -X POST .../api/v1/auth/login ...`

---

## RB-006: Rollback a production deployment

**Severity:** P1 (triggered by deployment regression)

### Immediate rollback (< 60 seconds)

```bash
# Option A: Kubernetes rollback (fastest — restores previous ReplicaSet)
kubectl rollout undo deployment/api        -n authentiguard
kubectl rollout undo deployment/worker-cpu -n authentiguard
kubectl rollout undo deployment/frontend   -n authentiguard

# Verify rollback
kubectl rollout status deployment/api -n authentiguard
kubectl get pods -n authentiguard -o wide
```

### ArgoCD rollback (GitOps-aware)

```bash
# List history
argocd app history authentiguard-production

# Roll back to specific revision
argocd app rollback authentiguard-production <REVISION_ID>

# Sync to verify
argocd app sync authentiguard-production
```

### Image tag rollback (if rollback was to wrong revision)

```bash
# Get the previous image tag from ECR
aws ecr describe-images \
  --repository-name authentiguard-api \
  --query 'sort_by(imageDetails, &imagePushedAt)[-2].imageTags[0]' \
  --output text

PREV_TAG=<output from above>

# Update kustomization to previous tag and push to trigger ArgoCD
cd infra/k8s/overlays/production
kustomize edit set image "$ECR_REGISTRY/authentiguard-api=$ECR_REGISTRY/authentiguard-api:$PREV_TAG"
git add . && git commit -m "chore: rollback to $PREV_TAG" && git push
```

### Post-incident
- Create a post-mortem within 24 hours
- Add regression test for the failing scenario
- Update CI smoke tests to catch the issue earlier
