# Deployment Runbook

**Rollout is human-in-the-loop.** CI prepares the release artifacts
(migrations, container images) automatically. The cluster-apply step
is a manual operation performed by an on-call operator.

This document is the single source of truth for what the operator does
and in what order.

---

## Why is cluster apply manual?

Wiring a fully-automated rollout requires architecture that does not
exist in this repo today:

- AWS OIDC credentials in GitHub Actions (for `aws eks update-kubeconfig`)
- ECR push (CI currently pushes to GHCR)
- Kustomize image substitution for the `${ECR_REGISTRY}/...` placeholders
  in `infra/k8s/overlays/production/kustomization.yaml`
- `kubectl rollout status` polling with proper failure handling
- A rollback path that understands partial failures across three Deployments

Until that work is scoped and landed, the repo does not pretend to do
any of it. CI does the parts that are safe to automate (migrations,
image builds) and leaves the cluster mutation to a human with the
right credentials and situational awareness.

---

## Deploy order (always the same)

1. **Migrations** — `alembic upgrade head` against the target DB.
2. **Image push** — new container images to the registry.
3. **Cluster rollout** — `kubectl` against the target EKS cluster.

**Never reorder.** Images that require a new column must never reach
pods before the migration that adds the column. This ordering is
enforced by the CI workflow for steps 1 and 2; the operator is
responsible for not running step 3 if step 1 or step 2 failed.

---

## What CI does for you

Every push to `main` runs the `release-production` job, which on
success means:

- ✅ `alembic upgrade head` was executed against the production database
  using the `DATABASE_URL` secret in the GitHub `production` Environment.
- ✅ `ghcr.io/<owner>/authentiguard-backend:<git-sha>` and `:latest`
  were built and pushed.
- ✅ `ghcr.io/<owner>/authentiguard-frontend:<git-sha>` and `:latest`
  were built and pushed.
- ✅ The Actions run summary contains a copy-pasteable rollout
  checklist for the exact commit you just pushed.

The `release-staging` job does the equivalent for `develop` pushes
against the staging environment.

If the migration or the push fails, the job fails and **no rollout
should happen**. Fix the underlying issue and re-run.

---

## Operator prerequisites (one-time per workstation)

You need:

1. **AWS CLI** configured with a role that can read the EKS cluster:
   ```bash
   aws sso login --profile authentiguard-prod
   aws eks update-kubeconfig \
     --name authentiguard-prod \
     --region us-east-1 \
     --profile authentiguard-prod
   ```

2. **kubectl** ≥ 1.28 and **kustomize** ≥ 5.0.

3. **Read access to the GHCR registry** from the cluster. The
   `image-pull-secret` is managed by the External Secrets Operator
   (see `infra/k8s/base/namespace.yaml`); verify with:
   ```bash
   kubectl -n authentiguard get secret ghcr-pull-secret -o name
   ```
   (If this is missing, the K8s manifests still reference the legacy
   ECR placeholders — see "Known gap" below.)

4. **Access to the `production` GitHub Environment** so you can read
   the CI run summary for the SHA you are about to roll out.

---

## Pre-rollout checklist

Before you run any `kubectl` command:

- [ ] The `release-production` (or `release-staging`) job for the
      commit you intend to roll out is **green**.
- [ ] You have opened the job's run page and can see the "Manual
      rollout checklist" summary with the exact commit SHA.
- [ ] `kubectl config current-context` points at the right cluster.
      Double-check; this is the single most common footgun.
- [ ] No other operator is rolling out the same services. Check
      `#deploys` (or your team's channel).
- [ ] You have a terminal window open with a rollback command ready
      to paste.

If any of these is false, **stop**.

---

## Production rollout

Copy the exact commands from the Actions run summary for your commit.
They look like this, with `<sha>` filled in:

```bash
# 1. Point kubectl at prod
aws eks update-kubeconfig --name authentiguard-prod --region us-east-1
kubectl config current-context   # must show the prod cluster

# 2. Verify migrations landed (CI already ran them; this is defence-in-depth)
kubectl -n authentiguard run --rm -it --restart=Never alembic-check \
  --image=ghcr.io/<owner>/authentiguard-backend:<sha> -- \
  alembic current

# 3. Roll out the new images
kubectl -n authentiguard set image deploy/api         api=ghcr.io/<owner>/authentiguard-backend:<sha>
kubectl -n authentiguard set image deploy/worker-cpu  worker=ghcr.io/<owner>/authentiguard-backend:<sha>
kubectl -n authentiguard set image deploy/frontend    frontend=ghcr.io/<owner>/authentiguard-frontend:<sha>

# 4. Wait for rollouts (they run in parallel)
kubectl -n authentiguard rollout status deploy/api         --timeout=10m
kubectl -n authentiguard rollout status deploy/worker-cpu  --timeout=10m
kubectl -n authentiguard rollout status deploy/frontend    --timeout=10m
```

**Do not move on until all three `rollout status` commands return
`successfully rolled out`.**

---

## Post-rollout smoke test

Takes ~60 seconds. If any of these fails, roll back (next section).

```bash
# Health endpoint — must return 200 with all checks "ok"
kubectl -n authentiguard exec deploy/api -- \
  curl -sf http://localhost:8000/api/v1/health | grep '"status": "ok"'

# Detector must NOT be in fallback mode
kubectl -n authentiguard exec deploy/api -- \
  curl -sf http://localhost:8000/metrics | grep '^detector_fallback_active 0'

# Celery workers must be reachable
kubectl -n authentiguard exec deploy/worker-cpu -- \
  celery -A app.workers.celery_app inspect ping --timeout 10

# Alerting webhook must be configured (empty string = alerts invisible externally)
kubectl -n authentiguard exec deploy/worker-cpu -- \
  python -c "from app.core.config import get_settings; import sys; sys.exit(0 if get_settings().ALERT_WEBHOOK_URL else 1)"
```

Then hit the actual product:

```bash
curl -sf https://api.authentiguard.io/api/v1/health | jq .
curl -sf https://api.authentiguard.io/metrics | grep detector_fallback_active
```

---

## Rollback

If any rollout fails, any smoke test fails, or anything looks wrong,
**roll back first and diagnose second**:

```bash
kubectl -n authentiguard rollout undo deploy/api
kubectl -n authentiguard rollout undo deploy/worker-cpu
kubectl -n authentiguard rollout undo deploy/frontend

# Watch them converge back to the previous ReplicaSet
kubectl -n authentiguard rollout status deploy/api         --timeout=10m
kubectl -n authentiguard rollout status deploy/worker-cpu  --timeout=10m
kubectl -n authentiguard rollout status deploy/frontend    --timeout=10m
```

**Schema rollback is not automatic.** Alembic migrations already ran
before the rollout. If the new migration is the problem, generate and
review a downgrade migration carefully — do not `alembic downgrade -1`
against production without reading the generated SQL first. In most
cases it is safer to roll the code back to a version that tolerates
the new schema and fix-forward with a new migration.

Post-incident, file a retro in `docs/guides/runbooks.md` so the next
operator learns from the failure mode.

---

## Staging rollout

Identical to production except:

- Cluster context: `authentiguard-staging` (not `-prod`)
- Image tag: `:staging` or the commit SHA from the `release-staging` run
- You can skip the pre-rollout checklist's "no other operator" step if
  staging is for your exclusive use at the time
- Post-rollout smoke test should additionally exercise a real
  `/api/v1/analyze/text` submission end-to-end

---

## Known gap: Kustomize image placeholders

The `infra/k8s/overlays/production/kustomization.yaml` file still
contains the legacy ECR placeholders:

```yaml
images:
  - name: "${ECR_REGISTRY}/authentiguard-api"
    newTag: "${IMAGE_TAG}"
```

These are **not substituted by CI** (CI does not run `kubectl apply -k`
at all). The rollout commands above use `kubectl set image` with the
full GHCR image reference, which sidesteps the issue entirely — the
Deployment's image field is updated in place.

**Do not** run `kubectl apply -k infra/k8s/overlays/production`
without first replacing those placeholders by hand or via `kustomize
edit set image`. Otherwise you will apply literal `${ECR_REGISTRY}`
strings to the cluster and Pods will fail to pull.

Closing this gap (making `kubectl apply -k` the canonical rollout
path) is what "fully automated rollout" requires, and it is tracked
as a separate piece of work — see the top of this document.

---

## Alerting sanity check

Before you declare a rollout "done":

1. Open the run summary in Actions for the commit you just rolled out.
2. Confirm the "Manual rollout checklist" step shows the commit SHA
   that matches `kubectl -n authentiguard get deploy/api -o
   jsonpath='{.spec.template.spec.containers[0].image}'`.
3. Confirm a test alert reaches the team channel:
   ```bash
   kubectl -n authentiguard exec deploy/worker-cpu -- python -c "
   from app.workers.alerting import _post_alert_webhook
   _post_alert_webhook('detector_fallback', 'critical', 'post-rollout smoke test')
   "
   ```
   A message should appear in the channel within a few seconds. If it
   does not, `ALERT_WEBHOOK_URL` is unset or the endpoint is down —
   treat this as a P2 incident and page the on-call. Alerting is the
   only external signal you have until Prometheus is deployed.

---

## Summary

| Step | Who | How | When to stop |
|------|-----|-----|--------------|
| 1. Migrations | CI (`release-production`) | `alembic upgrade head` | If the CI job is red |
| 2. Image push | CI (`release-production`) | `docker buildx build --push` | If the CI job is red |
| 3. Cluster rollout | **Operator (manual)** | `kubectl set image` + `rollout status` | If any rollout fails or any smoke check fails → rollback |

If you are reading this document because you are about to do step 3:
the commands you need are in the Actions run summary for your commit.
Read them from there, not from this document — the SHA is already
filled in for you.
