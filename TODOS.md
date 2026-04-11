# TODOS

## P2: Accuracy Benchmark Dashboard
Run detectors against standard datasets (RAID for text, FakeAVCeleb for video) and publish results on landing page. Honest public benchmarks build trust in a market full of inflated claims.

**Effort:** M (human) → S (CC+gstack)
**Priority:** P2
**Depends on:** DeBERTa model training (heuristic detector numbers not representative)
**Context:** Deferred during CEO review 2026-04-08. No competitor publishes honest benchmarks against paraphrased content. This is a differentiator once the real model is trained.

## P1: Migrate to Alembic-Only Migrations
Replace `Base.metadata.create_all` in `main.py` lifespan with Alembic migrations. Currently runs on every startup — safe for initial launch but will cause schema drift when columns or constraints change. Must happen before the first post-launch schema change.

**Effort:** M (human) → S (CC+gstack)
**Priority:** P1
**Depends on:** First schema change needed
**Context:** Flagged during CEO review 2026-04-11. Outside voice confirmed: create_all + Alembic = schema drift time bomb.

## P2: Demo User Job Cleanup Cron
Add a Celery Beat periodic task (daily) that deletes demo user jobs older than 24 hours. Anonymous demo requests create unbounded jobs — no cleanup currently exists.

**Effort:** S (human) → S (CC+gstack)
**Priority:** P2
**Depends on:** Nothing
**Context:** Flagged during CEO review 2026-04-11. Demo user UUID shared by all anonymous visitors.

## P2: Split Celery Workers by Modality
Run a dedicated text-only worker and a separate media worker (image/audio/video). Currently one worker handles all 5 queues with concurrency=4. A single video task (6 min) blocks 25% of capacity. Text (the primary product) shouldn't compete with experimental modalities for worker slots.

**Effort:** M (human) → S (CC+gstack)
**Priority:** P2
**Depends on:** Observing actual traffic patterns post-launch
**Context:** Flagged by outside voice during CEO review 2026-04-11.
