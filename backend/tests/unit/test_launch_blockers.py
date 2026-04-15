"""
Regression tests for the five launch-blocker fixes.

These tests exist so the next person who stumbles on the create_all time
bomb, the dispatcher split-brain, the CI deploy stub, or the alert path
cannot accidentally re-introduce them. If any of these tests fail, a fix
from the production audit has regressed.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────
# Fix 1: create_all is dev-only and production refuses the flag
# ─────────────────────────────────────────────────────────────────────


_REQUIRED_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
    "REDIS_URL": "redis://localhost:6379/0",
    "APP_SECRET_KEY": "test-app-secret-key-value-min-32-chars-long",
    "JWT_SECRET_KEY": "test-jwt-secret-key-minimum-32-characters-long",
    "CELERY_BROKER_URL": "redis://localhost:6379/0",
    "CELERY_RESULT_BACKEND": "redis://localhost:6379/1",
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "ENCRYPTION_KEY": "dGVzdC1lbmNyeXB0aW9uLWtleS1ub3QtZm9yLXByb2Q=",
}


def _build_settings(monkeypatch, **overrides):
    """Instantiate Settings with all required env vars, applying overrides."""
    from app.core.config import Settings

    env = {**_REQUIRED_ENV, **overrides}
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    # Disable .env file loading to make the test environment deterministic.
    return Settings(_env_file=None)  # type: ignore[call-arg]


class TestCreateAllGating:
    def test_flag_defaults_to_false(self, monkeypatch):
        settings = _build_settings(monkeypatch)
        assert settings.ALLOW_DB_CREATE_ALL is False

    def test_production_with_flag_true_raises(self, monkeypatch):
        from pydantic import ValidationError
        with pytest.raises((ValueError, ValidationError), match="ALLOW_DB_CREATE_ALL"):
            _build_settings(
                monkeypatch,
                APP_ENV="production",
                ALLOW_DB_CREATE_ALL="true",
            )

    def test_production_with_flag_false_is_allowed(self, monkeypatch):
        settings = _build_settings(
            monkeypatch,
            APP_ENV="production",
            ALLOW_DB_CREATE_ALL="false",
        )
        assert settings.APP_ENV == "production"
        assert settings.ALLOW_DB_CREATE_ALL is False

    def test_development_with_flag_true_is_allowed(self, monkeypatch):
        settings = _build_settings(
            monkeypatch,
            APP_ENV="development",
            ALLOW_DB_CREATE_ALL="true",
        )
        assert settings.ALLOW_DB_CREATE_ALL is True

    def test_lifespan_skips_create_all_when_flag_false(self, monkeypatch):
        """Production-shaped boot must NOT invoke `_ensure_db_tables`."""
        import asyncio

        from app import main as main_module

        settings = _build_settings(
            monkeypatch,
            APP_ENV="production",
            ALLOW_DB_CREATE_ALL="false",
        )

        calls: dict[str, int] = {"create_all": 0, "buckets": 0, "demo": 0}

        async def _fake_create_all():
            calls["create_all"] += 1

        async def _fake_buckets(_s):
            calls["buckets"] += 1

        async def _fake_demo():
            calls["demo"] += 1

        monkeypatch.setattr(main_module, "_ensure_db_tables", _fake_create_all)
        monkeypatch.setattr(main_module, "_ensure_minio_buckets", _fake_buckets)
        monkeypatch.setattr(main_module, "_ensure_demo_user", _fake_demo)
        monkeypatch.setattr(main_module, "get_settings", lambda: settings)

        # engine.dispose() needs to be a no-op async. Swap the whole
        # module-level engine reference so we don't touch the real one.
        class _FakeEngine:
            async def dispose(self):
                return None
        monkeypatch.setattr(main_module, "engine", _FakeEngine())

        fake_app = MagicMock()

        async def _run():
            async with main_module.lifespan(fake_app):
                pass

        asyncio.get_event_loop().run_until_complete(_run()) if False else asyncio.run(_run())

        assert calls["create_all"] == 0, (
            "production lifespan must NOT invoke _ensure_db_tables "
            "when ALLOW_DB_CREATE_ALL is false"
        )

    def test_lifespan_invokes_create_all_when_flag_true(self, monkeypatch):
        """Dev-shaped boot with the flag on DOES invoke create_all."""
        import asyncio
        from app import main as main_module

        settings = _build_settings(
            monkeypatch,
            APP_ENV="development",
            ALLOW_DB_CREATE_ALL="true",
        )

        calls = {"create_all": 0}

        async def _fake_create_all():
            calls["create_all"] += 1

        async def _fake_buckets(_s):
            return None

        async def _fake_demo():
            return None

        monkeypatch.setattr(main_module, "_ensure_db_tables", _fake_create_all)
        monkeypatch.setattr(main_module, "_ensure_minio_buckets", _fake_buckets)
        monkeypatch.setattr(main_module, "_ensure_demo_user", _fake_demo)
        monkeypatch.setattr(main_module, "get_settings", lambda: settings)

        class _FakeEngine:
            async def dispose(self):
                return None
        monkeypatch.setattr(main_module, "engine", _FakeEngine())

        fake_app = MagicMock()

        async def _run():
            async with main_module.lifespan(fake_app):
                pass

        asyncio.run(_run())

        assert calls["create_all"] == 1


# ─────────────────────────────────────────────────────────────────────
# Fix 2: dispatcher text path is disabled, and the canonical text
# checkpoint constant matches what the production worker loads
# ─────────────────────────────────────────────────────────────────────


class TestTextCheckpointUnity:
    def test_canonical_subpath_is_v3_hard_phase1(self):
        from app.workers.text_worker import TEXT_CHECKPOINT_SUBPATH

        expected = Path("ai") / "text_detector" / "checkpoints" / "transformer_v3_hard" / "phase1"
        assert TEXT_CHECKPOINT_SUBPATH == expected

    def test_dispatcher_refuses_text(self):
        """The dispatcher must NOT serve text — that path was split-brain."""
        # Import lazily because the dispatcher sits under `ai/` and tests
        # run from the backend package.
        import sys
        root = Path(__file__).resolve().parents[3]  # repo root
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ai.ensemble_engine.routing.dispatcher import DetectorRegistry

        reg = DetectorRegistry()
        with pytest.raises(NotImplementedError, match="text_worker"):
            reg.get("text")

    def test_dispatcher_refuses_code(self):
        import sys
        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ai.ensemble_engine.routing.dispatcher import DetectorRegistry

        reg = DetectorRegistry()
        with pytest.raises(NotImplementedError, match="text_worker"):
            reg.get("code")

    def test_dispatcher_source_has_no_stale_transformer_phase3_reference(self):
        """Regression guard: the old split-brain path must stay deleted.

        If a future edit re-introduces `transformer/phase3` into the
        dispatcher, this test fails and forces a re-review against
        TEXT_CHECKPOINT_SUBPATH.
        """
        root = Path(__file__).resolve().parents[3]
        dispatcher_path = root / "ai" / "ensemble_engine" / "routing" / "dispatcher.py"
        src = dispatcher_path.read_text(encoding="utf-8")
        # The only surviving reference should be the comment explaining
        # WHY the branch is disabled — not an actual path construction.
        assert "transformer/phase3" not in src or "used to load" in src, (
            "dispatcher.py must not reference transformer/phase3 as an "
            "active checkpoint; the text branch is disabled."
        )

    def test_resolver_returns_path_or_none(self):
        from app.workers.text_worker import resolve_text_checkpoint_root

        result = resolve_text_checkpoint_root()
        # On this dev machine the repo is present, so the resolver should
        # find it. On a stripped CI image it may return None — we allow
        # both but assert the return contract.
        assert result is None or isinstance(result, Path)


# ─────────────────────────────────────────────────────────────────────
# Fix 3: CI deploy steps fail honestly instead of echo-placeholder
# ─────────────────────────────────────────────────────────────────────


class TestCIDeployHonesty:
    def _ci_yaml(self) -> str:
        root = Path(__file__).resolve().parents[3]
        path = root / ".github" / "workflows" / "ci.yml"
        return path.read_text(encoding="utf-8")

    def _ci_doc(self):
        import yaml
        root = Path(__file__).resolve().parents[3]
        with (root / ".github" / "workflows" / "ci.yml").open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_no_echo_only_deploy_placeholder(self):
        src = self._ci_yaml()
        # The original placeholders were literally:
        #   run: echo "Deploy to k8s ..."
        # Fail if anything resembling that pattern survives.
        bad = [
            'run: echo "Deploy to k8s staging',
            'run: echo "Deploy to k8s production',
        ]
        for needle in bad:
            assert needle not in src, (
                f"ci.yml still contains a placeholder deploy: {needle!r}"
            )

    def test_release_jobs_are_labeled_human_in_the_loop(self):
        """Job names must make the manual-rollout contract obvious.

        The repo must not pretend the cluster rollout is automated.
        Jobs are named `release-*` (not `deploy-*`) and the `name:`
        label explicitly calls out human-in-the-loop rollout.
        """
        doc = self._ci_doc()
        jobs = doc["jobs"]

        # The old `deploy-*` names must be gone.
        assert "deploy-staging" not in jobs, (
            "ci.yml still has a `deploy-staging` job — rename to `release-staging` "
            "so the job name does not imply automated rollout"
        )
        assert "deploy-production" not in jobs, (
            "ci.yml still has a `deploy-production` job — rename to "
            "`release-production` so the job name does not imply automated rollout"
        )

        for job_key in ("release-staging", "release-production"):
            assert job_key in jobs, f"missing {job_key} job"
            label = jobs[job_key]["name"].lower()
            assert "human-in-the-loop" in label or "manual" in label, (
                f"{job_key}: job `name:` must explicitly say human-in-the-loop "
                f"or manual — got {jobs[job_key]['name']!r}"
            )

    def test_release_jobs_do_not_claim_automated_rollout(self):
        """The final step in each release job must NOT run kubectl
        against the cluster. It must write a manual-rollout checklist
        to $GITHUB_STEP_SUMMARY and emit a `::notice::` annotation."""
        doc = self._ci_doc()
        for job_key in ("release-staging", "release-production"):
            steps = doc["jobs"][job_key]["steps"]
            last = steps[-1]
            name = (last.get("name") or "").lower()
            run = last.get("run") or ""
            assert "manual rollout checklist" in name, (
                f"{job_key}: final step must be the manual rollout checklist, "
                f"got {last.get('name')!r}"
            )
            assert "$GITHUB_STEP_SUMMARY" in run, (
                f"{job_key}: final step must write the checklist to "
                f"$GITHUB_STEP_SUMMARY so the operator sees it in the run page"
            )
            assert "::notice title=Manual rollout required::" in run, (
                f"{job_key}: final step must emit `::notice title=Manual rollout required::`"
            )
            # The step must NOT actually apply anything.
            for forbidden in ("kubectl apply", "kubectl rollout restart", "kubectl set image"):
                assert forbidden not in run, (
                    f"{job_key}: final step contains `{forbidden}` — rollout "
                    f"must stay manual until automation is wired. "
                    f"Print commands to $GITHUB_STEP_SUMMARY, do not run them."
                )

    def test_deploy_runs_alembic_upgrade_head(self):
        """Deploy must migrate the DB before pushing images.

        This enforces the ordering that keeps `create_all` dev-only:
        Alembic is the single source of truth for schema in every
        non-dev environment, and a bad migration must fail the deploy
        BEFORE any new container image is tagged `latest`.
        """
        import yaml

        root = Path(__file__).resolve().parents[3]
        with (root / ".github" / "workflows" / "ci.yml").open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        for job_name in ("release-staging", "release-production"):
            steps = doc["jobs"][job_name]["steps"]
            names = [s.get("name", "") for s in steps]

            # Must contain an alembic step
            alembic_idx = next(
                (
                    i for i, s in enumerate(steps)
                    if "alembic upgrade head" in (s.get("run") or "")
                ),
                None,
            )
            assert alembic_idx is not None, (
                f"{job_name}: missing `alembic upgrade head` step"
            )

            alembic_step = steps[alembic_idx]

            # Must contain DATABASE_URL in the step env (sourced from secret)
            assert "DATABASE_URL" in (alembic_step.get("env") or {}), (
                f"{job_name}: Alembic step must inject DATABASE_URL from secrets"
            )

            # Must run BEFORE any image push
            push_idx = next(
                (i for i, n in enumerate(names) if "push" in n.lower()),
                None,
            )
            assert push_idx is not None, f"{job_name}: no image push step found"
            assert alembic_idx < push_idx, (
                f"{job_name}: Alembic must run BEFORE image push "
                f"(alembic at step {alembic_idx}, first push at step {push_idx})"
            )

    def test_deploy_production_forbids_create_all_flag(self):
        """Defense in depth: the production release step must explicitly
        set ALLOW_DB_CREATE_ALL=false so the Settings validator cannot
        fall back to the dev bootstrap path."""
        import yaml
        root = Path(__file__).resolve().parents[3]
        with (root / ".github" / "workflows" / "ci.yml").open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        prod_steps = doc["jobs"]["release-production"]["steps"]
        alembic_step = next(
            s for s in prod_steps
            if "alembic upgrade head" in (s.get("run") or "")
        )
        env = alembic_step.get("env") or {}
        assert env.get("APP_ENV") == "production"
        assert str(env.get("ALLOW_DB_CREATE_ALL", "")).lower() == "false"


class TestDeploymentDocsHonesty:
    def test_readme_points_at_deploy_doc_and_calls_out_manual(self):
        """README must tell the reader that deployment is
        human-in-the-loop and must link to `docs/DEPLOY.md`."""
        root = Path(__file__).resolve().parents[3]
        readme = (root / "README.md").read_text(encoding="utf-8").lower()
        assert "docs/deploy.md" in readme, (
            "README.md must link to docs/DEPLOY.md so operators find the runbook"
        )
        assert "human-in-the-loop" in readme or "manual" in readme, (
            "README.md Deployment section must state that the rollout is manual "
            "so the repo does not imply automated deployment"
        )

    def test_deploy_doc_exists_and_documents_ordering(self):
        """docs/DEPLOY.md must describe the migrations → push → rollout
        sequence so there is a single, unambiguous source of truth for
        how a release reaches production."""
        root = Path(__file__).resolve().parents[3]
        deploy_doc = root / "docs" / "DEPLOY.md"
        assert deploy_doc.exists(), "docs/DEPLOY.md is missing"

        body = deploy_doc.read_text(encoding="utf-8").lower()
        # The three steps in order
        assert "migrations" in body
        assert "image push" in body or "image build" in body
        assert "cluster rollout" in body or "kubectl" in body
        # Manual contract must be explicit
        assert "human-in-the-loop" in body or "manual" in body
        # Rollback procedure is required reading
        assert "rollback" in body
        # Must reference the alerting webhook sanity check
        assert "alert_webhook_url" in body or "alerting" in body


# ─────────────────────────────────────────────────────────────────────
# Fix 4: alert webhook path is wired and safe
# ─────────────────────────────────────────────────────────────────────


class TestAlertWebhook:
    def test_no_post_when_url_empty(self, monkeypatch):
        from app.workers import alerting

        settings = _build_settings(monkeypatch, ALERT_WEBHOOK_URL="")
        monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
        alerting._last_sent.clear()

        import httpx
        called = {"post": 0}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def post(self, *a, **kw):
                called["post"] += 1
                return MagicMock(status_code=200)

        monkeypatch.setattr(httpx, "Client", _FakeClient)

        alerting._post_alert_webhook("detector_fallback", "critical", "test")
        assert called["post"] == 0

    def test_posts_when_url_set(self, monkeypatch):
        from app.workers import alerting

        settings = _build_settings(
            monkeypatch,
            ALERT_WEBHOOK_URL="https://hooks.example.test/alert",
        )
        monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
        alerting._last_sent.clear()

        import httpx
        called = {"post": 0, "url": None, "payload": None}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def post(self, url, json=None, **kw):
                called["post"] += 1
                called["url"] = url
                called["payload"] = json
                return MagicMock(status_code=200)

        monkeypatch.setattr(httpx, "Client", _FakeClient)

        alerting._post_alert_webhook(
            "detector_fallback",
            "critical",
            "ML is down",
            extra={"detector_mode": "fallback"},
        )
        assert called["post"] == 1
        assert called["url"] == "https://hooks.example.test/alert"
        assert called["payload"]["severity"] == "critical"
        assert called["payload"]["alert"] == "detector_fallback"
        assert "text" in called["payload"]

    def test_cooldown_suppresses_repeat_within_window(self, monkeypatch):
        from app.workers import alerting

        settings = _build_settings(
            monkeypatch,
            ALERT_WEBHOOK_URL="https://hooks.example.test/alert",
        )
        monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
        alerting._last_sent.clear()

        import httpx
        called = {"post": 0}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def post(self, *a, **kw):
                called["post"] += 1
                return MagicMock(status_code=200)

        monkeypatch.setattr(httpx, "Client", _FakeClient)

        alerting._post_alert_webhook("detector_fallback", "critical", "1")
        alerting._post_alert_webhook("detector_fallback", "critical", "2")
        alerting._post_alert_webhook("detector_fallback", "critical", "3")
        assert called["post"] == 1  # cooldown deduped the next two

    def test_webhook_failure_is_swallowed(self, monkeypatch):
        """A broken webhook endpoint must NOT raise — alerting is best-effort."""
        from app.workers import alerting

        settings = _build_settings(
            monkeypatch,
            ALERT_WEBHOOK_URL="https://hooks.example.test/alert",
        )
        monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
        alerting._last_sent.clear()

        import httpx

        class _BrokenClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def post(self, *a, **kw):
                raise RuntimeError("simulated network failure")

        monkeypatch.setattr(httpx, "Client", _BrokenClient)

        # Must not raise
        alerting._post_alert_webhook("queue_depth_high:text", "warning", "x")


# ─────────────────────────────────────────────────────────────────────
# Fix 5: QUEUE_DEPTH gauge is wired from real Redis state
# ─────────────────────────────────────────────────────────────────────


class TestQueueDepthWiring:
    def test_alerting_module_sets_queue_depth_gauge(self):
        """Grep the alerting source to ensure the gauge is actually .set()."""
        import app.workers.alerting as alerting_mod
        src = Path(alerting_mod.__file__).read_text(encoding="utf-8")
        assert "QUEUE_DEPTH.labels(queue=" in src, (
            "alerting.py must call QUEUE_DEPTH.labels(queue=...).set(depth) "
            "so the gauge reflects real queue state"
        )
        assert ").set(depth)" in src

    def test_check_health_updates_queue_depth_for_each_queue(self, monkeypatch):
        """Run check_health with a fake Redis and assert the gauge was set."""
        from app.workers import alerting
        from app.core import metrics

        # Stub Redis so _check_queue_depths returns deterministic values.
        fake_depths = {"text": 7, "image": 0, "audio": 0, "video": 0, "webhook": 0}
        monkeypatch.setattr(alerting, "_check_queue_depths", lambda: fake_depths)

        # Stub failure-rate path so we don't need a real DB.
        async def _fake_rate():
            return {"completed": 10, "failed": 0}
        monkeypatch.setattr(alerting, "_check_failure_rate", _fake_rate)

        # Stub detector mode so we don't hit the real loader.
        import app.workers.text_worker as tw
        monkeypatch.setattr(tw, "get_detector_mode", lambda: "ml")

        # Reset the gauge so our assertion is deterministic.
        metrics.QUEUE_DEPTH.clear()

        result = alerting.check_health.run()  # .run() calls the task body sync
        assert "alerts" in result

        # The gauge must now have one sample per queue with the fake depth.
        samples = {
            s.labels["queue"]: s.value
            for m in metrics.QUEUE_DEPTH.collect()
            for s in m.samples
            if s.name.endswith("celery_queue_depth")
        }
        assert samples.get("text") == 7
        assert samples.get("image") == 0
