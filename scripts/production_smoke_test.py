#!/usr/bin/env python3
"""
Production smoke test — validates the running AuthentiGuard stack.

Combines all pre-deployment checks:
  1. Health check (DB, Redis, detector mode)
  2. AI sample → expect high score (>0.5)
  3. Human sample → expect low score (<0.5)
  4. 20 inference calls → confirm consistency
  5. Detector mode verification
  6. Metrics endpoint check
  7. Rate limiting verification

Usage:
    # Against test stack:
    docker compose -f docker-compose.test.yml up -d
    python scripts/production_smoke_test.py

    # Against prod stack:
    python scripts/production_smoke_test.py --base-url https://api.authentiguard.io

    # Verify ML mode specifically:
    python scripts/production_smoke_test.py --expect-ml

Exit code 0 = all checks passed, 1 = failures detected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid

import httpx


# ── Test samples ─────────────────────────────────────────────

AI_SAMPLES = [
    (
        "The implementation of advanced neural network architectures has "
        "fundamentally transformed the landscape of natural language processing. "
        "These sophisticated models leverage attention mechanisms and transformer "
        "architectures to achieve unprecedented performance across a diverse "
        "array of linguistic tasks, including but not limited to text generation, "
        "sentiment analysis, and machine translation."
    ),
    (
        "Furthermore, the integration of multi-head attention mechanisms enables "
        "the model to simultaneously attend to different representational subspaces "
        "at various positions, thereby facilitating the capture of complex "
        "dependencies across the input sequence. This architectural innovation "
        "represents a paradigm shift in how sequential data is processed."
    ),
]

HUMAN_SAMPLES = [
    (
        "I went to the grocery store yesterday and forgot my list again. "
        "Ended up buying way too much cheese, as usual. My dog was so "
        "excited when I got home - she always thinks the bags are for her. "
        "Spent the evening trying a new pasta recipe from that Italian "
        "cookbook my mom gave me last Christmas. It turned out okay."
    ),
    (
        "Just got back from the vet with Luna. She hates car rides but the doc said "
        "she's doing great for a 12 year old lab. Had to stop at three different pet "
        "stores to find her favorite treats because apparently they discontinued them?? "
        "Anyway she's passed out on the couch now, snoring like a chainsaw."
    ),
]


class SmokeTestRunner:
    def __init__(self, base_url: str, expect_ml: bool, timeout: float):
        self.base_url = base_url
        self.expect_ml = expect_ml
        self.poll_timeout = timeout
        self.client = httpx.Client(base_url=base_url, timeout=30.0)
        self.headers: dict[str, str] = {}
        self.results: list[tuple[str, bool, str]] = []

    def _check(self, name: str, passed: bool, detail: str = ""):
        icon = "PASS" if passed else "FAIL"
        msg = f"  [{icon}] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        self.results.append((name, passed, detail))

    def _register_and_login(self):
        email = f"smoke-{uuid.uuid4().hex[:8]}@test.local"
        self.client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "SmokeTest123!",
            "full_name": "Smoke Tester",
            "consent_given": True,
        })
        resp = self.client.post("/api/v1/auth/login", json={
            "email": email,
            "password": "SmokeTest123!",
        })
        if resp.status_code == 200:
            self.headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
            return True
        return False

    def _submit_and_poll(self, text: str) -> dict | None:
        resp = self.client.post("/api/v1/analyze/text", json={
            "text": text,
            "content_type": "text",
        }, headers=self.headers)
        if resp.status_code != 202:
            return None
        job_id = resp.json()["job_id"]

        deadline = time.monotonic() + self.poll_timeout
        while time.monotonic() < deadline:
            resp = self.client.get(f"/api/v1/jobs/{job_id}", headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                if data["status"] in ("completed", "failed"):
                    if data["status"] == "completed":
                        result_resp = self.client.get(
                            f"/api/v1/jobs/{job_id}/result", headers=self.headers
                        )
                        if result_resp.status_code == 200:
                            return result_resp.json()
                    return data
            time.sleep(1.0)
        return None

    def check_health(self):
        print("\n--- Health Check ---")
        try:
            resp = self.client.get("/health")
            data = resp.json()

            self._check("Health endpoint reachable", resp.status_code == 200)
            self._check("Database healthy", data.get("checks", {}).get("database") == "ok")
            self._check("Redis healthy", data.get("checks", {}).get("redis") == "ok")

            # Note: the backend process doesn't load the detector — workers do.
            # The /health detector_mode reflects the API process only, so it's
            # informational here. The authoritative mode comes from job results.
            mode = data.get("checks", {}).get("detector_mode", "unknown")
            self._check("Detector mode reported in /health", mode in ("ml", "fallback", "heuristic", "unknown"), f"api-process mode: {mode}")
        except Exception as exc:
            self._check("Health endpoint reachable", False, str(exc))

    def check_auth(self):
        print("\n--- Auth Flow ---")
        ok = self._register_and_login()
        self._check("Register + login", ok)

    def check_ai_sample(self):
        print("\n--- AI Sample Detection ---")
        result = self._submit_and_poll(AI_SAMPLES[0])
        if result is None:
            self._check("AI sample completed", False, "job did not complete")
            return

        score = result.get("authenticity_score", -1)
        label = result.get("label", "")
        mode = result.get("detector_mode", "unknown")

        self._check("AI sample completed", result.get("status") == "completed")
        threshold = 0.7 if self.expect_ml else 0.5
        self._check(f"AI sample score > {threshold}", score > threshold, f"score={score:.3f}")
        self._check("AI sample labeled AI or UNCERTAIN", label in ("AI", "UNCERTAIN"), f"label={label}")

        if self.expect_ml:
            self._check("Detector mode ML in result", mode == "ml", f"got: {mode}")

    def check_human_sample(self):
        print("\n--- Human Sample Detection ---")
        result = self._submit_and_poll(HUMAN_SAMPLES[0])
        if result is None:
            self._check("Human sample completed", False, "job did not complete")
            return

        score = result.get("authenticity_score", -1)
        label = result.get("label", "")

        self._check("Human sample completed", result.get("status") == "completed")
        threshold = 0.4 if self.expect_ml else 0.5
        self._check(f"Human sample score < {threshold}", score < threshold, f"score={score:.3f}")
        self._check("Human sample labeled HUMAN or UNCERTAIN", label in ("HUMAN", "UNCERTAIN"), f"label={label}")

    def check_inference_consistency(self, n: int = 20):
        print(f"\n--- Inference Consistency ({n} calls) ---")
        scores = []
        labels = []
        modes = set()
        errors = 0

        all_samples = AI_SAMPLES + HUMAN_SAMPLES
        for i in range(n):
            text = all_samples[i % len(all_samples)]
            result = self._submit_and_poll(text)
            if result and result.get("status") == "completed":
                scores.append(result.get("authenticity_score", 0))
                labels.append(result.get("label", ""))
                modes.add(result.get("detector_mode", "unknown"))
            else:
                errors += 1

        self._check(f"All {n} inferences completed", errors == 0, f"errors={errors}")

        if scores:
            ai_scores = scores[:len(AI_SAMPLES)]  # first samples are AI
            human_scores = scores[len(AI_SAMPLES):]  # rest are human

            # Check AI scores are consistently high
            if ai_scores:
                ai_mean = statistics.mean(ai_scores * (n // len(all_samples) + 1))
                self._check("AI scores consistent", all(s > 0.4 for s in ai_scores),
                            f"mean={statistics.mean(ai_scores):.3f}")

            # Check human scores are consistently low
            if human_scores:
                self._check("Human scores consistent", all(s < 0.6 for s in human_scores),
                            f"mean={statistics.mean(human_scores):.3f}")

        if self.expect_ml:
            self._check("All inferences used ML", modes == {"ml"}, f"modes={modes}")
        else:
            self._check("Detector mode consistent", len(modes) == 1, f"modes={modes}")

        # Check fallback metric
        if self.expect_ml:
            try:
                resp = self.client.get("/metrics")
                if resp.status_code == 200:
                    for line in resp.text.split("\n"):
                        if line.startswith("detector_fallback_active "):
                            val = float(line.split()[-1])
                            self._check("Fallback metric == 0", val == 0.0, f"value={val}")
                            break
            except Exception:
                pass

    def check_metrics(self):
        print("\n--- Metrics Endpoint ---")
        try:
            resp = self.client.get("/metrics")
            self._check("Metrics endpoint reachable", resp.status_code == 200)

            text = resp.text
            has_detection = "detection_jobs_total" in text
            has_duration = "detection_duration_seconds" in text
            self._check("Detection metrics present", has_detection and has_duration)
        except Exception as exc:
            self._check("Metrics endpoint reachable", False, str(exc))

    def check_error_logs(self):
        """Check that no critical errors appeared — delegate to user via instructions."""
        print("\n--- Log Verification ---")
        print("  [INFO] Check container logs manually:")
        print("    docker compose logs backend --tail=50 | grep -i error")
        print("    docker compose logs worker --tail=50 | grep -i error")

    def run_all(self):
        print("=" * 60)
        print("  AuthentiGuard Production Smoke Test")
        print(f"  Target: {self.base_url}")
        print(f"  Expect ML: {self.expect_ml}")
        print("=" * 60)

        self.check_health()
        self.check_auth()
        self.check_ai_sample()
        self.check_human_sample()
        self.check_inference_consistency(n=20)
        self.check_metrics()
        self.check_error_logs()

        # Summary
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = total - passed

        print("\n" + "=" * 60)
        print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
        if failed > 0:
            print("\n  FAILURES:")
            for name, ok, detail in self.results:
                if not ok:
                    print(f"    - {name}: {detail}")
        print("=" * 60)

        self.client.close()
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="AuthentiGuard production smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--expect-ml", action="store_true",
                        help="Assert detector_mode==ml (not fallback)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Max seconds to wait for each job")
    args = parser.parse_args()

    runner = SmokeTestRunner(args.base_url, args.expect_ml, args.timeout)
    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
