"""
Unit tests for Celery configuration — queues, priorities, mappings.
"""

from __future__ import annotations

import pytest


class TestCeleryQueues:
    def test_content_type_to_queue_mapping(self):
        from app.workers.celery_app import CONTENT_TYPE_TO_QUEUE
        assert CONTENT_TYPE_TO_QUEUE["text"] == "text"
        assert CONTENT_TYPE_TO_QUEUE["code"] == "text"  # code uses text queue
        assert CONTENT_TYPE_TO_QUEUE["image"] == "image"
        assert CONTENT_TYPE_TO_QUEUE["audio"] == "audio"
        assert CONTENT_TYPE_TO_QUEUE["video"] == "video"

    def test_tier_to_priority_mapping(self):
        from app.workers.celery_app import TIER_TO_PRIORITY
        assert TIER_TO_PRIORITY["free"] == 1
        assert TIER_TO_PRIORITY["pro"] == 5
        assert TIER_TO_PRIORITY["enterprise"] == 9
        # Enterprise > Pro > Free
        assert TIER_TO_PRIORITY["enterprise"] > TIER_TO_PRIORITY["pro"]
        assert TIER_TO_PRIORITY["pro"] > TIER_TO_PRIORITY["free"]

    def test_celery_app_config(self):
        from app.workers.celery_app import celery_app
        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.task_acks_late is True
        assert celery_app.conf.task_reject_on_worker_lost is True
        assert celery_app.conf.worker_prefetch_multiplier == 1
        assert celery_app.conf.timezone == "UTC"

    def test_five_queues_defined(self):
        from app.workers.celery_app import celery_app
        queues = celery_app.conf.task_queues
        assert "text" in queues
        assert "image" in queues
        assert "audio" in queues
        assert "video" in queues
        assert "webhook" in queues

    def test_timeout_settings(self):
        from app.workers.celery_app import celery_app
        assert celery_app.conf.task_soft_time_limit == 120
        assert celery_app.conf.task_time_limit == 180
