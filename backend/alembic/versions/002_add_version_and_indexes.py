"""Add version column for optimistic locking + composite indexes.

Revision ID: 002
Revises: 001
Create Date: 2026-04-03
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Optimistic locking: version column on detection_jobs
    op.add_column(
        "detection_jobs",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )

    # Correlation ID for request tracing
    op.add_column(
        "detection_jobs",
        sa.Column("correlation_id", sa.String(64), nullable=True),
    )

    # Composite indexes for dashboard and history queries
    op.create_index(
        "ix_detection_jobs_user_status",
        "detection_jobs",
        ["user_id", "status"],
    )
    op.create_index(
        "ix_detection_jobs_user_created",
        "detection_jobs",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_detection_jobs_user_created")
    op.drop_index("ix_detection_jobs_user_status")
    op.drop_column("detection_jobs", "correlation_id")
    op.drop_column("detection_jobs", "version")
