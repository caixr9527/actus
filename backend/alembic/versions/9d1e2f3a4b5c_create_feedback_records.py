"""create feedback records

Revision ID: 9d1e2f3a4b5c
Revises: 8c9d0e1f2a3b
Create Date: 2026-05-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "9d1e2f3a4b5c"
down_revision: Union[str, Sequence[str], None] = "8c9d0e1f2a3b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback_records",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=True),
        sa.Column("feedback_scope_kind", sa.String(length=64), nullable=False),
        sa.Column("scope_id", sa.String(length=255), nullable=False),
        sa.Column("source_run_id", sa.String(length=255), nullable=True),
        sa.Column("target_run_id", sa.String(length=255), nullable=True),
        sa.Column("step_id", sa.String(length=255), nullable=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=64), nullable=False),
        sa.Column("source_kind", sa.String(length=64), nullable=False),
        sa.Column("source_event_id", sa.String(length=255), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=255), nullable=False),
        sa.Column("target_revision_id", sa.String(length=255), nullable=True),
        sa.Column("target_content_hash", sa.String(length=255), nullable=True),
        sa.Column("feedback_key", sa.String(length=255), nullable=False),
        sa.Column("dedupe_key", sa.String(length=255), nullable=False),
        sa.Column("reason_code", sa.String(length=255), nullable=False),
        sa.Column("resolution_reason_code", sa.String(length=255), nullable=True),
        sa.Column("decay_policy", sa.String(length=255), nullable=False),
        sa.Column("ttl_scope", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("origin", sa.String(length=64), nullable=False),
        sa.Column("trust_level", sa.String(length=64), nullable=False),
        sa.Column("privacy_level", sa.String(length=64), nullable=False),
        sa.Column("retention_policy", sa.String(length=64), nullable=False),
        sa.Column("profile_hash", sa.String(length=255), nullable=True),
        sa.Column("source_record_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("source_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("target_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("feedback_summary", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("prompt_safe_summary", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("resolution", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("classification", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_feedback_records_id"),
        sa.UniqueConstraint(
            "user_id",
            "session_id",
            "feedback_scope_kind",
            "scope_id",
            "dedupe_key",
            name="uq_feedback_records_user_session_scope_dedupe",
        ),
    )
    op.create_index(
        "ix_feedback_user_session_run_created",
        "feedback_records",
        ["user_id", "session_id", "run_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_user_session_run_step",
        "feedback_records",
        ["user_id", "session_id", "run_id", "step_id"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_user_session_run_kind_status",
        "feedback_records",
        ["user_id", "session_id", "run_id", "kind", "status"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_user_session_run_severity_status",
        "feedback_records",
        ["user_id", "session_id", "run_id", "severity", "status"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_user_session_scope_status",
        "feedback_records",
        ["user_id", "session_id", "feedback_scope_kind", "scope_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_user_session_source_event",
        "feedback_records",
        ["user_id", "session_id", "source_event_id"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_user_session_target",
        "feedback_records",
        ["user_id", "session_id", "target_type", "target_id"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_user_session_target_revision",
        "feedback_records",
        ["user_id", "session_id", "target_type", "target_id", "target_revision_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_user_session_target_revision", table_name="feedback_records")
    op.drop_index("ix_feedback_user_session_target", table_name="feedback_records")
    op.drop_index("ix_feedback_user_session_source_event", table_name="feedback_records")
    op.drop_index("ix_feedback_user_session_scope_status", table_name="feedback_records")
    op.drop_index("ix_feedback_user_session_run_severity_status", table_name="feedback_records")
    op.drop_index("ix_feedback_user_session_run_kind_status", table_name="feedback_records")
    op.drop_index("ix_feedback_user_session_run_step", table_name="feedback_records")
    op.drop_index("ix_feedback_user_session_run_created", table_name="feedback_records")
    op.drop_table("feedback_records")
