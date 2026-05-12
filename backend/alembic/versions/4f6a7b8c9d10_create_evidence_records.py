"""create evidence records

Revision ID: 4f6a7b8c9d10
Revises: 7f8e9a0b1c2d
Create Date: 2026-05-07 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "4f6a7b8c9d10"
down_revision: Union[str, Sequence[str], None] = "7f8e9a0b1c2d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evidence_records",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=True),
        sa.Column("step_id", sa.String(length=255), nullable=True),
        sa.Column("evidence_scope", sa.String(length=64), nullable=False),
        sa.Column("evidence_kind", sa.String(length=128), nullable=False),
        sa.Column("action_key", sa.String(length=512), nullable=True),
        sa.Column("claim_key", sa.String(length=255), nullable=True),
        sa.Column("claim_text", sa.Text(), nullable=True),
        sa.Column("subject_key", sa.String(length=512), nullable=True),
        sa.Column("source_step_id", sa.String(length=255), nullable=True),
        sa.Column("support_level", sa.String(length=64), nullable=False),
        sa.Column("quality_status", sa.String(length=64), nullable=False),
        sa.Column("source_type", sa.String(length=128), nullable=False),
        sa.Column("source_event_id", sa.String(length=255), nullable=True),
        sa.Column("tool_call_id", sa.String(length=255), nullable=True),
        sa.Column("primary_fact_id", sa.String(length=255), nullable=True),
        sa.Column("primary_artifact_id", sa.String(length=255), nullable=True),
        sa.Column("profile_hash", sa.String(length=128), nullable=True),
        sa.Column("source_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("subject_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("result_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("related_evidence_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("payload_hash", sa.String(length=128), nullable=False),
        sa.Column("result_refs_hash", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("supersedes_evidence_id", sa.String(length=255), nullable=True),
        sa.Column("summary", sa.Text(), server_default=sa.text("''::text"), nullable=False),
        sa.Column("confidence", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("reusable", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("reuse_policy", sa.String(length=64), nullable=False),
        sa.Column("staleness_policy", sa.String(length=64), nullable=False),
        sa.Column("visibility", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=64), nullable=False),
        sa.Column("trust_level", sa.String(length=64), nullable=False),
        sa.Column("privacy_level", sa.String(length=64), nullable=False),
        sa.Column("retention_policy", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_evidence_records_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_evidence_records_idempotency_key"),
    )
    op.create_index("ix_evidence_user_session_created", "evidence_records", ["user_id", "session_id", "created_at"], unique=False)
    op.create_index("ix_evidence_user_run_step", "evidence_records", ["user_id", "run_id", "step_id"], unique=False)
    op.create_index("ix_evidence_user_scope", "evidence_records", ["user_id", "session_id", "evidence_scope"], unique=False)
    op.create_index("ix_evidence_source_event", "evidence_records", ["user_id", "session_id", "source_event_id"], unique=False)
    op.create_index("ix_evidence_fact", "evidence_records", ["user_id", "session_id", "primary_fact_id"], unique=False)
    op.create_index("ix_evidence_artifact", "evidence_records", ["user_id", "session_id", "primary_artifact_id"], unique=False)
    op.create_index("ix_evidence_claim", "evidence_records", ["user_id", "session_id", "run_id", "claim_key"], unique=False)
    op.create_index("ix_evidence_action_subject", "evidence_records", ["user_id", "session_id", "run_id", "action_key", "subject_key"], unique=False)
    op.create_index("ix_evidence_reusable_by_run", "evidence_records", ["user_id", "session_id", "run_id", "reusable"], unique=False)
    op.create_index("ix_evidence_result_refs_hash", "evidence_records", ["user_id", "session_id", "result_refs_hash"], unique=False)
    op.create_index("ix_evidence_supersedes", "evidence_records", ["user_id", "session_id", "supersedes_evidence_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_evidence_supersedes", table_name="evidence_records")
    op.drop_index("ix_evidence_result_refs_hash", table_name="evidence_records")
    op.drop_index("ix_evidence_reusable_by_run", table_name="evidence_records")
    op.drop_index("ix_evidence_action_subject", table_name="evidence_records")
    op.drop_index("ix_evidence_claim", table_name="evidence_records")
    op.drop_index("ix_evidence_artifact", table_name="evidence_records")
    op.drop_index("ix_evidence_fact", table_name="evidence_records")
    op.drop_index("ix_evidence_source_event", table_name="evidence_records")
    op.drop_index("ix_evidence_user_scope", table_name="evidence_records")
    op.drop_index("ix_evidence_user_run_step", table_name="evidence_records")
    op.drop_index("ix_evidence_user_session_created", table_name="evidence_records")
    op.drop_table("evidence_records")
