"""create sandbox facts

Revision ID: 7f8e9a0b1c2d
Revises: d7cfc3374fa2
Create Date: 2026-05-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "7f8e9a0b1c2d"
down_revision: Union[str, Sequence[str], None] = "d7cfc3374fa2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sandbox_facts",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("fact_scope", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=True),
        sa.Column("step_id", sa.String(length=255), nullable=True),
        sa.Column("sandbox_id", sa.String(length=255), nullable=True),
        sa.Column("fact_kind", sa.String(length=128), nullable=False),
        sa.Column("source_type", sa.String(length=128), nullable=False),
        sa.Column("source_event_id", sa.String(length=255), nullable=True),
        sa.Column("source_event_status", sa.String(length=64), server_default=sa.text("'missing'::character varying"), nullable=False),
        sa.Column("tool_event_id", sa.String(length=255), nullable=True),
        sa.Column("tool_call_id", sa.String(length=255), nullable=True),
        sa.Column("function_name", sa.String(length=255), nullable=True),
        sa.Column("subject_type", sa.String(length=64), nullable=False),
        sa.Column("subject_key", sa.String(length=512), nullable=False),
        sa.Column("profile_id", sa.String(length=255), nullable=True),
        sa.Column("profile_hash", sa.String(length=128), nullable=True),
        sa.Column("source_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("subject_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("profile_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("related_fact_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("supersedes_fact_id", sa.String(length=255), nullable=True),
        sa.Column("payload_hash", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("summary", sa.Text(), server_default=sa.text("''::text"), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("visibility", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=64), nullable=False),
        sa.Column("trust_level", sa.String(length=64), nullable=False),
        sa.Column("privacy_level", sa.String(length=64), nullable=False),
        sa.Column("retention_policy", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_sandbox_facts_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_sandbox_facts_idempotency_key"),
    )
    op.create_index("ix_sandbox_facts_user_session_created", "sandbox_facts", ["user_id", "session_id", "created_at"], unique=False)
    op.create_index("ix_sandbox_facts_user_run_step", "sandbox_facts", ["user_id", "run_id", "step_id"], unique=False)
    op.create_index("ix_sandbox_facts_user_scope", "sandbox_facts", ["user_id", "session_id", "fact_scope"], unique=False)
    op.create_index("ix_sandbox_facts_user_workspace_kind", "sandbox_facts", ["user_id", "workspace_id", "fact_kind"], unique=False)
    op.create_index("ix_sandbox_facts_source_event", "sandbox_facts", ["user_id", "session_id", "source_event_id"], unique=False)
    op.create_index("ix_sandbox_facts_tool_call", "sandbox_facts", ["user_id", "session_id", "tool_call_id"], unique=False)
    op.create_index("ix_sandbox_facts_profile_hash", "sandbox_facts", ["user_id", "workspace_id", "profile_hash"], unique=False)
    op.create_index("ix_sandbox_facts_supersedes", "sandbox_facts", ["user_id", "session_id", "supersedes_fact_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_sandbox_facts_supersedes", table_name="sandbox_facts")
    op.drop_index("ix_sandbox_facts_profile_hash", table_name="sandbox_facts")
    op.drop_index("ix_sandbox_facts_tool_call", table_name="sandbox_facts")
    op.drop_index("ix_sandbox_facts_source_event", table_name="sandbox_facts")
    op.drop_index("ix_sandbox_facts_user_workspace_kind", table_name="sandbox_facts")
    op.drop_index("ix_sandbox_facts_user_scope", table_name="sandbox_facts")
    op.drop_index("ix_sandbox_facts_user_run_step", table_name="sandbox_facts")
    op.drop_index("ix_sandbox_facts_user_session_created", table_name="sandbox_facts")
    op.drop_table("sandbox_facts")
