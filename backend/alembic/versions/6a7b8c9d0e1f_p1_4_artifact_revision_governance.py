"""p1-4 artifact revision governance

Revision ID: 6a7b8c9d0e1f
Revises: 4f6a7b8c9d10
Create Date: 2026-05-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "6a7b8c9d0e1f"
down_revision: Union[str, Sequence[str], None] = "4f6a7b8c9d10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("workspace_artifacts", sa.Column("current_revision_id", sa.String(length=255), nullable=True))
    op.add_column("workspace_artifacts", sa.Column("latest_content_hash", sa.String(length=128), nullable=True))
    op.add_column("workspace_artifacts", sa.Column("latest_size", sa.Integer(), nullable=True))
    op.add_column("workspace_artifacts", sa.Column("latest_mime_type", sa.String(length=255), nullable=True))
    op.add_column(
        "workspace_artifacts",
        sa.Column(
            "artifact_status",
            sa.String(length=64),
            server_default=sa.text("'active'::character varying"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_workspace_artifacts_user_workspace_current_revision",
        "workspace_artifacts",
        ["user_id", "workspace_id", "id", "current_revision_id"],
        unique=False,
    )

    op.create_table(
        "workspace_artifact_revisions",
        sa.Column("revision_id", sa.String(length=255), nullable=False),
        sa.Column("artifact_id", sa.String(length=255), nullable=False),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=True),
        sa.Column("step_id", sa.String(length=255), nullable=True),
        sa.Column("path", sa.String(length=2048), nullable=False),
        sa.Column("storage_ref", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("storage_hash", sa.String(length=128), nullable=True),
        sa.Column("hash_algorithm", sa.String(length=32), server_default=sa.text("'sha256'::character varying"), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("artifact_type", sa.String(length=128), nullable=False),
        sa.Column("delivery_state", sa.String(length=64), nullable=False),
        sa.Column("source_kind", sa.String(length=128), nullable=False),
        sa.Column("source_event_id", sa.String(length=255), nullable=True),
        sa.Column("source_run_id", sa.String(length=255), nullable=True),
        sa.Column("source_message_event_id", sa.String(length=255), nullable=True),
        sa.Column("source_revision_id", sa.String(length=255), nullable=True),
        sa.Column("source_fact_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("source_evidence_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("source_final_answer_hash", sa.String(length=128), nullable=True),
        sa.Column("derived_content_hash", sa.String(length=128), nullable=True),
        sa.Column("tool_call_id", sa.String(length=255), nullable=True),
        sa.Column("function_name", sa.String(length=255), nullable=True),
        sa.Column("profile_hash", sa.String(length=128), nullable=True),
        sa.Column("profile_status", sa.String(length=64), server_default=sa.text("'missing'::character varying"), nullable=False),
        sa.Column("origin", sa.String(length=64), nullable=False),
        sa.Column("trust_level", sa.String(length=64), nullable=False),
        sa.Column("privacy_level", sa.String(length=64), nullable=False),
        sa.Column("retention_policy", sa.String(length=64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.PrimaryKeyConstraint("revision_id", name="pk_workspace_artifact_revisions_revision_id"),
        sa.UniqueConstraint("artifact_id", "revision_no", name="uq_workspace_artifact_revisions_artifact_revision_no"),
    )
    op.create_index(
        "ix_workspace_artifact_revisions_user_workspace_revision",
        "workspace_artifact_revisions",
        ["user_id", "workspace_id", "revision_id"],
        unique=False,
    )
    op.create_index(
        "ix_workspace_artifact_revisions_user_workspace_artifact_hash",
        "workspace_artifact_revisions",
        ["user_id", "workspace_id", "artifact_id", "content_hash"],
        unique=False,
    )
    op.create_index(
        "ix_workspace_artifact_revisions_user_workspace_artifact",
        "workspace_artifact_revisions",
        ["user_id", "workspace_id", "artifact_id"],
        unique=False,
    )
    op.create_index(
        "ix_workspace_artifact_revisions_source_event",
        "workspace_artifact_revisions",
        ["user_id", "workspace_id", "source_event_id"],
        unique=False,
    )
    op.create_index(
        "uq_war_tool_idem",
        "workspace_artifact_revisions",
        ["user_id", "workspace_id", "source_event_id", "tool_call_id", "source_kind", "content_hash"],
        unique=True,
        postgresql_where=sa.text(
            "source_kind IN ('tool_write_file', 'tool_replace_file') "
            "AND source_event_id IS NOT NULL "
            "AND tool_call_id IS NOT NULL "
            "AND content_hash IS NOT NULL"
        ),
    )
    op.create_index(
        "uq_war_final_snapshot_idem",
        "workspace_artifact_revisions",
        ["user_id", "session_id", "source_run_id", "source_message_event_id", "source_final_answer_hash"],
        unique=True,
        postgresql_where=sa.text(
            "source_kind = 'final_answer_snapshot' "
            "AND source_run_id IS NOT NULL "
            "AND source_message_event_id IS NOT NULL "
            "AND source_final_answer_hash IS NOT NULL"
        ),
    )
    op.create_index(
        "uq_war_derived_export_idem",
        "workspace_artifact_revisions",
        ["user_id", "workspace_id", "source_revision_id", "source_kind", "content_hash"],
        unique=True,
        postgresql_where=sa.text(
            "source_kind = 'derived_export' "
            "AND source_revision_id IS NOT NULL "
            "AND content_hash IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index("uq_war_derived_export_idem", table_name="workspace_artifact_revisions")
    op.drop_index("uq_war_final_snapshot_idem", table_name="workspace_artifact_revisions")
    op.drop_index("uq_war_tool_idem", table_name="workspace_artifact_revisions")
    op.drop_index("ix_workspace_artifact_revisions_source_event", table_name="workspace_artifact_revisions")
    op.drop_index("ix_workspace_artifact_revisions_user_workspace_artifact", table_name="workspace_artifact_revisions")
    op.drop_index("ix_workspace_artifact_revisions_user_workspace_artifact_hash", table_name="workspace_artifact_revisions")
    op.drop_index("ix_workspace_artifact_revisions_user_workspace_revision", table_name="workspace_artifact_revisions")
    op.drop_table("workspace_artifact_revisions")
    op.drop_index("ix_workspace_artifacts_user_workspace_current_revision", table_name="workspace_artifacts")
    op.drop_column("workspace_artifacts", "artifact_status")
    op.drop_column("workspace_artifacts", "latest_mime_type")
    op.drop_column("workspace_artifacts", "latest_size")
    op.drop_column("workspace_artifacts", "latest_content_hash")
    op.drop_column("workspace_artifacts", "current_revision_id")
