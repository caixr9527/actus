"""workspace runtime initial cutover

Revision ID: 1a2b3c4d5e6f
Revises: f0a1b2c3d4e5, 5e4a7f9c2b31, e1f2a3b4c5d6
Create Date: 2026-04-12 17:35:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "1a2b3c4d5e6f"
down_revision: Union[str, Sequence[str], None] = ("f0a1b2c3d4e5", "5e4a7f9c2b31", "e1f2a3b4c5d6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("current_run_id", sa.String(length=255), nullable=True),
        sa.Column("sandbox_id", sa.String(length=255), nullable=True),
        sa.Column("task_id", sa.String(length=255), nullable=True),
        sa.Column("shell_session_id", sa.String(length=255), nullable=True),
        sa.Column("cwd", sa.String(length=1024), nullable=False, server_default=sa.text("''::character varying")),
        sa.Column("browser_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("environment_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(length=64), nullable=False, server_default=sa.text("'active'::character varying")),
        sa.Column("last_active_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workspaces_session_id", "workspaces", ["session_id"], unique=False)

    op.create_table(
        "workspace_artifacts",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=2048), nullable=False),
        sa.Column("artifact_type", sa.String(length=128), nullable=False),
        sa.Column("summary", sa.String(length=2048), nullable=False, server_default=sa.text("''::character varying")),
        sa.Column("source_step_id", sa.String(length=255), nullable=True),
        sa.Column("source_capability", sa.String(length=255), nullable=True),
        sa.Column("delivery_state", sa.String(length=128), nullable=False, server_default=sa.text("''::character varying")),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workspace_artifacts_workspace_id", "workspace_artifacts", ["workspace_id"], unique=False)
    op.create_index("ix_workspace_artifacts_workspace_id_path", "workspace_artifacts", ["workspace_id", "path"], unique=False)

    op.add_column("sessions", sa.Column("workspace_id", sa.String(length=255), nullable=True))
    op.drop_column("sessions", "sandbox_id")
    op.drop_column("sessions", "task_id")


def downgrade() -> None:
    op.add_column("sessions", sa.Column("task_id", sa.String(length=255), nullable=True))
    op.add_column("sessions", sa.Column("sandbox_id", sa.String(length=255), nullable=True))
    op.drop_column("sessions", "workspace_id")

    op.drop_index("ix_workspace_artifacts_workspace_id_path", table_name="workspace_artifacts")
    op.drop_index("ix_workspace_artifacts_workspace_id", table_name="workspace_artifacts")
    op.drop_table("workspace_artifacts")

    op.drop_index("ix_workspaces_session_id", table_name="workspaces")
    op.drop_table("workspaces")
