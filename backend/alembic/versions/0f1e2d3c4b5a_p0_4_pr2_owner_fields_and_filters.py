"""p0 4 pr2 owner fields and filters

Revision ID: 0f1e2d3c4b5a
Revises: 86bc17fa21eb
Create Date: 2026-04-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0f1e2d3c4b5a"
down_revision: Union[str, Sequence[str], None] = "86bc17fa21eb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    duplicate_sessions = op.get_bind().execute(
        sa.text(
            """
            SELECT session_id
            FROM workspaces
            GROUP BY session_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).fetchone()
    if duplicate_sessions is not None:
        raise RuntimeError(
            "同一 session 存在多个 workspace，拒绝静默迁移；请先清理历史脏数据"
        )

    op.add_column("workspaces", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.execute(
        """
        UPDATE workspaces AS w
        SET user_id = s.user_id
        FROM sessions AS s
        WHERE w.session_id = s.id
          AND w.user_id IS NULL
        """
    )
    op.create_index(
        "ix_workspaces_user_id_session_id",
        "workspaces",
        ["user_id", "session_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_workspaces_session_id",
        "workspaces",
        ["session_id"],
    )

    op.add_column("workspace_artifacts", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.add_column("workspace_artifacts", sa.Column("session_id", sa.String(length=255), nullable=True))
    op.add_column("workspace_artifacts", sa.Column("run_id", sa.String(length=255), nullable=True))
    op.add_column(
        "workspace_artifacts",
        sa.Column(
            "origin",
            sa.String(length=64),
            server_default=sa.text("'agent_generated'::character varying"),
            nullable=False,
        ),
    )
    op.add_column(
        "workspace_artifacts",
        sa.Column(
            "trust_level",
            sa.String(length=64),
            server_default=sa.text("'agent_generated'::character varying"),
            nullable=False,
        ),
    )
    op.add_column(
        "workspace_artifacts",
        sa.Column(
            "privacy_level",
            sa.String(length=64),
            server_default=sa.text("'private'::character varying"),
            nullable=False,
        ),
    )
    op.add_column(
        "workspace_artifacts",
        sa.Column(
            "retention_policy",
            sa.String(length=64),
            server_default=sa.text("'workspace_bound'::character varying"),
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE workspace_artifacts AS a
        SET user_id = w.user_id,
            session_id = w.session_id,
            run_id = w.current_run_id
        FROM workspaces AS w
        WHERE a.workspace_id = w.id
        """
    )
    op.create_index(
        "ix_workspace_artifacts_user_workspace",
        "workspace_artifacts",
        ["user_id", "workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_workspace_artifacts_user_session",
        "workspace_artifacts",
        ["user_id", "session_id"],
        unique=False,
    )

    op.add_column("workflow_run_events", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.execute(
        """
        UPDATE workflow_run_events AS e
        SET user_id = COALESCE(r.user_id, s.user_id)
        FROM workflow_runs AS r
        LEFT JOIN sessions AS s ON s.id = r.session_id
        WHERE e.run_id = r.id
          AND e.user_id IS NULL
        """
    )
    op.create_index(
        "ix_workflow_run_events_user_session_event",
        "workflow_run_events",
        ["user_id", "session_id", "event_id"],
        unique=False,
    )

    op.add_column("long_term_memories", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.add_column("long_term_memories", sa.Column("tenant_id", sa.String(length=255), nullable=True))
    op.add_column(
        "long_term_memories",
        sa.Column(
            "scope",
            sa.String(length=64),
            server_default=sa.text("'user'::character varying"),
            nullable=False,
        ),
    )
    op.add_column("long_term_memories", sa.Column("session_id", sa.String(length=255), nullable=True))
    op.add_column("long_term_memories", sa.Column("workspace_id", sa.String(length=255), nullable=True))
    op.add_column("long_term_memories", sa.Column("run_id", sa.String(length=255), nullable=True))
    op.add_column(
        "long_term_memories",
        sa.Column(
            "origin",
            sa.String(length=64),
            server_default=sa.text("'long_term_memory'::character varying"),
            nullable=False,
        ),
    )
    op.add_column(
        "long_term_memories",
        sa.Column(
            "trust_level",
            sa.String(length=64),
            server_default=sa.text("'system_generated'::character varying"),
            nullable=False,
        ),
    )
    op.add_column(
        "long_term_memories",
        sa.Column(
            "privacy_level",
            sa.String(length=64),
            server_default=sa.text("'sensitive'::character varying"),
            nullable=False,
        ),
    )
    op.add_column(
        "long_term_memories",
        sa.Column(
            "retention_policy",
            sa.String(length=64),
            server_default=sa.text("'user_memory'::character varying"),
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE long_term_memories
        SET user_id = split_part(namespace, '/', 2),
            tenant_id = split_part(namespace, '/', 2),
            scope = 'user'
        WHERE namespace LIKE 'user/%'
          AND split_part(namespace, '/', 2) <> ''
          AND user_id IS NULL
        """
    )
    op.create_index(
        "ix_long_term_memories_user_scope",
        "long_term_memories",
        ["user_id", "scope"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_long_term_memories_user_scope", table_name="long_term_memories")
    op.drop_column("long_term_memories", "retention_policy")
    op.drop_column("long_term_memories", "privacy_level")
    op.drop_column("long_term_memories", "trust_level")
    op.drop_column("long_term_memories", "origin")
    op.drop_column("long_term_memories", "run_id")
    op.drop_column("long_term_memories", "workspace_id")
    op.drop_column("long_term_memories", "session_id")
    op.drop_column("long_term_memories", "scope")
    op.drop_column("long_term_memories", "tenant_id")
    op.drop_column("long_term_memories", "user_id")

    op.drop_index("ix_workflow_run_events_user_session_event", table_name="workflow_run_events")
    op.drop_column("workflow_run_events", "user_id")

    op.drop_index("ix_workspace_artifacts_user_session", table_name="workspace_artifacts")
    op.drop_index("ix_workspace_artifacts_user_workspace", table_name="workspace_artifacts")
    op.drop_column("workspace_artifacts", "retention_policy")
    op.drop_column("workspace_artifacts", "privacy_level")
    op.drop_column("workspace_artifacts", "trust_level")
    op.drop_column("workspace_artifacts", "origin")
    op.drop_column("workspace_artifacts", "run_id")
    op.drop_column("workspace_artifacts", "session_id")
    op.drop_column("workspace_artifacts", "user_id")

    op.drop_constraint("uq_workspaces_session_id", "workspaces", type_="unique")
    op.drop_index("ix_workspaces_user_id_session_id", table_name="workspaces")
    op.drop_column("workspaces", "user_id")
