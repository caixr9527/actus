"""be lg 02 workflow run persistence

Revision ID: e9d1c7a2b4f6
Revises: c3e8b5a1d2f4
Create Date: 2026-03-21 10:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "e9d1c7a2b4f6"
down_revision: Union[str, Sequence[str], None] = "c3e8b5a1d2f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("thread_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=64), server_default=sa.text("'pending'::character varying"), nullable=False),
        sa.Column("checkpoint_namespace", sa.String(length=255), nullable=True),
        sa.Column("checkpoint_id", sa.String(length=255), nullable=True),
        sa.Column("current_step_id", sa.String(length=255), nullable=True),
        sa.Column("plan_snapshot", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("files_snapshot", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("memories_snapshot", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("runtime_metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("started_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("last_event_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_runs_id"),
    )
    op.create_index("ix_workflow_runs_session_id", "workflow_runs", ["session_id"], unique=False)
    op.create_index("ix_workflow_runs_user_id", "workflow_runs", ["user_id"], unique=False)
    op.create_index("ix_workflow_runs_status", "workflow_runs", ["status"], unique=False)

    op.create_table(
        "workflow_run_events",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=64), server_default=sa.text("''::character varying"), nullable=False),
        sa.Column("event_payload", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["workflow_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_run_events_id"),
        sa.UniqueConstraint("run_id", "event_id", name="uq_workflow_run_events_run_event_id"),
    )
    op.create_index("ix_workflow_run_events_run_id", "workflow_run_events", ["run_id"], unique=False)
    op.create_index("ix_workflow_run_events_session_id", "workflow_run_events", ["session_id"], unique=False)
    op.create_index("ix_workflow_run_events_created_at", "workflow_run_events", ["created_at"], unique=False)

    op.create_table(
        "workflow_run_steps",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("step_id", sa.String(length=255), nullable=False),
        sa.Column("step_index", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("description", sa.Text(), server_default=sa.text("''::text"), nullable=False),
        sa.Column("status", sa.String(length=64), server_default=sa.text("'pending'::character varying"), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("attachments", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["workflow_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_run_steps_id"),
        sa.UniqueConstraint("run_id", "step_id", name="uq_workflow_run_steps_run_step_id"),
    )
    op.create_index("ix_workflow_run_steps_run_id", "workflow_run_steps", ["run_id"], unique=False)
    op.create_index("ix_workflow_run_steps_step_index", "workflow_run_steps", ["step_index"], unique=False)

    op.add_column("sessions", sa.Column("current_run_id", sa.String(length=255), nullable=True))
    op.create_index("ix_sessions_current_run_id", "sessions", ["current_run_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_sessions_current_run_id", table_name="sessions")
    op.drop_column("sessions", "current_run_id")

    op.drop_index("ix_workflow_run_steps_step_index", table_name="workflow_run_steps")
    op.drop_index("ix_workflow_run_steps_run_id", table_name="workflow_run_steps")
    op.drop_table("workflow_run_steps")

    op.drop_index("ix_workflow_run_events_created_at", table_name="workflow_run_events")
    op.drop_index("ix_workflow_run_events_session_id", table_name="workflow_run_events")
    op.drop_index("ix_workflow_run_events_run_id", table_name="workflow_run_events")
    op.drop_table("workflow_run_events")

    op.drop_index("ix_workflow_runs_status", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_user_id", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_session_id", table_name="workflow_runs")
    op.drop_table("workflow_runs")

