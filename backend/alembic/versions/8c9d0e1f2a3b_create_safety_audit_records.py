"""create safety audit records

Revision ID: 8c9d0e1f2a3b
Revises: 6a7b8c9d0e1f
Create Date: 2026-05-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "8c9d0e1f2a3b"
down_revision: Union[str, Sequence[str], None] = "6a7b8c9d0e1f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "safety_audit_records",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("step_id", sa.String(length=255), nullable=True),
        sa.Column("action_id", sa.String(length=255), nullable=False),
        sa.Column("tool_call_id", sa.String(length=255), nullable=True),
        sa.Column("function_name", sa.String(length=255), nullable=False),
        sa.Column("normalized_function_name", sa.String(length=255), nullable=False),
        sa.Column("final_function_name", sa.String(length=255), nullable=False),
        sa.Column("final_normalized_function_name", sa.String(length=255), nullable=False),
        sa.Column("decision", sa.String(length=64), nullable=False),
        sa.Column("reason_code", sa.String(length=255), nullable=False),
        sa.Column("risk_level", sa.String(length=64), nullable=False),
        sa.Column("winning_policy", sa.String(length=255), nullable=False),
        sa.Column("tool_call_fingerprint", sa.String(length=255), nullable=False),
        sa.Column("capability_id", sa.String(length=255), nullable=False),
        sa.Column("tool_family", sa.String(length=128), nullable=False),
        sa.Column("decision_event_id", sa.String(length=255), nullable=True),
        sa.Column("tool_event_source_event_id", sa.String(length=255), nullable=True),
        sa.Column("confirmation_event_id", sa.String(length=255), nullable=True),
        sa.Column("source_event_type", sa.String(length=128), nullable=True),
        sa.Column("source_linked_at", sa.DateTime(), nullable=True),
        sa.Column("rewrite_applied", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("rewrite_reason", sa.String(length=255), nullable=True),
        sa.Column("confirmation_id", sa.String(length=255), nullable=True),
        sa.Column("origin", sa.String(length=64), nullable=False),
        sa.Column("trust_level", sa.String(length=64), nullable=False),
        sa.Column("privacy_level", sa.String(length=64), nullable=False),
        sa.Column("retention_policy", sa.String(length=64), nullable=False),
        sa.Column("profile_hash", sa.String(length=128), nullable=True),
        sa.Column("requested_args_digest", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("final_args_digest", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("policy_trace", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("rewrite_metadata_digest", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("related_refs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("classification", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("risk_classification_digest", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_safety_audit_records_id"),
        sa.UniqueConstraint(
            "user_id",
            "session_id",
            "run_id",
            "action_id",
            name="uq_safety_audit_records_user_session_run_action",
        ),
    )
    op.create_index(
        "ix_safety_audit_user_session_run_created",
        "safety_audit_records",
        ["user_id", "session_id", "run_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_safety_audit_user_session_run_step",
        "safety_audit_records",
        ["user_id", "session_id", "run_id", "step_id"],
        unique=False,
    )
    op.create_index(
        "ix_safety_audit_user_session_decision_risk",
        "safety_audit_records",
        ["user_id", "session_id", "decision", "risk_level"],
        unique=False,
    )
    op.create_index(
        "ix_safety_audit_tool_event_source",
        "safety_audit_records",
        ["user_id", "session_id", "tool_event_source_event_id"],
        unique=False,
    )
    op.create_index(
        "ix_safety_audit_decision_event",
        "safety_audit_records",
        ["user_id", "session_id", "decision_event_id"],
        unique=False,
    )
    op.create_index(
        "ix_safety_audit_confirmation_event",
        "safety_audit_records",
        ["user_id", "session_id", "confirmation_event_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_safety_audit_confirmation_event", table_name="safety_audit_records")
    op.drop_index("ix_safety_audit_decision_event", table_name="safety_audit_records")
    op.drop_index("ix_safety_audit_tool_event_source", table_name="safety_audit_records")
    op.drop_index("ix_safety_audit_user_session_decision_risk", table_name="safety_audit_records")
    op.drop_index("ix_safety_audit_user_session_run_step", table_name="safety_audit_records")
    op.drop_index("ix_safety_audit_user_session_run_created", table_name="safety_audit_records")
    op.drop_table("safety_audit_records")
