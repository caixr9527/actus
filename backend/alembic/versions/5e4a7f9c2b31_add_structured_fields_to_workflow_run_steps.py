"""add structured fields to workflow run steps

Revision ID: 5e4a7f9c2b31
Revises: fb012b452f27
Create Date: 2026-04-09 20:30:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5e4a7f9c2b31"
down_revision: Union[str, Sequence[str], None] = "fb012b452f27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("workflow_run_steps", sa.Column("task_mode_hint", sa.String(length=64), nullable=True))
    op.add_column("workflow_run_steps", sa.Column("output_mode", sa.String(length=64), nullable=True))
    op.add_column("workflow_run_steps", sa.Column("artifact_policy", sa.String(length=64), nullable=True))
    op.add_column("workflow_run_steps", sa.Column("delivery_role", sa.String(length=64), nullable=True))
    op.add_column("workflow_run_steps", sa.Column("delivery_context_state", sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("workflow_run_steps", "delivery_context_state")
    op.drop_column("workflow_run_steps", "delivery_role")
    op.drop_column("workflow_run_steps", "artifact_policy")
    op.drop_column("workflow_run_steps", "output_mode")
    op.drop_column("workflow_run_steps", "task_mode_hint")
