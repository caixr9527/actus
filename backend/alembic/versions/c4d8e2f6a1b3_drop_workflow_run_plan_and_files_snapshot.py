"""drop workflow run plan and files snapshot

Revision ID: c4d8e2f6a1b3
Revises: b2f7c1a9d4e6
Create Date: 2026-03-29 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c4d8e2f6a1b3"
down_revision: Union[str, Sequence[str], None] = "b2f7c1a9d4e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("workflow_runs", "files_snapshot")
    op.drop_column("workflow_runs", "plan_snapshot")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "workflow_runs",
        sa.Column(
            "plan_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "workflow_runs",
        sa.Column(
            "files_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
