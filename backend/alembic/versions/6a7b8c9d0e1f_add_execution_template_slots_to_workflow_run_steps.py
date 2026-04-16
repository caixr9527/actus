"""add execution template slots to workflow run steps

Revision ID: 6a7b8c9d0e1f
Revises: 3c4d5e6f7a8b
Create Date: 2026-04-16 20:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "6a7b8c9d0e1f"
down_revision: Union[str, Sequence[str], None] = "3c4d5e6f7a8b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "workflow_run_steps",
        sa.Column("execution_template", sa.Text(), nullable=False, server_default=sa.text("''::text")),
    )
    op.add_column(
        "workflow_run_steps",
        sa.Column(
            "required_slots",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "workflow_run_steps",
        sa.Column(
            "execution_slots",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("workflow_run_steps", "execution_slots")
    op.drop_column("workflow_run_steps", "required_slots")
    op.drop_column("workflow_run_steps", "execution_template")
