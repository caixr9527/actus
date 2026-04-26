"""drop workflow run memories snapshot

Revision ID: b2f7c1a9d4e6
Revises: e9d1c7a2b4f6
Create Date: 2026-03-29 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "b2f7c1a9d4e6"
down_revision: Union[str, Sequence[str], None] = "e9d1c7a2b4f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("workflow_runs", "memories_snapshot")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "workflow_runs",
        sa.Column(
            "memories_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
