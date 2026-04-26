"""drop session memories

Revision ID: d5f6a7b8c9d0
Revises: c4d8e2f6a1b3
Create Date: 2026-03-29 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "c4d8e2f6a1b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("sessions", "memories")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "sessions",
        sa.Column(
            "memories",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
