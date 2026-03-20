"""drop users password_salt column

Revision ID: a8f3d1b2c4e5
Revises: 43ab3c53add6
Create Date: 2026-03-16 14:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a8f3d1b2c4e5"
down_revision: Union[str, Sequence[str], None] = "43ab3c53add6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("users", "password_salt")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "users",
        sa.Column(
            "password_salt",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("'legacy-salt'::character varying"),
        ),
    )
    op.alter_column("users", "password_salt", server_default=None)
