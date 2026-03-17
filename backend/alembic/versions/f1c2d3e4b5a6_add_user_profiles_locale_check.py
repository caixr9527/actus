"""add locale check constraint for user_profiles

Revision ID: f1c2d3e4b5a6
Revises: a8f3d1b2c4e5
Create Date: 2026-03-17 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f1c2d3e4b5a6"
down_revision: Union[str, Sequence[str], None] = "a8f3d1b2c4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CHECK_CONSTRAINT_NAME = "ck_user_profiles_locale_supported"
CHECK_CONDITION = "locale IN ('zh-CN', 'en-US')"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_check_constraint(
        CHECK_CONSTRAINT_NAME,
        "user_profiles",
        CHECK_CONDITION,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        CHECK_CONSTRAINT_NAME,
        "user_profiles",
        type_="check",
    )
