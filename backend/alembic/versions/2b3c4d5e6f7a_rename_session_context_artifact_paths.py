"""rename session context artifact paths

Revision ID: 2b3c4d5e6f7a
Revises: 1a2b3c4d5e6f
Create Date: 2026-04-12 19:10:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2b3c4d5e6f7a"
down_revision: Union[str, Sequence[str], None] = "1a2b3c4d5e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "session_context_snapshots",
        "artifact_refs",
        new_column_name="artifact_paths",
    )


def downgrade() -> None:
    op.alter_column(
        "session_context_snapshots",
        "artifact_paths",
        new_column_name="artifact_refs",
    )
