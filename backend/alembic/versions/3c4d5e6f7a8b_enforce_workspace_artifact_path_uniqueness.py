"""enforce workspace artifact path uniqueness

Revision ID: 3c4d5e6f7a8b
Revises: 2b3c4d5e6f7a
Create Date: 2026-04-13 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c4d5e6f7a8b"
down_revision: Union[str, Sequence[str], None] = "2b3c4d5e6f7a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM workspace_artifacts AS older
        USING workspace_artifacts AS newer
        WHERE older.workspace_id = newer.workspace_id
          AND older.path = newer.path
          AND (
              older.updated_at < newer.updated_at
              OR (
                  older.updated_at = newer.updated_at
                  AND older.created_at < newer.created_at
              )
              OR (
                  older.updated_at = newer.updated_at
                  AND older.created_at = newer.created_at
                  AND older.id < newer.id
              )
          )
        """
    )
    op.drop_index("ix_workspace_artifacts_workspace_id_path", table_name="workspace_artifacts")
    op.create_unique_constraint(
        "uq_workspace_artifacts_workspace_id_path",
        "workspace_artifacts",
        ["workspace_id", "path"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_workspace_artifacts_workspace_id_path",
        "workspace_artifacts",
        type_="unique",
    )
    op.create_index(
        "ix_workspace_artifacts_workspace_id_path",
        "workspace_artifacts",
        ["workspace_id", "path"],
        unique=False,
    )
