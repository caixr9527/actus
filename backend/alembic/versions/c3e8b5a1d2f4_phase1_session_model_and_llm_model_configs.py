"""phase1 session current_model_id and llm model configs

Revision ID: c3e8b5a1d2f4
Revises: f1c2d3e4b5a6
Create Date: 2026-03-18 14:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c3e8b5a1d2f4"
down_revision: Union[str, Sequence[str], None] = "f1c2d3e4b5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "llm_model_configs",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("base_url", sa.String(length=2048), nullable=False),
        sa.Column("api_key", sa.String(length=1024), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("sort_order", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP(0)"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_llm_model_configs_id"),
    )
    op.create_index("ix_llm_model_configs_enabled", "llm_model_configs", ["enabled"], unique=False)
    op.create_index("ix_llm_model_configs_sort_order", "llm_model_configs", ["sort_order"], unique=False)

    op.add_column("sessions", sa.Column("current_model_id", sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("sessions", "current_model_id")

    op.drop_index("ix_llm_model_configs_sort_order", table_name="llm_model_configs")
    op.drop_index("ix_llm_model_configs_enabled", table_name="llm_model_configs")
    op.drop_table("llm_model_configs")
