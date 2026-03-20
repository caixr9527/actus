"""m1 b01 user models and user id links

Revision ID: 43ab3c53add6
Revises: d69fb8ec2c1e
Create Date: 2026-03-13 10:33:21.996915

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '43ab3c53add6'
down_revision: Union[str, Sequence[str], None] = 'd69fb8ec2c1e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password", sa.String(length=255), nullable=False),
        sa.Column("password_salt", sa.String(length=255), nullable=False),
        sa.Column(
            "auth_provider",
            sa.String(length=255),
            server_default=sa.text("'email'::character varying"),
            nullable=False,
        ),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.String(length=255),
            server_default=sa.text("'active'::character varying"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
            nullable=False,
        ),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("last_login_ip", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_users_id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("nickname", sa.String(length=255), nullable=True),
        sa.Column("avatar_url", sa.String(length=2048), nullable=True),
        sa.Column(
            "timezone",
            sa.String(length=64),
            server_default=sa.text("'Asia/Shanghai'::character varying"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.String(length=32),
            server_default=sa.text("'zh-CN'::character varying"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id", name="pk_user_profiles_user_id"),
    )

    op.add_column("sessions", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"], unique=False)

    op.add_column("files", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.create_index("ix_files_user_id", "files", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_files_user_id", table_name="files")
    op.drop_column("files", "user_id")

    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_column("sessions", "user_id")

    op.drop_table("user_profiles")
    op.drop_table("users")
