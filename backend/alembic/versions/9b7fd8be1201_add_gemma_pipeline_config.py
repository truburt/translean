"""add gemma pipeline config fields

Revision ID: 9b7fd8be1201
Revises: 6b2e2c9f4c3a
Create Date: 2026-04-09 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9b7fd8be1201"
down_revision: Union[str, None] = "6b2e2c9f4c3a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "app_config",
        sa.Column("pipeline_mode", sa.String(length=50), nullable=False, server_default="legacy_whisper_ollama"),
    )
    op.add_column(
        "app_config",
        sa.Column("gemma_base_url", sa.String(length=255), nullable=False, server_default="http://localhost:8010"),
    )
    op.add_column(
        "app_config",
        sa.Column("gemma_model", sa.String(length=255), nullable=False, server_default="google/gemma-4-E4B-it"),
    )
    op.add_column(
        "app_config",
        sa.Column("gemma_keep_alive_seconds", sa.Integer(), nullable=False, server_default="900"),
    )


def downgrade() -> None:
    op.drop_column("app_config", "gemma_keep_alive_seconds")
    op.drop_column("app_config", "gemma_model")
    op.drop_column("app_config", "gemma_base_url")
    op.drop_column("app_config", "pipeline_mode")
