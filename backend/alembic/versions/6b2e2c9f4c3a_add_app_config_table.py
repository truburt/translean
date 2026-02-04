"""Add app_config table

Revision ID: 6b2e2c9f4c3a
Revises: 1c4b1785e9a3
Create Date: 2026-01-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6b2e2c9f4c3a"
down_revision: Union[str, None] = "1c4b1785e9a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_config",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("whisper_base_url", sa.String(length=255), nullable=False),
        sa.Column("whisper_model", sa.String(length=255), nullable=False),
        sa.Column("whisper_keep_alive_seconds", sa.Integer(), nullable=False),
        sa.Column("ollama_base_url", sa.String(length=255), nullable=False),
        sa.Column("llm_model_translation", sa.String(length=255), nullable=False),
        sa.Column("ollama_keep_alive_seconds", sa.Integer(), nullable=False),
        sa.Column("commit_timeout_seconds", sa.Float(), nullable=False),
        sa.Column("silence_finalize_seconds", sa.Float(), nullable=False),
        sa.Column("min_preview_buffer_seconds", sa.Float(), nullable=False),
        sa.Column("stable_window_seconds", sa.Float(), nullable=False),
        sa.Column("no_speech_prob_skip", sa.Float(), nullable=False),
        sa.Column("no_speech_prob_logprob_skip", sa.Float(), nullable=False),
        sa.Column("avg_logprob_skip", sa.Float(), nullable=False),
        sa.Column("compression_ratio_skip", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("app_config")
