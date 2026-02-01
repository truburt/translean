"""
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.

Initial schema for users, conversations, and paragraphs.
"""
from alembic import op
import sqlalchemy as sa
import uuid

# revision identifiers, used by Alembic.
revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("oidc_sub", sa.String(length=255), nullable=False, unique=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("source_language", sa.String(length=8), nullable=False),
        sa.Column("target_language", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "paragraphs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE")),
        sa.Column("paragraph_index", sa.Integer, nullable=False),
        sa.Column("source_text", sa.Text, nullable=False),
        sa.Column("translated_text", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("paragraphs")
    op.drop_table("conversations")
    op.drop_table("users")
