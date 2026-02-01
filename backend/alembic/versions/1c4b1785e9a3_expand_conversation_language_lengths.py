"""Expand conversation language column lengths

Revision ID: 1c4b1785e9a3
Revises: 4a17af6269d8
Create Date: 2025-02-05 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1c4b1785e9a3"
down_revision: Union[str, None] = "4a17af6269d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("conversations") as batch_op:
        batch_op.alter_column(
            "source_language",
            existing_type=sa.String(length=8),
            type_=sa.String(length=255),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "target_language",
            existing_type=sa.String(length=8),
            type_=sa.String(length=50),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("conversations") as batch_op:
        batch_op.alter_column(
            "target_language",
            existing_type=sa.String(length=50),
            type_=sa.String(length=8),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "source_language",
            existing_type=sa.String(length=255),
            type_=sa.String(length=8),
            existing_nullable=False,
        )
