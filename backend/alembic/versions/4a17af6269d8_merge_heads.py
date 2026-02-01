"""merge_heads

Revision ID: 4a17af6269d8
Revises: 0003_add_pending_deletion, 08bfb0e5cec0
Create Date: 2025-12-18 10:32:39.338580

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a17af6269d8'
down_revision: Union[str, None] = ('0003_add_pending_deletion', '08bfb0e5cec0')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
