"""add type enum

Revision ID: 0002_add_type_enum
Revises: 0001_init
Create Date: 2025-12-13 10:55:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision = '0002_add_type_enum'
down_revision = '0001_init'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [c['name'] for c in inspector.get_columns('paragraphs')]

    # Add 'type' column
    # We set a server_default temporarily to fill existing rows, then we can remove it if needed.
    # But defaulting to 'active' or 'stable' is fine.
    op.add_column('paragraphs', sa.Column('type', sa.String(length=50), nullable=False, server_default='stable'))

    if 'is_refined' in columns:
        # Migrate Data
        # Note: SQLite booleans are 1/0, Postgres are true/false.
        # simpler to just update where it translates to true.
        
        # refined
        op.execute("UPDATE paragraphs SET type = 'refined' WHERE is_refined = 1 OR is_refined = 'true' OR is_refined = 't'")
        
        # summary (override refined if text starts with SUMMARY:)
        op.execute("UPDATE paragraphs SET type = 'summary' WHERE type = 'refined' AND source_text LIKE 'SUMMARY:%'")
        
        # stable/active (default is stable, but let's be explicit for non-refined)
        op.execute("UPDATE paragraphs SET type = 'stable' WHERE is_refined = 0 OR is_refined = 'false' OR is_refined = 'f'")
        
        # Drop is_refined
        with op.batch_alter_table('paragraphs') as batch_op:
            batch_op.drop_column('is_refined')


def downgrade():
    op.add_column('paragraphs', sa.Column('is_refined', sa.Boolean(), server_default='false', nullable=False))
    
    op.execute("UPDATE paragraphs SET is_refined = 1 WHERE type IN ('refined', 'summary')")
    op.execute("UPDATE paragraphs SET is_refined = 0 WHERE type NOT IN ('refined', 'summary')")
    
    with op.batch_alter_table('paragraphs') as batch_op:
        batch_op.drop_column('type')
