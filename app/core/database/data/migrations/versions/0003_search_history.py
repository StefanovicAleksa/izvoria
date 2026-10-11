"""search history: past standard and verified searches with their ranked hits

chunk_ids is deliberately not a foreign key to vector_chunks: re-ingesting a file cascades its chunks away, and the
record of what was searched for has to outlive that.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-28 23:36:50.756860
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('search_history',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('query', sa.Text(), nullable=False),
    sa.Column('mode', sa.Text(), nullable=False),
    sa.Column('result_limit', sa.Integer(), nullable=False),
    sa.Column('chunk_ids', postgresql.ARRAY(sa.UUID()), server_default=sa.text("'{}'::uuid[]"), nullable=False),
    sa.Column('scores', postgresql.ARRAY(sa.DOUBLE_PRECISION()), server_default=sa.text("'{}'::double precision[]"), nullable=False),
    sa.Column('execution_time_ms', sa.Float(), server_default='0', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_search_history_created_at', 'search_history', ['created_at'], unique=False)


def downgrade() -> None:
    # The appliance only moves forward; a restore from backup is the way back.
    raise NotImplementedError("Izvoria migrations are forward-only.")
