"""source profiles: each source's title and topic, so a question can name it and every excerpt says where it is from

Empty after the upgrade: new ingestions write a profile, and `python -m scripts.backfill_source_profiles` writes one
for every source indexed before this revision.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-20 14:02:12.312927
"""
from alembic import op
import sqlalchemy as sa


revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('source_profiles',
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('topic', sa.Text(), server_default='', nullable=False),
    sa.Column('origin', sa.String(length=16), nullable=False),
    sa.Column('language', sa.String(length=16), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('source_id')
    )


def downgrade() -> None:
    # The appliance only moves forward; a restore from backup is the way back.
    raise NotImplementedError("Izvoria migrations are forward-only.")
