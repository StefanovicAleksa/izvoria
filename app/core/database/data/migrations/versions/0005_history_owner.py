"""history owner: which user a conversation turn, a search or a report belongs to

Nullable and NULL on the appliance, where history is shared by whoever sits at it. The hosted multi-user API
(web_deploy/) writes the signed-in user's id and filters every history read by it, so users of one firm (one
database) each see only their own conversations, searches and reports.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-04 20:20:43.896224
"""
from alembic import op
import sqlalchemy as sa


revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('research_reports', sa.Column('owner_id', sa.String(length=64), nullable=True))
    op.create_index(op.f('ix_research_reports_owner_id'), 'research_reports', ['owner_id'], unique=False)
    op.add_column('search_history', sa.Column('owner_id', sa.String(length=64), nullable=True))
    op.create_index(op.f('ix_search_history_owner_id'), 'search_history', ['owner_id'], unique=False)
    op.add_column('session_turns', sa.Column('owner_id', sa.String(length=64), nullable=True))
    op.create_index(op.f('ix_session_turns_owner_id'), 'session_turns', ['owner_id'], unique=False)


def downgrade() -> None:
    # The appliance only moves forward; a restore from backup is the way back.
    raise NotImplementedError("Izvoria migrations are forward-only.")
