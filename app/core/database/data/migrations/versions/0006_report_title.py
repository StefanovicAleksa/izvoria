"""report title: a research report's short name, apart from the request it was asked with

Nullable: a report from before titles keeps NULL and is shown by its request (`topic`) until it is renamed.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-07 19:05:11.716023
"""
from alembic import op
import sqlalchemy as sa


revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('research_reports', sa.Column('title', sa.Text(), nullable=True))


def downgrade() -> None:
    # The appliance only moves forward; a restore from backup is the way back.
    raise NotImplementedError("Izvoria migrations are forward-only.")
