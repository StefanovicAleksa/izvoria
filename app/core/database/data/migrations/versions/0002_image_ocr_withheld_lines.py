"""image ocr withheld lines: low-confidence OCR lines (handwriting) the vision model did not confirm

Rows from before it had nothing withheld, which is what the default says.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-27 21:58:28.923214
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('image_descriptions', sa.Column('ocr_withheld_lines', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False))


def downgrade() -> None:
    # The appliance only moves forward; a restore from backup is the way back.
    raise NotImplementedError("Izvoria migrations are forward-only.")
