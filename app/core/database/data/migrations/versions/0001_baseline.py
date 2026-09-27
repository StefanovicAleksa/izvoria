"""baseline: the schema as it stood when migrations were introduced

Revision ID: 0001
Revises:
Create Date: 2026-09-27 20:06:33.264962

A database created before this revision is caught up to it and stamped by upgrade_to_head(), not run through it.
"""
from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.core.config import settings

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('files',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('file_path', sa.String(), nullable=False),
    sa.Column('file_name', sa.String(), nullable=False),
    sa.Column('file_size_bytes', sa.Integer(), nullable=False),
    sa.Column('mime_type', sa.String(), nullable=True),
    sa.Column('file_metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('is_processed', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('research_reports',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=True),
    sa.Column('topic', sa.Text(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('scope_note', sa.Text(), server_default='', nullable=False),
    sa.Column('plan_revision', sa.Integer(), server_default='1', nullable=False),
    sa.Column('executive_summary', sa.Text(), server_default='', nullable=False),
    sa.Column('markdown_path', sa.Text(), nullable=True),
    sa.Column('corpus_digest', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('indexed_source_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('swept_source_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('scrub_enabled', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('mode', sa.String(), server_default='deep', nullable=False),
    sa.Column('rounds_used', sa.Integer(), server_default='1', nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_research_reports_session', 'research_reports', ['session_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_research_reports_session_id'), 'research_reports', ['session_id'], unique=False)
    op.create_index(op.f('ix_research_reports_status'), 'research_reports', ['status'], unique=False)
    op.create_table('session_turns',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('answer_text', sa.Text(), nullable=False),
    sa.Column('mode', sa.Text(), server_default='chat', nullable=False),
    sa.Column('standalone_query', sa.Text(), nullable=True),
    sa.Column('evidence_chunk_ids', postgresql.ARRAY(sa.UUID()), server_default=sa.text("'{}'::uuid[]"), nullable=False),
    sa.Column('first_doc_number', sa.Integer(), server_default='1', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_session_turns_session_history', 'session_turns', ['session_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_session_turns_session_id'), 'session_turns', ['session_id'], unique=False)
    op.create_table('research_report_evidence',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('report_id', sa.UUID(), nullable=False),
    sa.Column('doc_number', sa.Integer(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('source_name', sa.Text(), server_default='', nullable=False),
    sa.Column('start_sec', sa.Float(), server_default='0', nullable=False),
    sa.Column('end_sec', sa.Float(), server_default='0', nullable=False),
    sa.Column('anchor_kind', sa.String(), server_default='time', nullable=False),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('section_path', sa.Text(), nullable=True),
    sa.Column('speaker_labels', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('digest', sa.Text(), server_default='', nullable=False),
    sa.Column('origin', sa.String(), server_default='fan_out', nullable=False),
    sa.Column('found_by', sa.Text(), nullable=True),
    sa.Column('rerank_score', sa.Float(), nullable=True),
    sa.Column('scrubbed', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.ForeignKeyConstraint(['report_id'], ['research_reports.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('report_id', 'chunk_id', name='uq_research_report_evidence_chunk'),
    sa.UniqueConstraint('report_id', 'doc_number', name='uq_research_report_evidence_doc_number')
    )
    op.create_index(op.f('ix_research_report_evidence_report_id'), 'research_report_evidence', ['report_id'], unique=False)
    op.create_table('research_report_sections',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('report_id', sa.UUID(), nullable=False),
    sa.Column('section_key', sa.String(), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('order_index', sa.Integer(), server_default='0', nullable=False),
    sa.Column('brief', sa.Text(), server_default='', nullable=False),
    sa.Column('status', sa.String(), server_default='pending', nullable=False),
    sa.Column('markdown_body', sa.Text(), server_default='', nullable=False),
    sa.Column('evidence_doc_numbers', postgresql.ARRAY(sa.Integer()), server_default=sa.text("'{}'::integer[]"), nullable=False),
    sa.Column('cited_doc_numbers', postgresql.ARRAY(sa.Integer()), server_default=sa.text("'{}'::integer[]"), nullable=False),
    sa.Column('written_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['report_id'], ['research_reports.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('report_id', 'section_key', name='uq_research_report_section_key')
    )
    op.create_index('idx_research_report_sections_order', 'research_report_sections', ['report_id', 'order_index'], unique=False)
    op.create_index(op.f('ix_research_report_sections_report_id'), 'research_report_sections', ['report_id'], unique=False)
    op.create_table('research_report_sub_questions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('report_id', sa.UUID(), nullable=False),
    sa.Column('section_key', sa.String(), nullable=False),
    sa.Column('question_text', sa.Text(), nullable=False),
    sa.Column('order_index', sa.Integer(), server_default='0', nullable=False),
    sa.Column('chunks_found', sa.Integer(), server_default='0', nullable=False),
    sa.ForeignKeyConstraint(['report_id'], ['research_reports.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_research_report_sub_questions_report_id'), 'research_report_sub_questions', ['report_id'], unique=False)
    op.create_table('sources',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('source_type', sa.String(), nullable=False),
    sa.Column('content_hash', sa.String(), nullable=False),
    sa.Column('file_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['file_id'], ['files.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sources_content_hash'), 'sources', ['content_hash'], unique=True)
    op.create_index(op.f('ix_sources_file_id'), 'sources', ['file_id'], unique=True)
    op.create_table('audio_extractions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('extracted_file_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['extracted_file_id'], ['files.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('extracted_file_id')
    )
    op.create_index(op.f('ix_audio_extractions_source_id'), 'audio_extractions', ['source_id'], unique=False)
    op.create_table('chunk_visual_analyses',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('analyses', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chunk_visual_analyses_chunk_id'), 'chunk_visual_analyses', ['chunk_id'], unique=True)
    op.create_index(op.f('ix_chunk_visual_analyses_source_id'), 'chunk_visual_analyses', ['source_id'], unique=False)
    op.create_table('contextualized_transcripts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('chunks', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('converted_documents',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.String(), nullable=False),
    sa.Column('pages', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('diarization_maps',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('segments', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('fallback_applied', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('document_glossaries',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('domain_category', sa.String(), nullable=False),
    sa.Column('items', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('domain_corrected_transcripts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('sentences', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('audit_trail', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('domain_glossaries',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('industry', sa.String(), nullable=False),
    sa.Column('sub_domains', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
    sa.Column('hotword_string', sa.Text(), nullable=False),
    sa.Column('initial_prompt_string', sa.Text(), server_default='', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('domain_phonetic_corrected_transcripts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('sentences', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('audit_trail', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('extracted_chunk_frames',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('frames', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_extracted_chunk_frames_chunk_id'), 'extracted_chunk_frames', ['chunk_id'], unique=True)
    op.create_table('hallucination_scrubbed_transcripts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('segments', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('audit_trail', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('image_descriptions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('file_name', sa.String(), nullable=False),
    sa.Column('shows', sa.Text(), nullable=False),
    sa.Column('marks', sa.Text(), nullable=True),
    sa.Column('ocr_text', sa.Text(), nullable=False),
    sa.Column('ocr_lines_detected', sa.Integer(), nullable=False),
    sa.Column('ocr_lines_read', sa.Integer(), nullable=False),
    sa.Column('ocr_checked', sa.Boolean(), nullable=False),
    sa.Column('verified_items', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('unverified_items', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('dropped_items', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('taken_at', sa.String(), nullable=True),
    sa.Column('width', sa.Integer(), nullable=False),
    sa.Column('height', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('jobs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('job_type', sa.String(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('worker_id', sa.String(), nullable=True),
    sa.Column('input_params', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('output_params', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('stack_trace', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_jobs_source_history', 'jobs', ['source_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_jobs_job_type'), 'jobs', ['job_type'], unique=False)
    op.create_index(op.f('ix_jobs_source_id'), 'jobs', ['source_id'], unique=False)
    op.create_index(op.f('ix_jobs_status'), 'jobs', ['status'], unique=False)
    op.create_table('rag_visual_timelines',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('visual_chunks', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('reconstructed_transcripts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('sentences', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('scene_maps',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('scenes', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('transcripts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('segments', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('unified_transcripts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('chunks', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('vad_maps',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('segments', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_id')
    )
    op.create_table('vector_chunks',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('anchor_kind', sa.String(), server_default='time', nullable=False),
    sa.Column('start_sec', sa.Float(), nullable=False),
    sa.Column('end_sec', sa.Float(), nullable=False),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('section_path', sa.String(), nullable=True),
    sa.Column('position', sa.Integer(), nullable=True),
    sa.Column('text_payload', sa.String(), nullable=False),
    sa.Column('raw_text', sa.String(), nullable=False),
    sa.Column('context_text', sa.String(), nullable=True),
    sa.Column('speaker_labels', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=False),
    # Stemmed with this appliance's language; RetrievalAPI.verify_index_language() refuses a mismatch on boot
    sa.Column('fts_document', postgresql.TSVECTOR(), sa.Computed(f"to_tsvector('{settings.FTS_LANGUAGE_CONFIG}', text_payload)", persisted=True), nullable=True),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_vector_chunks_chunk_id'), 'vector_chunks', ['chunk_id'], unique=True)
    op.create_index('ix_vector_chunks_fts', 'vector_chunks', ['fts_document'], unique=False, postgresql_using='gin')
    op.create_index(op.f('ix_vector_chunks_source_id'), 'vector_chunks', ['source_id'], unique=False)
    op.create_index('ix_vector_chunks_source_position', 'vector_chunks', ['source_id', 'position'], unique=False)
    op.create_table('video_clips',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('extracted_file_id', sa.UUID(), nullable=False),
    sa.Column('start_sec', sa.Float(), nullable=False),
    sa.Column('end_sec', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['extracted_file_id'], ['files.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('extracted_file_id')
    )
    op.create_index(op.f('ix_video_clips_chunk_id'), 'video_clips', ['chunk_id'], unique=True)
    op.create_index(op.f('ix_video_clips_source_id'), 'video_clips', ['source_id'], unique=False)
    op.create_table('visual_hallucination_evaluations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('source_id', sa.UUID(), nullable=False),
    sa.Column('audits', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['sources.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_visual_hallucination_evaluations_source_id'), 'visual_hallucination_evaluations', ['source_id'], unique=True)


def downgrade() -> None:
    # The appliance only moves forward; a restore from backup is the way back.
    raise NotImplementedError("Onyx migrations are forward-only.")
