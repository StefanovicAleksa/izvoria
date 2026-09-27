# Path: app/features/document_structuring/service/structuring_api.py

import logging

from app.core.anchors.domain.enums import AnchorKind
from app.core.config import settings
from app.core.language.service.language_api import get_language_pack
from app.core.language.utils.script import normalize_script
from app.features.document_conversion.domain.entities import ConvertedDocument
from app.features.document_conversion.domain.enums import DocumentKind
from app.features.document_structuring.domain.entities import StructuredDocument
from app.features.document_structuring.exceptions import EmptyDocumentError
from app.features.document_structuring.utils.blocks import parse_blocks
from app.features.document_structuring.utils.chunker import build_chunks
from app.features.document_structuring.utils.error_mapper import map_structuring_error
from app.features.document_structuring.utils.headings import compile_clause_patterns

logger = logging.getLogger(__name__)

# Sources a reader can open and count lines in: their chunks cite line ranges. Other unpaged kinds (DOCX, HTML,
# RTF, EPUB) are converted before their lines exist, so they cite the section a chunk sits in.
_LINE_ANCHORED = {DocumentKind.TXT, DocumentKind.MARKDOWN}


class DocumentStructuringAPI:
    """
    Cuts a converted document into retrieval chunks along its own structure: clause and title headings start a
    chunk, tables keep their rows together, a short document stays whole, and every chunk knows its pages and
    the headings it sits under (its pages, or for plain text its lines). Pure and fast -- no model, no storage.
    """

    def execute(self, document: ConvertedDocument) -> StructuredDocument:
        try:
            pack = get_language_pack()
            script = pack.transcription.script
            pages = [(page.page_no, page.text) for page in document.pages if page.text.strip()]
            if not pages:
                raise EmptyDocumentError(f"Source {document.source_id} has no readable text on any page.")

            blocks = parse_blocks(
                pages,
                compile_clause_patterns(pack.documents.clause_patterns),
                # A Serbian appliance indexes Latin script, the same as its transcripts
                normalize=lambda text: normalize_script(text, script),
            )
            chunks = build_chunks(
                blocks,
                max_words=settings.DOC_CHUNK_MAX_WORDS,
                short_document_words=settings.DOC_SHORT_DOCUMENT_WORDS,
                separator=pack.documents.section_separator,
            )
            anchor_kind = _anchor_kind(document.kind)
            logger.info(f"Structured Source {document.source_id}: {len(blocks)} blocks -> {len(chunks)} chunks.")
            return StructuredDocument(source_id=document.source_id, anchor_kind=anchor_kind, chunks=chunks)

        except Exception as e:
            raise map_structuring_error(e, context=f"Structuring Source {document.source_id}") from e


def _anchor_kind(kind: DocumentKind) -> AnchorKind:
    if kind is DocumentKind.PDF:
        return AnchorKind.PAGE
    if kind in _LINE_ANCHORED:
        return AnchorKind.LINE
    return AnchorKind.SECTION
