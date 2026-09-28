# Path: app/features/document_conversion/data/repositories.py

import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.document_conversion.data.models import ConvertedDocumentModel
from app.features.document_conversion.domain.entities import ConvertedDocument, PageContent
from app.features.document_conversion.domain.enums import DocumentKind
from app.features.document_conversion.domain.interfaces import IConvertedDocumentRepository
from app.features.document_conversion.utils.error_mapper import map_conversion_error


class ConvertedDocumentRepository(IConvertedDocumentRepository):
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, document: ConvertedDocument) -> ConvertedDocument:
        try:
            with self._db.session() as session:
                session.add(ConvertedDocumentModel(
                    id=document.id,
                    source_id=document.source_id,
                    kind=document.kind.value,
                    pages=[page.model_dump(mode="json") for page in document.pages],
                    created_at=document.created_at,
                ))
            return document
        except Exception as e:
            raise map_conversion_error(e, context=f"Saving converted document for source {document.source_id}") from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ConvertedDocument]:
        try:
            with self._db.session() as session:
                row = session.execute(
                    select(ConvertedDocumentModel).where(ConvertedDocumentModel.source_id == source_id)
                ).scalar_one_or_none()
                if row is None:
                    return None
                return ConvertedDocument(
                    id=row.id,
                    source_id=row.source_id,
                    kind=DocumentKind(row.kind),
                    pages=[PageContent(**page) for page in row.pages],
                    created_at=row.created_at,
                )
        except Exception as e:
            raise map_conversion_error(e, context=f"Fetching converted document for source {source_id}") from e
