# Canvas: domain_adaptation_repositories_v3
# Path: app/features/domain_adaptation/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.domain_adaptation.domain.interfaces import IDomainAdaptationRepository
from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.features.domain_adaptation.data.models import DomainGlossaryModel
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error

class DomainAdaptationRepository(IDomainAdaptationRepository):
    """
    PostgreSQL implementation of the Domain Adaptation Repository.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save_glossary(self, glossary: DomainGlossary) -> DomainGlossary:
        try:
            db_model = DomainGlossaryModel(
                id=glossary.id,
                source_id=glossary.source_id,
                industry=glossary.industry,
                sub_domains=glossary.sub_domains,
                hotword_string=glossary.hotword_string,
                initial_prompt_string=glossary.initial_prompt_string,  # Added mapping
                created_at=glossary.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return glossary

        except Exception as e:
            raise map_domain_adaptation_error(e, context="Save Multi-Domain Glossary to Postgres") from e

    def get_glossary_by_source_id(self, source_id: uuid.UUID) -> Optional[DomainGlossary]:
        try:
            with self._db.session() as session:
                stmt = select(DomainGlossaryModel).where(DomainGlossaryModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                return DomainGlossary(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    industry=db_model.industry,
                    sub_domains=db_model.sub_domains,
                    hotword_string=db_model.hotword_string,
                    initial_prompt_string=db_model.initial_prompt_string,  # Added mapping
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_domain_adaptation_error(e, context="Fetch Glossary by Source ID") from e