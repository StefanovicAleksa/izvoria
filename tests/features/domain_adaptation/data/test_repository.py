# Canvas: test_domain_repository_v3
# Path: tests/features/domain_adaptation/data/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.features.domain_adaptation.data.models import DomainGlossaryModel
from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.exceptions.database import DomainDatabaseError


class TestDomainAdaptationRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    @pytest.fixture
    def sample_glossary(self):
        return DomainGlossary(
            source_id=uuid.uuid4(),
            industry="FINANCE",
            sub_domains=["UK_MARKETS", "OPTIONS_TRADING"],
            hotword_string="Gilts:15.0, strike:15.0",
            initial_prompt_string="This dictation covers UK Markets."
        )

    def test_save_success(self, mock_db_manager, sample_glossary):
        manager, session = mock_db_manager
        repo = DomainAdaptationRepository(manager)

        result = repo.save_glossary(sample_glossary)

        assert result == sample_glossary
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, DomainGlossaryModel)
        assert added_model.source_id == sample_glossary.source_id
        assert added_model.sub_domains == ["UK_MARKETS", "OPTIONS_TRADING"]

    def test_get_by_source_id_found(self, mock_db_manager, sample_glossary):
        manager, session = mock_db_manager

        mock_db_model = DomainGlossaryModel(
            id=sample_glossary.id,
            source_id=sample_glossary.source_id,
            industry=sample_glossary.industry,
            sub_domains=sample_glossary.sub_domains,
            hotword_string=sample_glossary.hotword_string,
            initial_prompt_string=sample_glossary.initial_prompt_string,
            created_at=sample_glossary.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = DomainAdaptationRepository(manager)
        result = repo.get_glossary_by_source_id(sample_glossary.source_id)

        assert result is not None
        assert result.sub_domains == ["UK_MARKETS", "OPTIONS_TRADING"]
        assert result.hotword_string == "Gilts:15.0, strike:15.0"

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_glossary):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Connection Lost")

        repo = DomainAdaptationRepository(manager)

        with pytest.raises(DomainDatabaseError):
            repo.save_glossary(sample_glossary)