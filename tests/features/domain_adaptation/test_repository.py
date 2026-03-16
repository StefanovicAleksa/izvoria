# Canvas: test_domain_repository_v1
# Path: tests/features/domain_adaptation/test_repository.py

import uuid
from unittest.mock import MagicMock
from app.features.domain_adaptation.data.repositories import DomainAdaptationRepository
from app.features.domain_adaptation.domain.entities import DomainGlossary


def test_save_glossary(mocker):
    # Arrange
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.session.return_value.__enter__.return_value = mock_session

    repo = DomainAdaptationRepository(mock_db_manager)
    glossary = DomainGlossary(
        source_id=uuid.uuid4(),
        industry="FINANCE",
        sub_domain="UK_MARKETS",
        hotword_string="Gilts:15.0"
    )

    # Act
    result = repo.save_glossary(glossary)

    # Assert
    mock_session.add.assert_called_once()
    assert result.sub_domain == "UK_MARKETS"