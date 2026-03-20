# Canvas: test_adaptation_api_v5
# Path: tests/features/domain_adaptation/test_adaptation_api.py

import uuid
from unittest.mock import MagicMock
from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.domain_adaptation.domain.entities import ScoutResult


def test_process_domain_adaptation_orchestration(mocker):
    # Arrange
    source_id = uuid.uuid4()
    mock_repo, mock_orch, mock_scout, mock_classifier, mock_sampler = [MagicMock() for _ in range(5)]

    mock_repo.get_glossary_by_source_id.return_value = None
    mock_sampler.extract_samples.return_value = ["sample1", "sample2"]

    mock_scout.transcribe_samples.return_value = ScoutResult(
        source_id=source_id, samples=[], combined_text="We are looking at the strike price."
    )

    # Mock the new GlossaryBuilder Utility
    mocker.patch(
        "app.features.domain_adaptation.utils.glossary_builder.GlossaryBuilder.get_candidate_labels",
        return_value=["OPTIONS_TRADING"]
    )
    mock_classifier.classify_domain.return_value = "OPTIONS_TRADING"

    mocker.patch(
        "app.features.domain_adaptation.utils.glossary_builder.GlossaryBuilder.validate_and_build",
        return_value=("OPTIONS_TRADING", "EBITDA:15.0, strike:15.0")
    )

    api = DomainAdaptationAPI(mock_repo, mock_orch, mock_scout, mock_classifier, mock_sampler)
    vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=10, confidence=0.9)])

    # Act
    api.process_domain_adaptation(source_id, "/mock.mp4", vad_map)

    # Assert
    # Prove the API properly connected the Scout, Classifier, and GlossaryBuilder
    mock_repo.save_glossary.assert_called_once()
    saved_glossary = mock_repo.save_glossary.call_args[0][0]

    assert saved_glossary.sub_domain == "OPTIONS_TRADING"
    assert saved_glossary.hotword_string == "EBITDA:15.0, strike:15.0"