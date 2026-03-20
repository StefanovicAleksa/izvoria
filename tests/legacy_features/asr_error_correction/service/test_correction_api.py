# Canvas: test_correction_api_v3
# Path: tests/features/asr_error_correction/test_correction_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.legacy_features.asr_error_correction.service.correction_api import AsrCorrectionAPI
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.legacy_features.asr_error_correction.domain.entities import DocumentGlossary, DomainCorrectedTranscript, AnomalyTag


class TestAsrCorrectionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "llama_adapter": MagicMock(),
            "t5_adapter": MagicMock(),
            "qwen_adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return AsrCorrectionAPI(
            repository=mocks["repo"],
            llama_adapter=mocks["llama_adapter"],
            t5_adapter=mocks["t5_adapter"],
            qwen_adapter=mocks["qwen_adapter"]
        )

    @pytest.fixture
    def transcript(self):
        source_id = uuid.uuid4()
        return ReconstructedTranscript(
            source_id=source_id,
            sentences=[
                ReconstructedSentence(sentence_id=uuid.uuid4(), text="I like m r i.", speaker_id="S1", start_sec=0.0,
                                      end_sec=1.0, segments=[])
            ]
        )

    @patch("app.legacy_features.asr_error_correction.service.correction_api.chunk_sentences_by_exact_tokens")
    def test_execute_passthrough_when_no_anomalies(self, mock_chunker, api, mocks, transcript):
        mock_chunker.return_value = [transcript.sentences]

        glossary = DocumentGlossary(source_id=transcript.source_id, domain_category="General", items=[])
        mocks["llama_adapter"].extract_glossary.return_value = glossary

        # Prevent AttributeError when the chunker tries to extract the tokenizer func
        mock_model = MagicMock()
        mocks["llama_adapter"].load.return_value = mock_model

        # T5 detects NO anomalies
        mocks["t5_adapter"].detect_anomalies.return_value = []

        result = api.execute(transcript)

        assert isinstance(result, DomainCorrectedTranscript)
        assert len(result.audit_trail) == 0
        assert result.sentences == transcript.sentences

        mocks["repo"].save_glossary.assert_called_once()
        mocks["qwen_adapter"].apply_corrections.assert_not_called()

    @patch("app.legacy_features.asr_error_correction.service.correction_api.chunk_sentences_by_exact_tokens")
    def test_execute_full_correction_pipeline(self, mock_chunker, api, mocks, transcript):
        mock_chunker.return_value = [transcript.sentences]

        glossary = DocumentGlossary(source_id=transcript.source_id, domain_category="Medical", items=[])
        mocks["llama_adapter"].extract_glossary.return_value = glossary

        mock_model = MagicMock()
        mocks["llama_adapter"].load.return_value = mock_model

        anomalies = [MagicMock(spec=AnomalyTag)]
        mocks["t5_adapter"].detect_anomalies.return_value = anomalies

        corrected = DomainCorrectedTranscript(source_id=transcript.source_id, sentences=[], audit_trail=[])
        mocks["qwen_adapter"].apply_corrections.return_value = corrected

        result = api.execute(transcript)

        assert result == corrected
        mocks["repo"].save_glossary.assert_called_once()
        mocks["repo"].save_corrected_transcript.assert_called_once_with(corrected)
        mocks["qwen_adapter"].apply_corrections.assert_called_once()