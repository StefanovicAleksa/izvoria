# Canvas: test_domain_correction_api_v21
# Path: tests/features/domain_phonetic_correction/service/test_correction_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.domain_phonetic_correction.service.correction_api import DomainPhoneticCorrectionAPI
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticCorrectedTranscript, \
    DomainPhoneticPatch


class TestDomainPhoneticCorrectionAPI:
    @pytest.fixture
    def mock_repo(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_orchestrator(self) -> MagicMock:
        return MagicMock(spec=ModelManagerOrchestrator)

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def api(self, mock_repo: MagicMock, mock_orchestrator: MagicMock,
            mock_adapter: MagicMock) -> DomainPhoneticCorrectionAPI:
        return DomainPhoneticCorrectionAPI(
            repository=mock_repo,
            orchestrator=mock_orchestrator,
            patch_adapter=mock_adapter
        )

    def test_execute_idempotent_return(self, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock) -> None:
        transcript_id = uuid.uuid4()
        mock_existing = DomainPhoneticCorrectedTranscript(source_id=transcript_id, sentences=[], audit_trail=[])
        mock_repo.get_by_source_id.return_value = mock_existing

        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[])
        result = api.execute(dummy_transcript, "GENERAL", [])

        assert result == mock_existing
        mock_repo.get_by_source_id.assert_called_once_with(transcript_id)

    def test_execute_passthrough_empty_transcript(self, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
                                                  mock_orchestrator: MagicMock) -> None:
        mock_repo.get_by_source_id.return_value = None
        mock_repo.save.side_effect = lambda x: x

        transcript_id = uuid.uuid4()
        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[])

        result = api.execute(dummy_transcript, "GENERAL", [])

        assert result.source_id == transcript_id
        assert len(result.sentences) == 0
        # Persisted: the next phase loads this feature's output from the repository
        mock_repo.save.assert_called_once_with(result)
        mock_orchestrator.acquire_model.assert_not_called()

    @patch("app.features.domain_phonetic_correction.service.correction_api.DictionaryLoader")
    @patch("app.features.domain_phonetic_correction.service.correction_api.split_massive_sentences")
    def test_execute_sanitizes_and_applies_patches(
            self, mock_splitter: MagicMock, mock_dict_loader: MagicMock, api: DomainPhoneticCorrectionAPI,
            mock_repo: MagicMock, mock_orchestrator: MagicMock, mock_adapter: MagicMock
    ) -> None:
        mock_repo.get_by_source_id.return_value = None
        mock_dict_loader.build_prompt_dictionary.return_value = "- acute\n- angina\n- CAGR\n- EBITDA\n- hypoglycemia"

        transcript_id = uuid.uuid4()
        dummy_sentence = ReconstructedSentence(
            sentence_id=uuid.uuid4(),
            start_sec=0.0,
            end_sec=1.0,
            text="I 0presented with a cute angina today0.",  # Injecting 0 artifacts
            speaker_id="S0",
            words=[]
        )
        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[dummy_sentence])

        # Mocking the split return to match the SANITIZED sentence text
        sanitized_dummy = dummy_sentence.model_copy(update={"text": "I presented with a cute angina today."})
        mock_splitter.return_value = [sanitized_dummy]

        mock_loaded_model = MagicMock()
        mock_orchestrator.acquire_model.return_value = mock_loaded_model

        mock_adapter.generate_patches.return_value = [
            DomainPhoneticPatch(original="with a cute angina today.", correction="with acute angina today.")
        ]

        mock_repo.save.side_effect = lambda x: x

        result = api.execute(dummy_transcript, "MEDICAL", [], release_after=True)

        mock_splitter.assert_called_once()
        mock_orchestrator.acquire_model.assert_called_once()
        mock_adapter.generate_patches.assert_called_once()
        mock_orchestrator.release_model.assert_called_once_with(mock_loaded_model, mock_adapter, flush=True)

        # Assert that the text was sanitized AND patched
        assert result.sentences[0].text == "I presented with acute angina today."
        assert len(result.audit_trail) == 1
        assert result.audit_trail[0].original_text == "I presented with a cute angina today."

    def test_apply_patches_deduplicates_overlapping_signatures(self) -> None:
        sentence_id = uuid.uuid4()
        sentences = [
            ReconstructedSentence(
                sentence_id=sentence_id,
                start_sec=0.0,
                end_sec=1.0,
                text="The patient has a cute angina.",
                speaker_id="S0",
                words=[]
            )
        ]

        target_patch = DomainPhoneticPatch(original="has a cute angina", correction="has acute angina")
        audit_records_map = {}

        result_1 = DomainPhoneticCorrectionAPI._apply_patches_to_chunk(sentences, [target_patch], audit_records_map)
        assert result_1[0].text == "The patient has acute angina."
        assert len(audit_records_map[sentence_id].applied_patches) == 1

        result_2 = DomainPhoneticCorrectionAPI._apply_patches_to_chunk(result_1, [target_patch], audit_records_map)

        assert result_2[0].text == "The patient has acute angina."
        assert len(audit_records_map[sentence_id].applied_patches) == 1

    @patch("app.features.domain_phonetic_correction.service.correction_api.DictionaryLoader")
    def test_execute_skips_model_when_dictionary_is_empty(
            self, mock_dict_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
            mock_orchestrator: MagicMock, mock_adapter: MagicMock
    ) -> None:
        mock_repo.get_by_source_id.return_value = None
        mock_repo.save.side_effect = lambda x: x
        mock_dict_loader.build_prompt_dictionary.return_value = "No specific terms available."

        sentence = ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=0.0, end_sec=1.0,
                                         text="So0 we drill0 here.", speaker_id="S0", words=[])
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[sentence])

        result = api.execute(transcript, "GENERAL", [])

        mock_orchestrator.acquire_model.assert_not_called()
        mock_adapter.generate_patches.assert_not_called()
        assert [s.text for s in result.sentences] == ["So we drill here."]
        assert result.audit_trail == []
        mock_repo.save.assert_called_once_with(result)

    @patch("app.features.domain_phonetic_correction.service.correction_api.DictionaryLoader")
    def test_execute_rejects_patches_that_do_not_introduce_a_dictionary_term(
            self, mock_dict_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
            mock_orchestrator: MagicMock, mock_adapter: MagicMock
    ) -> None:
        mock_repo.get_by_source_id.return_value = None
        mock_repo.save.side_effect = lambda x: x
        mock_dict_loader.build_prompt_dictionary.return_value = "- acute\n- angina\n- CAGR\n- EBITDA\n- hypoglycemia"

        sentence = ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=0.0, end_sec=1.0,
                                         text="The patient has high pole glycemia today.", speaker_id="S0", words=[])
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[sentence])
        mock_adapter.generate_patches.return_value = [
            DomainPhoneticPatch(original="The patient has", correction="The patient had"),
            DomainPhoneticPatch(original="has high pole glycemia", correction="has hypoglycemia"),
        ]

        result = api.execute(transcript, "MEDICAL", [])

        assert result.sentences[0].text == "The patient has hypoglycemia today."
        assert [p.correction for p in result.audit_trail[0].applied_patches] == ["has hypoglycemia"]

    @pytest.mark.parametrize("original, correction, expected", [
        ("high pole glycemia", "hypoglycemia", True),
        ("has a cute angina", "has acute angina", True),
        ("ebitda margin", "EBITDA margin", True),
        ("the cagr", "the CAGR", True),
        # A spelled-out acronym isn't letter-similar; rejected rather than risk writing words nobody said
        ("the sea a grr", "the CAGR", False),
        # Ordinary words turned into dictionary terms, as seen on a real ingestion
        ("downside", "hypoglycemia", False),
        ("the Bank of England decision", "the angina decision", False),
        # Two unrelated edits are judged separately, not averaged over the words they share
        ("buyers that come in on the release", "angina that come in on the hypoglycemia", False),
        # Case-only change of a term that isn't acronym-like
        ("the bunds", "the Bunds", False),
        # Inserting a word nobody said
        ("patient has pain", "patient has angina pain", False),
        ("angina pain", "angina pains", False),
        ("patient has", "patient had", False),
    ])
    def test_is_grounded_patch(self, original: str, correction: str, expected: bool) -> None:
        terms = ["hypoglycemia", "CAGR", "EBITDA", "angina", "acute angina", "Bunds"]
        patch_obj = DomainPhoneticPatch(original=original, correction=correction)
        assert DomainPhoneticCorrectionAPI._is_grounded_patch(patch_obj, terms) is expected
