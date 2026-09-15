# Path: tests/features/domain_phonetic_correction/service/test_correction_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.domain_phonetic_correction.exceptions import PhoneticCorrectionError
from app.features.domain_phonetic_correction.service.correction_api import DomainPhoneticCorrectionAPI
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.domain_phonetic_correction.domain.entities import (
    AppliedDomainCorrectionRecord,
    DomainPhoneticCorrectedTranscript,
    DomainPhoneticPatch
)

LOADER = "app.features.domain_phonetic_correction.service.correction_api.DictionaryLoader"


def _sentence(text: str, start: float = 0.0) -> ReconstructedSentence:
    return ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=start, end_sec=start + 1.0, text=text,
                                 speaker_id="S0", words=[])


class TestDomainPhoneticCorrectionAPI:
    @pytest.fixture
    def mock_repo(self) -> MagicMock:
        repo = MagicMock()
        repo.get_by_source_id.return_value = None
        repo.save.side_effect = lambda x: x
        return repo

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

        result = api.execute(ReconstructedTranscript(source_id=transcript_id, sentences=[]), "GENERAL", [])

        assert result == mock_existing
        mock_repo.get_by_source_id.assert_called_once_with(transcript_id)

    def test_execute_passthrough_empty_transcript(self, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
                                                  mock_orchestrator: MagicMock) -> None:
        transcript_id = uuid.uuid4()

        result = api.execute(ReconstructedTranscript(source_id=transcript_id, sentences=[]), "GENERAL", [])

        assert result.source_id == transcript_id
        assert result.sentences == []
        # Persisted: the next phase loads this feature's output from the repository
        mock_repo.save.assert_called_once_with(result)
        mock_orchestrator.acquire_model.assert_not_called()

    @patch(LOADER)
    def test_execute_skips_model_when_dictionary_is_empty(
            self, mock_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
            mock_orchestrator: MagicMock, mock_adapter: MagicMock
    ) -> None:
        mock_loader.load_terms.return_value = []
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[_sentence("So we drill0 here.")])

        result = api.execute(transcript, "PROP_TRADING_MACRO", ["EUROPEAN_CENTRAL_BANK"])

        mock_loader.load_terms.assert_called_once_with("PROP_TRADING_MACRO", ["EUROPEAN_CENTRAL_BANK"])
        mock_orchestrator.acquire_model.assert_not_called()
        mock_adapter.verify_candidates.assert_not_called()
        assert [s.text for s in result.sentences] == ["So we drill here."]
        assert result.audit_trail == []
        mock_repo.save.assert_called_once_with(result)

    @patch(LOADER)
    def test_execute_skips_model_when_nothing_sounds_like_a_misheard_term(
            self, mock_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
            mock_orchestrator: MagicMock, mock_adapter: MagicMock
    ) -> None:
        mock_loader.load_terms.return_value = ["Bunds"]
        texts = ["So again showing that there is still a little bit of weakness in the bund.",
                 "You have the fade from the top of the range up at those 73s."]
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[_sentence(t) for t in texts])

        result = api.execute(transcript, "PROP_TRADING_MACRO", [])

        mock_orchestrator.acquire_model.assert_not_called()
        mock_adapter.verify_candidates.assert_not_called()
        assert [s.text for s in result.sentences] == texts
        assert result.audit_trail == []

    @patch(LOADER)
    def test_execute_applies_confirmed_candidates_and_records_them(
            self, mock_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
            mock_orchestrator: MagicMock, mock_adapter: MagicMock
    ) -> None:
        mock_loader.load_terms.return_value = ["Bunds", "Mario Draghi"]
        loaded_model = MagicMock()
        mock_orchestrator.acquire_model.return_value = loaded_model
        mock_adapter.verify_candidates.side_effect = lambda **kwargs: [True] * len(kwargs["candidates"])
        first = _sentence("This is where Mario Draggy spoke.", 0.0)
        second = _sentence("We saw the burn fall.", 1.0)
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[first, second])

        result = api.execute(transcript, "PROP_TRADING_MACRO", ["EUROPEAN_CENTRAL_BANK"], release_after=True)

        mock_orchestrator.acquire_model.assert_called_once_with(api._manifest, mock_adapter)
        first_call, second_call = mock_adapter.verify_candidates.call_args_list
        assert first_call.kwargs["model"] is loaded_model
        assert first_call.kwargs["sentence_text"] == "This is where Mario Draggy spoke."
        assert first_call.kwargs["previous_text"] == ""
        assert [c.heard for c in first_call.kwargs["candidates"]] == ["Mario Draggy"]
        assert first_call.kwargs["domain"] == "prop trading macro (european central bank)"
        # The next sentence sees the corrected previous sentence as context
        assert second_call.kwargs["previous_text"] == "This is where Mario Draghi spoke."

        assert [s.text for s in result.sentences] == ["This is where Mario Draghi spoke.", "We saw the Bund fall."]
        assert result.audit_trail == [
            AppliedDomainCorrectionRecord(
                sentence_id=first.sentence_id, original_text=first.text, corrected_text="This is where Mario Draghi spoke.",
                applied_patches=[DomainPhoneticPatch(original="Mario Draggy", correction="Mario Draghi")]
            ),
            AppliedDomainCorrectionRecord(
                sentence_id=second.sentence_id, original_text=second.text, corrected_text="We saw the Bund fall.",
                applied_patches=[DomainPhoneticPatch(original="burn", correction="Bund")]
            ),
        ]
        mock_orchestrator.release_model.assert_called_once_with(loaded_model, mock_adapter, flush=True)
        mock_repo.save.assert_called_once_with(result)

    @patch(LOADER)
    def test_execute_does_not_apply_rejected_candidates(
            self, mock_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_orchestrator: MagicMock,
            mock_adapter: MagicMock
    ) -> None:
        mock_loader.load_terms.return_value = ["Bunds"]
        mock_adapter.verify_candidates.return_value = [False]
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[_sentence("We saw the burn fall.")])

        result = api.execute(transcript, "PROP_TRADING_MACRO", [], release_after=True)

        mock_adapter.verify_candidates.assert_called_once()
        assert [s.text for s in result.sentences] == ["We saw the burn fall."]
        assert result.audit_trail == []

    @patch(LOADER)
    def test_execute_keeps_the_model_loaded_when_release_after_is_false(
            self, mock_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_orchestrator: MagicMock,
            mock_adapter: MagicMock
    ) -> None:
        mock_loader.load_terms.return_value = ["Bunds"]
        mock_adapter.verify_candidates.return_value = [True]
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[_sentence("We saw the burn fall.")])

        api.execute(transcript, "PROP_TRADING_MACRO", [], release_after=False)

        mock_orchestrator.acquire_model.assert_called_once_with(api._manifest, mock_adapter)
        mock_orchestrator.release_model.assert_not_called()

    @patch(LOADER)
    def test_execute_releases_the_model_when_verification_fails(
            self, mock_loader: MagicMock, api: DomainPhoneticCorrectionAPI, mock_repo: MagicMock,
            mock_orchestrator: MagicMock, mock_adapter: MagicMock
    ) -> None:
        mock_loader.load_terms.return_value = ["Bunds"]
        loaded_model = MagicMock()
        mock_orchestrator.acquire_model.return_value = loaded_model
        mock_adapter.verify_candidates.side_effect = RuntimeError("llama_cpp decode failed")
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[_sentence("We saw the burn fall.")])

        with pytest.raises(PhoneticCorrectionError):
            api.execute(transcript, "PROP_TRADING_MACRO", [], release_after=True)

        mock_orchestrator.release_model.assert_called_once_with(loaded_model, mock_adapter, flush=True)
        mock_repo.save.assert_not_called()

    @pytest.mark.parametrize("industry, sub_domains, expected", [
        ("PROP_TRADING_MACRO", ["CENTRAL_BANK_MACRO", "EUROPEAN_CENTRAL_BANK"],
         "prop trading macro (central bank macro, european central bank)"),
        ("MEDICAL", [], "medical"),
    ])
    def test_describe_domain(self, industry: str, sub_domains: list, expected: str) -> None:
        assert DomainPhoneticCorrectionAPI._describe_domain(industry, sub_domains) == expected
