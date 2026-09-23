# Canvas: test_contextualizer_api_v3
# Path: tests/features/contextual_retrieval/service/test_contextualizer_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch, call

from app.core.anchors.domain.enums import AnchorKind
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.contextual_retrieval.data.prompts import DOCUMENT_CONTEXTUAL_SYSTEM_PROMPT
from app.features.document_structuring.domain.entities import StructuredChunk, StructuredDocument
from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticCorrectedTranscript
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.contextual_retrieval.exceptions.model import SLMInferenceError


def _sentence(text: str, speaker: str = "S0", start: float = 0.0) -> ReconstructedSentence:
    return ReconstructedSentence(
        sentence_id=uuid.uuid4(), text=text, speaker_id=speaker, start_sec=start, end_sec=start + 1, words=[]
    )


class TestContextualizerAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "orchestrator": MagicMock(spec=ModelManagerOrchestrator),
            "slm_adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return ContextualizerAPI(
            repository=mocks["repo"],
            orchestrator=mocks["orchestrator"],
            slm_adapter=mocks["slm_adapter"]
        )

    def test_update_transcript_delegates_to_repo(self, api, mocks):
        """Verifies the new boundary method correctly proxies to the repository."""
        mock_transcript = MagicMock(spec=ContextualizedTranscript)
        mocks["repo"].update.return_value = mock_transcript

        result = api.update_transcript(mock_transcript)

        assert result == mock_transcript
        mocks["repo"].update.assert_called_once_with(mock_transcript)

    def test_execute_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        mock_existing = ContextualizedTranscript(source_id=source_id, chunks=[])
        mocks["repo"].get_by_source_id.return_value = mock_existing

        dummy_input = DomainPhoneticCorrectedTranscript(source_id=source_id, sentences=[], audit_trail=[])
        result = api.execute(dummy_input)

        assert result == mock_existing
        mocks["repo"].get_by_source_id.assert_called_once_with(source_id)
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_execute_passthrough_empty_transcript(self, api, mocks):
        mocks["repo"].get_by_source_id.return_value = None

        source_id = uuid.uuid4()
        dummy_input = DomainPhoneticCorrectedTranscript(source_id=source_id, sentences=[], audit_trail=[])

        api.execute(dummy_input)

        mocks["repo"].save.assert_called_once()
        saved_arg = mocks["repo"].save.call_args[0][0]
        assert len(saved_arg.chunks) == 0
        mocks["orchestrator"].acquire_model.assert_not_called()

    @patch("app.features.contextual_retrieval.service.contextualizer_api.build_rag_chunks")
    def test_execute_success_flow_and_vram_release(self, mock_chunker, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        dummy_sentence = _sentence("I have angina.")
        dummy_input = DomainPhoneticCorrectedTranscript(
            source_id=source_id, sentences=[dummy_sentence], audit_trail=[]
        )

        mock_chunker.return_value = [[dummy_sentence]]

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model
        mocks["slm_adapter"].count_tokens.return_value = 12
        mocks["slm_adapter"].generate_context.return_value = "Patient statement."
        mocks["repo"].save.side_effect = lambda x: x

        result = api.execute(dummy_input, release_after=True)

        labelled_text = "[S0 @ 00:00:00]: I have angina."
        mocks["orchestrator"].acquire_model.assert_called_once()
        # The SLM reads plain turns; the stored chunk keeps full timestamped labels
        mocks["slm_adapter"].generate_context.assert_called_once_with(
            model=mock_loaded_model,
            full_document_text="S0: I have angina.",
            chunk_text="S0: I have angina."
        )
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["slm_adapter"], flush=True
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].context_string == "Patient statement."
        assert result.chunks[0].original_text == labelled_text
        assert "Patient statement." in result.chunks[0].combined_text

    @patch("app.features.contextual_retrieval.service.contextualizer_api.settings")
    @patch("app.features.contextual_retrieval.service.contextualizer_api.build_rag_chunks")
    def test_execute_situates_chunks_in_sections_when_transcript_overflows(
            self, mock_chunker, mock_settings, api, mocks
    ):
        """A transcript larger than the context budget must not fail; chunks get a document section."""
        mock_settings.CONTEXTUAL_RETRIEVAL_N_CTX = 1502
        mock_settings.CONTEXTUAL_RETRIEVAL_RESERVED_TOKENS = 1500  # budget = 2 tokens

        mocks["repo"].get_by_source_id.return_value = None
        s0, s1, s2 = _sentence("First.", "S0", 0), _sentence("Second.", "S1", 5), _sentence("Third.", "S0", 10)
        dummy_input = DomainPhoneticCorrectedTranscript(source_id=uuid.uuid4(), sentences=[s0, s1, s2], audit_trail=[])
        mock_chunker.return_value = [[s0], [s1], [s2]]

        mocks["orchestrator"].acquire_model.return_value = MagicMock()
        # The whole document overflows; each single sentence costs 1 token
        mocks["slm_adapter"].count_tokens.side_effect = lambda model, text: 100 if "\n" in text else 1
        mocks["slm_adapter"].generate_context.return_value = "ctx"
        mocks["repo"].save.side_effect = lambda x: x

        api.execute(dummy_input)

        documents = [c.kwargs["full_document_text"] for c in mocks["slm_adapter"].generate_context.call_args_list]
        first_section = "S0: First.\nS1: Second."
        second_section = "S1: Second.\nS0: Third."
        assert documents == [first_section, first_section, second_section]

    def test_execute_guarantees_vram_release_on_crash(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        dummy_input = DomainPhoneticCorrectedTranscript(
            source_id=source_id, sentences=[_sentence("Test.")], audit_trail=[]
        )

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model
        mocks["slm_adapter"].count_tokens.return_value = 5

        mocks["slm_adapter"].generate_context.side_effect = SLMInferenceError("Out of Memory")

        with pytest.raises(SLMInferenceError):
            api.execute(dummy_input, release_after=True)

        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["slm_adapter"], flush=True
        )


class TestExecuteDocument:
    @pytest.fixture
    def mocks(self):
        mocks = {"repo": MagicMock(), "orchestrator": MagicMock(spec=ModelManagerOrchestrator), "slm_adapter": MagicMock()}
        mocks["repo"].get_by_source_id.return_value = None
        mocks["repo"].save.side_effect = lambda transcript: transcript
        mocks["slm_adapter"].count_tokens.return_value = 10
        mocks["slm_adapter"].generate_context.side_effect = ["A lease between Acme and Beta.", "Its termination clause."]
        return mocks

    @pytest.fixture
    def api(self, mocks):
        return ContextualizerAPI(repository=mocks["repo"], orchestrator=mocks["orchestrator"],
                                 slm_adapter=mocks["slm_adapter"])

    @pytest.fixture
    def document(self):
        return StructuredDocument(source_id=uuid.uuid4(), anchor_kind=AnchorKind.PAGE, chunks=[
            StructuredChunk(ordinal=0, text="LEASE\n\nMade between Acme and Beta.", page_start=1, page_end=1,
                            section_path="LEASE"),
            StructuredChunk(ordinal=1, text="Article 7 Termination\n\nThirty days' notice.", page_start=2,
                            page_end=3, section_path="LEASE › Article 7 Termination"),
        ])

    def test_chunks_keep_their_verbatim_text_and_page_anchor(self, api, mocks, document):
        result = api.execute_document(document)

        assert [c.original_text for c in result.chunks] == [c.text for c in document.chunks]
        assert [c.context_string for c in result.chunks] == ["A lease between Acme and Beta.", "Its termination clause."]
        second = result.chunks[1]
        assert (second.anchor_kind, second.page_start, second.page_end) == (AnchorKind.PAGE, 2, 3)
        assert second.section_path == "LEASE › Article 7 Termination"
        assert second.start_sec == 0.0 and second.sentences == []

    def test_slm_reads_the_whole_document_and_the_chunk_under_its_headings(self, api, mocks, document):
        api.execute_document(document)

        second_call = mocks["slm_adapter"].generate_context.call_args_list[1].kwargs
        assert second_call["full_document_text"] == "\n\n".join(c.text for c in document.chunks)
        assert second_call["chunk_text"] == "[LEASE › Article 7 Termination]\nArticle 7 Termination\n\nThirty days' notice."
        assert second_call["system_prompt"] == DOCUMENT_CONTEXTUAL_SYSTEM_PROMPT

    def test_model_is_released_after_the_document(self, api, mocks, document):
        api.execute_document(document)

        loaded = mocks["orchestrator"].acquire_model.return_value
        mocks["orchestrator"].release_model.assert_called_once_with(loaded, mocks["slm_adapter"], flush=True)

    def test_release_after_false_keeps_the_model_for_the_next_document(self, api, mocks, document):
        api.execute_document(document, release_after=False)

        mocks["orchestrator"].release_model.assert_not_called()

    def test_existing_record_skips_the_model(self, api, mocks, document):
        existing = ContextualizedTranscript(source_id=document.source_id, chunks=[])
        mocks["repo"].get_by_source_id.return_value = existing

        assert api.execute_document(document) is existing
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_long_document_is_situated_within_sections(self, api, mocks, document):
        # The whole document overflows the context window; each chunk on its own fits
        mocks["slm_adapter"].count_tokens.side_effect = [10 ** 6, 10, 10]

        with patch("app.features.contextual_retrieval.service.contextualizer_api.build_document_sections",
                   return_value=[(0, 1), (1, 2)]) as planner:
            api.execute_document(document)

        planner.assert_called_once()
        calls = mocks["slm_adapter"].generate_context.call_args_list
        assert calls[0].kwargs["full_document_text"] == document.chunks[0].text
        assert calls[1].kwargs["full_document_text"] == document.chunks[1].text

    def test_failure_releases_the_model_and_is_mapped(self, api, mocks, document):
        mocks["slm_adapter"].generate_context.side_effect = RuntimeError("llama_cpp generation crashed")

        with pytest.raises(SLMInferenceError):
            api.execute_document(document)
        mocks["orchestrator"].release_model.assert_called_once()
