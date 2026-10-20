# Canvas: test_storage_api_v4
# Path: tests/features/rag_storage/service/test_storage_api.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk
from app.features.rag_storage.service.storage_api import RAGStorageService
from app.features.rag_storage.exceptions.base import RAGStorageError
from app.core.model_manager.domain.entities import LoadedModel
from app.features.rag_storage.domain.entities import StoredChunkText
from app.features.source_profile.domain.entities import SourceProfile
from app.features.source_profile.domain.enums import ProfileOrigin


class TestRAGStorageService:
    @pytest.fixture
    def mocks(self):
        return {
            "rag_repo": MagicMock(),
            "embedder": MagicMock(),
            "orchestrator": MagicMock()
        }

    @pytest.fixture
    def service(self, mocks):
        return RAGStorageService(
            rag_repo=mocks["rag_repo"],
            orchestrator=mocks["orchestrator"],
            embedder=mocks["embedder"]
        )

    def test_index_transcript_success_flow(self, service, mocks):
        source_id = uuid.uuid4()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Test​ string")
        transcript = UnifiedTranscript(source_id=source_id, chunks=[chunk])

        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        mocks["embedder"].embed_batch.return_value = [[0.1, 0.2]]
        mocks["rag_repo"].replace_rag_chunks.return_value = 1

        result = service.index_transcript(source_id, transcript)

        assert result.success is True
        assert result.telemetry.dense_vectors_saved == 1

        # Verify Payload Normalization reached the embedder
        normalized_texts = mocks["embedder"].embed_batch.call_args[0][1]
        assert normalized_texts == ["Test string"]

        # Verify a single atomic replace ran with the normalized chunks and their embeddings
        replace_args = mocks["rag_repo"].replace_rag_chunks.call_args[0]
        assert replace_args[0] == source_id
        assert [c.text_payload for c in replace_args[1]] == ["Test string"]
        assert replace_args[2] == [[0.1, 0.2]]
        mocks["rag_repo"].delete_by_source_id.assert_not_called()

        # Verify Orchestrator Lifecycle
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["embedder"], flush=True
        )

    def test_index_transcript_empty_chunks_clears_existing_index(self, service, mocks):
        source_id = uuid.uuid4()
        transcript = UnifiedTranscript(source_id=source_id, chunks=[])

        result = service.index_transcript(source_id, transcript)

        assert result.success is True
        mocks["rag_repo"].delete_by_source_id.assert_called_once_with(source_id)
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_index_transcript_raises_and_keeps_old_index_on_embedding_error(self, service, mocks):
        source_id = uuid.uuid4()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Crash text")
        transcript = UnifiedTranscript(source_id=source_id, chunks=[chunk])

        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        mocks["embedder"].embed_batch.side_effect = Exception("Inference Error")

        with pytest.raises(RAGStorageError, match="Inference Error"):
            service.index_transcript(source_id, transcript)

        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["embedder"], flush=True
        )
        mocks["rag_repo"].replace_rag_chunks.assert_not_called()
        mocks["rag_repo"].delete_by_source_id.assert_not_called()


class TestRAGStorageServiceSourceLine:
    """Each chunk's indexed context names its source, so a question naming the source finds its chunks."""

    @pytest.fixture
    def mocks(self):
        mocks = {"rag_repo": MagicMock(), "embedder": MagicMock(), "orchestrator": MagicMock(),
                 "profile_api": MagicMock()}
        mocks["profile_api"].get.return_value = SourceProfile(
            source_id=uuid.uuid4(), title="Statistics 110, Lecture 4", topic="Conditional probability.",
            origin=ProfileOrigin.GENERATED)
        mocks["embedder"].embed_batch.side_effect = lambda model, texts: [[0.5] for _ in texts]
        return mocks

    @pytest.fixture
    def service(self, mocks):
        return RAGStorageService(rag_repo=mocks["rag_repo"], orchestrator=mocks["orchestrator"],
                                 embedder=mocks["embedder"], profile_api=mocks["profile_api"])

    def test_indexing_puts_the_source_line_in_context_and_payload_never_raw_text(self, service, mocks):
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1,
                             text_payload="[Audio Context: Bayes.]\n\nSpeaker_0: hello",
                             raw_text="Speaker_0: hello", context_text="[Audio Context: Bayes.]")

        service.index_transcript(uuid.uuid4(), UnifiedTranscript(source_id=uuid.uuid4(), chunks=[chunk]))

        stored = mocks["rag_repo"].replace_rag_chunks.call_args.args[1][0]
        line = "[Source: Statistics 110, Lecture 4 — Conditional probability.]"
        assert stored.context_text == f"{line}\n[Audio Context: Bayes.]"
        assert stored.text_payload == f"{line} [Audio Context: Bayes.] Speaker_0: hello"
        assert stored.raw_text == "Speaker_0: hello"

    def test_a_failed_profile_read_indexes_without_the_line(self, service, mocks):
        mocks["profile_api"].get.side_effect = RuntimeError("db down")
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="text")

        result = service.index_transcript(uuid.uuid4(), UnifiedTranscript(source_id=uuid.uuid4(), chunks=[chunk]))

        assert result.success is True
        assert mocks["rag_repo"].replace_rag_chunks.call_args.args[1][0].text_payload == "text"

    def test_refresh_rewrites_the_line_and_re_embeds_in_place(self, service, mocks):
        source_id = uuid.uuid4()
        first, second = uuid.uuid4(), uuid.uuid4()
        mocks["rag_repo"].read_chunk_texts.return_value = [
            StoredChunkText(chunk_id=first, context_text="[Source: old]\n[Audio Context: a]", raw_text="one"),
            StoredChunkText(chunk_id=second, context_text=None, raw_text="two"),
        ]
        mocks["rag_repo"].update_chunk_payloads.return_value = 2

        assert service.refresh_source_line(source_id) == 2

        line = "[Source: Statistics 110, Lecture 4 — Conditional probability.]"
        args = mocks["rag_repo"].update_chunk_payloads.call_args.args
        assert args[0] == source_id
        assert [(c.chunk_id, c.context_text, c.raw_text) for c in args[1]] == [
            (first, f"{line}\n[Audio Context: a]", "one"), (second, line, "two")]
        assert args[2] == [f"{line} [Audio Context: a] one", f"{line} two"]
        assert args[3] == [[0.5], [0.5]]
        assert mocks["orchestrator"].acquire_model.call_args.kwargs == {"wait_for_clear": True}
        mocks["orchestrator"].release_model.assert_called_once()

    def test_refresh_without_a_profile_touches_nothing(self, service, mocks):
        mocks["profile_api"].get.return_value = None

        assert service.refresh_source_line(uuid.uuid4()) == 0
        mocks["orchestrator"].acquire_model.assert_not_called()
        mocks["rag_repo"].update_chunk_payloads.assert_not_called()

    def test_stored_texts_are_the_raw_texts_in_order(self, service, mocks):
        mocks["rag_repo"].read_chunk_texts.return_value = [
            StoredChunkText(chunk_id=uuid.uuid4(), context_text="c", raw_text="one"),
            StoredChunkText(chunk_id=uuid.uuid4(), context_text="c", raw_text=""),
            StoredChunkText(chunk_id=uuid.uuid4(), context_text=None, raw_text="two")]

        assert service.stored_texts(uuid.uuid4()) == ["one", "two"]
