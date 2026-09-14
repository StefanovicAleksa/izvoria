# Path: tests/features/rag_reranking/service/test_rerank_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.core.model_manager.domain.enums import ModelType
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.rag_reranking.service.rerank_api import RerankAPI
from app.features.rag_reranking.exceptions.model import RerankInferenceError


def _chunk(payload: str, rrf: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=payload,
                          start_sec=0, end_sec=1, relevance_score=rrf)


class TestRerankAPI:
    @pytest.fixture
    def mocks(self):
        return {"orchestrator": MagicMock(), "adapter": MagicMock()}

    @pytest.fixture
    def api(self, mocks):
        return RerankAPI(orchestrator=mocks["orchestrator"], adapter=mocks["adapter"])

    def test_manifest_uses_reranker_settings(self, api):
        assert api.manifest.model_type == ModelType.RAG_RERANKING
        assert api.manifest.model_path == settings.RAG_RERANKER_MODEL_PATH
        assert api.manifest.required_vram_mb == settings.RAG_RERANKER_VRAM_REQUIREMENT_MB

    @patch("app.features.rag_reranking.service.rerank_api.os.path.exists", return_value=False)
    def test_missing_weights_pass_through_in_rrf_order(self, _, api, mocks):
        chunks = [_chunk("a", 0.3), _chunk("b", 0.2), _chunk("c", 0.1)]

        result = api.rerank("query", chunks, top_n=2)

        assert result == chunks[:2]
        mocks["orchestrator"].acquire_model.assert_not_called()

    @patch("app.features.rag_reranking.service.rerank_api.os.path.exists", return_value=True)
    def test_rerank_orders_by_cross_encoder_score_and_truncates(self, _, api, mocks):
        chunks = [_chunk("a", 0.3), _chunk("b", 0.2), _chunk("c", 0.1)]
        loaded = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded
        mocks["adapter"].score.return_value = [0.1, 0.9, 0.5]

        result = api.rerank("what happened?", chunks, top_n=2)

        assert [c.chunk_id for c in result] == [chunks[1].chunk_id, chunks[2].chunk_id]
        assert [c.rerank_score for c in result] == [0.9, 0.5]
        mocks["orchestrator"].acquire_model.assert_called_once_with(api.manifest, mocks["adapter"], wait_for_clear=True)
        mocks["adapter"].score.assert_called_once_with(loaded, "what happened?", ["a", "b", "c"])
        mocks["orchestrator"].release_model.assert_called_once_with(loaded, mocks["adapter"], flush=True)

    @patch("app.features.rag_reranking.service.rerank_api.os.path.exists", return_value=True)
    def test_equal_scores_keep_rrf_order(self, _, api, mocks):
        chunks = [_chunk("a", 0.3), _chunk("b", 0.2)]
        mocks["adapter"].score.return_value = [0.5, 0.5]

        result = api.rerank("q", chunks, top_n=2)

        assert [c.chunk_id for c in result] == [chunks[0].chunk_id, chunks[1].chunk_id]

    @patch("app.features.rag_reranking.service.rerank_api.os.path.exists", return_value=True)
    def test_release_after_false_keeps_model(self, _, api, mocks):
        mocks["adapter"].score.return_value = [0.5]

        api.rerank("q", [_chunk("a", 0.1)], top_n=1, release_after=False)

        mocks["orchestrator"].release_model.assert_not_called()

    @patch("app.features.rag_reranking.service.rerank_api.os.path.exists", return_value=True)
    def test_score_count_mismatch_raises_and_releases(self, _, api, mocks):
        loaded = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded
        mocks["adapter"].score.return_value = [0.5]

        with pytest.raises(RerankInferenceError, match="1 scores for 2 chunks"):
            api.rerank("q", [_chunk("a", 0.1), _chunk("b", 0.1)], top_n=2)

        mocks["orchestrator"].release_model.assert_called_once_with(loaded, mocks["adapter"], flush=True)

    @patch("app.features.rag_reranking.service.rerank_api.os.path.exists", return_value=True)
    def test_inference_crash_is_mapped_and_releases(self, _, api, mocks):
        loaded = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded
        mocks["adapter"].score.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(RerankInferenceError):
            api.rerank("q", [_chunk("a", 0.1)], top_n=1)

        mocks["orchestrator"].release_model.assert_called_once_with(loaded, mocks["adapter"], flush=True)

    def test_empty_candidates_return_empty(self, api, mocks):
        assert api.rerank("q", [], top_n=5) == []
        mocks["orchestrator"].acquire_model.assert_not_called()
