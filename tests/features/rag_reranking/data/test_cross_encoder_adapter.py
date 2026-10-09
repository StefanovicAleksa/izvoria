# Path: tests/features/rag_reranking/data/test_cross_encoder_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.rag_reranking.data.cross_encoder_adapter import CrossEncoderRerankerAdapter
from app.features.rag_reranking.exceptions.model import RerankerModelLoadError, RerankInferenceError


class TestCrossEncoderRerankerAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.RAG_RERANKING,
            model_path="/mock/path/bge-reranker-v2-m3",
            required_vram_mb=1500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(CrossEncoderRerankerAdapter, "_get_cross_encoder")
    def test_load_success(self, mock_get_encoder, _, manifest):
        encoder = MagicMock()
        mock_get_encoder.return_value = encoder

        loaded = CrossEncoderRerankerAdapter().load(manifest)

        assert loaded.model_instance == encoder
        mock_get_encoder.assert_called_once_with(manifest.model_path)

    @patch("os.path.exists", return_value=False)
    def test_load_missing_weights_raises(self, _, manifest):
        with pytest.raises(RerankerModelLoadError):
            CrossEncoderRerankerAdapter().load(manifest)

    def test_score_requires_loaded_model(self, manifest):
        with pytest.raises(RerankInferenceError, match="not loaded"):
            CrossEncoderRerankerAdapter().score(LoadedModel(manifest=manifest, model_instance=None), "q", ["a"])

    def test_score_pairs_query_with_each_text(self, manifest):
        encoder = MagicMock()
        encoder.predict.return_value = [0.25, 0.75]
        loaded = LoadedModel(manifest=manifest, model_instance=encoder)

        scores = CrossEncoderRerankerAdapter().score(loaded, "what was prescribed?", ["chunk a", "chunk b"])

        assert scores == [0.25, 0.75]
        encoder.predict.assert_called_once_with(
            [("what was prescribed?", "chunk a"), ("what was prescribed?", "chunk b")],
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def test_score_empty_texts_skips_inference(self, manifest):
        encoder = MagicMock()
        loaded = LoadedModel(manifest=manifest, model_instance=encoder)

        assert CrossEncoderRerankerAdapter().score(loaded, "q", []) == []
        encoder.predict.assert_not_called()

    @patch("app.features.rag_reranking.data.cross_encoder_adapter.gc.collect")
    @patch("app.features.rag_reranking.data.cross_encoder_adapter.torch.cuda.is_available", return_value=False)
    def test_unload_drops_instance(self, _, mock_gc, manifest):
        loaded = LoadedModel(manifest=manifest, model_instance=MagicMock())

        assert CrossEncoderRerankerAdapter().unload(loaded) is True
        assert not hasattr(loaded, "model_instance")
        mock_gc.assert_called_once()


class TestConcurrentScoring:
    """One resident reranker serves every concurrent request; its fast tokenizer must never run on two threads."""

    def test_scores_from_several_threads_take_turns(self):
        import threading
        import time
        state = {"active": 0, "most": 0}
        lock = threading.Lock()

        def predict(pairs, **_):
            with lock:
                state["active"] += 1
                state["most"] = max(state["most"], state["active"])
            time.sleep(0.02)
            with lock:
                state["active"] -= 1
            return [0.5] * len(pairs)

        encoder = MagicMock()
        encoder.predict.side_effect = predict
        model = LoadedModel(manifest=ModelManifest(model_type=ModelType.RAG_RERANKING, model_path="/m",
                                                   required_vram_mb=1.0), model_instance=encoder)
        adapter = CrossEncoderRerankerAdapter()

        threads = [threading.Thread(target=adapter.score, args=(model, "q", ["a", "b"])) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert encoder.predict.call_count == 5
        assert state["most"] == 1

    def test_a_failed_predict_releases_the_lock(self):
        encoder = MagicMock()
        encoder.predict.side_effect = [RuntimeError("CUDA error"), [0.9]]
        model = LoadedModel(manifest=ModelManifest(model_type=ModelType.RAG_RERANKING, model_path="/m",
                                                   required_vram_mb=1.0), model_instance=encoder)
        adapter = CrossEncoderRerankerAdapter()

        with pytest.raises(RerankInferenceError):
            adapter.score(model, "q", ["a"])

        assert adapter.score(model, "q", ["b"]) == [0.9]
