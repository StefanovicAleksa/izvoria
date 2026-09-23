# Canvas: test_rag_retrieval_api_v2
# Path: tests/features/rag_retrieval/service/test_retrieval_api.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.core.config import settings
from app.features.rag_retrieval.exceptions import IndexLanguageMismatchError
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.rag_retrieval.domain.entities import CorpusSweep, SearchQuery, RetrievedChunk
from app.core.model_manager.domain.entities import LoadedModel


class TestRetrievalAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "orchestrator": MagicMock(),
            "embedding_adapter": MagicMock(),
            "search_repo": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return RetrievalAPI(
            orchestrator=mocks["orchestrator"],
            embedding_adapter=mocks["embedding_adapter"],
            search_repo=mocks["search_repo"]
        )

    def test_search_success_flow(self, api, mocks):
        query = SearchQuery(prompt="test search", limit=5)

        # Mock VRAM Load
        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        # Mock Embedder
        mocks["embedding_adapter"].embed_prompt.return_value = [0.1, 0.2]

        # Mock DB Search
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            text_payload="Match",
            start_sec=0,
            end_sec=5,
            relevance_score=0.05,
            extracted_file_id=uuid.uuid4()
        )
        mocks["search_repo"].execute_hybrid_search.return_value = [chunk]

        result = api.search(query)

        assert result.query == "test search"
        assert len(result.results) == 1

        # Verify Orchestrator Lifecycle (Zero-Leakage check)
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["embedding_adapter"], flush=True
        )

    def test_search_guarantees_vram_release_on_error(self, api, mocks):
        query = SearchQuery(prompt="crash test", limit=5)

        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        # Force a crash during embedding
        mocks["embedding_adapter"].embed_prompt.side_effect = Exception("Model Crash")

        with pytest.raises(Exception):
            api.search(query)

        # Ensure the finally block released the VRAM even during a crash
        mocks["orchestrator"].release_model.assert_called_once()

    def test_search_fetches_the_reranker_candidate_pool(self, api, mocks):
        from app.core.config import settings

        source_id = uuid.uuid4()
        mocks["orchestrator"].acquire_model.return_value = MagicMock(spec=LoadedModel)
        mocks["embedding_adapter"].embed_prompt.return_value = [0.1, 0.2]
        mocks["search_repo"].execute_hybrid_search.return_value = []

        api.search(SearchQuery(prompt="what did the witness say", limit=5, source_id_filter=source_id))

        mocks["search_repo"].execute_hybrid_search.assert_called_once_with(
            query_vector=[0.1, 0.2],
            query_text="what did the witness say",
            limit=settings.RAG_RETRIEVAL_CANDIDATES,
            source_id_filter=source_id
        )

    @staticmethod
    def _chunk(source_id=None, start=0.0) -> RetrievedChunk:
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=source_id or uuid.uuid4(), text_payload="t",
                              start_sec=start, end_sec=start + 1, relevance_score=0.5)

    def test_expand_neighbors_appends_new_neighbours_of_the_top_chunks(self, api, mocks):
        source = uuid.uuid4()
        chunks = [self._chunk(source, 50), self._chunk(source, 60), self._chunk(source, 90)]
        after, before = self._chunk(source, 70), self._chunk(source, 40)
        # chunks[1] is itself a neighbour of chunks[0] and must not be added twice
        mocks["search_repo"].fetch_neighbors.return_value = [after, chunks[1], before]

        result = api.expand_neighbors(chunks, top_k=2)

        mocks["search_repo"].fetch_neighbors.assert_called_once_with([chunks[0].chunk_id, chunks[1].chunk_id])
        assert result == chunks + [before, after]
        mocks["orchestrator"].acquire_model.assert_not_called()

    @pytest.mark.parametrize("top_k, chunk_count", [(0, 2), (3, 0)])
    def test_expand_neighbors_noop_cases_skip_the_database(self, api, mocks, top_k, chunk_count):
        chunks = [self._chunk() for _ in range(chunk_count)]

        assert api.expand_neighbors(chunks, top_k=top_k) == chunks
        mocks["search_repo"].fetch_neighbors.assert_not_called()

    def test_expand_neighbors_maps_errors(self, api, mocks):
        mocks["search_repo"].fetch_neighbors.side_effect = RuntimeError("db down")

        with pytest.raises(Exception) as exc_info:
            api.expand_neighbors([self._chunk()], top_k=1)
        assert "neighbouring" in str(exc_info.value)

    def test_get_chunks_delegates_to_the_repository_without_a_model(self, api, mocks):
        ids = [uuid.uuid4(), uuid.uuid4()]
        mocks["search_repo"].fetch_chunks.return_value = ["chunk"]

        assert api.get_chunks(ids) == ["chunk"]
        mocks["search_repo"].fetch_chunks.assert_called_once_with(ids)
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_get_chunks_maps_errors(self, api, mocks):
        mocks["search_repo"].fetch_chunks.side_effect = RuntimeError("db down")

        with pytest.raises(Exception) as exc_info:
            api.get_chunks([uuid.uuid4()])
        assert "by id" in str(exc_info.value)


class TestSearchBatch:
    """
    `search_batch` is what the Deep Research Report mode's model-swap budget rests on: a whole fan-out
    plus a per-document sweep under ONE embedder acquisition, instead of one 2.2 GB load per query.
    """

    @pytest.fixture
    def mocks(self):
        return {
            "orchestrator": MagicMock(),
            "embedding_adapter": MagicMock(),
            "search_repo": MagicMock(),
        }

    @pytest.fixture
    def api(self, mocks):
        return RetrievalAPI(
            orchestrator=mocks["orchestrator"],
            embedding_adapter=mocks["embedding_adapter"],
            search_repo=mocks["search_repo"],
        )

    @staticmethod
    def _chunk(source_id=None) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=uuid.uuid4(),
            source_id=source_id or uuid.uuid4(),
            text_payload="Match",
            start_sec=0,
            end_sec=10,
            relevance_score=0.9,
        )

    def test_the_embedder_is_acquired_and_released_exactly_once_for_the_whole_batch(self, api, mocks):
        queries = [SearchQuery(prompt=f"question {i}", limit=10) for i in range(8)]
        mocks["embedding_adapter"].embed_prompts.return_value = [[0.1]] * 8
        mocks["search_repo"].execute_hybrid_search.return_value = []

        api.search_batch(queries)

        assert mocks["orchestrator"].acquire_model.call_count == 1
        assert mocks["orchestrator"].release_model.call_count == 1
        assert mocks["embedding_adapter"].embed_prompts.call_count == 1
        assert mocks["embedding_adapter"].embed_prompt.call_count == 0

    def test_every_query_still_runs_its_own_search(self, api, mocks):
        queries = [SearchQuery(prompt=f"question {i}", limit=10) for i in range(3)]
        mocks["embedding_adapter"].embed_prompts.return_value = [[0.1], [0.2], [0.3]]
        mocks["search_repo"].execute_hybrid_search.return_value = [self._chunk()]

        result = api.search_batch(queries)

        assert mocks["search_repo"].execute_hybrid_search.call_count == 3
        assert [r.query for r in result.results] == ["question 0", "question 1", "question 2"]

    def test_each_query_is_searched_with_its_own_vector(self, api, mocks):
        queries = [SearchQuery(prompt="alpha", limit=10), SearchQuery(prompt="beta", limit=10)]
        mocks["embedding_adapter"].embed_prompts.return_value = [[1.0], [2.0]]
        mocks["search_repo"].execute_hybrid_search.return_value = []

        api.search_batch(queries)

        vectors = [c.kwargs["query_vector"] for c in mocks["search_repo"].execute_hybrid_search.call_args_list]
        assert vectors == [[1.0], [2.0]]

    def test_a_repeated_prompt_is_embedded_once(self, api, mocks):
        queries = [SearchQuery(prompt="same", limit=10), SearchQuery(prompt="same", limit=10)]
        mocks["embedding_adapter"].embed_prompts.return_value = [[0.5]]
        mocks["search_repo"].execute_hybrid_search.return_value = []

        api.search_batch(queries)

        assert mocks["embedding_adapter"].embed_prompts.call_args.args[1] == ["same"]
        assert mocks["search_repo"].execute_hybrid_search.call_count == 2

    def test_the_sweep_reuses_the_topic_vector_rather_than_re_embedding_per_source(self, api, mocks):
        """
        The sweep issues one query per picked source, all with the same text. Embedding that text once
        is the difference between one forward pass and fifty.
        """
        from app.features.rag_retrieval.domain.entities import CorpusSweep

        sweep = CorpusSweep(prompt="the topic", per_source_limit=3, max_sources=50, ann_depth=2000)
        mocks["embedding_adapter"].embed_prompts.return_value = [[0.7]]
        mocks["search_repo"].pick_sweep_sources.return_value = [uuid.uuid4() for _ in range(4)]
        mocks["search_repo"].execute_hybrid_search.return_value = [self._chunk()]

        result = api.search_batch([], sweep=sweep)

        assert mocks["embedding_adapter"].embed_prompts.call_args.args[1] == ["the topic"]
        assert mocks["search_repo"].execute_hybrid_search.call_count == 4
        assert result.swept_source_count == 4
        assert len(result.sweep_results) == 4

    def test_each_swept_source_gets_a_filtered_hybrid_search_not_a_raw_ann_slice(self, api, mocks):
        """
        The ANN probe only decides WHERE to look. Letting it also decide what to return would drop the
        keyword half of retrieval on exactly the sources the fan-out already failed to surface.
        """
        from app.features.rag_retrieval.domain.entities import CorpusSweep

        source_ids = [uuid.uuid4(), uuid.uuid4()]
        mocks["embedding_adapter"].embed_prompts.return_value = [[0.7]]
        mocks["search_repo"].pick_sweep_sources.return_value = source_ids
        mocks["search_repo"].execute_hybrid_search.return_value = []

        api.search_batch([], sweep=CorpusSweep(prompt="topic", per_source_limit=3))

        filters = [c.kwargs["source_id_filter"] for c in mocks["search_repo"].execute_hybrid_search.call_args_list]
        assert filters == source_ids
        for call in mocks["search_repo"].execute_hybrid_search.call_args_list:
            assert call.kwargs["query_text"] == "topic"
            assert call.kwargs["limit"] == 3

    def test_a_sweep_that_picks_no_sources_returns_cleanly(self, api, mocks):
        from app.features.rag_retrieval.domain.entities import CorpusSweep

        mocks["embedding_adapter"].embed_prompts.return_value = [[0.7]]
        mocks["search_repo"].pick_sweep_sources.return_value = []

        result = api.search_batch([], sweep=CorpusSweep(prompt="topic"))

        assert result.sweep_results == []
        assert result.swept_source_count == 0

    def test_an_empty_batch_loads_no_model_at_all(self, api, mocks):
        result = api.search_batch([])

        assert result.results == []
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_the_embedder_is_released_even_when_embedding_raises(self, api, mocks):
        mocks["embedding_adapter"].embed_prompts.side_effect = RuntimeError("CUDA fell over")

        with pytest.raises(Exception):
            api.search_batch([SearchQuery(prompt="q", limit=10)])

        mocks["orchestrator"].release_model.assert_called_once()

    def test_the_candidate_pool_is_widened_for_the_reranker(self, api, mocks):
        mocks["embedding_adapter"].embed_prompts.return_value = [[0.1]]
        mocks["search_repo"].execute_hybrid_search.return_value = []

        api.search_batch([SearchQuery(prompt="q", limit=10)], candidates=40)

        assert mocks["search_repo"].execute_hybrid_search.call_args.kwargs["limit"] == 40


class TestListIndexedSources:
    @pytest.fixture
    def mocks(self):
        return {
            "orchestrator": MagicMock(),
            "embedding_adapter": MagicMock(),
            "search_repo": MagicMock(),
        }

    @pytest.fixture
    def api(self, mocks):
        return RetrievalAPI(
            orchestrator=mocks["orchestrator"],
            embedding_adapter=mocks["embedding_adapter"],
            search_repo=mocks["search_repo"],
        )

    def test_reads_the_indexed_set_without_loading_a_model(self, api, mocks):
        """
        Distinct from storage's list_sources(), which reports what was *ingested*. Corpus-coverage
        claims have to be based on what is actually retrievable.
        """
        mocks["search_repo"].list_indexed_sources.return_value = []

        api.list_indexed_sources()

        mocks["search_repo"].list_indexed_sources.assert_called_once_with(1)
        mocks["orchestrator"].acquire_model.assert_not_called()


class TestRetrievalAPIScript:
    """A Serbian index is Latin, so a query typed in Cyrillic is searched in Latin."""

    @pytest.fixture
    def mocks(self):
        return {
            "orchestrator": MagicMock(),
            "embedding_adapter": MagicMock(**{"embed_prompt.return_value": [0.1],
                                              "embed_prompts.side_effect": lambda _, prompts: [[0.1]] * len(prompts)}),
            "search_repo": MagicMock(**{"execute_hybrid_search.return_value": [],
                                        "pick_sweep_sources.return_value": [uuid.uuid4()]}),
        }

    @pytest.fixture
    def api(self, mocks):
        return RetrievalAPI(mocks["orchestrator"], mocks["embedding_adapter"], mocks["search_repo"])

    def test_search_embeds_and_keyword_matches_the_latin_text(self, api, mocks, appliance_language):
        appliance_language("sr")

        result = api.search(SearchQuery(prompt="Шта је лекар рекао?", limit=5))

        assert mocks["embedding_adapter"].embed_prompt.call_args.args[1] == "Šta je lekar rekao?"
        assert mocks["search_repo"].execute_hybrid_search.call_args.kwargs["query_text"] == "Šta je lekar rekao?"
        assert result.query == "Шта је лекар рекао?"

    def test_search_batch_keeps_results_keyed_by_the_callers_prompt(self, api, mocks, appliance_language):
        appliance_language("sr")

        result = api.search_batch([SearchQuery(prompt="терапија", limit=5)],
                                  sweep=CorpusSweep(prompt="дијагноза"))

        assert mocks["embedding_adapter"].embed_prompts.call_args.args[1] == ["terapija", "dijagnoza"]
        texts = [c.kwargs["query_text"] for c in mocks["search_repo"].execute_hybrid_search.call_args_list]
        assert texts == ["terapija", "dijagnoza"]
        assert [r.query for r in result.results] == ["терапија"]

    def test_english_prompt_is_searched_as_written(self, api, mocks):
        api.search(SearchQuery(prompt="What did Лекар say?", limit=5))

        assert mocks["embedding_adapter"].embed_prompt.call_args.args[1] == "What did Лекар say?"


class TestVerifyIndexLanguage:
    @pytest.fixture
    def repo(self):
        return MagicMock()

    @pytest.fixture
    def api(self, repo):
        return RetrievalAPI(MagicMock(), MagicMock(), repo)

    @pytest.mark.parametrize("indexed", ["english", None])
    def test_matching_or_absent_index_passes(self, api, repo, indexed):
        repo.get_index_fts_config.return_value = indexed

        api.verify_index_language()

    def test_index_built_for_another_language_is_refused(self, api, repo, monkeypatch):
        monkeypatch.setattr(settings, "FTS_LANGUAGE_CONFIG", "serbian")
        repo.get_index_fts_config.return_value = "english"

        with pytest.raises(IndexLanguageMismatchError, match="'english'.*'serbian'"):
            api.verify_index_language()
