# Path: integration_tests/features/qa_generation/test_conversation_generation_integration.py

import os
import uuid
import pytest

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.qa_generation.data.llama_cpp_generation_adapter import LlamaCppGenerationAdapter
from app.features.qa_generation.data.prompts import NO_EVIDENCE_ANSWER
from app.features.qa_generation.domain.entities import ConversationHistory, ConversationTurn, NumberedChunk
from app.features.qa_generation.service.generation_api import QAGenerationAPI
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.orchestration.utils.follow_up import answer_question

SOURCE_ID = uuid.uuid4()


def _chunk(start: float, text: str) -> RetrievedChunk:
    stamp = f"{int(start) // 3600:02d}:{int(start) % 3600 // 60:02d}:{int(start) % 60:02d}"
    return RetrievedChunk(
        chunk_id=uuid.uuid4(), source_id=SOURCE_ID, text_payload=text, raw_text=f"[Speaker_0 @ {stamp}]: {text}",
        speaker_labels=["Speaker_0"], start_sec=start, end_sec=start + 30, relevance_score=0.0
    )


@pytest.mark.integration
class TestConversationGenerationIntegration:
    """
    Real Qwen3.5-9B on the GPU, no database: a two-turn conversation whose follow-up can only be answered from the
    first turn's evidence, cited under the number that turn gave it.
    """

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.QA_GENERATION_MODEL_PATH):
            pytest.skip(f"QA generation model missing at: {settings.QA_GENERATION_MODEL_PATH}")

    @pytest.fixture
    def orchestrator(self):
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        yield orchestrator
        orchestrator.release_all({})

    @pytest.fixture
    def api(self, orchestrator) -> QAGenerationAPI:
        return QAGenerationAPI(adapter=LlamaCppGenerationAdapter(), orchestrator=orchestrator)

    def test_follow_up_cites_evidence_from_the_earlier_turn(self, api: QAGenerationAPI):
        entered = _chunk(540, "The defendant, wearing a red jacket, entered the bank at exactly nine in the morning.")
        weather = _chunk(900, "It was raining heavily across the city that entire day.")

        first = api.generate_conversation("What was the defendant wearing when he entered the bank?",
                                          [entered, weather], ConversationHistory())

        print(f"\n[TURN 1] {first.response.generated_answer}")
        assert first.response.generated_answer != NO_EVIDENCE_ANSWER
        assert "red" in first.response.generated_answer.lower()
        assert "<think>" not in first.response.generated_answer
        assert first.new_chunk_ids == [entered.chunk_id, weather.chunk_id]
        assert entered.chunk_id in {c.chunk_id for c in first.response.citations}

        history = ConversationHistory(
            turns=[ConversationTurn(
                question="What was the defendant wearing when he entered the bank?",
                answer=first.response.generated_answer,
                evidence=[NumberedChunk(doc_number=1, chunk=entered), NumberedChunk(doc_number=2, chunk=weather)]
            )],
            next_doc_number=3
        )
        # The follow-up's own search found nothing about the time; only turn 1's evidence has it
        unrelated = _chunk(3000, "The parking lot was repaved the following week.")

        # As LiveQAPipeline asks it: the user's words plus the standalone form the follow-up was searched with
        question = answer_question("And at what time did he walk in?",
                                   "At what time did the defendant walk into the bank?")
        second = api.generate_conversation(question, [unrelated], history)

        print(f"[TURN 2] {second.response.generated_answer}")
        answer = second.response.generated_answer.lower()
        assert "cannot answer" not in answer
        assert "nine" in answer or "9" in answer
        assert "[Doc 1]" in second.response.generated_answer
        cited = {c.doc_id: c.chunk_id for c in second.response.citations}
        assert cited.get("[Doc 1]") == entered.chunk_id
        assert second.new_chunk_ids == [unrelated.chunk_id]

    def test_long_context_window_loads_within_the_vram_budget(self, orchestrator):
        """The configured 128K window (q8_0 KV cache) must fit next to the weights on this GPU."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA device available.")

        adapter = LlamaCppGenerationAdapter()
        manifest = ModelManifest(model_type=ModelType.QA_GENERATION, model_path=settings.QA_GENERATION_MODEL_PATH,
                                 required_vram_mb=settings.QA_GENERATION_VRAM_REQUIREMENT_MB)
        loaded = orchestrator.acquire_model(manifest, adapter)
        try:
            assert loaded.model_instance.n_ctx() == settings.QA_GENERATION_N_CTX
            free, total = torch.cuda.mem_get_info()
            used_mb = (total - free) / (1024 * 1024)
            print(f"\n[VRAM] {used_mb:.0f} MB used of {total / (1024 * 1024):.0f} MB at n_ctx={settings.QA_GENERATION_N_CTX}")
            assert free > 0
        finally:
            orchestrator.release_model(loaded, adapter, flush=True)
