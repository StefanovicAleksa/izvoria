# Path: tests/features/domain_adaptation/service/test_adaptation_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI
from app.features.domain_adaptation.domain.entities import DomainGlossary, ScoutResult, TimeSlice
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.domain_adaptation.exceptions.model import ScoutInferenceError


class TestDomainAdaptationAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "orchestrator": MagicMock(),
            "scout_adapter": MagicMock(),
            "llama_adapter": MagicMock(),
            "sampler": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return DomainAdaptationAPI(
            repository=mocks["repo"],
            orchestrator=mocks["orchestrator"],
            scout_adapter=mocks["scout_adapter"],
            llama_adapter=mocks["llama_adapter"],
            smart_sampler=mocks["sampler"]
        )

    def test_process_domain_adaptation_idempotent(self, api, mocks):
        source_id = uuid.uuid4()
        existing_glossary = DomainGlossary(
            source_id=source_id, industry="FINANCE", sub_domains=[], hotword_string="", initial_prompt_string=""
        )
        mocks["repo"].get_glossary_by_source_id.return_value = existing_glossary

        result = api.process_domain_adaptation(source_id, "/dummy.wav", VadMap(source_id=source_id, segments=[]))

        assert result == existing_glossary
        mocks["orchestrator"].acquire_model.assert_not_called()

    @patch("app.features.domain_adaptation.service.adaptation_api.settings")
    @patch("app.features.domain_adaptation.service.adaptation_api.GlossaryBuilder")
    def test_process_domain_adaptation_success_flow(self, mock_builder, mock_settings, api, mocks):
        # FIX: Explicitly map all possible paths to strings so Pydantic doesn't receive a MagicMock
        mock_settings.APPLIANCE_INDUSTRY = "FINANCE"
        mock_settings.DISCOVERY_SCOUT_MODEL_PATH = "/mock/scout"
        mock_settings.DOMAIN_CLASSIFIER_MODEL_PATH = "/mock/llama.gguf"
        mock_settings.ASR_PHASE_1_MODEL_PATH = "/mock/llama.gguf"

        source_id = uuid.uuid4()
        mocks["repo"].get_glossary_by_source_id.return_value = None

        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=10, confidence=0.9)])
        mocks["sampler"].calculate_optimal_slices.return_value = [TimeSlice(start_sec=0, end_sec=10)]

        loaded_scout_mock = MagicMock()
        loaded_llama_mock = MagicMock()
        mocks["orchestrator"].acquire_model.side_effect = [loaded_scout_mock, loaded_llama_mock]

        mocks["scout_adapter"].transcribe_samples.return_value = ScoutResult(
            source_id=source_id, combined_text="We are talking about mergers."
        )

        mocks["llama_adapter"].predict_subdomains.return_value = ["M_AND_A"]

        mock_builder.get_candidate_labels.return_value = ["M_AND_A"]
        mock_builder.build_injection_payloads.return_value = ("merger:15.0", "Prompt text.")

        saved_glossary = DomainGlossary(
            source_id=source_id,
            industry="FINANCE",
            sub_domains=["M_AND_A"],
            hotword_string="merger:15.0",
            initial_prompt_string="Prompt text."
        )
        mocks["repo"].save_glossary.return_value = saved_glossary

        result = api.process_domain_adaptation(source_id, "/dummy.wav", vad_map)

        assert result == saved_glossary
        assert mocks["orchestrator"].acquire_model.call_count == 2
        assert mocks["orchestrator"].release_model.call_count == 2

    @patch("app.features.domain_adaptation.service.adaptation_api.map_domain_adaptation_error")
    @patch("app.features.domain_adaptation.service.adaptation_api.settings")
    def test_process_guarantees_vram_release_on_crash(self, mock_settings, mock_error_mapper, api, mocks):
        # FIX: Explicitly map all possible paths to strings here as well
        mock_settings.APPLIANCE_INDUSTRY = "FINANCE"
        mock_settings.DISCOVERY_SCOUT_MODEL_PATH = "/mock/scout"
        mock_settings.DOMAIN_CLASSIFIER_MODEL_PATH = "/mock/llama.gguf"
        mock_settings.ASR_PHASE_1_MODEL_PATH = "/mock/llama.gguf"

        source_id = uuid.uuid4()
        mocks["repo"].get_glossary_by_source_id.return_value = None

        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=10, confidence=0.9)])
        mocks["sampler"].calculate_optimal_slices.return_value = [TimeSlice(start_sec=0, end_sec=10)]

        loaded_scout_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_scout_mock

        mocks["scout_adapter"].transcribe_samples.side_effect = RuntimeError("CUDA Out of Memory")
        mock_error_mapper.side_effect = ScoutInferenceError("Mapped Crash")

        with pytest.raises(ScoutInferenceError):
            api.process_domain_adaptation(source_id, "/dummy.wav", vad_map)

        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_scout_mock, mocks["scout_adapter"], flush=True
        )