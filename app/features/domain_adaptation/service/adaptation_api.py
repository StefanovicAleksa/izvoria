# Canvas: adaptation_api_v8
# Path: app/features/domain_adaptation/service/adaptation_api.py

import uuid

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.vad.domain.entities import VadMap
from app.features.domain_adaptation.domain.interfaces import (
    IScoutWhisperAdapter,
    ILlamaAdapter,
    IDomainAdaptationRepository
)
from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.features.domain_adaptation.utils.smart_sampler import SmartSampler
from app.features.domain_adaptation.utils.glossary_builder import GlossaryBuilder
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error


class DomainAdaptationAPI:
    """
    Strict Orchestrator for the Domain Adaptation phase.
    Manages VRAM handoffs between the Whisper Scout and Llama Router.
    """

    def __init__(
            self,
            repository: IDomainAdaptationRepository,
            orchestrator: ModelManagerOrchestrator,
            scout_adapter: IScoutWhisperAdapter,
            llama_adapter: ILlamaAdapter,
            smart_sampler: SmartSampler
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._scout_adapter = scout_adapter
        self._llama_adapter = llama_adapter
        self._sampler = smart_sampler

    def process_domain_adaptation(
            self, source_id: uuid.UUID, audio_path: str, vad_map: VadMap
    ) -> DomainGlossary:

        try:
            # 1. Idempotency Check
            existing_glossary = self._repo.get_glossary_by_source_id(source_id)
            if existing_glossary:
                return existing_glossary

            industry = settings.APPLIANCE_INDUSTRY

            # 2. Smart Sampling
            if not vad_map.segments:
                return self._finalize_glossary(source_id, industry, [], "", "")

            time_slices = self._sampler.calculate_optimal_slices(vad_map)

            # 3. GPU Phase 1: Scout
            scout_result = self._execute_scouting(audio_path, time_slices, source_id)
            if not scout_result.combined_text.strip():
                return self._finalize_glossary(source_id, industry, [], "", "")

            # 4. GPU Phase 2: Router
            candidate_labels = GlossaryBuilder.get_candidate_labels(industry)
            predicted_domains = self._execute_llm_routing(scout_result.combined_text, candidate_labels)

            # 5. Token-Safe Glossary Building
            hotword_string, initial_prompt = GlossaryBuilder.build_injection_payloads(
                industry=industry,
                predicted_domains=predicted_domains
            )

            # 6. Persistence
            return self._finalize_glossary(
                source_id=source_id,
                industry=industry,
                sub_domains=predicted_domains,
                hotword_string=hotword_string,
                initial_prompt_string=initial_prompt
            )

        except Exception as e:
            raise map_domain_adaptation_error(e, context=f"Domain Adaptation Flow for Source {source_id}") from e

    def _execute_scouting(self, audio_path: str, time_slices: list, source_id: uuid.UUID):
        manifest = ModelManifest(
            model_type=ModelType.DOMAIN_SCOUT,
            model_path=settings.DISCOVERY_SCOUT_MODEL_PATH,
            required_vram_mb=1000.0
        )
        loaded_model = None
        try:
            loaded_model = self._orchestrator.acquire_model(manifest, self._scout_adapter)
            return self._scout_adapter.transcribe_samples(loaded_model, audio_path, time_slices, source_id)
        finally:
            if loaded_model:
                self._orchestrator.release_model(loaded_model, self._scout_adapter, flush=True)

    def _execute_llm_routing(self, text: str, candidate_labels: list):
        manifest = ModelManifest(
            model_type=ModelType.DOMAIN_CLASSIFIER,
            model_path=settings.ASR_PHASE_1_MODEL_PATH,
            required_vram_mb=3500.0
        )
        loaded_model = None
        try:
            loaded_model = self._orchestrator.acquire_model(manifest, self._llama_adapter)
            return self._llama_adapter.predict_subdomains(loaded_model, text, candidate_labels)
        finally:
            if loaded_model:
                self._orchestrator.release_model(loaded_model, self._llama_adapter, flush=True)

    def _finalize_glossary(
            self, source_id, industry, sub_domains, hotword_string, initial_prompt_string
    ) -> DomainGlossary:

        # PyCharm warning resolved: actually passing the parameter
        glossary = DomainGlossary(
            source_id=source_id,
            industry=industry,
            sub_domains=sub_domains,
            hotword_string=hotword_string,
            initial_prompt_string=initial_prompt_string
        )
        return self._repo.save_glossary(glossary)