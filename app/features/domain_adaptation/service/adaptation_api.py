# Canvas: adaptation_api_v6
# Path: app/features/domain_adaptation/service/adaptation_api.py

import uuid

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.vad.domain.entities import VadMap
from app.features.domain_adaptation.domain.interfaces import (
    IScoutWhisperAdapter,
    IClassifierAdapter,
    IDomainAdaptationRepository
)
from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.features.domain_adaptation.utils.audio_sampler import AudioSampler
from app.features.domain_adaptation.utils.glossary_builder import GlossaryBuilder
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error


class DomainAdaptationAPI:
    """
    Orchestrates the GPU-bound 'Scouting' phase to determine the specific subdomain
    of the media file and generate heavily weighted hotwords for the main transcriber.
    """

    def __init__(
            self,
            repository: IDomainAdaptationRepository,
            orchestrator: ModelManagerOrchestrator,
            scout_adapter: IScoutWhisperAdapter,
            classifier_adapter: IClassifierAdapter,
            audio_sampler: AudioSampler
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._scout_adapter = scout_adapter
        self._classifier_adapter = classifier_adapter
        self._sampler = audio_sampler

    def process_domain_adaptation(
            self, source_id: uuid.UUID, audio_path: str, vad_map: VadMap
    ) -> DomainGlossary:

        # 1. Idempotency Check
        existing_glossary = self._repo.get_glossary_by_source_id(source_id)
        if existing_glossary:
            return existing_glossary

        industry = settings.APPLIANCE_INDUSTRY
        samples = []

        try:
            # 2. Extract Audio using VAD limits
            if not vad_map.segments:
                return self._finalize_glossary(source_id, industry, "GENERAL", "")

            samples = self._sampler.extract_samples(audio_path, vad_map)
            if not samples:
                return self._finalize_glossary(source_id, industry, "GENERAL", "")

            # 3. GPU Phase 1: Scout
            scout_result = self._execute_scouting(samples, source_id)
            if not scout_result.combined_text.strip():
                return self._finalize_glossary(source_id, industry, "GENERAL", "")

            # 4. GPU Phase 2: Classification
            candidate_labels = GlossaryBuilder.get_candidate_labels(industry)
            best_subdomain = self._execute_classification(scout_result.combined_text, candidate_labels)

            # 5. Domain Logic: Anchor Validation & String Building
            return self._finalize_glossary(
                source_id=source_id,
                industry=industry,
                guessed_subdomain=best_subdomain,
                scouted_text=scout_result.combined_text
            )

        except Exception as e:
            raise map_domain_adaptation_error(e, context=f"Domain Adaptation for Source {source_id}") from e

        finally:
            if samples:
                self._sampler.cleanup_samples(samples)

    # --- Private Execution Helpers ---

    def _execute_scouting(self, samples, source_id):
        scout_manifest = ModelManifest(
            model_type=ModelType.DOMAIN_SCOUT,
            model_path=settings.DISCOVERY_SCOUT_MODEL_PATH,
            required_vram_mb=settings.DISCOVERY_SCOUT_VRAM_REQUIREMENT_MB
        )
        loaded_scout = None
        try:
            loaded_scout = self._orchestrator.acquire_model(scout_manifest, self._scout_adapter)
            return self._scout_adapter.transcribe_samples(loaded_scout, samples, source_id)
        finally:
            if loaded_scout:
                self._orchestrator.release_model(loaded_scout, self._scout_adapter, flush=True)

    def _execute_classification(self, text, candidate_labels):
        classifier_manifest = ModelManifest(
            model_type=ModelType.DOMAIN_CLASSIFIER,
            model_path=settings.DISCOVERY_CLASSIFIER_MODEL_PATH,
            required_vram_mb=settings.DISCOVERY_CLASSIFIER_VRAM_REQUIREMENT_MB
        )
        loaded_classifier = None
        try:
            loaded_classifier = self._orchestrator.acquire_model(classifier_manifest, self._classifier_adapter)
            return self._classifier_adapter.classify_domain(loaded_classifier, text, candidate_labels)
        finally:
            if loaded_classifier:
                self._orchestrator.release_model(loaded_classifier, self._classifier_adapter, flush=True)

    def _finalize_glossary(self, source_id, industry, guessed_subdomain, scouted_text) -> DomainGlossary:
        # Offloads the complex string building and validation rules to the utility
        final_subdomain, hotword_string = GlossaryBuilder.validate_and_build(
            industry=industry,
            guessed_subdomain=guessed_subdomain,
            scouted_text=scouted_text
        )

        glossary = DomainGlossary(
            source_id=source_id,
            industry=industry,
            sub_domain=final_subdomain,
            hotword_string=hotword_string
        )
        return self._repo.save_glossary(glossary)