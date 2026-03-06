import os

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript

from app.features.asr_error_correction.domain.interfaces import (
    ICorrectionRepository,
    ILlamaGlossaryAdapter,
    IT5AnomalyAdapter,
    IQwenCorrectionAdapter
)
from app.features.asr_error_correction.domain.entities import (
    DomainCorrectedTranscript,
    DocumentGlossary
)
from app.features.asr_error_correction.utils.chunker import chunk_sentences_by_exact_tokens
from app.features.asr_error_correction.utils.error_mapper import map_asr_correction_error


class AsrCorrectionAPI:
    """
    Public Service API for the ASR Error Correction feature.
    Coordinates the 3-pass pipeline with strict VRAM lifecycle management.
    """

    def __init__(
            self,
            repository: ICorrectionRepository,
            llama_adapter: ILlamaGlossaryAdapter,
            t5_adapter: IT5AnomalyAdapter,
            qwen_adapter: IQwenCorrectionAdapter
    ):
        self._repo = repository
        self._llama_adapter = llama_adapter
        self._t5_adapter = t5_adapter
        self._qwen_adapter = qwen_adapter

    def execute(self, transcript: ReconstructedTranscript) -> DomainCorrectedTranscript:
        try:
            glossary = self._run_glossary_phase(transcript)
            self._repo.save_glossary(glossary)

            anomalies = self._run_anomaly_phase(transcript)
            if not anomalies:
                return self._create_passthrough(transcript)

            corrected = self._run_correction_phase(transcript, anomalies, glossary)
            self._repo.save_corrected_transcript(corrected)
            return corrected

        except Exception as e:
            raise map_asr_correction_error(e, context="AsrCorrectionAPI.execute") from e

    # --- Private Phase Logic ---

    def _run_glossary_phase(self, transcript: ReconstructedTranscript) -> DocumentGlossary:
        manifest = ModelManifest(
            model_type=ModelType.ASR_CORRECTION_PHASE_1,
            model_path=settings.ASR_PHASE_1_MODEL_PATH,
            required_vram_mb=settings.ASR_PHASE_1_VRAM_REQUIREMENT_MB
        )
        tokenizer_path = os.path.dirname(manifest.model_path)

        model = None
        try:
            model = self._llama_adapter.load(manifest)
            chunks = chunk_sentences_by_exact_tokens(transcript.sentences, tokenizer_path)

            all_items = []
            final_domain = "General"

            for chunk in chunks:
                chunk_transcript = ReconstructedTranscript(source_id=transcript.source_id, sentences=chunk)
                partial = self._llama_adapter.extract_glossary(model, chunk_transcript)
                all_items.extend(partial.items)
                if partial.domain_category != "Unknown":
                    final_domain = partial.domain_category

            unique_items = list({item.term.lower(): item for item in all_items}.values())
            return DocumentGlossary(
                source_id=transcript.source_id,
                domain_category=final_domain,
                items=unique_items
            )
        finally:
            if model:
                self._llama_adapter.unload(model)

    def _run_anomaly_phase(self, transcript: ReconstructedTranscript) -> list:
        manifest = ModelManifest(
            model_type=ModelType.ASR_CORRECTION_PHASE_2,
            model_path=settings.ASR_PHASE_2_MODEL_PATH,
            required_vram_mb=settings.ASR_PHASE_2_VRAM_REQUIREMENT_MB
        )
        model = None
        try:
            model = self._t5_adapter.load(manifest)
            return self._t5_adapter.detect_anomalies(model, transcript)
        finally:
            if model:
                self._t5_adapter.unload(model)

    def _run_correction_phase(self, transcript, anomalies, glossary) -> DomainCorrectedTranscript:
        manifest = ModelManifest(
            model_type=ModelType.ASR_CORRECTION_PHASE_3,
            model_path=settings.ASR_PHASE_3_MODEL_PATH,
            required_vram_mb=settings.ASR_PHASE_3_VRAM_REQUIREMENT_MB
        )
        model = None
        try:
            model = self._qwen_adapter.load(manifest)
            return self._qwen_adapter.apply_corrections(model, transcript, anomalies, glossary)
        finally:
            if model:
                self._qwen_adapter.unload(model)

    @staticmethod
    def _create_passthrough(transcript: ReconstructedTranscript) -> DomainCorrectedTranscript:
        return DomainCorrectedTranscript(
            source_id=transcript.source_id,
            sentences=transcript.sentences,
            audit_trail=[]
        )