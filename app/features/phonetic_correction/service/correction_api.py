# Canvas: correction_api_v14
# Path: app/features/phonetic_correction/service/correction_api.py

import logging
import re
import uuid
from typing import List, Dict

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence

from app.features.phonetic_correction.domain.interfaces import IPhoneticPatchAdapter, IPhoneticRepository
from app.features.phonetic_correction.domain.entities import (
    PhoneticCorrectedTranscript,
    AppliedCorrectionRecord,
    PhoneticPatch
)
from app.features.phonetic_correction.utils.sentence_splitter import split_massive_sentences
from app.features.phonetic_correction.utils.chunker import sliding_window_chunker
from app.features.phonetic_correction.utils.dictionary_loader import DictionaryLoader
from app.features.phonetic_correction.utils.error_mapper import map_phonetic_error

logger = logging.getLogger(__name__)


class PhoneticCorrectionAPI:
    def __init__(
            self,
            repository: IPhoneticRepository,
            orchestrator: ModelManagerOrchestrator,
            patch_adapter: IPhoneticPatchAdapter
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._patch_adapter = patch_adapter

    def execute(
            self,
            transcript: ReconstructedTranscript,
            industry: str,
            sub_domains: List[str],
            release_after: bool = False
    ) -> PhoneticCorrectedTranscript:
        try:
            existing_record = self._repo.get_by_source_id(transcript.source_id)
            if existing_record:
                return existing_record

            if not transcript.sentences:
                return self._create_passthrough(transcript)

            domain_dictionary_str = DictionaryLoader.build_prompt_dictionary(industry, sub_domains)

            manifest = ModelManifest(
                model_type=ModelType.ASR_CORRECTION_PHASE_3,
                model_path=settings.ASR_PHASE_3_MODEL_PATH,
                required_vram_mb=settings.ASR_PHASE_3_VRAM_REQUIREMENT_MB
            )

            loaded_model = None
            audit_records: List[AppliedCorrectionRecord] = []
            processed_sentences_map: Dict[uuid.UUID, ReconstructedSentence] = {}

            # 1. Bruteforce protection: Pre-split massive runaway sentences
            safe_sentences = split_massive_sentences(transcript.sentences, max_chars=10000)

            try:
                loaded_model = self._orchestrator.acquire_model(manifest, self._patch_adapter)

                # 2. Dynamic Character-Aware Chunking
                chunks = sliding_window_chunker(
                    items=safe_sentences,
                    text_extractor=lambda s: s.text,
                    max_chars=12000,
                    overlap_items=2
                )

                for idx, chunk in enumerate(chunks):
                    logger.info(f"  -> Processing Phonetic Window {idx + 1}/{len(chunks)}...")
                    chunk_text = " ".join([s.text for s in chunk])

                    patches: List[PhoneticPatch] = self._patch_adapter.generate_patches(
                        model=loaded_model,
                        text_chunk=chunk_text,
                        domain_dictionary=domain_dictionary_str
                    )

                    patched_chunk = self._apply_patches_to_chunk(chunk, patches, audit_records)

                    for s in patched_chunk:
                        processed_sentences_map[s.sentence_id] = s

            finally:
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._patch_adapter, flush=True)

            all_patched_sentences = list(processed_sentences_map.values())
            all_patched_sentences.sort(key=lambda x: x.start_sec)
            cleaned_sentences = [s for s in all_patched_sentences if s.text.strip()]

            final_transcript = PhoneticCorrectedTranscript(
                source_id=transcript.source_id,
                sentences=cleaned_sentences,
                audit_trail=audit_records
            )

            return self._repo.save(final_transcript)

        except Exception as e:
            raise map_phonetic_error(e, context="PhoneticCorrectionAPI.execute") from e

    @staticmethod
    def _apply_patches_to_chunk(
            chunk_sentences: List[ReconstructedSentence],
            patches: List[PhoneticPatch],
            audit_records: List[AppliedCorrectionRecord]
    ) -> List[ReconstructedSentence]:
        patched_sentences = []

        for sentence in chunk_sentences:
            new_text = sentence.text
            applied_to_this_sentence = []

            if patches:
                for patch in patches:
                    if not patch.original:
                        continue

                    escaped_target = re.escape(patch.original)
                    pattern = re.compile(rf'(?<!\w){escaped_target}(?!\w)', re.IGNORECASE)

                    if pattern.search(new_text):
                        # ==========================================================
                        # CRITICAL FIX: Regex Replacement Safety
                        # ==========================================================
                        # Pass correction via lambda so re.sub treats it as a raw literal,
                        # preventing hallucinated backslashes from being parsed as group IDs.
                        new_text = pattern.sub(lambda m: patch.correction, new_text).strip()

                        applied_to_this_sentence.append(patch)

                        if patch.correction == "":
                            logger.info(f"Hallucination Scrubbed: Removed '{patch.original}'")
                        else:
                            logger.info(f"Phonetic Patch Applied: '{patch.original}' -> '{patch.correction}'")

                if applied_to_this_sentence:
                    audit_records.append(
                        AppliedCorrectionRecord(
                            sentence_id=sentence.sentence_id,
                            original_text=sentence.text,
                            corrected_text=new_text,
                            applied_patches=applied_to_this_sentence
                        )
                    )
                    patched_sentences.append(sentence.model_copy(update={"text": new_text}))
                else:
                    patched_sentences.append(sentence)
            else:
                patched_sentences.append(sentence)

        return patched_sentences

    @staticmethod
    def _create_passthrough(transcript: ReconstructedTranscript) -> PhoneticCorrectedTranscript:
        return PhoneticCorrectedTranscript(
            source_id=transcript.source_id,
            sentences=transcript.sentences,
            audit_trail=[]
        )