# Canvas: phonetic_api_v5
# Path: app/features/phonetic_correction/service/correction_api.py

import logging
from typing import List

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence

from app.features.phonetic_correction.domain.interfaces import IPhoneticPatchAdapter, IPhoneticRepository
from app.features.phonetic_correction.domain.entities import (
    PhoneticCorrectedTranscript,
    AppliedCorrectionRecord,
    PhoneticPatch
)
from app.features.phonetic_correction.utils.chunker import chunk_sentences_by_exact_tokens
from app.features.phonetic_correction.utils.dictionary_loader import DictionaryLoader
from app.features.phonetic_correction.utils.error_mapper import map_phonetic_error

logger = logging.getLogger(__name__)


class PhoneticCorrectionAPI:
    """
    Orchestrates Phase 8: Static Dictionary Phonetic Patching & Hallucination Scrubbing.
    Loads Qwen 2.5, chunks the transcript, retrieves patches, and deterministically applies them.
    """

    def __init__(
            self,
            repository: IPhoneticRepository,
            patch_adapter: IPhoneticPatchAdapter
    ):
        self._repo = repository
        self._patch_adapter = patch_adapter

    def execute(
            self,
            transcript: ReconstructedTranscript,
            industry: str,
            sub_domain: str
    ) -> PhoneticCorrectedTranscript:
        try:
            existing_record = self._repo.get_by_source_id(transcript.source_id)
            if existing_record:
                logger.info(f"Phonetic correction already completed for source {transcript.source_id}")
                return existing_record

            if not transcript.sentences:
                return self._create_passthrough(transcript)

            # Delegate File I/O to the Utils layer
            domain_dictionary_str = DictionaryLoader.build_prompt_dictionary(industry, sub_domain)

            if not domain_dictionary_str:
                logger.warning(
                    f"No dictionary loaded for '{industry}/{sub_domain}'. Proceeding with hallucination scrubbing only.")

            manifest = ModelManifest(
                model_type=ModelType.ASR_CORRECTION_PHASE_3,
                model_path=settings.ASR_PHASE_3_MODEL_PATH,
                required_vram_mb=settings.ASR_PHASE_3_VRAM_REQUIREMENT_MB
            )

            loaded_model = None
            audit_records: List[AppliedCorrectionRecord] = []
            all_patched_sentences: List[ReconstructedSentence] = []

            try:
                loaded_model = self._patch_adapter.load(manifest)

                tokenize_func = lambda text: loaded_model.model_instance.tokenize(text.encode("utf-8"), add_bos=False)

                chunks = chunk_sentences_by_exact_tokens(
                    sentences=transcript.sentences,
                    tokenize_func=tokenize_func,
                    max_tokens_per_chunk=4000
                )

                for chunk in chunks:
                    chunk_text = " ".join([s.text for s in chunk])

                    patches: List[PhoneticPatch] = self._patch_adapter.generate_patches(
                        model=loaded_model,
                        text_chunk=chunk_text,
                        domain_dictionary=domain_dictionary_str
                    )

                    # FIXED: Apply patches immutably and accumulate the new objects
                    patched_chunk = self._apply_patches_to_chunk(chunk, patches, audit_records)
                    all_patched_sentences.extend(patched_chunk)

            finally:
                if loaded_model:
                    self._patch_adapter.unload(loaded_model)

            # Filter out any sentences that were completely scrubbed empty by the hallucination remover
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
        """Safely applies patches using immutable Pydantic copies and logs the audit trail."""
        patched_sentences = []

        for sentence in chunk_sentences:
            new_text = sentence.text
            applied_to_this_sentence = []

            if patches:
                for patch in patches:
                    if not patch.original:
                        continue

                    if patch.original in new_text:
                        new_text = new_text.replace(patch.original, patch.correction).strip()
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
                # FIXED: Immutably create a new sentence object instead of mutating the frozen one
                patched_sentences.append(sentence.model_copy(update={"text": new_text}))
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