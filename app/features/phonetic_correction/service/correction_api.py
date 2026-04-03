# Canvas: correction_api_v5
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

        self._manifest = ModelManifest(
            model_type=ModelType.PHONETIC_CORRECTION,
            model_path=settings.PHONETIC_CORRECTION_MODEL_PATH,
            required_vram_mb=settings.PHONETIC_CORRECTION_VRAM_REQUIREMENT_MB
        )

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
            loaded_model = None

            # Use a map to track audit records by sentence ID for deduplication
            audit_records_map: Dict[uuid.UUID, AppliedCorrectionRecord] = {}
            processed_sentences_map: Dict[uuid.UUID, ReconstructedSentence] = {}

            # FIXED: Standardized to 8,000 characters to match test expectations and improve stability
            safe_sentences = split_massive_sentences(transcript.sentences, max_chars=8000)

            try:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._patch_adapter)

                # FIXED: Reduced max_chars from 12,000 to 8,000 to prevent JSON truncation/OOM
                chunks = sliding_window_chunker(
                    items=safe_sentences,
                    text_extractor=lambda s: s.text,
                    max_chars=8000,
                    overlap_items=2
                )

                for idx, chunk in enumerate(chunks):
                    logger.info(f"  -> Processing Phonetic Window {idx + 1}/{len(chunks)}...")

                    # Ensure we pass the LATEST patched version of sentences into the next window
                    current_chunk_text_list = []
                    for s in chunk:
                        text = processed_sentences_map[
                            s.sentence_id].text if s.sentence_id in processed_sentences_map else s.text
                        current_chunk_text_list.append(text)

                    chunk_text = " ".join(current_chunk_text_list)

                    patches: List[PhoneticPatch] = self._patch_adapter.generate_patches(
                        model=loaded_model,
                        text_chunk=chunk_text,
                        domain_dictionary=domain_dictionary_str
                    )

                    # Pass the stateful audit map to prevent double-patching across overlaps
                    patched_chunk = self._apply_patches_to_chunk(chunk, patches, audit_records_map)

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
                audit_trail=list(audit_records_map.values())
            )

            return self._repo.save(final_transcript)

        except Exception as e:
            raise map_phonetic_error(e, context="PhoneticCorrectionAPI.execute") from e

    @staticmethod
    def _apply_patches_to_chunk(
            chunk_sentences: List[ReconstructedSentence],
            patches: List[PhoneticPatch],
            audit_records: Dict[uuid.UUID, AppliedCorrectionRecord]
    ) -> List[ReconstructedSentence]:
        """
        Applies phonetic patches to a batch of sentences while ensuring
        signatures are deduplicated across overlapping sliding windows.
        """
        patched_sentences = []

        for sentence in chunk_sentences:
            new_text = sentence.text

            # Retrieve or initialize the audit record for this specific sentence
            existing_record = audit_records.get(sentence.sentence_id)
            applied_patches = existing_record.applied_patches.copy() if existing_record else []

            # Generate signatures (orig, corr) to prevent the same patch being applied twice
            existing_sigs = {(p.original.lower(), p.correction.lower()) for p in applied_patches}

            if patches:
                for patch in patches:
                    if not patch.original:
                        continue

                    sig = (patch.original.lower(), patch.correction.lower())
                    if sig in existing_sigs:
                        continue

                    escaped_target = re.escape(patch.original)
                    pattern = re.compile(rf'(?<!\w){escaped_target}(?!\w)', re.IGNORECASE)

                    if pattern.search(new_text):
                        # Use a lambda to handle regex backreference hallucinations like \1
                        new_text = pattern.sub(lambda m: patch.correction, new_text).strip()
                        applied_patches.append(patch)
                        existing_sigs.add(sig)

                        if patch.correction == "":
                            logger.info(f"Hallucination Scrubbed: Removed '{patch.original}'")
                        else:
                            logger.info(f"Phonetic Patch Applied: '{patch.original}' -> '{patch.correction}'")

                if applied_patches:
                    audit_records[sentence.sentence_id] = AppliedCorrectionRecord(
                        sentence_id=sentence.sentence_id,
                        original_text=sentence.text,
                        corrected_text=new_text,
                        applied_patches=applied_patches
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