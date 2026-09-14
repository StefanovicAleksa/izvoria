# Canvas: domain_correction_api_v21
# Path: app/features/domain_phonetic_correction/service/correction_api.py

import logging
import re
import uuid
from difflib import SequenceMatcher
from typing import List, Dict

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.domain_phonetic_correction.domain.interfaces import IDomainPhoneticPatchAdapter, IDomainPhoneticRepository
from app.features.domain_phonetic_correction.domain.entities import (
    DomainPhoneticCorrectedTranscript,
    AppliedDomainCorrectionRecord,
    DomainPhoneticPatch
)
from app.features.domain_phonetic_correction.utils.sentence_splitter import split_massive_sentences
from app.features.domain_phonetic_correction.utils.chunker import sliding_window_chunker
from app.features.domain_phonetic_correction.utils.dictionary_loader import DictionaryLoader
from app.features.domain_phonetic_correction.utils.anchor_applicator import AnchorPatchApplicator
from app.features.domain_phonetic_correction.utils.error_mapper import map_phonetic_error
from app.features.domain_phonetic_correction.utils.text_sanitizer import PhoneticTextSanitizer

logger = logging.getLogger(__name__)

class DomainPhoneticCorrectionAPI:
    def __init__(
            self,
            repository: IDomainPhoneticRepository,
            orchestrator: ModelManagerOrchestrator,
            patch_adapter: IDomainPhoneticPatchAdapter
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._patch_adapter = patch_adapter

        self._manifest = ModelManifest(
            model_type=ModelType.DOMAIN_PHONETIC_CORRECTION,
            model_path=settings.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH,
            required_vram_mb=settings.DOMAIN_PHONETIC_CORRECTION_VRAM_REQUIREMENT_MB
        )

    def release_model(self, loaded_model: LoadedModel) -> None:
        self._orchestrator.release_model(loaded_model, self._patch_adapter, flush=True)

    def execute(
            self,
            transcript: ReconstructedTranscript,
            industry: str,
            sub_domains: List[str],
            release_after: bool = False
    ) -> DomainPhoneticCorrectedTranscript:
        try:
            existing_record = self._repo.get_by_source_id(transcript.source_id)
            if existing_record:
                return existing_record

            # A passthrough must be persisted too: the next phase loads this feature's output from the DB
            if not transcript.sentences:
                return self._repo.save(self._create_passthrough(transcript.source_id, []))

            # PRE-PROCESSING: Sanitize all rogue '0' tokens immediately before processing
            sanitized_sentences = []
            for s in transcript.sentences:
                clean_text = PhoneticTextSanitizer.sanitize(s.text)
                if clean_text:
                    sanitized_sentences.append(s.model_copy(update={"text": clean_text}))

            domain_dictionary_str = DictionaryLoader.build_prompt_dictionary(industry, sub_domains)
            dictionary_terms = self._extract_dictionary_terms(domain_dictionary_str)

            # With (almost) no dictionary there is nothing to correct against, and a small LLM left
            # free to "correct" ordinary words is a risk with no upside. Skip the model entirely.
            if len(dictionary_terms) < settings.PHONETIC_MIN_DICTIONARY_TERMS:
                logger.info(
                    f"Domain dictionary for {industry}/{sub_domains} has {len(dictionary_terms)} terms "
                    f"(< {settings.PHONETIC_MIN_DICTIONARY_TERMS}); skipping phonetic correction."
                )
                return self._repo.save(self._create_passthrough(transcript.source_id, sanitized_sentences))

            loaded_model = None

            audit_records_map: Dict[uuid.UUID, AppliedDomainCorrectionRecord] = {}
            processed_sentences_map: Dict[uuid.UUID, ReconstructedSentence] = {}

            CHUNK_SIZE_LIMIT = 1200
            safe_sentences = split_massive_sentences(sanitized_sentences, max_chars=CHUNK_SIZE_LIMIT)

            try:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._patch_adapter)

                chunks = sliding_window_chunker(
                    items=safe_sentences,
                    text_extractor=lambda item: item.text,
                    max_chars=CHUNK_SIZE_LIMIT,
                    overlap_items=2
                )

                for idx, chunk in enumerate(chunks):
                    logger.info(f"  -> Processing Domain Phonetic Window {idx + 1}/{len(chunks)}...")

                    current_chunk_text_list = []
                    for sentence_obj in chunk:
                        text = processed_sentences_map.get(sentence_obj.sentence_id, sentence_obj).text
                        current_chunk_text_list.append(text)

                    chunk_text = " ".join(current_chunk_text_list)

                    patches: List[DomainPhoneticPatch] = self._patch_adapter.generate_patches(
                        model=loaded_model,
                        text_chunk=chunk_text,
                        domain_dictionary=domain_dictionary_str
                    )

                    grounded_patches = [p for p in patches if self._is_grounded_patch(p, dictionary_terms)]
                    if len(grounded_patches) != len(patches):
                        logger.info(
                            f"Rejected {len(patches) - len(grounded_patches)} ungrounded patch(es): no dictionary term "
                            f"introduced, or the replaced words don't sound like the correction."
                        )

                    patched_chunk = self._apply_patches_to_chunk(chunk, grounded_patches, audit_records_map)

                    for s in patched_chunk:
                        processed_sentences_map[s.sentence_id] = s

            finally:
                if loaded_model and release_after:
                    self.release_model(loaded_model)

            all_patched_sentences = list(processed_sentences_map.values())
            all_patched_sentences.sort(key=lambda x: x.start_sec)
            cleaned_sentences = [s for s in all_patched_sentences if s.text.strip()]

            final_transcript = DomainPhoneticCorrectedTranscript(
                source_id=transcript.source_id,
                sentences=cleaned_sentences,
                audit_trail=list(audit_records_map.values())
            )

            return self._repo.save(final_transcript)

        except Exception as e:
            raise map_phonetic_error(e, context="DomainPhoneticCorrectionAPI.execute") from e

    @staticmethod
    def _extract_dictionary_terms(domain_dictionary: str) -> List[str]:
        """Parses the '- term' lines produced by DictionaryLoader.build_prompt_dictionary."""
        return [line[2:].strip() for line in domain_dictionary.splitlines() if line.startswith("- ") and line[2:].strip()]

    @staticmethod
    def _is_grounded_patch(patch: DomainPhoneticPatch, dictionary_terms: List[str]) -> bool:
        """
        A patch fixes a mis-hearing, so it must (a) introduce a dictionary term and (b) only change words into
        words that sound like them. Introducing a term alone let the model turn ordinary words into terms
        ("Bank of England" -> "Onyx", "decision" -> "prognosis").

        Every changed word region is judged on its own, so a patch with two unrelated edits can't pass on the
        words they share: inserted or deleted words are rejected, a case-only change is accepted only for
        acronym-like terms (at least two capitals), and anything else needs letters-only similarity of at least
        PHONETIC_MIN_SIMILARITY. Spelled-out acronyms ("the sea a grr" -> "CAGR") are rejected as a result.
        """
        if not DomainPhoneticCorrectionAPI._introduces_dictionary_term(patch, dictionary_terms):
            return False

        heard_words = patch.original.split()
        written_words = patch.correction.split()
        regions = [op for op in SequenceMatcher(None, heard_words, written_words, autojunk=False).get_opcodes()
                   if op[0] != "equal"]
        if not regions:
            return False

        for _, heard_start, heard_end, written_start, written_end in regions:
            heard = " ".join(heard_words[heard_start:heard_end])
            written = " ".join(written_words[written_start:written_end])

            if not heard or not written:
                return False

            if heard.lower() == written.lower():
                if sum(1 for ch in written if ch.isupper()) < 2:
                    return False
                continue

            similarity = SequenceMatcher(
                None, DomainPhoneticCorrectionAPI._letters_only(heard),
                DomainPhoneticCorrectionAPI._letters_only(written), autojunk=False
            ).ratio()
            if similarity < settings.PHONETIC_MIN_SIMILARITY:
                return False

        return True

    @staticmethod
    def _letters_only(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def _introduces_dictionary_term(patch: DomainPhoneticPatch, dictionary_terms: List[str]) -> bool:
        """
        A patch is only trusted if its correction contains a dictionary term (case-insensitive) that
        the original did not already contain in that exact spelling. This admits real fixes
        ("high pole glycemia" -> "hypoglycemia", "ebitda" -> "EBITDA") and rejects the model
        rewriting ordinary words that have nothing to do with the dictionary.
        """
        for term in dictionary_terms:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            if re.search(pattern, patch.correction, re.IGNORECASE) and not re.search(pattern, patch.original):
                return True
        return False

    @staticmethod
    def _apply_patches_to_chunk(
            chunk_sentences: List[ReconstructedSentence],
            patches: List[DomainPhoneticPatch],
            audit_records: Dict[uuid.UUID, AppliedDomainCorrectionRecord]
    ) -> List[ReconstructedSentence]:
        patched_sentences = []

        for sentence in chunk_sentences:
            new_text = sentence.text
            existing_record = audit_records.get(sentence.sentence_id)
            applied_patches = existing_record.applied_patches.copy() if existing_record else []
            existing_sigs = {(p.original.lower(), p.correction.lower()) for p in applied_patches}

            if patches:
                for patch in patches:
                    if not patch.original: continue
                    sig = (patch.original.lower(), patch.correction.lower())
                    if sig in existing_sigs: continue

                    updated_text = AnchorPatchApplicator.apply(new_text, patch.original, patch.correction)

                    if updated_text != new_text:
                        applied_patches.append(patch)
                        existing_sigs.add(sig)
                        new_text = updated_text

                if applied_patches:
                    audit_records[sentence.sentence_id] = AppliedDomainCorrectionRecord(
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
    def _create_passthrough(source_id: uuid.UUID, sentences: List[ReconstructedSentence]) -> DomainPhoneticCorrectedTranscript:
        return DomainPhoneticCorrectedTranscript(
            source_id=source_id,
            sentences=sentences,
            audit_trail=[]
        )
