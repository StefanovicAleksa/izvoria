# Path: app/features/domain_phonetic_correction/service/correction_api.py

import logging
import uuid
from typing import Dict, List

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.domain_phonetic_correction.domain.interfaces import IDomainPhoneticPatchAdapter, IDomainPhoneticRepository
from app.features.domain_phonetic_correction.domain.entities import (
    DomainPhoneticCorrectedTranscript,
    AppliedDomainCorrectionRecord,
    DomainPhoneticPatch,
    PhoneticCandidate
)
from app.features.domain_phonetic_correction.utils.dictionary_loader import DictionaryLoader
from app.features.domain_phonetic_correction.utils.error_mapper import map_phonetic_error
from app.features.domain_phonetic_correction.utils.span_replacer import replace_spans
from app.features.domain_phonetic_correction.utils.term_candidate_finder import find_term_candidates
from app.features.domain_phonetic_correction.utils.text_sanitizer import PhoneticTextSanitizer

logger = logging.getLogger(__name__)

# Previous-sentence context handed to the verifier
MAX_PROMPT_PREVIOUS_CHARS = 400


class DomainPhoneticCorrectionAPI:
    """
    Fixes genuine mis-hearings of glossary terms and nothing else.

    Candidates are found deterministically (a heard span that sounds like exactly one dictionary term, never touching
    digits, a correctly spelled term, or neighbouring words); the model only confirms or rejects each candidate; the
    confirmed span is replaced by the term with the rest of the sentence copied unchanged. The model never writes text,
    so it can't paraphrase, insert or delete words.
    """

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
            sanitized_sentences: List[ReconstructedSentence] = []
            for s in transcript.sentences:
                clean_text = PhoneticTextSanitizer.sanitize(s.text)
                if clean_text:
                    sanitized_sentences.append(s.model_copy(update={"text": clean_text}))

            terms = DictionaryLoader.load_terms(industry, sub_domains)
            if len(terms) < settings.PHONETIC_MIN_DICTIONARY_TERMS:
                logger.info(
                    f"Domain dictionary for {industry}/{sub_domains} has {len(terms)} terms "
                    f"(< {settings.PHONETIC_MIN_DICTIONARY_TERMS}); skipping phonetic correction."
                )
                return self._repo.save(self._create_passthrough(transcript.source_id, sanitized_sentences))

            candidates_by_sentence: Dict[uuid.UUID, List[PhoneticCandidate]] = {}
            for sentence in sanitized_sentences:
                candidates = find_term_candidates(
                    text=sentence.text,
                    words=sentence.words,
                    terms=terms,
                    min_similarity=settings.PHONETIC_CANDIDATE_MIN_SIMILARITY,
                    length_ratio_min=settings.PHONETIC_LENGTH_RATIO_MIN,
                    length_ratio_max=settings.PHONETIC_LENGTH_RATIO_MAX,
                    max_word_probability=settings.PHONETIC_MAX_WORD_PROBABILITY
                )
                if candidates:
                    candidates_by_sentence[sentence.sentence_id] = candidates

            # Nothing sounds like a mis-heard term: the model isn't needed at all
            if not candidates_by_sentence:
                logger.info("No phonetic candidates found; transcript passes through unchanged.")
                return self._repo.save(self._create_passthrough(transcript.source_id, sanitized_sentences))

            logger.info(
                f"Verifying {sum(len(c) for c in candidates_by_sentence.values())} phonetic candidate(s) "
                f"in {len(candidates_by_sentence)} sentence(s)."
            )

            corrected_sentences: List[ReconstructedSentence] = []
            audit_trail: List[AppliedDomainCorrectionRecord] = []
            domain = self._describe_domain(industry, sub_domains)
            loaded_model = None

            try:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._patch_adapter)
                previous_text = ""

                for sentence in sanitized_sentences:
                    candidates = candidates_by_sentence.get(sentence.sentence_id)
                    if candidates:
                        sentence = self._correct_sentence(loaded_model, sentence, previous_text, candidates, domain,
                                                          audit_trail)
                    corrected_sentences.append(sentence)
                    previous_text = sentence.text

            finally:
                if loaded_model and release_after:
                    self.release_model(loaded_model)

            final_transcript = DomainPhoneticCorrectedTranscript(
                source_id=transcript.source_id,
                sentences=corrected_sentences,
                audit_trail=audit_trail
            )
            return self._repo.save(final_transcript)

        except Exception as e:
            raise map_phonetic_error(e, context="DomainPhoneticCorrectionAPI.execute") from e

    def _correct_sentence(
            self,
            loaded_model: LoadedModel,
            sentence: ReconstructedSentence,
            previous_text: str,
            candidates: List[PhoneticCandidate],
            domain: str,
            audit_trail: List[AppliedDomainCorrectionRecord]
    ) -> ReconstructedSentence:
        verdicts = self._patch_adapter.verify_candidates(
            model=loaded_model,
            sentence_text=sentence.text,
            previous_text=previous_text[-MAX_PROMPT_PREVIOUS_CHARS:],
            candidates=candidates,
            domain=domain
        )
        confirmed = [candidate for candidate, verdict in zip(candidates, verdicts) if verdict is True]
        if not confirmed:
            return sentence

        corrected_text = replace_spans(sentence.text, confirmed)
        if corrected_text == sentence.text:
            return sentence

        audit_trail.append(AppliedDomainCorrectionRecord(
            sentence_id=sentence.sentence_id,
            original_text=sentence.text,
            corrected_text=corrected_text,
            applied_patches=[DomainPhoneticPatch(original=c.heard, correction=c.replacement) for c in confirmed]
        ))
        logger.info(f"Phonetic correction: '{sentence.text}' -> '{corrected_text}'")
        return sentence.model_copy(update={"text": corrected_text})

    @staticmethod
    def _describe_domain(industry: str, sub_domains: List[str]) -> str:
        readable = lambda label: label.replace("_", " ").lower()
        return f"{readable(industry)} ({', '.join(readable(s) for s in sub_domains)})" if sub_domains else readable(industry)

    @staticmethod
    def _create_passthrough(source_id: uuid.UUID, sentences: List[ReconstructedSentence]) -> DomainPhoneticCorrectedTranscript:
        return DomainPhoneticCorrectedTranscript(
            source_id=source_id,
            sentences=sentences,
            audit_trail=[]
        )
