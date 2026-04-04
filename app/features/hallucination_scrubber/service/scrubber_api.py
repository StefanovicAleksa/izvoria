# Canvas: scrubber_api_v4
# Path: app/features/hallucination_scrubber/service/scrubber_api.py

import uuid
import logging
from typing import List, Tuple, Optional

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

from app.features.transcription.domain.interfaces import ITranscriptRepository
from app.features.transcription.domain.entities import TranscriptSegment
from app.features.domain_adaptation.domain.interfaces import IDomainAdaptationRepository
from app.features.domain_adaptation.domain.entities import DomainGlossary

from app.features.hallucination_scrubber.domain.interfaces import IZeroShotAdapter, IHallucinationRepository
from app.features.hallucination_scrubber.domain.entities import (
    HallucinationScrubbedTranscript,
    AppliedScrubbingRecord,
    ScrubbingPatch
)
from app.features.hallucination_scrubber.utils.error_mapper import map_scrubber_error
from app.features.hallucination_scrubber.utils.text_segmenter import TextSegmenter

logger = logging.getLogger(__name__)


class HallucinationScrubberAPI:
    """
    Evaluates transcript segments against dynamically generated domain labels
    to identify and strip semantic hallucinations without using Regex.
    """

    # If the segment hits 40% on a hallucination label, we investigate clause-by-clause
    FAST_PATH_SUSPICION_THRESHOLD = 0.40

    # We only actually delete text if the specific clause hits 85% certainty
    CLAUSE_DELETION_THRESHOLD = 0.85

    def __init__(
            self,
            transcript_repo: ITranscriptRepository,
            domain_repo: IDomainAdaptationRepository,
            hallucination_repo: IHallucinationRepository,
            orchestrator: ModelManagerOrchestrator,
            adapter: IZeroShotAdapter
    ):
        self._transcript_repo = transcript_repo
        self._domain_repo = domain_repo
        self._hallucination_repo = hallucination_repo
        self._orchestrator = orchestrator
        self._adapter = adapter

        self._manifest = ModelManifest(
            model_type=ModelType.HALLUCINATION_SCRUBBER,
            model_path=settings.HALLUCINATION_SCRUBBER_MODEL_PATH,
            required_vram_mb=settings.HALLUCINATION_SCRUBBER_VRAM_REQUIREMENT_MB
        )

    def execute(self, source_id: uuid.UUID, release_after: bool = False) -> HallucinationScrubbedTranscript:
        try:
            # 1. Idempotency Check
            existing_record = self._hallucination_repo.get_by_source_id(source_id)
            if existing_record:
                return existing_record

            # 2. Fetch Prerequisites
            transcript = self._transcript_repo.get_by_source_id(source_id)
            if not transcript or not transcript.segments:
                raise ValueError(f"Dirty Transcript for Source {source_id} missing.")

            glossary = self._domain_repo.get_glossary_by_source_id(source_id)

            # 3. Build Dynamic Semantic Labels
            positive_label, negative_labels = self._build_labels(glossary)
            candidate_labels = [positive_label] + negative_labels

            scrubbed_segments: List[TranscriptSegment] = []
            audit_trail: List[AppliedScrubbingRecord] = []
            loaded_model = None

            try:
                # 4. Acquire DeBERTa
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter)

                # 5. Process Segments
                for segment in transcript.segments:
                    clean_segment, audit_record = self._evaluate_and_scrub(
                        model=loaded_model,
                        segment=segment,
                        negative_labels=negative_labels,
                        candidate_labels=candidate_labels
                    )

                    if clean_segment:
                        scrubbed_segments.append(clean_segment)
                    if audit_record:
                        audit_trail.append(audit_record)

                # 6. Save Aggregate Root
                final_transcript = HallucinationScrubbedTranscript(
                    source_id=source_id,
                    segments=scrubbed_segments,
                    audit_trail=audit_trail
                )
                return self._hallucination_repo.save(final_transcript)

            finally:
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

        except Exception as e:
            raise map_scrubber_error(e, context=f"HallucinationScrubberAPI.execute for Source {source_id}") from e

    def _evaluate_and_scrub(
            self,
            model: LoadedModel,
            segment: TranscriptSegment,
            negative_labels: List[str],
            candidate_labels: List[str]
    ) -> Tuple[Optional[TranscriptSegment], Optional[AppliedScrubbingRecord]]:

        # Fast Path: Check if the segment is suspicious at all
        scores = self._adapter.classify(model, segment.text, candidate_labels)

        is_suspicious = any(scores.get(neg, 0.0) >= self.FAST_PATH_SUSPICION_THRESHOLD for neg in negative_labels)

        if not is_suspicious:
            return segment, None

        # Slow Path: If flagged, split into clauses to salvage valid speech
        clauses = TextSegmenter.split_into_clauses(segment.text)

        if len(clauses) <= 1:
            # Re-verify the deletion threshold before complete deletion
            highest_score = max(scores.get(neg, 0.0) for neg in negative_labels)
            if highest_score < self.CLAUSE_DELETION_THRESHOLD:
                return segment, None

            patch = ScrubbingPatch(
                removed_text=segment.text,
                detected_label=max(negative_labels, key=lambda n: scores.get(n, 0.0)),
                confidence_score=highest_score
            )
            audit = AppliedScrubbingRecord(
                segment_start_sec=segment.start_sec,
                segment_end_sec=segment.end_sec,
                original_text=segment.text,
                scrubbed_text="",
                applied_patches=[patch]
            )
            return None, audit

        # Batch evaluate individual clauses
        clause_scores = self._adapter.classify_batch(model, clauses, candidate_labels)

        valid_clauses = []
        deleted_clauses = []
        patches = []

        for clause, score_dict in zip(clauses, clause_scores):
            highest_neg_score = 0.0
            highest_neg_label = ""

            for neg in negative_labels:
                score = score_dict.get(neg, 0.0)
                if score > highest_neg_score:
                    highest_neg_score = score
                    highest_neg_label = neg

            if highest_neg_score >= self.CLAUSE_DELETION_THRESHOLD:
                deleted_clauses.append(clause)
                patches.append(ScrubbingPatch(
                    removed_text=clause,
                    detected_label=highest_neg_label,
                    confidence_score=highest_neg_score
                ))
            else:
                valid_clauses.append(clause)

        if not deleted_clauses:
            return segment, None

        if not valid_clauses:
            audit = AppliedScrubbingRecord(
                segment_start_sec=segment.start_sec,
                segment_end_sec=segment.end_sec,
                original_text=segment.text,
                scrubbed_text="",
                applied_patches=patches
            )
            return None, audit

        # Reconstruct the segment safely
        clean_segment = TextSegmenter.reconstruct_segment(segment, valid_clauses, deleted_clauses)

        audit = AppliedScrubbingRecord(
            segment_start_sec=segment.start_sec,
            segment_end_sec=segment.end_sec,
            original_text=segment.text,
            scrubbed_text=clean_segment.text if clean_segment else "",
            applied_patches=patches
        )

        return clean_segment, audit

    @staticmethod
    def _build_labels(glossary: Optional[DomainGlossary]) -> Tuple[str, List[str]]:
        domain_str = "general discussion"
        if glossary:
            domain_str = glossary.industry.replace('_', ' ').lower()
            if glossary.sub_domains:
                domain_str += f", specifically {', '.join(glossary.sub_domains).replace('_', ' ')}"

        positive_label = f"valid spoken dialogue regarding {domain_str}"
        negative_labels = [
            "youtube outro, subscription request, or video formatting text",
            "a polite sign-off, customer service greeting, or closing remark",
            "audio transcription watermark or auto-generated subtitle notice"
        ]
        return positive_label, negative_labels