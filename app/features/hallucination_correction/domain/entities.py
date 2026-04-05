# Canvas: hallucination_correction_entities_v1
# Path: app/features/hallucination_correction/domain/entities.py

from typing import List
from pydantic import BaseModel, ConfigDict, Field


class CorrectionConfig(BaseModel):
    """
    Deterministic thresholds for the Zero-VRAM filtering gauntlet.
    Values are tuned based on OpenAI Whisper's standard failure modes.
    """
    # VAD Overlap Tolerance: If Whisper generates text, but less than 15% of that
    # segment's duration contains actual human speech, we flag it.
    min_vad_overlap_ratio: float = Field(default=0.15)

    # Metadata Thresholds
    max_no_speech_prob: float = Field(default=0.60)
    min_avg_logprob: float = Field(default=-1.0)
    max_compression_ratio: float = Field(default=2.4)

    # Global Regex Blacklist
    blacklist_patterns: List[str] = Field(default_factory=lambda: [
        r"(?i)(subscribe|like and subscribe|thank you for watching)",
        r"(?i)(amara\.org|subtitles by)",
        r"(?i)(^MBC Network$|^KBS$|^NHK$)",  # Asian broadcasting network hallucinations
        r"(?i)(\[Music\]|\[Silence\]|\[Applause\])"
    ])

    model_config = ConfigDict(frozen=True)


class CorrectionTelemetry(BaseModel):
    """
    Tracks exactly how many segments were amputated and by which filter.
    Highly useful for debugging and Orchestrator phase logging.
    """
    total_original_segments: int = 0
    removed_by_vad: int = 0
    removed_by_metadata: int = 0
    removed_by_regex: int = 0

    @property
    def total_surviving_segments(self) -> int:
        removed = self.removed_by_vad + self.removed_by_metadata + self.removed_by_regex
        return self.total_original_segments - removed

    model_config = ConfigDict(frozen=False)