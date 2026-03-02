# Canvas: nlp_router_entities_v1
# Path: app/features/nlp_router/domain/entities.py

from typing import List
from pydantic import BaseModel, Field, ConfigDict

# We reuse the exact TimeWindow entity from the Scene Detector
from app.features.scene_detector.domain.entities import TimeWindow


class TranscriptChunk(BaseModel):
    """
    A 3-to-5 minute chronological slice of the Whisper transcript.
    This acts as the specific payload we hand to the Phi-3.5 model.
    """
    start_sec: float
    end_sec: float
    text_block: str

    model_config = ConfigDict(frozen=True)


class RouterReasoning(BaseModel):
    """
    The strictly validated JSON output expected from the LLM.
    Forcing 'reasoning' before 'windows' drastically reduces timestamp hallucinations.
    """
    reasoning: str = Field(
        description="The LLM's explanation of why it did or did not extract a window based on literal visual cues."
    )
    windows: List[TimeWindow] = Field(
        default_factory=list,
        description="The precise timestamps where visual references occur."
    )

    model_config = ConfigDict(frozen=True)