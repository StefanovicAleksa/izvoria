import uuid
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.core.model_manager.domain.enums import ModelType

class VramStatus(BaseModel):
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float

    model_config = ConfigDict(frozen=True)

class ModelManifest(BaseModel):
    model_type: ModelType
    model_path: str  # Local path to the weights on the Onyx Appliance
    required_vram_mb: float

    model_config = ConfigDict(frozen=True)


class LoadedModel(BaseModel):
    """
    Represents a model that currently resides in GPU VRAM.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    manifest: ModelManifest

    # Typed as 'Any' because the Domain layer must remain ignorant of
    # external libraries like PyTorch or CTranslate2.
    model_instance: Any = Field(description="The raw model object in memory.")

    model_config = ConfigDict(arbitrary_types_allowed=True)