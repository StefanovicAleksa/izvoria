# filename: audio_extractor_entities_v1.py
import uuid
from pydantic import BaseModel, ConfigDict

class AudioExtraction(BaseModel):
    """
    Represents the relational link between a Video Source,
    the Job that processed it, and the resulting Audio File.
    """
    id: uuid.UUID
    source_id: uuid.UUID
    extracted_file_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)