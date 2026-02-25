# Canvas: diarization_exceptions_model_v1
# Path: app/features/diarization/exceptions/model.py

from app.features.diarization.exceptions.base import DiarizationError

class DiarizationInferenceError(DiarizationError):
    """Raised when the NeMo embedding model fails during the forward pass."""
    pass

class DiarizationAudioSegmentError(DiarizationError):
    """
    Raised when the VAD segments are too short for TitaNet to extract a meaningful voiceprint.
    TitaNet typically requires at least 1.5 seconds of continuous audio.
    """
    pass

class DiarizationClusteringError(DiarizationError):
    """
    Raised when Spectral Clustering fails to converge on distinct speaker groups.
    Usually caused by extreme background noise, overlapping speech, or linear algebra faults.
    """
    pass

class DiarizationManifestError(DiarizationError):
    """
    Raised when NeMo fails to construct or read its internal JSON manifest for the audio segments,
    or if the OmegaConf configuration graph is invalid.
    """
    pass

class DiarizationZeroSpeakersError(DiarizationError):
    """
    Raised when the algorithm completes but groups all audio as background noise,
    resulting in 0 identified speakers.
    """
    pass