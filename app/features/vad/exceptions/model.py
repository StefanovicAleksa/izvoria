# File: app/features/vad/exceptions/model.py
from app.features.vad.exceptions.base import VadError

class VadInferenceError(VadError):
    """Raised when the NeMo model fails during the actual forward pass."""
    pass

class AudioFormatError(VadError):
    """Raised when the provided audio file is corrupted or unsupported."""
    pass

class AudioSampleRateError(VadError):
    """
    Raised when the audio sample rate cannot be processed.
    NeMo VAD typically strictly requires 16kHz mono audio.
    """
    pass

class VadDeviceError(VadError):
    """
    Raised when there is a PyTorch device mismatch
    (e.g., the NeMo model is on GPU, but the audio tensor was loaded onto CPU).
    """
    pass

class VadModelConfigError(VadError):
    """
    Raised when NeMo's OmegaConf/Hydra configuration graph is corrupted,
    or the model signature does not match the inputs.
    """
    pass

class VadModelFilePathError(VadError):
    """Raised when the specified .nemo model file cannot be found on disk."""
    pass