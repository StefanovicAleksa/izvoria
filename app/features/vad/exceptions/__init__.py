# File: app/features/vad/exceptions/__init__.py
from .base import VadError
from .model import (
    VadInferenceError,
    AudioFormatError,
    AudioSampleRateError,
    VadDeviceError,
    VadModelConfigError
)
from .database import VadDatabaseError

__all__ = [
    "VadError",
    "VadInferenceError",
    "AudioFormatError",
    "AudioSampleRateError",
    "VadDeviceError",
    "VadModelConfigError",
    "VadDatabaseError"
]