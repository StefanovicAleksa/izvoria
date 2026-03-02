# Canvas: scene_detector_exceptions_processing_v1
# Path: app/features/scene_detector/exceptions/processing.py

from app.features.scene_detector.exceptions.base import SceneDetectorError

class SceneDetectionWindowError(SceneDetectorError):
    """
    Raised when the Micro-LLM provides invalid time windows
    (e.g., start_sec > end_sec, or timestamps outside the video duration).
    """
    pass

class SceneDeduplicationError(SceneDetectorError):
    """
    Raised when the CPU-bound Perceptual Hash (pHash) fails to process an extracted frame
    (e.g., the JPEG is corrupted or I/O permission was denied).
    """
    pass