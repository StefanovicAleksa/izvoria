# Canvas: scene_detector_exceptions_base_v1
# Path: app/features/scene_detector/exceptions/base.py

class SceneDetectorError(Exception):
    """
    The Root Exception for the Scene Detector feature.
    All other scene detector exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception