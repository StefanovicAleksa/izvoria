# Canvas: scene_detector_exceptions_database_v1
# Path: app/features/scene_detector/exceptions/database.py

from app.features.scene_detector.exceptions.base import SceneDetectorError

class SceneDetectorDatabaseError(SceneDetectorError):
    """
    Raised when the core database layer fails to save or retrieve the SceneMap.
    """
    pass