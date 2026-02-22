# filename: orchestration_exceptions_base_v1.py
from typing import Optional, Dict, Any

class OrchestrationError(Exception):
    """
    The Root Exception for the Orchestration Layer.
    All other orchestration exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.details = details or {}