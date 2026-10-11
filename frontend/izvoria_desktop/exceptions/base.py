# Canvas: desktop_exceptions_base_v1
# Path: frontend/izvoria_desktop/exceptions/base.py


class DesktopError(Exception):
    """
    Root exception for the Izvoria desktop app.

    `user_message` is what the UI shows: a sentence a lawyer or clinician can act on, with no
    stack trace and no CUDA jargon. `original_exception` keeps the underlying engine error for
    the log pane.
    """

    def __init__(self, user_message: str, original_exception: Exception = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.original_exception = original_exception
