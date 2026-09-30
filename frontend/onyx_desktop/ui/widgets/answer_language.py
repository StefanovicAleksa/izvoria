# Path: frontend/onyx_desktop/ui/widgets/answer_language.py

from typing import Optional

from PySide6.QtWidgets import QLabel, QWidget

from onyx_desktop.ui import translate as tl


class AnswerLanguageLabel(QLabel):
    """
    Which language the next answer is written in. The index holds every language and evidence is quoted
    in its own, so a view that answers says which language the answer itself will be in.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("", parent)
        self.setProperty("role", "subtle")

    def set_language(self, name: str) -> None:
        tl.text(self, "Answers in {language}", language=name)
