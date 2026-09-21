# Canvas: desktop_test_evidence_card_v1
# Path: frontend/tests/ui/test_evidence_card_unit.py

"""
The card is where a reader acts on a doubt, so its two gestures have to be unambiguous: the button
opens the source in one click, and clicking the card reviews the excerpt without launching anything.

An earlier version had only "Review source", which merely filled a side panel and left the user
needing a second click to actually see the video. These tests pin the corrected behaviour.
"""

import uuid

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QLabel, QPushButton  # noqa: E402

from onyx_desktop.domain.entities import Evidence  # noqa: E402
from onyx_desktop.domain.enums import MediaKind  # noqa: E402
from onyx_desktop.ui.widgets.evidence_card import EvidenceCard  # noqa: E402

pytestmark = pytest.mark.gui


def make_evidence(media_kind=MediaKind.CLIP, **overrides):
    values = dict(
        doc_number=2,
        chunk_id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        start_sec=2001.0,
        end_sec=2093.0,
        excerpt="SPEAKER_01: some very key reference points.",
        speakers=("SPEAKER_01",),
        clip_path=None if media_kind is MediaKind.UNRESOLVED else "/vault/clip.mp4",
        media_kind=media_kind,
        relevance_score=0.77,
        rerank_score=0.91,
    )
    values.update(overrides)
    return Evidence(**values)


def card_for(qt_app, evidence, show_scores=False):
    widget = EvidenceCard(evidence, show_scores=show_scores)
    widget.show()
    return widget


def open_button(card) -> QPushButton:
    return next(b for b in card.findChildren(QPushButton) if b.text().startswith("Open"))


def label_text(card) -> str:
    return " ".join(label.text() for label in card.findChildren(QLabel))


class TestOneClickOpen:
    def test_the_button_opens_in_one_click(self, qt_app):
        evidence = make_evidence()
        card = card_for(qt_app, evidence)
        opened = []
        card.openRequested.connect(opened.append)

        open_button(card).click()

        assert opened == [evidence]
        card.close()

    def test_a_clip_button_says_clip(self, qt_app):
        card = card_for(qt_app, make_evidence(media_kind=MediaKind.CLIP))
        assert open_button(card).text().startswith("Open clip")
        card.close()

    def test_a_full_source_button_names_the_recording_and_the_timestamp(self, qt_app):
        """
        The label has to warn that the whole recording opens, because a system player cannot be
        told where to start.
        """
        card = card_for(qt_app, make_evidence(media_kind=MediaKind.FULL_SOURCE))
        button = open_button(card)
        assert button.text().startswith("Open recording")
        assert "33:21" in button.toolTip()
        card.close()

    def test_an_unresolved_source_disables_the_button_with_a_reason(self, qt_app):
        card = card_for(qt_app, make_evidence(media_kind=MediaKind.UNRESOLVED))
        button = open_button(card)
        assert button.isEnabled() is False
        assert button.toolTip()
        card.close()

    def test_an_unresolved_source_never_emits_open(self, qt_app):
        card = card_for(qt_app, make_evidence(media_kind=MediaKind.UNRESOLVED))
        opened = []
        card.openRequested.connect(opened.append)
        open_button(card).click()
        assert opened == []
        card.close()


class TestReviewGesture:
    def test_clicking_the_card_reviews_without_opening(self, qt_app):
        evidence = make_evidence()
        card = card_for(qt_app, evidence)
        reviewed, opened = [], []
        card.reviewRequested.connect(reviewed.append)
        card.openRequested.connect(opened.append)

        card.mousePressEvent(QMouseEvent(
            QMouseEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        ))

        assert reviewed == [evidence]
        assert opened == []          # reviewing must not launch a player
        card.close()


class TestScores:
    def test_scores_are_shown_only_when_asked(self, qt_app):
        with_scores = card_for(qt_app, make_evidence(), show_scores=True)
        assert "rrf 0.770" in label_text(with_scores)
        assert "rerank 0.910" in label_text(with_scores)
        with_scores.close()

        without = card_for(qt_app, make_evidence(), show_scores=False)
        assert "rrf" not in label_text(without)
        without.close()
