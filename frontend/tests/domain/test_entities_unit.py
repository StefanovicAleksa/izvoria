# Canvas: desktop_test_entities_v1
# Path: frontend/tests/domain/test_entities_unit.py

"""
The derived properties on these dataclasses encode product rules, so they are tested directly
rather than through the widgets that read them.
"""

import uuid

import pytest

from onyx_desktop.domain.entities import ChatMessage, Evidence, SearchHit
from onyx_desktop.domain.enums import MediaKind, MessageRole, TurnOutcome

CHUNK_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SOURCE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def make_evidence(**overrides) -> Evidence:
    values = dict(
        doc_number=2,
        chunk_id=CHUNK_ID,
        source_id=SOURCE_ID,
        start_sec=72.0,
        end_sec=108.0,
        excerpt="SPEAKER_01: verbatim words.",
        clip_path="/vault/clip.mp4",
        media_kind=MediaKind.CLIP,
    )
    values.update(overrides)
    return Evidence(**values)


class TestEvidence:
    def test_doc_label_matches_the_engine_citation_format(self):
        assert make_evidence().doc_label == "[Doc 2]"

    def test_a_clip_needs_no_seek_because_the_file_is_the_excerpt(self):
        assert make_evidence(media_kind=MediaKind.CLIP).seek_to_sec == 0.0

    def test_a_full_source_must_be_seeked_to_the_cited_moment(self):
        evidence = make_evidence(media_kind=MediaKind.FULL_SOURCE)
        assert evidence.seek_to_sec == 72.0

    @pytest.mark.parametrize(
        "clip_path, media_kind, expected",
        [
            ("/vault/clip.mp4", MediaKind.CLIP, True),
            ("/vault/whole.wav", MediaKind.FULL_SOURCE, True),
            (None, MediaKind.UNRESOLVED, False),
            # A path with no kind is still not playable: the player would not know how to treat it.
            ("/vault/clip.mp4", MediaKind.UNRESOLVED, False),
        ],
    )
    def test_playability(self, clip_path, media_kind, expected):
        evidence = make_evidence(clip_path=clip_path, media_kind=media_kind)
        assert evidence.is_playable is expected

    def test_entities_are_frozen(self):
        with pytest.raises(Exception):
            make_evidence().doc_number = 9


class TestChatMessageGrounding:
    def test_an_answer_with_no_evidence_is_ungrounded(self):
        message = ChatMessage(
            role=MessageRole.ASSISTANT, text="It was the fourteenth.",
            outcome=TurnOutcome.ANSWERED,
        )
        assert message.is_ungrounded is True

    def test_an_answer_with_evidence_is_grounded(self):
        message = ChatMessage(
            role=MessageRole.ASSISTANT, text="It was the fourteenth [Doc 1].",
            outcome=TurnOutcome.ANSWERED, evidence=(make_evidence(),),
        )
        assert message.is_ungrounded is False

    def test_a_clarification_request_is_not_branded_ungrounded(self):
        """
        Reasoned Answer asks for clarification more often than it answers. Those turns legitimately
        carry no evidence and must not be flagged as untraceable answers.
        """
        message = ChatMessage(
            role=MessageRole.ASSISTANT, text="Which schedule do you mean?",
            outcome=TurnOutcome.NEEDS_CLARIFICATION,
        )
        assert message.is_ungrounded is False

    def test_a_user_message_is_never_ungrounded(self):
        message = ChatMessage(role=MessageRole.USER, text="Who signed it?")
        assert message.is_ungrounded is False


class TestSearchHit:
    def test_a_hit_converts_to_evidence_preserving_rank_and_scores(self):
        hit = SearchHit(
            rank=3, chunk_id=CHUNK_ID, source_id=SOURCE_ID, text="verbatim",
            start_sec=10.0, end_sec=20.0, relevance_score=0.77, rerank_score=0.91,
            speakers=("SPEAKER_01",), clip_path="/vault/clip.mp4", media_kind=MediaKind.CLIP,
        )

        evidence = hit.as_evidence()

        assert evidence.doc_number == 3          # rank becomes the visible label
        assert evidence.excerpt == "verbatim"
        assert evidence.relevance_score == 0.77
        assert evidence.rerank_score == 0.91
        assert evidence.is_playable is True
