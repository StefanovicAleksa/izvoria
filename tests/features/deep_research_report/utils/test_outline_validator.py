# Path: tests/features/deep_research_report/utils/test_outline_validator.py

import uuid

import pytest

from app.core.config import settings
from app.features.deep_research_report.domain.entities import OutlineSection, ResearchPlan, SubQuestion
from app.features.deep_research_report.exceptions.pipeline import OutlineValidationError
from app.features.deep_research_report.utils.outline_validator import validate_and_normalise


def _plan(report_id=None, sections=None, sub_questions=None, revision=1) -> ResearchPlan:
    report_id = report_id or uuid.uuid4()
    return ResearchPlan(
        report_id=report_id,
        topic="Chain of custody",
        sections=sections if sections is not None else [
            OutlineSection(key="s1", title="Handling", order_index=0, brief="How it was handled."),
            OutlineSection(key="s2", title="Storage", order_index=1, brief="Where it was kept."),
        ],
        sub_questions=sub_questions if sub_questions is not None else [
            SubQuestion(text="Who sealed the bag?", section_key="s1", order_index=0),
            SubQuestion(text="Where was it stored?", section_key="s2", order_index=1),
        ],
        revision=revision,
    )


class TestValidateAndNormalise:
    def test_an_unedited_plan_passes_through_with_a_bumped_revision(self):
        proposed = _plan()

        result = validate_and_normalise(proposed, proposed)

        assert result.revision == proposed.revision + 1
        assert [s.key for s in result.sections] == ["s1", "s2"]

    def test_reordering_preserves_keys_so_evidence_stays_attached(self):
        """
        Identity is the key, not the position. If reordering reassigned keys, evidence assigned in one
        revision would silently land on a different section in the next.
        """
        proposed = _plan()
        reordered = proposed.model_copy(update={"sections": list(reversed(proposed.sections))})

        result = validate_and_normalise(proposed, reordered)

        assert [s.key for s in result.sections] == ["s2", "s1"]
        assert [s.order_index for s in result.sections] == [0, 1]
        assert result.sections[0].title == "Storage"

    def test_renaming_keeps_the_key(self):
        proposed = _plan()
        renamed = proposed.model_copy(update={
            "sections": [proposed.sections[0].model_copy(update={"title": "Evidence handling"}),
                         proposed.sections[1]]
        })

        result = validate_and_normalise(proposed, renamed)

        assert result.sections[0].key == "s1"
        assert result.sections[0].title == "Evidence handling"

    def test_deleting_a_section_drops_only_its_sub_questions(self):
        proposed = _plan()
        trimmed = proposed.model_copy(update={
            "sections": [proposed.sections[0]],
            "sub_questions": [proposed.sub_questions[0]],
        })

        result = validate_and_normalise(proposed, trimmed)

        assert [s.key for s in result.sections] == ["s1"]
        assert [q.text for q in result.sub_questions] == ["Who sealed the bag?"]

    def test_a_user_added_section_gets_a_fresh_non_colliding_key(self):
        proposed = _plan()
        added = proposed.model_copy(update={
            "sections": proposed.sections + [OutlineSection(key="", title="Transfers", order_index=2)]
        })

        result = validate_and_normalise(proposed, added)

        new_section = result.sections[-1]
        assert new_section.title == "Transfers"
        assert new_section.key.startswith("u")
        assert new_section.key not in {"s1", "s2"}

    def test_a_user_added_section_gets_a_synthesised_sub_question_without_a_model_call(self):
        """
        A section nothing searches for would never gather evidence and would always be 'not covered'.
        The fallback is built from the user's own words -- loading a 5.7 GB model to write one sentence
        they already wrote would be absurd.
        """
        proposed = _plan()
        added = proposed.model_copy(update={
            "sections": proposed.sections + [
                OutlineSection(key="", title="Transfers", order_index=2, brief="Who moved it and when")
            ]
        })

        result = validate_and_normalise(proposed, added)

        new_key = result.sections[-1].key
        synthesised = [q for q in result.sub_questions if q.section_key == new_key]
        assert len(synthesised) == 1
        assert "Transfers" in synthesised[0].text
        assert "Who moved it and when" in synthesised[0].text

    def test_sections_are_linked_to_their_sub_question_ids(self):
        proposed = _plan()

        result = validate_and_normalise(proposed, proposed)

        by_key = {s.key: s for s in result.sections}
        ids = {q.section_key: q.id for q in result.sub_questions}
        assert by_key["s1"].sub_question_ids == [ids["s1"]]

    def test_stale_evidence_assignments_are_cleared(self):
        """Assignments belong to a gathered bank; one carried over from the proposal points at nothing."""
        proposed = _plan()
        stale = proposed.model_copy(update={
            "sections": [proposed.sections[0].model_copy(update={"evidence_doc_numbers": [1, 2, 3]}),
                         proposed.sections[1]]
        })

        result = validate_and_normalise(proposed, stale)

        assert result.sections[0].evidence_doc_numbers == []

    def test_a_blanked_sub_question_is_dropped_and_its_section_gets_a_synthesised_one(self):
        """
        Blanking s1's only sub-question leaves that section with nothing to search for, so the
        synthesiser steps in. A section that is kept but never searched would always come back
        "not covered", which would look like an absence of evidence rather than an editing slip.
        """
        proposed = _plan()
        edited = proposed.model_copy(update={
            "sub_questions": [proposed.sub_questions[0].model_copy(update={"text": "   "}),
                              proposed.sub_questions[1]]
        })

        result = validate_and_normalise(proposed, edited)

        assert "Who sealed the bag?" not in [q.text for q in result.sub_questions]
        assert [q.text for q in result.sub_questions if q.section_key == "s2"] == ["Where was it stored?"]
        s1_questions = [q.text for q in result.sub_questions if q.section_key == "s1"]
        assert s1_questions == ["Handling. How it was handled."]


class TestValidationFailures:
    def test_a_mismatched_report_id_is_rejected(self):
        proposed = _plan()
        other = _plan(report_id=uuid.uuid4())

        with pytest.raises(OutlineValidationError, match="report"):
            validate_and_normalise(proposed, other)

    def test_an_empty_outline_is_rejected(self):
        proposed = _plan()
        empty = proposed.model_copy(update={"sections": [], "sub_questions": []})

        with pytest.raises(OutlineValidationError, match="no sections"):
            validate_and_normalise(proposed, empty)

    def test_duplicate_section_keys_are_rejected(self):
        proposed = _plan()
        duped = proposed.model_copy(update={
            "sections": [proposed.sections[0], proposed.sections[0].model_copy(update={"title": "Copy"})]
        })

        with pytest.raises(OutlineValidationError, match="Duplicate section keys"):
            validate_and_normalise(proposed, duped)

    def test_a_sub_question_pointing_at_a_deleted_section_is_rejected(self):
        """
        Caught here rather than forty minutes in: the search would run, gather evidence, and have
        nowhere to put it.
        """
        proposed = _plan()
        orphaned = proposed.model_copy(update={"sections": [proposed.sections[0]]})

        with pytest.raises(OutlineValidationError, match="not in the\n?\\s*approved outline|approved outline"):
            validate_and_normalise(proposed, orphaned)

    def test_a_section_without_a_title_is_rejected(self):
        proposed = _plan()
        untitled = proposed.model_copy(update={
            "sections": [proposed.sections[0].model_copy(update={"title": "  "}), proposed.sections[1]]
        })

        with pytest.raises(OutlineValidationError, match="no title"):
            validate_and_normalise(proposed, untitled)

    def test_too_many_sections_is_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_MAX_SECTIONS", 2)
        proposed = _plan()
        oversized = proposed.model_copy(update={
            "sections": proposed.sections + [OutlineSection(key="s3", title="Third", order_index=2)]
        })

        with pytest.raises(OutlineValidationError, match="REPORT_MAX_SECTIONS"):
            validate_and_normalise(proposed, oversized)

    def test_too_many_sub_questions_is_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_MAX_SUB_QUESTIONS", 1)
        proposed = _plan()

        with pytest.raises(OutlineValidationError, match="REPORT_MAX_SUB_QUESTIONS"):
            validate_and_normalise(proposed, proposed)
