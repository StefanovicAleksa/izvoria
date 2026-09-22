# Path: app/features/deep_research_report/utils/outline_validator.py

"""
Validation for the outline the user hands back after editing the proposed plan.

This runs before the long run starts, not during it. A dangling reference or a duplicate key discovered
forty minutes in costs the whole run; discovered here it costs a sentence of feedback.
"""

import uuid
from typing import List

from app.core.config import settings
from app.features.deep_research_report.domain.entities import OutlineSection, ResearchPlan, SubQuestion
from app.features.deep_research_report.exceptions.pipeline import OutlineValidationError

USER_SECTION_PREFIX = "u"


def validate_and_normalise(proposed: ResearchPlan, approved: ResearchPlan) -> ResearchPlan:
    """
    Checks the approved outline against the one that was proposed, and returns the plan to actually run.

    Section `key`s carried over from `proposed` are preserved rather than reassigned by position. That is
    what lets a user reorder or rename sections without silently detaching them from the evidence a later
    revision will assign to them -- the key, not the position, is the identity.

    A section the user added with no sub-questions gets one synthesised from its own title and brief. That
    is done here, deterministically, rather than with a model call: asking the planner to fill it in would
    mean loading a 5.7 GB model to write one sentence the user already wrote.
    """
    if approved.report_id != proposed.report_id:
        raise OutlineValidationError(
            f"Approved plan is for report {approved.report_id}, but {proposed.report_id} was proposed."
        )

    sections = list(approved.sections)
    if not sections:
        raise OutlineValidationError("The approved outline has no sections; there would be nothing to write.")

    if len(sections) > settings.REPORT_MAX_SECTIONS:
        raise OutlineValidationError(
            f"The approved outline has {len(sections)} sections; the maximum is "
            f"{settings.REPORT_MAX_SECTIONS} (REPORT_MAX_SECTIONS)."
        )

    keys = [s.key for s in sections if s.key]
    if len(keys) != len(set(keys)):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        raise OutlineValidationError(f"Duplicate section keys in the approved outline: {', '.join(duplicates)}")

    known_keys = {s.key for s in proposed.sections}
    normalised_sections: List[OutlineSection] = []
    next_user_index = 1

    for order_index, section in enumerate(sections):
        if not section.title.strip():
            raise OutlineValidationError(f"Section {order_index + 1} of the approved outline has no title.")

        key = section.key
        if not key or key not in known_keys:
            # A section the user added. Give it a fresh key that cannot collide with a planner key.
            while f"{USER_SECTION_PREFIX}{next_user_index}" in set(keys) | {s.key for s in normalised_sections}:
                next_user_index += 1
            key = f"{USER_SECTION_PREFIX}{next_user_index}"
            next_user_index += 1

        normalised_sections.append(section.model_copy(update={
            "key": key,
            "order_index": order_index,
            # Evidence is assigned after gathering; anything carried over from the proposal is stale.
            "evidence_doc_numbers": [],
        }))

    key_by_old = {old.key: new.key for old, new in zip(sections, normalised_sections)}
    valid_keys = {s.key for s in normalised_sections}

    sub_questions = _normalise_sub_questions(approved.sub_questions, key_by_old, valid_keys)
    sub_questions = _fill_empty_sections(normalised_sections, sub_questions)

    if len(sub_questions) > settings.REPORT_MAX_SUB_QUESTIONS:
        raise OutlineValidationError(
            f"The approved plan has {len(sub_questions)} sub-questions; the maximum is "
            f"{settings.REPORT_MAX_SUB_QUESTIONS} (REPORT_MAX_SUB_QUESTIONS)."
        )

    linked = [
        section.model_copy(update={
            "sub_question_ids": [q.id for q in sub_questions if q.section_key == section.key]
        })
        for section in normalised_sections
    ]

    return approved.model_copy(update={
        "sections": linked,
        "sub_questions": sub_questions,
        "revision": proposed.revision + 1,
    })


def _normalise_sub_questions(
        sub_questions: List[SubQuestion],
        key_by_old: dict,
        valid_keys: set,
) -> List[SubQuestion]:
    normalised: List[SubQuestion] = []
    for order_index, question in enumerate(sub_questions):
        text = question.text.strip()
        if not text:
            continue

        section_key = key_by_old.get(question.section_key, question.section_key)
        if section_key not in valid_keys:
            raise OutlineValidationError(
                f"Sub-question {text!r} points at section {question.section_key!r}, which is not in the "
                f"approved outline. Delete the sub-question or restore the section."
            )

        normalised.append(question.model_copy(update={
            "text": text,
            "section_key": section_key,
            "order_index": order_index,
        }))
    return normalised


def _fill_empty_sections(
        sections: List[OutlineSection],
        sub_questions: List[SubQuestion],
) -> List[SubQuestion]:
    """A section with no sub-question would never be searched for, so it gets one from its own wording."""
    covered = {q.section_key for q in sub_questions}
    filled = list(sub_questions)

    for section in sections:
        if section.key in covered:
            continue
        text = f"{section.title.strip()}. {section.brief.strip()}".strip().rstrip(".") + "."
        filled.append(SubQuestion(
            id=uuid.uuid4(),
            text=text,
            section_key=section.key,
            order_index=len(filled),
        ))

    return filled
