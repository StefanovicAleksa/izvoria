import json
import uuid
from types import SimpleNamespace as NS

import pytest

from web_deploy.api import mapping
from web_deploy.tests.api.fakes import FakeEngine, FakeHost, Plan, Section, chunk, citation


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path)


class TestEvidence:
    def test_a_document_chunk_is_located_by_page_and_never_exposes_its_path(self, host):
        c = chunk("Article 7", path="/workspace/tenants/f/vault/x.pdf", anchor="page", page_start=7, page_end=7)
        evidence = mapping.chunk_evidence(host, 3, c)

        assert (evidence["doc_number"], evidence["location"], evidence["media_kind"]) == (3, "p. 7", "document")
        assert evidence["has_source"] is True
        assert "/workspace" not in json.dumps(evidence)

    @pytest.mark.parametrize("extracted, path, kind", [(uuid.uuid4(), "/v/c.mp4", "clip"), (None, "/v/a.wav", "full_source"),
                                                       (None, None, "unresolved")])
    def test_recordings_are_clips_or_whole_sources(self, host, extracted, path, kind):
        c = chunk(path=path, extracted_file_id=extracted)
        assert mapping.chunk_evidence(host, 1, c)["media_kind"] == kind

    def test_the_excerpt_is_the_verbatim_text_not_the_generated_context(self, host):
        assert mapping.chunk_evidence(host, 1, chunk("verbatim"))["excerpt"] == "verbatim"


class TestChat:
    def test_citations_are_hydrated_with_their_excerpts(self, host):
        engine = FakeEngine(host)
        c = engine.add_chunk(chunk("Thirty days."))
        response = NS(generated_answer="Thirty [Doc 2].", citations=[citation(c, 2)], inference_time_ms=5)

        message = mapping.chat_message(host, engine, response)

        assert message["evidence"][0]["doc_number"] == 2
        assert message["evidence"][0]["excerpt"] == "Thirty days."
        assert message["scoped_source"] is None

    def test_an_answer_read_from_one_named_source_says_which(self, host):
        response = NS(generated_answer="A.", citations=[], inference_time_ms=5,
                      scoped_source_title="Lecture 4 Conditional Probability")

        assert mapping.chat_message(host, FakeEngine(host), response)["scoped_source"] == \
            "Lecture 4 Conditional Probability"

    def test_a_stored_conversation_keeps_conversation_wide_numbers(self, host):
        engine = FakeEngine(host)
        first, second = engine.add_chunk(chunk("one")), engine.add_chunk(chunk("two"))
        turns = [
            NS(question="Q1", answer_text="A [Doc 1].", evidence_chunk_ids=[first.chunk_id], first_doc_number=1,
               mode=NS(value="chat"), created_at=None),
            NS(question="Q2", answer_text="B [Doc 2] and [Doc 1].", evidence_chunk_ids=[second.chunk_id],
               first_doc_number=2, mode=NS(value="chat"), created_at=None),
        ]

        messages = mapping.conversation(host, engine, turns)

        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        assert [(e["doc_number"], e["excerpt"]) for e in messages[3]["evidence"]] == [(2, "two"), (1, "one")]


class TestPlanEdits:
    def _plan(self):
        return Plan(report_id=uuid.uuid4(), topic="t", title="Proposed", scope_note="",
                    sections=[Section(key="s1", title="A", brief="", order_index=0),
                              Section(key="s2", title="B", brief="", order_index=1)],
                    sub_questions=[])

    def test_keys_survive_reordering_and_deleted_sections_disappear(self, host):
        edited = mapping.apply_edits(host, self._plan(), {"sections": [
            {"key": "s2", "title": "B first", "questions": ["Why?"]},
        ]})
        assert [(s.key, s.title, s.order_index) for s in edited.sections] == [("s2", "B first", 0)]
        assert [(q.text, q.section_key) for q in edited.sub_questions] == [("Why?", "s2")]

    def test_an_edited_title_is_carried_into_the_plan(self, host):
        edited = mapping.apply_edits(host, self._plan(), {"title": "  Notice\n Periods ", "sections": [{"key": "s1", "title": "A"}]})
        assert edited.title == "Notice Periods"

    def test_a_draft_without_a_title_keeps_the_proposed_one(self, host):
        edited = mapping.apply_edits(host, self._plan(), {"sections": [{"key": "s1", "title": "A"}]})
        assert edited.title == "Proposed"

    def test_a_long_title_is_capped(self, host):
        edited = mapping.apply_edits(host, self._plan(), {"title": "x" * 500, "sections": [{"key": "s1", "title": "A"}]})
        assert len(edited.title) == mapping.MAX_TITLE_CHARS

    def test_the_draft_shows_the_proposed_title_beside_the_request(self):
        draft = mapping.plan_draft("r1", self._plan())
        assert (draft["title"], draft["topic"]) == ("Proposed", "t")

    @pytest.mark.parametrize("draft", [{}, {"sections": []}, {"sections": [{"title": " "}]}, {"sections": ["x"]}])
    def test_bad_outlines_are_refused(self, host, draft):
        with pytest.raises(mapping.InvalidPlan):
            mapping.apply_edits(host, self._plan(), draft)
