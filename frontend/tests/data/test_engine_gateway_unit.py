# Canvas: desktop_test_engine_gateway_v1
# Path: frontend/tests/data/test_engine_gateway_unit.py

"""
The gateway's mapping is where every subtle engine rule is encoded, so it gets the closest tests.

Chunks and citations are built as *real* pydantic entities rather than MagicMocks. A MagicMock
auto-creates attributes, so `getattr(chunk, "physical_path", None)` would return a MagicMock
instead of None and the unresolved-path branch could never fail -- the test would pass while the
behaviour was broken. Importing these entities costs no torch import.
"""

import uuid

import pytest
from unittest.mock import MagicMock

from onyx_desktop.domain.entities import ReportOutlineDraft, ReportPlanDraft
from onyx_desktop.domain.enums import MediaKind, MessageRole, ReportMode, TurnOutcome

pytest.importorskip("app.features.rag_retrieval.domain.entities")

from app.features.qa_generation.domain.entities import QACitation, QAResponse  # noqa: E402
from app.features.rag_retrieval.domain.entities import (  # noqa: E402
    RetrievedChunk,
    SearchResult,
)
from app.orchestration.domain.entities import (  # noqa: E402
    ReasoningIterationTrace,
    ReasonedChatResponse,
)

from app.features.deep_research_report.domain.entities import (  # noqa: E402
    EvidenceEntry,
    OutlineSection,
    Report,
    ReportSection,
    ResearchPlan,
    SubQuestion,
)
from app.features.deep_research_report.domain.enums import (  # noqa: E402
    ReportStatus,
    SectionStatus,
)

from onyx_desktop.data.engine_gateway import OnyxEngineGateway  # noqa: E402

CHUNK_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SOURCE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
CLIP_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
SESSION_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")
REPORT_ID = uuid.UUID("55555555-5555-5555-5555-555555555555")


def make_chunk(extracted_file_id=CLIP_ID, physical_path=None, **overrides):
    values = dict(
        chunk_id=CHUNK_ID,
        source_id=SOURCE_ID,
        text_payload="Generated situating context. NOT citable evidence.",
        raw_text="SPEAKER_01: The schedule was circulated on the fourteenth.",
        speaker_labels=["SPEAKER_01"],
        start_sec=72.0,
        end_sec=108.0,
        relevance_score=0.83,
        rerank_score=0.91,
        extracted_file_id=extracted_file_id,
    )
    values.update(overrides)
    chunk = RetrievedChunk(**values)
    if physical_path is not None:
        # Exactly what OnyxEngine._resolve_physical_paths does to these frozen entities.
        object.__setattr__(chunk, "physical_path", physical_path)
    return chunk


@pytest.fixture
def gateway_and_engine():
    gateway = OnyxEngineGateway()
    engine = MagicMock()
    gateway._engine = engine
    return gateway, engine


class TestSearchMapping:
    def test_uses_raw_text_never_the_contextualized_payload(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.search_standard.return_value = SearchResult(
            query="schedule", results=[make_chunk(physical_path="/vault/clip.mp4")],
            execution_time_ms=412,
        )

        outcome = gateway.search("schedule", __import__(
            "onyx_desktop.domain.enums", fromlist=["SearchMode"]).SearchMode.STANDARD, 5)

        hit = outcome.hits[0]
        assert hit.text == "SPEAKER_01: The schedule was circulated on the fourteenth."
        assert "NOT citable" not in hit.text
        assert outcome.execution_time_ms == 412

    def test_an_extracted_clip_is_marked_clip_and_needs_no_seek(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.search_standard.return_value = SearchResult(
            query="q", results=[make_chunk(physical_path="/vault/clip.mp4")],
            execution_time_ms=1,
        )
        from onyx_desktop.domain.enums import SearchMode

        evidence = gateway.search("q", SearchMode.STANDARD, 5).hits[0].as_evidence()
        assert evidence.media_kind is MediaKind.CLIP
        assert evidence.seek_to_sec == 0.0     # the clip already IS the excerpt
        assert evidence.is_playable is True

    def test_without_a_clip_the_path_is_the_full_source_and_must_be_seeked(
        self, gateway_and_engine
    ):
        gateway, engine = gateway_and_engine
        engine.search_standard.return_value = SearchResult(
            query="q",
            results=[make_chunk(extracted_file_id=None, physical_path="/vault/whole.wav")],
            execution_time_ms=1,
        )
        from onyx_desktop.domain.enums import SearchMode

        evidence = gateway.search("q", SearchMode.STANDARD, 5).hits[0].as_evidence()
        assert evidence.media_kind is MediaKind.FULL_SOURCE
        assert evidence.seek_to_sec == 72.0    # seek into the recording
        assert evidence.is_playable is True

    def test_an_unresolved_path_is_not_playable(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        # No object.__setattr__ at all: this is what a chunk looks like when the engine could not
        # resolve a file for it.
        engine.search_standard.return_value = SearchResult(
            query="q", results=[make_chunk()], execution_time_ms=1
        )
        from onyx_desktop.domain.enums import SearchMode

        evidence = gateway.search("q", SearchMode.STANDARD, 5).hits[0].as_evidence()
        assert evidence.clip_path is None
        assert evidence.media_kind is MediaKind.UNRESOLVED
        assert evidence.is_playable is False

    def test_verified_mode_calls_the_verified_engine_method(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.search_verified.return_value = SearchResult(
            query="q", results=[], execution_time_ms=1
        )
        from onyx_desktop.domain.enums import SearchMode

        gateway.search("q", SearchMode.VERIFIED, 7)

        engine.search_standard.assert_not_called()
        query = engine.search_verified.call_args.args[0]
        assert query.prompt == "q"
        assert query.limit == 7


class TestChatMapping:
    def test_citation_excerpts_are_hydrated_from_chunk_ids(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.chat.return_value = QAResponse(
            query="q",
            generated_answer="It was circulated on the fourteenth [Doc 1].",
            citations=[QACitation(
                doc_id="[Doc 1]", chunk_id=CHUNK_ID, source_id=SOURCE_ID,
                extracted_file_id=CLIP_ID, physical_path="/vault/clip.mp4",
                start_sec=72.0, end_sec=108.0,
            )],
            inference_time_ms=24000,
        )
        engine.get_chunks.return_value = [make_chunk()]
        from onyx_desktop.domain.enums import ChatMode

        message = gateway.chat("q", SESSION_ID, ChatMode.SINGLE_PASS, 5)

        engine.get_chunks.assert_called_once_with([CHUNK_ID])
        assert message.role is MessageRole.ASSISTANT
        assert message.outcome is TurnOutcome.ANSWERED
        assert len(message.evidence) == 1
        assert message.evidence[0].doc_number == 1
        # The text comes from the citation-hydrating read, not from the citation itself.
        assert message.evidence[0].excerpt.startswith("SPEAKER_01:")
        assert message.evidence[0].clip_path == "/vault/clip.mp4"
        assert message.is_ungrounded is False

    def test_grouped_citation_tags_are_normalized_so_the_ui_can_link_them(
        self, gateway_and_engine
    ):
        gateway, engine = gateway_and_engine
        other_chunk = uuid.UUID("55555555-5555-5555-5555-555555555555")
        engine.chat.return_value = QAResponse(
            query="q",
            # The engine returns the RAW model text, which can still hold a grouped tag.
            generated_answer="Both agreed [Doc 1, 2].",
            citations=[
                QACitation(doc_id="[Doc 1]", chunk_id=CHUNK_ID, source_id=SOURCE_ID,
                           start_sec=1.0, end_sec=2.0),
                QACitation(doc_id="[Doc 2]", chunk_id=other_chunk, source_id=SOURCE_ID,
                           start_sec=3.0, end_sec=4.0),
            ],
            inference_time_ms=1,
        )
        engine.get_chunks.return_value = []
        from onyx_desktop.domain.enums import ChatMode

        message = gateway.chat("q", SESSION_ID, ChatMode.SINGLE_PASS, 5)

        assert message.text == "Both agreed [Doc 1][Doc 2]."

    def test_an_answer_with_no_citations_is_flagged_ungrounded(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.chat.return_value = QAResponse(
            query="q", generated_answer="I could not find that.", citations=[],
            inference_time_ms=1,
        )
        from onyx_desktop.domain.enums import ChatMode

        message = gateway.chat("q", SESSION_ID, ChatMode.SINGLE_PASS, 5)

        assert message.evidence == ()
        assert message.is_ungrounded is True

    def test_a_failed_excerpt_lookup_still_returns_the_answer_and_its_cards(
        self, gateway_and_engine
    ):
        gateway, engine = gateway_and_engine
        engine.chat.return_value = QAResponse(
            query="q", generated_answer="Signed [Doc 1].",
            citations=[QACitation(
                doc_id="[Doc 1]", chunk_id=CHUNK_ID, source_id=SOURCE_ID,
                extracted_file_id=CLIP_ID, physical_path="/vault/clip.mp4",
                start_sec=72.0, end_sec=108.0,
            )],
            inference_time_ms=1,
        )
        engine.get_chunks.side_effect = RuntimeError("database went away")
        from onyx_desktop.domain.enums import ChatMode

        message = gateway.chat("q", SESSION_ID, ChatMode.SINGLE_PASS, 5)

        # An answer must never be dropped because the excerpt lookup failed.
        assert len(message.evidence) == 1
        assert message.evidence[0].excerpt == ""
        assert message.evidence[0].clip_path == "/vault/clip.mp4"


class TestReasonedMapping:
    def test_a_clarification_request_is_its_own_outcome(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.chat_reasoned.return_value = ReasonedChatResponse(
            qa_response=None,
            clarification_question="Which schedule, the 2019 or the 2024 revision?",
            iteration_trace=[ReasoningIterationTrace(
                iteration=1, decision="CLARIFY",
                interim_analysis="Two candidate documents match.", chunks_added=3,
            )],
            total_iterations=1,
        )
        from onyx_desktop.domain.enums import ChatMode

        message = gateway.chat("q", SESSION_ID, ChatMode.REASONING, 5)

        assert message.outcome is TurnOutcome.NEEDS_CLARIFICATION
        assert message.text.startswith("Which schedule")
        assert message.evidence == ()
        # A clarification has no evidence, but it must NOT be branded untraceable.
        assert message.is_ungrounded is False
        assert len(message.reasoning) == 1
        assert message.reasoning[0].decision == "CLARIFY"

    def test_an_unresolved_critique_is_carried_through(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.chat_reasoned.return_value = ReasonedChatResponse(
            qa_response=QAResponse(
                query="q", generated_answer="Probably [Doc 1].",
                citations=[QACitation(doc_id="[Doc 1]", chunk_id=CHUNK_ID,
                                      source_id=SOURCE_ID, start_sec=1.0, end_sec=2.0)],
                inference_time_ms=1,
            ),
            critique_unresolved=True,
            total_iterations=3,
        )
        engine.get_chunks.return_value = []
        from onyx_desktop.domain.enums import ChatMode

        message = gateway.chat("q", SESSION_ID, ChatMode.REASONING, 5)

        assert message.critique_unresolved is True
        assert message.total_iterations == 3


class TestHydration:
    def test_results_are_keyed_by_id_because_missing_chunks_are_dropped_upstream(
        self, gateway_and_engine
    ):
        gateway, engine = gateway_and_engine
        missing = uuid.UUID("66666666-6666-6666-6666-666666666666")
        # get_chunks silently skips ids that no longer exist, so the reply is shorter than the
        # request and cannot be zipped positionally.
        engine.get_chunks.return_value = [make_chunk(physical_path="/vault/clip.mp4")]

        evidence = gateway.hydrate_evidence([missing, CHUNK_ID])

        assert len(evidence) == 1
        assert evidence[0].chunk_id == CHUNK_ID

    def test_hydrated_evidence_reports_no_scores(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.get_chunks.return_value = [make_chunk(physical_path="/vault/clip.mp4")]

        evidence = gateway.hydrate_evidence([CHUNK_ID])[0]

        # An unranked read has relevance hard-coded to 0.0 upstream; surfacing that as a score
        # would invent precision.
        assert evidence.relevance_score is None
        assert evidence.rerank_score is None


class TestErrorTranslation:
    def test_engine_failures_become_desktop_errors(self, gateway_and_engine):
        from onyx_desktop.exceptions.base import DesktopError
        from onyx_desktop.domain.enums import SearchMode

        gateway, engine = gateway_and_engine
        engine.search_standard.side_effect = RuntimeError("cuda is sulking")

        with pytest.raises(DesktopError) as caught:
            gateway.search("q", SearchMode.STANDARD, 5)

        assert isinstance(caught.value.original_exception, RuntimeError)
        assert caught.value.user_message

    def test_calling_before_boot_is_refused(self):
        from onyx_desktop.domain.enums import SearchMode
        from onyx_desktop.exceptions.gateway import ApplianceOfflineError

        with pytest.raises(ApplianceOfflineError):
            OnyxEngineGateway().search("q", SearchMode.STANDARD, 5)


class TestReportMapping:
    """
    The report path, with REAL engine entities.

    These exist because the first version of this feature shipped broken: `_import_engine()` never
    registered the symbols `_apply_edits` and `_map_report` look up, so both raised KeyError the first
    time the desktop ran a report -- and the error mapper, which matches on class name, turned that
    into "The appliance could not complete that request." The view tests mocked the gateway and the
    stub gateway has its own implementation, so nothing exercised this code until a human clicked it.
    """

    @staticmethod
    def _plan(report_id=REPORT_ID):
        questions = [
            SubQuestion(text="Who sealed the bag?", section_key="s1", order_index=0),
            SubQuestion(text="Where was it stored?", section_key="s2", order_index=1),
        ]
        return ResearchPlan(
            report_id=report_id,
            topic="chain of custody",
            scope_note="Covers handling and storage.",
            sections=[
                OutlineSection(key="s1", title="Handling", order_index=0, brief="How it was handled.",
                               sub_question_ids=[questions[0].id]),
                OutlineSection(key="s2", title="Storage", order_index=1, brief="Where it was kept.",
                               sub_question_ids=[questions[1].id]),
            ],
            sub_questions=questions,
        )

    @staticmethod
    def _draft(sections):
        return ReportPlanDraft(report_id=REPORT_ID, topic="chain of custody", sections=tuple(sections))

    def _report(self, status=ReportStatus.COMPLETED, evidence=None):
        plan = self._plan()
        return Report(
            id=REPORT_ID,
            topic="chain of custody",
            status=status,
            plan=plan,
            mode=ReportMode.EXHAUSTIVE,
            rounds_used=2,
            sections=[
                ReportSection(key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
                              markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1]),
                ReportSection(key="s2", title="Storage", order_index=1,
                              status=SectionStatus.SKIPPED_NO_EVIDENCE),
            ],
            evidence=evidence if evidence is not None else [
                EvidenceEntry(doc_number=1, chunk_id=CHUNK_ID, source_id=SOURCE_ID,
                              source_name="Deposition", start_sec=10, end_sec=40,
                              speaker_labels=["speaker_0"], digest="The bag was sealed."),
            ],
            markdown_path="/reports/r.md",
            indexed_source_count=3,
            swept_source_count=3,
        )

    # ---------- _apply_edits ----------

    def test_an_unedited_plan_round_trips(self, gateway_and_engine):
        gateway, _ = gateway_and_engine
        plan = self._plan()
        draft = self._draft([
            ReportOutlineDraft(key=s.key, title=s.title, brief=s.brief,
                               questions=tuple(q.text for q in plan.sub_questions_for(s.key)))
            for s in plan.sections
        ])

        approved = gateway._apply_edits(plan, draft)

        assert [s.key for s in approved.sections] == ["s1", "s2"]
        assert [q.text for q in approved.sub_questions] == ["Who sealed the bag?", "Where was it stored?"]

    def test_a_renamed_section_keeps_its_key(self, gateway_and_engine):
        """The engine re-attaches evidence by key across revisions; regenerating one would detach it."""
        gateway, _ = gateway_and_engine
        plan = self._plan()
        draft = self._draft([
            ReportOutlineDraft(key="s1", title="Evidence handling", brief="Reworded.",
                               questions=("Who sealed the bag?",)),
            ReportOutlineDraft(key="s2", title="Storage", brief="", questions=()),
        ])

        approved = gateway._apply_edits(plan, draft)

        assert approved.sections[0].key == "s1"
        assert approved.sections[0].title == "Evidence handling"
        assert approved.sections[0].brief == "Reworded."

    def test_a_dropped_section_takes_its_questions_with_it(self, gateway_and_engine):
        gateway, _ = gateway_and_engine
        draft = self._draft([
            ReportOutlineDraft(key="s1", title="Handling", questions=("Who sealed the bag?",)),
        ])

        approved = gateway._apply_edits(self._plan(), draft)

        assert [s.key for s in approved.sections] == ["s1"]
        assert [q.section_key for q in approved.sub_questions] == ["s1"]

    def test_a_user_added_section_gets_a_key_that_cannot_collide(self, gateway_and_engine):
        gateway, _ = gateway_and_engine
        draft = self._draft([
            ReportOutlineDraft(key="s1", title="Handling", questions=("Who sealed the bag?",)),
            ReportOutlineDraft(key="", title="Transfers", brief="Who moved it.", questions=("Who moved it?",)),
        ])

        approved = gateway._apply_edits(self._plan(), draft)

        assert approved.sections[1].key not in {"s1", "s2"}
        assert approved.sections[1].title == "Transfers"

    def test_sub_question_order_is_sequential(self, gateway_and_engine):
        """It was len(questions) + question_index, which double-counted once a section had two."""
        gateway, _ = gateway_and_engine
        draft = self._draft([
            ReportOutlineDraft(key="s1", title="Handling", questions=("one", "two", "three")),
            ReportOutlineDraft(key="s2", title="Storage", questions=("four", "five")),
        ])

        approved = gateway._apply_edits(self._plan(), draft)

        assert [q.order_index for q in approved.sub_questions] == [0, 1, 2, 3, 4]

    def test_blank_questions_are_dropped(self, gateway_and_engine):
        gateway, _ = gateway_and_engine
        draft = self._draft([
            ReportOutlineDraft(key="s1", title="Handling", questions=("  ", "Who sealed the bag?", "")),
        ])

        approved = gateway._apply_edits(self._plan(), draft)

        assert [q.text for q in approved.sub_questions] == ["Who sealed the bag?"]

    def test_the_rebuilt_plan_passes_the_engines_own_validator(self, gateway_and_engine):
        """
        The gateway and the validator have to agree. Anything the editor can produce must survive
        validate_and_normalise, or the user gets a failure they cannot act on.
        """
        from app.features.deep_research_report.utils.outline_validator import validate_and_normalise

        gateway, _ = gateway_and_engine
        plan = self._plan()
        draft = self._draft([
            ReportOutlineDraft(key="s2", title="Storage", questions=("Where was it stored?",)),
            ReportOutlineDraft(key="", title="Transfers", brief="Who moved it.", questions=("Who moved it?",)),
        ])

        final = validate_and_normalise(plan, gateway._apply_edits(plan, draft))

        assert [s.title for s in final.sections] == ["Storage", "Transfers"]
        assert final.revision == plan.revision + 1

    # ---------- _map_report ----------

    def test_citations_are_rendered_as_clickable_links(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.get_chunks.return_value = [make_chunk(physical_path="/vault/clip.mp4")]

        document = gateway._map_report(self._report())

        assert "[Doc 1](onyx://doc/1)" in document.markdown
        assert document.evidence_for(1) is not None

    def test_evidence_uses_raw_text_and_resolves_its_clip(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        chunk = make_chunk(physical_path="/vault/clip.mp4")
        engine.get_chunks.return_value = [chunk]

        evidence = gateway._map_report(self._report()).evidence[0]

        # raw_text, never text_payload: the latter carries generated situating context that is
        # explicitly not citable source material.
        assert evidence.excerpt == chunk.raw_text
        assert evidence.excerpt != chunk.text_payload
        assert evidence.clip_path == "/vault/clip.mp4"
        assert evidence.media_kind is MediaKind.CLIP

    def test_a_chunk_deleted_by_a_re_ingest_falls_back_to_the_stored_digest(self, gateway_and_engine):
        """
        chunk_id is deliberately not a foreign key, so a signed-off report outlives its chunks. The
        citation must still render with real provenance rather than becoming blank.
        """
        gateway, engine = gateway_and_engine
        engine.get_chunks.return_value = []

        evidence = gateway._map_report(self._report()).evidence[0]

        assert evidence.excerpt == "The bag was sealed."
        assert evidence.clip_path is None
        assert evidence.media_kind is MediaKind.UNRESOLVED

    def test_only_cited_evidence_is_carried_over(self, gateway_and_engine):
        """The bank holds everything gathered; an uncited entry is noise in a document to be checked."""
        gateway, engine = gateway_and_engine
        engine.get_chunks.return_value = []
        report = self._report(evidence=[
            EvidenceEntry(doc_number=1, chunk_id=CHUNK_ID, source_id=SOURCE_ID,
                          source_name="Deposition", start_sec=10, end_sec=40, digest="cited"),
            EvidenceEntry(doc_number=9, chunk_id=uuid.uuid4(), source_id=SOURCE_ID,
                          source_name="Deposition", start_sec=99, end_sec=120, digest="never cited"),
        ])

        document = gateway._map_report(report)

        assert [e.doc_number for e in document.evidence] == [1]

    def test_mode_status_and_round_count_survive_the_mapping(self, gateway_and_engine):
        gateway, engine = gateway_and_engine
        engine.get_chunks.return_value = []

        document = gateway._map_report(self._report(status=ReportStatus.CANCELLED))

        assert document.mode is ReportMode.EXHAUSTIVE
        assert document.rounds_used == 2
        assert document.is_partial is True
        assert document.section_count == 1
        assert document.uncovered_count == 1

    def test_a_failed_hydration_still_returns_a_readable_report(self, gateway_and_engine):
        """A report whose citations cannot be hydrated is still worth reading."""
        gateway, engine = gateway_and_engine
        engine.get_chunks.side_effect = RuntimeError("database gone")

        document = gateway._map_report(self._report())

        assert "[Doc 1](onyx://doc/1)" in document.markdown
        assert document.evidence == ()
