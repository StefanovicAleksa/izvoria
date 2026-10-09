# Path: tests/features/deep_research_report/service/test_report_api.py

import uuid
from unittest.mock import MagicMock, Mock

import pytest

from app.core.config import settings
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, FollowUpQuery, OutlineAssignment, OutlineSection, QueryAttempt,
    Report, ReportSection, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import ReportStatus, SectionStatus
from app.features.deep_research_report.service.report_api import ReportAuthoringAPI
from app.features.deep_research_report.utils.query_log import normalise_query
from app.features.rag_retrieval.domain.entities import RetrievedChunk


@pytest.fixture
def adapter():
    adapter = MagicMock()
    adapter.count_tokens.side_effect = lambda _model, text: max(1, len(text) // 4)
    adapter.get_section_token_budget.return_value = 8000
    return adapter


@pytest.fixture
def orchestrator():
    orchestrator = Mock()
    orchestrator.acquire_model.return_value = LoadedModel(
        manifest=ModelManifest(
            model_type=ModelType.REPORT_AUTHORING, model_path="models/x.gguf", required_vram_mb=1.0
        ),
        model_instance=MagicMock(),
    )
    return orchestrator


@pytest.fixture
def repository():
    return Mock()


@pytest.fixture
def file_store():
    store = Mock()
    store.write_markdown.return_value = "/reports/2026/09/21/report.md"
    return store


@pytest.fixture
def api(adapter, orchestrator, repository, file_store):
    return ReportAuthoringAPI(adapter, orchestrator, repository, file_store)


def _chunk(chunk_id=None, raw_text="[SPEAKER_00 @ 00:00:10]: The bag was sealed.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        source_id=uuid.uuid4(),
        text_payload="GENERATED SITUATING CONTEXT. " + raw_text,
        raw_text=raw_text,
        speaker_labels=["SPEAKER_00"],
        start_sec=10,
        end_sec=40,
        relevance_score=0.5,
    )


def _evidence(chunk: RetrievedChunk, doc_number: int = 1) -> EvidenceEntry:
    return EvidenceEntry(
        doc_number=doc_number,
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_name="Deposition",
        start_sec=chunk.start_sec,
        end_sec=chunk.end_sec,
        speaker_labels=list(chunk.speaker_labels),
        digest="The bag was sealed.",
    )


class TestModelLifecycle:
    def test_the_manifest_comes_from_settings(self, api):
        assert api.manifest.model_type == ModelType.REPORT_AUTHORING
        assert api.manifest.model_path == settings.REPORT_AUTHORING_MODEL_PATH
        assert api.manifest.required_vram_mb == settings.REPORT_AUTHORING_VRAM_REQUIREMENT_MB

    def test_acquires_with_wait_for_clear_and_releases_with_flush(self, api, adapter, orchestrator):
        api.scope_plan("topic", CorpusDigest(indexed_source_count=1, indexed_chunk_count=1))

        orchestrator.acquire_model.assert_called_once_with(api.manifest, adapter, wait_for_clear=True)
        orchestrator.release_model.assert_called_once_with(
            orchestrator.acquire_model.return_value, adapter, flush=True
        )

    def test_releases_the_model_even_when_inference_raises(self, api, adapter, orchestrator):
        adapter.scope.side_effect = RuntimeError("llama_cpp exploded")

        with pytest.raises(Exception):
            api.scope_plan("topic", CorpusDigest(indexed_source_count=1, indexed_chunk_count=1))

        orchestrator.release_model.assert_called_once()

    def test_release_after_false_keeps_the_model_resident(self, api, orchestrator, adapter):
        """
        A report run sets this for everything after the first call, which is what lets one acquisition
        of the ~5.7 GB model cover outline optimisation and every section write.
        """
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t",
                            sections=[OutlineSection(key="s1", title="T", order_index=0)])
        chunk = _chunk()

        adapter.optimize_outline.return_value = OutlineAssignment(sections=plan.sections)

        api.optimize_outline(plan, [_evidence(chunk)], release_after=False)

        orchestrator.acquire_model.assert_called_once()
        orchestrator.release_model.assert_not_called()


class TestScopePlan:
    def test_persists_the_proposed_plan_and_returns_the_report(self, api, adapter, repository):
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t")
        adapter.scope.return_value = plan
        corpus = CorpusDigest(indexed_source_count=2, indexed_chunk_count=40)
        session_id = uuid.uuid4()

        api.scope_plan("t", corpus, session_id=session_id)

        repository.create_report.assert_called_once_with(
            topic="t", plan=plan, corpus=corpus, session_id=session_id, language=settings.APPLIANCE_LANGUAGE
        )

    def test_records_the_requests_answer_language_not_the_appliances(self, api, adapter, repository,
                                                                      appliance_language):
        """The hosted API scopes each request to the language its user picked; the report is written in that one."""
        from app.core.language.service.language_api import language_scope
        appliance_language("en")
        adapter.scope.return_value = ResearchPlan(report_id=uuid.uuid4(), topic="t")

        with language_scope("sr"):
            api.scope_plan("t", CorpusDigest(indexed_source_count=1, indexed_chunk_count=1))

        assert repository.create_report.call_args.kwargs["language"] == "sr"


class TestOptimizeOutline:
    def test_an_empty_bank_marks_every_section_uncovered_without_loading_a_model(
            self, api, orchestrator, adapter
    ):
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t", sections=[
            OutlineSection(key="s1", title="A", order_index=0, evidence_doc_numbers=[9]),
        ])

        assignment = api.optimize_outline(plan, [])

        assert assignment.sections[0].evidence_doc_numbers == []
        orchestrator.acquire_model.assert_not_called()
        adapter.optimize_outline.assert_not_called()
        # The case that most needs a second round is the one the model is never asked about, so the
        # fallback has to supply the query or an exhaustive run would stop after round one.
        assert [f.section_key for f in assignment.follow_ups] == ["s1"]

    def test_hallucinated_doc_numbers_are_filtered_against_the_bank(self, api, adapter, repository):
        """A number that is not in the bank would become a citation pointing at nothing."""
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t",
                            sections=[OutlineSection(key="s1", title="A", order_index=0)])
        chunk = _chunk()
        adapter.optimize_outline.return_value = OutlineAssignment(sections=[
            OutlineSection(key="s1", title="A", order_index=0, evidence_doc_numbers=[1, 42, 99])
        ])

        assignment = api.optimize_outline(plan, [_evidence(chunk, doc_number=1)])

        assert assignment.sections[0].evidence_doc_numbers == [1]
        repository.save_section_assignments.assert_called_once()

    def test_every_uncovered_section_gets_a_follow_up_even_when_the_model_gave_none(
            self, api, adapter, repository
    ):
        """A worse, derived query beats no second gathering round at all."""
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t", sections=[
            OutlineSection(key="s1", title="Handling", order_index=0, evidence_doc_numbers=[]),
            OutlineSection(key="s2", title="Storage", order_index=1, brief="Where it was kept"),
        ])
        chunk = _chunk()
        adapter.optimize_outline.return_value = OutlineAssignment(sections=[
            OutlineSection(key="s1", title="Handling", order_index=0, evidence_doc_numbers=[1]),
            OutlineSection(key="s2", title="Storage", order_index=1, brief="Where it was kept"),
        ])

        assignment = api.optimize_outline(plan, [_evidence(chunk, doc_number=1)])

        assert [f.section_key for f in assignment.follow_ups] == ["s2"]
        assert "Storage" in assignment.follow_ups[0].text

    def test_a_model_written_follow_up_is_preferred_over_the_derived_one(self, api, adapter):
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t",
                            sections=[OutlineSection(key="s1", title="Handling", order_index=0)])
        chunk = _chunk()
        adapter.optimize_outline.return_value = OutlineAssignment(
            sections=[OutlineSection(key="s1", title="Handling", order_index=0)],
            follow_ups=[FollowUpQuery(section_key="s1", text="who sealed the bag at the scene")],
        )

        assignment = api.optimize_outline(plan, [_evidence(chunk, doc_number=1)])

        assert [f.text for f in assignment.follow_ups] == ["who sealed the bag at the scene"]


class TestSearchHistory:
    """
    A follow-up is only worth a gathering round if it asks something the run has not asked. Each round
    costs an embedder load, a reranker load and a ~5.7 GB report-model load, and the fallback query is a
    pure function of a title and brief that do not change between rounds -- so without this the same
    query is re-issued every round until the budget runs out.
    """

    @staticmethod
    def _uncovered_plan():
        return ResearchPlan(report_id=uuid.uuid4(), topic="t", sections=[
            OutlineSection(key="s1", title="Storage", order_index=0, brief="Where it was kept"),
        ])

    @staticmethod
    def _asked(text: str, chunks_added: int = 0):
        return [QueryAttempt(text=text, round_index=1, chunks_added=chunks_added)]

    def test_the_history_is_shown_to_the_model(self, api, adapter):
        plan = self._uncovered_plan()
        adapter.optimize_outline.return_value = OutlineAssignment(sections=list(plan.sections))

        api.optimize_outline(plan, [_evidence(_chunk(), doc_number=1)],
                             asked=self._asked("who sealed the bag", chunks_added=2))

        asked_block = adapter.optimize_outline.call_args[0][3]
        assert "who sealed the bag" in asked_block
        assert "2 new excerpt(s)" in asked_block

    def test_no_history_means_no_block(self, api, adapter):
        """Deep mode and round 1 must send the prompt they have always sent."""
        plan = self._uncovered_plan()
        adapter.optimize_outline.return_value = OutlineAssignment(sections=list(plan.sections))

        api.optimize_outline(plan, [_evidence(_chunk(), doc_number=1)])

        assert adapter.optimize_outline.call_args[0][3] == ""

    def test_a_model_follow_up_that_repeats_an_earlier_search_is_dropped(self, api, adapter):
        """
        The repeat never reaches retrieval. The section is still uncovered afterwards, so it falls back
        to its one derived query -- which is a different search, not the one already run.
        """
        plan = self._uncovered_plan()
        adapter.optimize_outline.return_value = OutlineAssignment(
            sections=list(plan.sections),
            follow_ups=[FollowUpQuery(section_key="s1", text="Who sealed the bag?")],
        )

        assignment = api.optimize_outline(plan, [_evidence(_chunk(), doc_number=1)],
                                          asked=self._asked("who sealed the bag"))

        texts = [f.text for f in assignment.follow_ups]
        assert "Who sealed the bag?" not in texts
        assert texts == ["Storage. Where it was kept."]

    def test_the_derived_fallback_is_not_regenerated_once_it_has_been_asked(self, api, adapter):
        """
        The exact loop that was spinning: the model writes no follow-up, so the fallback fills the gap
        from the section's title and brief -- identically, every round.
        """
        plan = self._uncovered_plan()
        adapter.optimize_outline.return_value = OutlineAssignment(sections=list(plan.sections))

        first = api.optimize_outline(plan, [_evidence(_chunk(), doc_number=1)])
        assert len(first.follow_ups) == 1

        second = api.optimize_outline(
            plan, [_evidence(_chunk(), doc_number=1)],
            asked=[QueryAttempt(text=first.follow_ups[0].text, round_index=1, chunks_added=0)],
        )

        assert second.follow_ups == []

    def test_a_still_unasked_section_keeps_its_follow_up(self, api, adapter):
        """Exhausting one section must not silence the others."""
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t", sections=[
            OutlineSection(key="s1", title="Storage", order_index=0, brief="Where it was kept"),
            OutlineSection(key="s2", title="Handling", order_index=1, brief="Who touched it"),
        ])
        adapter.optimize_outline.return_value = OutlineAssignment(sections=list(plan.sections))

        exhausted = api.optimize_outline(plan, [_evidence(_chunk(), doc_number=1)]).follow_ups
        storage = next(f for f in exhausted if f.section_key == "s1")

        assignment = api.optimize_outline(
            plan, [_evidence(_chunk(), doc_number=1)],
            asked=[QueryAttempt(text=storage.text, round_index=1, chunks_added=0)],
        )

        assert [f.section_key for f in assignment.follow_ups] == ["s2"]

    def test_no_query_is_issued_twice_in_one_pass(self, api, adapter):
        """
        Two sections can be briefed similarly enough that the model writes one query twice. Searching it
        twice in the same round costs two retrievals to return one set of chunks.
        """
        plan = ResearchPlan(report_id=uuid.uuid4(), topic="t", sections=[
            OutlineSection(key="s1", title="A", order_index=0),
            OutlineSection(key="s2", title="B", order_index=1),
        ])
        adapter.optimize_outline.return_value = OutlineAssignment(
            sections=list(plan.sections),
            follow_ups=[FollowUpQuery(section_key="s1", text="what was sealed"),
                        FollowUpQuery(section_key="s2", text="What was sealed?")],
        )

        assignment = api.optimize_outline(plan, [_evidence(_chunk(), doc_number=1)])

        texts = [normalise_query(f.text) for f in assignment.follow_ups]
        assert len(texts) == len(set(texts))
        assert texts.count("what was sealed") == 1

    def test_an_empty_bank_still_respects_the_history(self, api, adapter, orchestrator):
        """The no-model path has its own return and would otherwise bypass the filter entirely."""
        plan = self._uncovered_plan()

        derived = api.optimize_outline(plan, []).follow_ups[0]
        assignment = api.optimize_outline(
            plan, [], asked=[QueryAttempt(text=derived.text, round_index=1, chunks_added=0)]
        )

        assert assignment.follow_ups == []
        orchestrator.acquire_model.assert_not_called()


class TestWriteSection:
    def _setup(self, adapter, body):
        adapter.write_section.return_value = body

    def test_the_writer_sees_only_verbatim_transcript_never_the_generated_payload(self, api, adapter):
        """
        The product's non-negotiable: an answer grounded in model-generated situating context would
        still come out carrying a valid-looking citation.
        """
        chunk = _chunk(raw_text="[SPEAKER_00 @ 00:00:10]: We logged it at nine.")
        self._setup(adapter, "Logged at nine [Doc 1].")
        section = OutlineSection(key="s1", title="T", order_index=0, evidence_doc_numbers=[1])

        api.write_section(uuid.uuid4(), "topic", section, [_evidence(chunk)], {chunk.chunk_id: chunk})

        evidence_block = adapter.write_section.call_args.kwargs["evidence_block"]
        assert "We logged it at nine." in evidence_block
        assert "GENERATED SITUATING CONTEXT" not in evidence_block

    def test_citations_are_restricted_to_this_sections_allowed_numbers(self, api, adapter):
        """
        Numbers are report-wide, so without this restriction a section could cite a document it was
        never shown.
        """
        chunk = _chunk()
        self._setup(adapter, "Claim [Doc 1]. Other claim [Doc 7].")
        section = OutlineSection(key="s1", title="T", order_index=0, evidence_doc_numbers=[1])

        written = api.write_section(
            uuid.uuid4(), "topic", section, [_evidence(chunk)], {chunk.chunk_id: chunk}
        )

        assert written.cited_doc_numbers == [1]
        assert "[Doc 7]" not in written.markdown_body

    def test_a_section_with_no_resolvable_evidence_is_skipped_without_a_model_call(
            self, api, adapter, orchestrator
    ):
        section = OutlineSection(key="s1", title="T", order_index=0, evidence_doc_numbers=[1])

        written = api.write_section(uuid.uuid4(), "topic", section, [], {})

        assert written.status == SectionStatus.SKIPPED_NO_EVIDENCE
        orchestrator.acquire_model.assert_not_called()
        adapter.write_section.assert_not_called()

    def test_a_section_that_still_cites_nothing_becomes_not_covered_not_uncited_prose(
            self, api, adapter
    ):
        chunk = _chunk()
        self._setup(adapter, "The bag was sealed and then stored.")
        section = OutlineSection(key="s1", title="T", order_index=0, evidence_doc_numbers=[1])

        written = api.write_section(
            uuid.uuid4(), "topic", section, [_evidence(chunk)], {chunk.chunk_id: chunk}
        )

        assert written.status == SectionStatus.SKIPPED_NO_EVIDENCE
        assert written.markdown_body == ""

    def test_a_heading_the_model_repeated_is_stripped(self, api, adapter):
        """
        The renderer owns section headings. Models emit one anyway often enough that the rendered
        report showed every heading twice.
        """
        chunk = _chunk()
        self._setup(adapter, "## Handling\n\nThe bag was sealed [Doc 1].")
        section = OutlineSection(key="s1", title="Handling", order_index=0, evidence_doc_numbers=[1])

        written = api.write_section(
            uuid.uuid4(), "topic", section, [_evidence(chunk)], {chunk.chunk_id: chunk}
        )

        assert written.markdown_body == "The bag was sealed [Doc 1]."

    def test_sub_headings_inside_the_body_are_kept(self, api, adapter):
        chunk = _chunk()
        self._setup(adapter, "Sealed [Doc 1].\n\n### Transfers\n\nMoved [Doc 1].")
        section = OutlineSection(key="s1", title="Handling", order_index=0, evidence_doc_numbers=[1])

        written = api.write_section(
            uuid.uuid4(), "topic", section, [_evidence(chunk)], {chunk.chunk_id: chunk}
        )

        assert "### Transfers" in written.markdown_body

    def test_grouped_citations_are_normalised(self, api, adapter):
        chunk_a, chunk_b = _chunk(), _chunk()
        self._setup(adapter, "Both agree [Docs 1-2].")
        section = OutlineSection(key="s1", title="T", order_index=0, evidence_doc_numbers=[1, 2])
        evidence = [_evidence(chunk_a, 1), _evidence(chunk_b, 2)]

        written = api.write_section(
            uuid.uuid4(), "topic", section, evidence,
            {chunk_a.chunk_id: chunk_a, chunk_b.chunk_id: chunk_b},
        )

        assert "[Doc 1][Doc 2]" in written.markdown_body
        assert written.cited_doc_numbers == [1, 2]

    def test_every_finished_section_is_committed_immediately(self, api, adapter, repository):
        """A crash costs at most the section in flight."""
        chunk = _chunk()
        self._setup(adapter, "Sealed [Doc 1].")
        section = OutlineSection(key="s1", title="T", order_index=0, evidence_doc_numbers=[1])
        report_id = uuid.uuid4()

        written = api.write_section(
            report_id, "topic", section, [_evidence(chunk)], {chunk.chunk_id: chunk}
        )

        repository.save_section.assert_called_once_with(report_id, written)

    def test_evidence_beyond_the_budget_is_dropped_rather_than_overflowing(self, api, adapter):
        chunk = _chunk()
        adapter.get_section_token_budget.return_value = 0
        section = OutlineSection(key="s1", title="T", order_index=0, evidence_doc_numbers=[1])

        written = api.write_section(
            uuid.uuid4(), "topic", section, [_evidence(chunk)], {chunk.chunk_id: chunk}
        )

        assert written.status == SectionStatus.SKIPPED_NO_EVIDENCE
        adapter.write_section.assert_not_called()


class TestWriteSummary:
    def test_summary_may_only_cite_numbers_the_body_already_used(self, api, adapter):
        """
        The summary is the one place a report synthesises across sections, and so the easiest place to
        assert a connection no source supports.
        """
        adapter.write_summary.return_value = "Overall [Doc 1] and also [Doc 9]."
        sections = [ReportSection(key="s1", title="A", order_index=0, status=SectionStatus.WRITTEN,
                                  markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1])]

        summary = api.write_summary("topic", sections)

        assert "[Doc 1]" in summary
        assert "[Doc 9]" not in summary
        assert adapter.write_summary.call_args.kwargs["allowed_doc_numbers"] == [1]

    def test_no_written_sections_means_no_summary_and_no_model_load(self, api, orchestrator, adapter):
        sections = [ReportSection(key="s1", title="A", order_index=0,
                                  status=SectionStatus.SKIPPED_NO_EVIDENCE)]

        assert api.write_summary("topic", sections) == ""
        orchestrator.acquire_model.assert_not_called()
        adapter.write_summary.assert_not_called()

    def test_the_summary_reads_section_prose_not_evidence(self, api, adapter):
        adapter.write_summary.return_value = "Overall [Doc 1]."
        sections = [ReportSection(key="s1", title="Handling", order_index=0,
                                  status=SectionStatus.WRITTEN,
                                  markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1])]

        api.write_summary("topic", sections)

        digest = adapter.write_summary.call_args.kwargs["section_digest"]
        assert "## Handling" in digest
        assert "Sealed [Doc 1]." in digest


class TestRenderAndFinalize:
    def _report(self, report_id) -> Report:
        return Report(
            id=report_id,
            topic="t",
            status=ReportStatus.WRITING,
            plan=ResearchPlan(report_id=report_id, topic="t"),
            sections=[ReportSection(key="s1", title="A", order_index=0, status=SectionStatus.WRITTEN,
                                    markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1])],
            evidence=[_evidence(_chunk(), 1)],
        )

    def test_renders_from_the_persisted_report_then_finalizes(self, api, repository, file_store):
        report_id = uuid.uuid4()
        repository.get_report.return_value = self._report(report_id)

        api.render_and_finalize(report_id, "Summary [Doc 1].")

        markdown = file_store.write_markdown.call_args.args[1]
        assert "Summary [Doc 1]." in markdown
        assert "## References" in markdown
        repository.finalize.assert_called_once_with(
            report_id, "Summary [Doc 1].", "/reports/2026/09/21/report.md"
        )

    def test_render_partial_writes_a_file_without_marking_the_report_complete(
            self, api, repository, file_store
    ):
        """A stopped report is still worth reading; the status stays whatever the caller set."""
        report_id = uuid.uuid4()
        repository.get_report.return_value = self._report(report_id)

        report, path = api.render_partial(report_id)

        assert path == "/reports/2026/09/21/report.md"
        assert report is not None
        repository.finalize.assert_not_called()

    def test_render_partial_returns_nothing_when_the_report_cannot_be_read_at_all(
            self, api, repository, file_store
    ):
        """
        This runs on the cancel path. A repository that fails the first read is likely to fail the
        re-read too, and an unmapped SQLAlchemy error escaping the feature boundary there would turn a
        clean user-requested stop into a crash.
        """
        repository.get_report.side_effect = RuntimeError("connection reset")

        report, path = api.render_partial(uuid.uuid4())

        assert report is None
        assert path is None
        file_store.write_markdown.assert_not_called()

    def test_render_partial_still_returns_the_report_when_only_the_file_write_failed(
            self, api, repository, file_store
    ):
        report_id = uuid.uuid4()
        stored = self._report(report_id)
        repository.get_report.return_value = stored
        file_store.write_markdown.side_effect = OSError("disk full")

        report, path = api.render_partial(report_id)

        assert report is stored
        assert path is None

    def test_render_partial_writes_nothing_when_no_section_was_written(self, api, repository, file_store):
        report_id = uuid.uuid4()
        empty = self._report(report_id)
        repository.get_report.return_value = empty.model_copy(update={"sections": []})

        report, path = api.render_partial(report_id)

        assert path is None
        file_store.write_markdown.assert_not_called()


class TestOwner:
    """Hosted API: a report belongs to the user who scoped it. Without an owner the calls are the appliance's."""

    def test_set_owner_reaches_the_repository(self, api, repository):
        report_id = uuid.uuid4()

        api.set_owner(report_id, "user-7")

        repository.set_owner.assert_called_once_with(report_id, "user-7")

    def test_set_owner_failure_is_a_typed_report_error(self, api, repository):
        from app.features.deep_research_report.exceptions.base import DeepResearchReportError
        repository.set_owner.side_effect = RuntimeError("connection reset")

        with pytest.raises(DeepResearchReportError):
            api.set_owner(uuid.uuid4(), "user-7")

    def test_get_report_passes_the_owner(self, api, repository):
        report_id = uuid.uuid4()

        api.get_report(report_id, owner_id="user-7")

        repository.get_report.assert_called_once_with(report_id, owner_id="user-7")

    def test_get_report_without_an_owner_calls_as_before(self, api, repository):
        report_id = uuid.uuid4()

        api.get_report(report_id)

        repository.get_report.assert_called_once_with(report_id)

    def test_list_reports_passes_the_owner(self, api, repository):
        session_id = uuid.uuid4()

        api.list_reports(session_id, 20, owner_id="user-7")

        repository.list_reports.assert_called_once_with(session_id, 20, owner_id="user-7")

    def test_list_reports_without_an_owner_calls_as_before(self, api, repository):
        api.list_reports(None, 20)

        repository.list_reports.assert_called_once_with(None, 20)


class TestTitles:
    def _report(self, report_id, title="", markdown_path=None) -> Report:
        return Report(
            id=report_id, topic="create a report on custody", status=ReportStatus.COMPLETED,
            plan=ResearchPlan(report_id=report_id, topic="create a report on custody", title=title),
            sections=[ReportSection(key="s1", title="A", order_index=0, status=SectionStatus.WRITTEN,
                                    markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1])],
            evidence=[_evidence(_chunk(), 1)], markdown_path=markdown_path,
        )

    def test_approving_tidies_an_edited_title(self, api, repository):
        report_id = uuid.uuid4()
        plan = ResearchPlan(report_id=report_id, topic="t", title="  # Chain of Custody.  ")

        api.approve_plan(report_id, plan)

        assert repository.replace_plan.call_args.args[1].title == "Chain of Custody"

    def test_renaming_an_unwritten_report_touches_no_file(self, api, repository, file_store):
        report_id = uuid.uuid4()
        repository.get_report.side_effect = [self._report(report_id), self._report(report_id, "Chain of Custody")]

        renamed = api.rename_report(report_id, '"Chain of Custody."')

        repository.rename_report.assert_called_once_with(report_id, "Chain of Custody")
        file_store.write_markdown.assert_not_called()
        assert renamed.display_title == "Chain of Custody"

    def test_renaming_a_written_report_renders_its_file_again_with_the_new_title(
            self, api, repository, file_store, monkeypatch
    ):
        report_id = uuid.uuid4()
        old_path = "/reports/2026/09/30/report.md"
        repository.get_report.side_effect = [
            self._report(report_id, markdown_path=old_path),
            self._report(report_id, "Chain of Custody", markdown_path=old_path),
        ]
        removed = []
        monkeypatch.setattr("app.features.deep_research_report.service.report_api._remove_replaced",
                            lambda old, new: removed.append((old, new)))

        renamed = api.rename_report(report_id, "Chain of Custody")

        assert file_store.write_markdown.call_args.args[1].startswith("# Chain of Custody\n")
        repository.set_markdown_path.assert_called_once_with(report_id, "/reports/2026/09/21/report.md")
        assert removed == [(old_path, "/reports/2026/09/21/report.md")]
        assert renamed.markdown_path == "/reports/2026/09/21/report.md"

    def test_renaming_a_missing_report_returns_none(self, api, repository):
        repository.get_report.return_value = None

        assert api.rename_report(uuid.uuid4(), "x") is None
        repository.rename_report.assert_not_called()

    def test_a_rename_failure_is_a_typed_report_error(self, api, repository):
        from app.features.deep_research_report.exceptions.base import DeepResearchReportError
        repository.get_report.side_effect = RuntimeError("connection reset")

        with pytest.raises(DeepResearchReportError):
            api.rename_report(uuid.uuid4(), "x")
