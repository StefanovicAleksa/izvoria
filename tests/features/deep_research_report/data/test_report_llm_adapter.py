# Path: tests/features/deep_research_report/data/test_report_llm_adapter.py

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.deep_research_report.data.report_llm_adapter import LlamaCppReportAdapter
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, IndexedSource, OutlineSection, ResearchPlan,
)
from app.features.deep_research_report.exceptions.pipeline import JSONPlanParseError


@pytest.fixture
def adapter():
    return LlamaCppReportAdapter()


@pytest.fixture
def model():
    llm = MagicMock()
    llm.n_ctx.return_value = 131072
    llm.tokenize.return_value = [0] * 10
    return LoadedModel(
        manifest=ModelManifest(
            model_type=ModelType.REPORT_AUTHORING, model_path="models/x.gguf", required_vram_mb=1.0
        ),
        model_instance=llm,
    )


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _messages(mock_chat, call_index: int = -1):
    """chat_completion(llm, messages, ...) passes messages positionally."""
    return mock_chat.call_args_list[call_index].args[1]


def _corpus() -> CorpusDigest:
    return CorpusDigest(
        indexed_source_count=2,
        indexed_chunk_count=40,
        sources=[
            IndexedSource(source_id=uuid.uuid4(), name="Deposition", chunk_count=30,
                          first_start_sec=0, last_end_sec=3600),
            IndexedSource(source_id=uuid.uuid4(), name="Interview", chunk_count=10),
        ],
        sample_digests=["[SPEAKER_00 @ 00:00:01]: The bag was sealed."],
    )


class TestLifecycle:
    def test_load_raises_when_the_weights_are_missing(self, adapter):
        manifest = ModelManifest(
            model_type=ModelType.REPORT_AUTHORING, model_path="models/nope.gguf", required_vram_mb=1.0
        )

        with pytest.raises(Exception) as excinfo:
            adapter.load(manifest)

        assert "missing" in str(excinfo.value).lower()

    def test_unload_closes_the_model_and_nulls_the_instance(self, adapter, model):
        instance = model.model_instance

        assert adapter.unload(model) is True
        instance.close.assert_called_once()
        assert model.model_instance is None

    def test_unload_tolerates_an_already_unloaded_model(self, adapter, model):
        model.model_instance = None

        assert adapter.unload(model) is True


class TestSectionTokenBudget:
    def test_budget_is_capped_by_the_evidence_setting_not_by_the_window(self, adapter, model, monkeypatch):
        """
        The window is 128K, but prefill is what costs wall-clock time. A section allowed to fill the
        window would make an eight-section report take hours.
        """
        from app.core.config import settings
        monkeypatch.setattr(settings, "REPORT_SECTION_EVIDENCE_TOKENS", 8000)
        section = OutlineSection(key="s1", title="T", order_index=0)

        budget = adapter.get_section_token_budget(model, section, "topic")

        assert budget == 8000

    def test_budget_never_goes_negative(self, adapter, model, monkeypatch):
        model.model_instance.n_ctx.return_value = 8
        model.model_instance.tokenize.return_value = [0] * 100_000
        section = OutlineSection(key="s1", title="T", order_index=0)

        assert adapter.get_section_token_budget(model, section, "topic") == 0


class TestScope:
    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_parses_a_plan_and_links_sub_questions_to_sections(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({
            "scope_note": "Covers handling and storage.",
            "sections": [
                {"key": "s1", "title": "Handling", "brief": "How it was handled."},
                {"key": "s2", "title": "Storage", "brief": "Where it was kept."},
            ],
            "sub_questions": [
                {"section_key": "s1", "text": "Who sealed the bag?"},
                {"section_key": "s2", "text": "Where was it stored?"},
            ],
        }))

        plan = adapter.scope(model, "Chain of custody", _corpus())

        assert plan.topic == "Chain of custody"
        assert plan.scope_note == "Covers handling and storage."
        assert [s.key for s in plan.sections] == ["s1", "s2"]
        assert len(plan.sections[0].sub_question_ids) == 1
        assert plan.revision == 1

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_the_proposed_title_is_kept_tidied_and_the_request_stays_the_topic(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({
            "title": '"Supply, Demand and Price Elasticity."',
            "sections": [{"key": "s1", "title": "Handling"}],
        }))

        plan = adapter.scope(model, "create a report on microeconomics", _corpus())

        assert plan.title == "Supply, Demand and Price Elasticity"
        assert plan.topic == "create a report on microeconomics"

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_a_plan_without_a_title_falls_back_to_its_request(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({"sections": [{"key": "s1", "title": "Handling"}]}))

        plan = adapter.scope(model, "Chain of custody", _corpus())

        assert (plan.title, plan.display_title) == ("", "Chain of custody")

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_the_corpus_overview_and_real_excerpts_reach_the_prompt(self, mock_chat, adapter, model):
        """
        The grounding checkpoint: the planner sees real source names and real transcript, so it cannot
        propose a section about material this archive does not contain.
        """
        mock_chat.return_value = _reply(json.dumps({"sections": [{"key": "s1", "title": "T"}]}))

        adapter.scope(model, "Chain of custody", _corpus())

        user_content = _messages(mock_chat)[1]["content"]
        assert "Deposition" in user_content
        assert "Interview" in user_content
        assert "The bag was sealed." in user_content

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_a_sub_question_aimed_at_an_unknown_section_is_reattached_not_dropped(
            self, mock_chat, adapter, model
    ):
        mock_chat.return_value = _reply(json.dumps({
            "sections": [{"key": "s1", "title": "Handling"}],
            "sub_questions": [{"section_key": "s9", "text": "Who sealed it?"}],
        }))

        plan = adapter.scope(model, "topic", _corpus())

        assert plan.sub_questions[0].section_key == "s1"

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_no_sections_fails_closed(self, mock_chat, adapter, model):
        """
        There is no safe default outline. Inventing one would put the user's approval on a plan the
        corpus was never consulted about.
        """
        mock_chat.return_value = _reply(json.dumps({"sections": []}))

        with pytest.raises(JSONPlanParseError):
            adapter.scope(model, "topic", _corpus())

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_unparseable_json_fails_closed(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply("I think the outline should be about...")

        with pytest.raises(JSONPlanParseError):
            adapter.scope(model, "topic", _corpus())

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_sections_without_titles_are_skipped(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({
            "sections": [{"key": "s1", "title": "  "}, {"key": "s2", "title": "Storage"}],
        }))

        plan = adapter.scope(model, "topic", _corpus())

        assert [s.title for s in plan.sections] == ["Storage"]

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_respects_the_section_and_sub_question_caps(self, mock_chat, adapter, model, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "REPORT_MAX_SECTIONS", 2)
        monkeypatch.setattr(settings, "REPORT_MAX_SUB_QUESTIONS", 3)
        mock_chat.return_value = _reply(json.dumps({
            "sections": [{"key": f"s{i}", "title": f"T{i}"} for i in range(6)],
            "sub_questions": [{"section_key": "s0", "text": f"q{i}"} for i in range(10)],
        }))

        plan = adapter.scope(model, "topic", _corpus())

        assert len(plan.sections) == 2
        assert len(plan.sub_questions) == 3


class TestOptimizeOutline:
    def _plan(self):
        return ResearchPlan(
            report_id=uuid.uuid4(),
            topic="Chain of custody",
            sections=[
                OutlineSection(key="s1", title="Handling", order_index=0),
                OutlineSection(key="s2", title="Storage", order_index=1),
                OutlineSection(key="s3", title="Transfers", order_index=2),
            ],
        )

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_assigns_doc_numbers_and_applies_the_models_ordering(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({"sections": [
            {"key": "s3", "doc_numbers": [5, 2]},
            {"key": "s1", "doc_numbers": [1]},
        ]}))

        assignment = adapter.optimize_outline(model, self._plan(), "index")
        sections = assignment.sections

        assert [s.key for s in sections[:2]] == ["s3", "s1"]
        assert sections[0].evidence_doc_numbers == [2, 5]
        assert [s.order_index for s in sections] == [0, 1, 2]

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_a_section_the_model_omitted_is_kept_with_no_evidence_not_deleted(
            self, mock_chat, adapter, model
    ):
        """
        Dropping a section the user approved without telling them would quietly change the report they
        signed off on. It becomes 'not covered' instead.
        """
        mock_chat.return_value = _reply(json.dumps({"sections": [{"key": "s1", "doc_numbers": [1]}]}))

        sections = adapter.optimize_outline(model, self._plan(), "index").sections

        assert {s.key for s in sections} == {"s1", "s2", "s3"}
        assert [s.evidence_doc_numbers for s in sections if s.key == "s2"] == [[]]

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_invented_section_keys_are_ignored(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({"sections": [
            {"key": "s99", "doc_numbers": [1]}, {"key": "s1", "doc_numbers": [2]},
        ]}))

        sections = adapter.optimize_outline(model, self._plan(), "index").sections

        assert "s99" not in {s.key for s in sections}

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_duplicate_and_non_positive_doc_numbers_are_cleaned(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({"sections": [
            {"key": "s1", "doc_numbers": [3, 3, 1, 0, -2]},
        ]}))

        sections = adapter.optimize_outline(model, self._plan(), "index").sections

        assert sections[0].evidence_doc_numbers == [1, 3]


    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_follow_ups_are_kept_only_for_sections_that_got_nothing(self, mock_chat, adapter, model):
        """
        A follow-up aimed at a covered section would spend a whole extra gathering round searching for
        evidence the report already has.
        """
        mock_chat.return_value = _reply(json.dumps({
            "sections": [{"key": "s1", "doc_numbers": [1]}, {"key": "s2", "doc_numbers": []}],
            "follow_up_questions": [
                {"section_key": "s1", "text": "already covered"},
                {"section_key": "s2", "text": "re-worded query"},
                {"section_key": "s9", "text": "invented section"},
            ],
        }))

        assignment = adapter.optimize_outline(model, self._plan(), "index")

        assert [(f.section_key, f.text) for f in assignment.follow_ups] == [("s2", "re-worded query")]

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_uncovered_exposes_exactly_the_sections_with_no_evidence(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply(json.dumps({
            "sections": [{"key": "s1", "doc_numbers": [1]}, {"key": "s2", "doc_numbers": []}],
        }))

        assignment = adapter.optimize_outline(model, self._plan(), "index")

        assert {s.key for s in assignment.uncovered} == {"s2", "s3"}


class TestWriteSection:
    def _section(self):
        return OutlineSection(key="s1", title="Handling", order_index=0, brief="How it was handled.")

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_returns_a_cited_draft_without_retrying(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply("The bag was sealed [Doc 1].")

        body = adapter.write_section(model, "topic", self._section(), "evidence", [1], "outline")

        assert body == "The bag was sealed [Doc 1]."
        assert mock_chat.call_count == 1

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_an_uncited_draft_is_re_prompted_once_in_the_same_acquisition(self, mock_chat, adapter, model):
        """
        An uncited draft is more often a formatting lapse than a genuine absence of evidence, and one
        extra generation is far cheaper than discarding a section the corpus could support.
        """
        mock_chat.side_effect = [
            _reply("The bag was sealed."),
            _reply("The bag was sealed [Doc 1]."),
        ]

        body = adapter.write_section(model, "topic", self._section(), "evidence", [1], "outline")

        assert body == "The bag was sealed [Doc 1]."
        assert mock_chat.call_count == 2
        retry_messages = _messages(mock_chat)
        assert retry_messages[-2]["role"] == "assistant"
        assert "cites no sources" in retry_messages[-1]["content"]

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_an_explicit_no_evidence_reply_is_not_retried(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply("NO_EVIDENCE")

        body = adapter.write_section(model, "topic", self._section(), "", [], "outline")

        assert body == "NO_EVIDENCE"
        assert mock_chat.call_count == 1

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_the_allowed_doc_numbers_are_stated_in_the_prompt(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply("x [Doc 7].")

        adapter.write_section(model, "topic", self._section(), "evidence", [7, 12], "outline")

        content = _messages(mock_chat)[1]["content"]
        assert "[Doc 7], [Doc 12]" in content

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_only_this_sections_evidence_reaches_the_prompt(self, mock_chat, adapter, model):
        """Pruning is what keeps peak context max-over-sections rather than the sum of them."""
        mock_chat.return_value = _reply("x [Doc 1].")

        adapter.write_section(model, "topic", self._section(), "ONLY THIS SECTION", [1], "outline")

        content = _messages(mock_chat)[1]["content"]
        assert "ONLY THIS SECTION" in content


class TestWriteSummary:
    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_writes_from_section_prose_and_states_the_allowed_numbers(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply("Custody held [Doc 1].")

        summary = adapter.write_summary(model, "topic", "## Handling\nSealed [Doc 1].", [1, 3])

        assert summary == "Custody held [Doc 1]."
        content = _messages(mock_chat)[1]["content"]
        assert "## Handling" in content
        assert "[Doc 1], [Doc 3]" in content


class TestReportLanguage:
    """A report on a Serbian archive is planned, written and summarised in Serbian; NO_EVIDENCE stays a keyword."""

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_section_and_retry_prompts_ask_for_serbian(self, mock_chat, adapter, model, appliance_language):
        appliance_language("sr")
        mock_chat.side_effect = [_reply("Kesa je zapečaćena."), _reply("Kesa je zapečaćena [Doc 1].")]

        adapter.write_section(model, "tema", OutlineSection(key="s1", title="Rukovanje", order_index=0, brief="b"),
                              "evidence", [1], "outline")

        first, retry = _messages(mock_chat, 0), _messages(mock_chat, 1)
        assert "in Serbian, using the Latin script" in first[0]["content"]
        assert "such as NO_EVIDENCE" in first[0]["content"]
        assert "in Serbian" in retry[-1]["content"]

    @patch("app.features.deep_research_report.data.report_llm_adapter.chat_completion")
    def test_english_plan_prompt_is_unchanged(self, mock_chat, adapter, model):
        mock_chat.return_value = _reply("{}")

        with pytest.raises(JSONPlanParseError):
            adapter.scope(model, "topic", _corpus())

        assert "<OUTPUT_LANGUAGE>" not in _messages(mock_chat)[0]["content"]
