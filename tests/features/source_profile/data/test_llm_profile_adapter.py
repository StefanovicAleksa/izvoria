# Path: tests/features/source_profile/data/test_llm_profile_adapter.py

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.core.model_manager.domain.entities import LoadedModel
from app.features.source_profile.data.llm_profile_adapter import LlmProfileAdapter
from app.features.source_profile.domain.entities import (
    MAX_TITLE_CHARS, ProfileDraft, ProfiledSource, SourceIdentity, SourceProfile
)
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.exceptions import ProfileGenerationError, ProfileParseError

IDENTITY = SourceIdentity(source_id=uuid.uuid4(), name="module_01.mp4", file_name="module_01.mp4")
MODULE = "app.features.source_profile.data.llm_profile_adapter"


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def adapter():
    return LlmProfileAdapter()


@pytest.fixture
def model():
    loaded = MagicMock(spec=LoadedModel)
    loaded.model_instance = MagicMock()
    return loaded


class TestWriteProfile:
    def test_parses_the_json_reply(self, adapter, model):
        model.model_instance.create_chat_completion.return_value = _reply(
            '{"title": "Contract Law, Module 1", "topic": "Offer and acceptance."}')

        draft = adapter.write_profile(model, IDENTITY, "the transcript")

        assert draft == ProfileDraft(title="Contract Law, Module 1", topic="Offer and acceptance.")
        kwargs = model.model_instance.create_chat_completion.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.0
        user = kwargs["messages"][1]["content"]
        assert "module_01.mp4" in user and "the transcript" in user

    def test_an_escaped_apostrophe_is_read(self, adapter, model):
        model.model_instance.create_chat_completion.return_value = _reply(
            r'{"title": "Lecture 7 Gambler\'s Ruin", "topic": "The gambler\'s ruin."}')

        draft = adapter.write_profile(model, IDENTITY, "t")

        assert (draft.title, draft.topic) == ("Lecture 7 Gambler's Ruin", "The gambler's ruin.")

    def test_section_notes_use_the_notes_template(self, adapter, model):
        model.model_instance.create_chat_completion.return_value = _reply('{"title": "T", "topic": ""}')

        adapter.write_profile(model, IDENTITY, "1. note", from_notes=True)

        user = model.model_instance.create_chat_completion.call_args.kwargs["messages"][1]["content"]
        assert "<section_notes>" in user and "1. note" in user

    def test_an_overlong_title_is_cut_not_rejected(self, adapter, model):
        model.model_instance.create_chat_completion.return_value = _reply(
            '{"title": "%s", "topic": "x"}' % ("word " * 80))

        assert len(adapter.write_profile(model, IDENTITY, "t").title) <= MAX_TITLE_CHARS

    @pytest.mark.parametrize("content", ["not json at all", '{"topic": "no title"}', '["a list"]', '{"title": ""}'])
    def test_an_unusable_reply_raises_a_parse_error(self, adapter, model, content):
        model.model_instance.create_chat_completion.return_value = _reply(content)

        with pytest.raises(ProfileParseError):
            adapter.write_profile(model, IDENTITY, "t")

    def test_an_unloaded_model_raises(self, adapter):
        with pytest.raises(ProfileGenerationError):
            adapter.write_profile(MagicMock(spec=LoadedModel, model_instance=None), IDENTITY, "t")


class TestNoteSection:
    def test_returns_the_note_on_one_line(self, adapter, model):
        model.model_instance.create_chat_completion.return_value = _reply("Covers Bayes'\nrule. Lecture 4.")

        assert adapter.note_section(model, "section") == "Covers Bayes' rule. Lecture 4."
        assert "response_format" not in model.model_instance.create_chat_completion.call_args.kwargs


class TestPickScope:
    @pytest.fixture
    def candidates(self):
        def profiled(title):
            source_id = uuid.uuid4()
            return ProfiledSource(
                identity=SourceIdentity(source_id=source_id, name=f"{title}.mp4", file_name=f"{title}.mp4"),
                profile=SourceProfile(source_id=source_id, title=title, topic="A topic.",
                                      origin=ProfileOrigin.GENERATED))
        return [profiled("Lecture 4"), profiled("Lecture 14")]

    @pytest.mark.parametrize("content, expected", [
        ('{"source": 2}', 1),
        ('{"source": "1"}', 0),
        ('{"source": null}', None),
        ('{"source": 3}', None),
        ('{"source": 0}', None),
        ('{"source": true}', None),
        ('<think></think>{"source": 1}', 0),
    ])
    def test_maps_the_reply_to_a_candidate_index(self, adapter, model, candidates, content, expected):
        with patch(f"{MODULE}.chat_completion", return_value=_reply(content)) as chat:
            assert adapter.pick_scope(model, "lecture 4?", candidates) == expected
        assert chat.call_args.kwargs["enable_thinking"] is False

    def test_lists_each_candidate_numbered_with_its_file_and_topic(self, adapter, model, candidates):
        with patch(f"{MODULE}.chat_completion", return_value=_reply('{"source": null}')) as chat:
            adapter.pick_scope(model, "lecture 4?", candidates)

        user = chat.call_args.args[1][1]["content"]
        assert "1. Lecture 4 (file: Lecture 4.mp4) — A topic." in user
        assert "2. Lecture 14 (file: Lecture 14.mp4) — A topic." in user
        assert "lecture 4?" in user
