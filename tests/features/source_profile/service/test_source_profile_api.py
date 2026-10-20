# Path: tests/features/source_profile/service/test_source_profile_api.py

import uuid
from unittest.mock import MagicMock

import pytest

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.exceptions import ModelManagerError
from app.features.source_profile.domain.entities import (
    ProfileDraft, ProfiledSource, SourceIdentity, SourceProfile, SourceScope
)
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.exceptions import ProfileParseError, SourceProfileDataError
from app.features.source_profile.service.source_profile_api import MAX_NOTED_SECTIONS, SourceProfileAPI

SOURCE_ID = uuid.uuid4()
IDENTITY = SourceIdentity(source_id=SOURCE_ID, name="module_01.mp4", file_name="module_01.mp4", language="en")
JUDGE = ModelManifest(model_type=ModelType.QA_GENERATION, model_path="models/qa.gguf", required_vram_mb=1.0)


@pytest.fixture
def repo():
    repository = MagicMock()
    repository.get.return_value = None
    repository.get_identity.return_value = IDENTITY
    repository.save.side_effect = lambda profile: profile
    return repository


@pytest.fixture
def adapter():
    return MagicMock()


@pytest.fixture
def orchestrator():
    return MagicMock()


@pytest.fixture
def judge_loader():
    return MagicMock()


@pytest.fixture
def api(repo, adapter, orchestrator, judge_loader):
    return SourceProfileAPI(repo, adapter, orchestrator=orchestrator, judge_manifest=JUDGE, judge_loader=judge_loader)


@pytest.fixture
def model():
    return MagicMock(spec=LoadedModel)


def _profiled(name, title=None):
    source_id = uuid.uuid4()
    profile = SourceProfile(source_id=source_id, title=title, origin=ProfileOrigin.GENERATED) if title else None
    return ProfiledSource(identity=SourceIdentity(source_id=source_id, name=name), profile=profile)


class TestSourceProfileAPIConstruction:
    def test_judge_manifest_and_loader_come_together(self, repo, adapter):
        with pytest.raises(ValueError):
            SourceProfileAPI(repo, adapter, judge_manifest=JUDGE)


class TestGenerate:
    def test_one_section_is_profiled_from_its_whole_text(self, api, repo, adapter, model):
        adapter.write_profile.return_value = ProfileDraft(title="Contract Law, Module 1", topic="Offer and acceptance.")

        profile = api.generate(SOURCE_ID, ["the whole transcript"], model)

        adapter.write_profile.assert_called_once_with(model, IDENTITY, "the whole transcript")
        adapter.note_section.assert_not_called()
        assert profile == SourceProfile(source_id=SOURCE_ID, title="Contract Law, Module 1",
                                        topic="Offer and acceptance.", origin=ProfileOrigin.GENERATED, language="en")
        repo.save.assert_called_once_with(profile)

    def test_a_long_source_is_profiled_from_section_notes(self, api, adapter, model):
        adapter.note_section.side_effect = ["Note one.", "Note two."]
        adapter.write_profile.return_value = ProfileDraft(title="T")

        api.generate(SOURCE_ID, ["first part", "second part"], model)

        assert [c.args for c in adapter.note_section.call_args_list] == [(model, "first part"), (model, "second part")]
        adapter.write_profile.assert_called_once_with(model, IDENTITY, "1. Note one.\n2. Note two.", from_notes=True)

    def test_a_very_long_source_notes_a_spread_of_sections_including_the_first(self, api, adapter, model):
        adapter.note_section.return_value = "note"
        adapter.write_profile.return_value = ProfileDraft(title="T")
        sections = [f"part {i}" for i in range(30)]

        api.generate(SOURCE_ID, sections, model)

        noted = [c.args[1] for c in adapter.note_section.call_args_list]
        assert len(noted) == MAX_NOTED_SECTIONS
        assert noted[0] == "part 0" and noted[-1] == "part 29"

    def test_an_unusable_reply_falls_back_to_the_file_name(self, api, adapter, model):
        adapter.write_profile.side_effect = ProfileParseError("not JSON")

        profile = api.generate(SOURCE_ID, ["text"], model)

        assert (profile.title, profile.topic, profile.origin) == ("module 01", "", ProfileOrigin.NAMED)

    def test_a_source_without_text_is_named_by_its_file(self, api, adapter, model):
        profile = api.generate(SOURCE_ID, ["", "  "], model)

        adapter.write_profile.assert_not_called()
        assert profile.origin is ProfileOrigin.NAMED

    def test_model_manager_errors_are_raised(self, api, adapter, model):
        adapter.write_profile.side_effect = ModelManagerError("VRAM")

        with pytest.raises(ModelManagerError):
            api.generate(SOURCE_ID, ["text"], model)

    def test_an_edited_profile_is_never_replaced(self, api, repo, adapter, model):
        edited = SourceProfile(source_id=SOURCE_ID, title="Mine", origin=ProfileOrigin.EDITED)
        repo.get.return_value = edited

        assert api.generate(SOURCE_ID, ["text"], model) is edited
        adapter.write_profile.assert_not_called()
        repo.save.assert_not_called()

    def test_a_missing_source_is_not_profiled(self, api, repo, model):
        repo.get_identity.return_value = None

        assert api.generate(SOURCE_ID, ["text"], model) is None
        repo.save.assert_not_called()


class TestNameOnly:
    def test_an_image_is_titled_by_its_file_with_the_description_as_topic(self, api, repo):
        repo.get_identity.return_value = IDENTITY.model_copy(update={"name": "site_photo_04.jpg"})

        profile = api.name_only(SOURCE_ID, "A cracked wall. Two windows.")

        assert (profile.title, profile.topic, profile.origin) == ("site photo 04", "A cracked wall.",
                                                                  ProfileOrigin.NAMED)


class TestEdit:
    def test_an_edit_is_saved_as_edited(self, api, repo):
        profile = api.edit(SOURCE_ID, "  Contract Law 1 ", "Offer and acceptance")

        assert (profile.title, profile.topic, profile.origin) == ("Contract Law 1", "Offer and acceptance",
                                                                  ProfileOrigin.EDITED)
        repo.save.assert_called_once_with(profile)

    def test_an_empty_title_is_refused(self, api, repo):
        with pytest.raises(SourceProfileDataError):
            api.edit(SOURCE_ID, "   ")
        repo.save.assert_not_called()

    def test_an_unknown_source_is_refused(self, api, repo):
        repo.get_identity.return_value = None
        with pytest.raises(SourceProfileDataError):
            api.edit(SOURCE_ID, "Title")


class TestResolveScope:
    def test_a_question_naming_nothing_costs_no_model(self, api, repo, orchestrator):
        repo.list_indexed.return_value = [_profiled("Lecture 4 Statistics 110.mp4")]

        assert api.resolve_scope("what is conditional probability") is None
        orchestrator.acquire_model.assert_not_called()

    def test_the_answer_model_picks_among_the_shortlist_and_is_released(self, api, repo, adapter, orchestrator,
                                                                        judge_loader):
        lecture4 = _profiled("Lecture 4 Statistics 110.mp4", title="Statistics 110, Lecture 4")
        lecture5 = _profiled("Lecture 5 Statistics 110.mp4", title="Statistics 110, Lecture 5")
        repo.list_indexed.return_value = [lecture4, lecture5]
        loaded = orchestrator.acquire_model.return_value
        adapter.pick_scope.return_value = 0

        scope = api.resolve_scope("all formulas in lecture 4 of stat 110")

        assert scope == SourceScope(source_id=lecture4.source_id, title="Statistics 110, Lecture 4")
        orchestrator.acquire_model.assert_called_once_with(JUDGE, judge_loader, wait_for_clear=True)
        assert adapter.pick_scope.call_args.args[2][0] is lecture4
        orchestrator.release_model.assert_called_once_with(loaded, judge_loader, flush=True)

    def test_the_model_saying_none_scopes_nothing(self, api, repo, adapter, orchestrator):
        repo.list_indexed.return_value = [_profiled("Lecture 4 Statistics 110.mp4")]
        adapter.pick_scope.return_value = None

        assert api.resolve_scope("lecture 4") is None
        orchestrator.release_model.assert_called_once()

    def test_the_model_is_released_when_picking_fails(self, api, repo, adapter, orchestrator):
        repo.list_indexed.return_value = [_profiled("Lecture 4 Statistics 110.mp4")]
        adapter.pick_scope.side_effect = ProfileParseError("bad")

        with pytest.raises(ProfileParseError):
            api.resolve_scope("lecture 4")
        orchestrator.release_model.assert_called_once()

    def test_without_a_judge_nothing_is_scoped(self, repo, adapter):
        api = SourceProfileAPI(repo, adapter)
        assert api.can_resolve_scope() is False
        assert api.resolve_scope("lecture 4") is None
        repo.list_indexed.assert_not_called()


class TestGeneratedTitleGrounding:
    def test_a_title_with_an_invented_number_falls_back_to_the_name_and_keeps_the_topic(self, api, repo, adapter,
                                                                                        model):
        repo.get_identity.return_value = IDENTITY.model_copy(update={
            "name": "Lecture 20 Multinomial Statistics 110 [xiVWNkQUqKk].mp4",
            "file_name": "Lecture 20 Multinomial Statistics 110 [xiVWNkQUqKk].mp4"})
        adapter.write_profile.return_value = ProfileDraft(title="Statistics 110, Lecture 4", topic="Multinomials.")

        profile = api.generate(SOURCE_ID, ["transcript"], model)

        assert (profile.title, profile.topic) == ("Lecture 20 Multinomial Statistics 110", "Multinomials.")
        assert profile.origin is ProfileOrigin.GENERATED

    def test_a_title_dropping_the_lecture_number_falls_back_to_the_name(self, api, repo, adapter, model):
        name = "Lecture 16 Exponential Distribution Statistics 110 [bM6nFDjvEns].mp4"
        repo.get_identity.return_value = IDENTITY.model_copy(update={"name": name, "file_name": name})
        adapter.write_profile.return_value = ProfileDraft(title="Exponential Distribution Statistics 110", topic="t")

        assert api.generate(SOURCE_ID, ["transcript"], model).title == \
            "Lecture 16 Exponential Distribution Statistics 110"
