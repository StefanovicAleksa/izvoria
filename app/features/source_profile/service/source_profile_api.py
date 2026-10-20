# Path: app/features/source_profile/service/source_profile_api.py

import logging
import uuid
from typing import List, Optional

from app.core.language.service.language_api import active_language
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions import ModelManagerError
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.source_profile.domain.entities import (
    ProfileDraft, ProfiledSource, SourceIdentity, SourceProfile, SourceScope
)
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.domain.interfaces import IProfileLanguageAdapter, ISourceProfileRepository
from app.features.source_profile.exceptions import SourceProfileDataError, SourceProfileError
from app.features.source_profile.utils.error_mapper import map_source_profile_error
from app.features.source_profile.utils.grounding import dropped_numbers, invented_numbers
from app.features.source_profile.utils.profile_text import clipped_draft, first_sentence, title_from_name
from app.features.source_profile.utils.scope_matcher import shortlist

logger = logging.getLogger(__name__)

# A long source is profiled from notes on at most this many of its sections, spread over it (the first always
# included, where a title is usually stated): each note re-reads a whole section, so a book would otherwise pay
# one long prefill per section on top of contextualising it
MAX_NOTED_SECTIONS = 8


def _spread(sections: List[str], limit: int) -> List[str]:
    if len(sections) <= limit:
        return sections
    step = (len(sections) - 1) / (limit - 1)
    return [sections[round(i * step)] for i in range(limit)]


class SourceProfileAPI:
    """
    Every source's title and topic: written at ingestion by the model that contextualises it, editable by a person,
    shown with each excerpt and indexed beside it, and used to tell which source a question names.
    """

    def __init__(
            self,
            repository: ISourceProfileRepository,
            language_adapter: IProfileLanguageAdapter,
            orchestrator: Optional[ModelManagerOrchestrator] = None,
            judge_manifest: Optional[ModelManifest] = None,
            judge_loader: Optional[IModelLoader] = None
    ):
        """
        `judge_manifest`/`judge_loader` are the answer model's, borrowed as the follow-up rewrite borrows it: it is
        usually warm when a question arrives. Without them `resolve_scope` never scopes a question.
        """
        if (judge_manifest is None) != (judge_loader is None):
            raise ValueError("judge_manifest and judge_loader are given together or not at all.")
        self._repo = repository
        self._adapter = language_adapter
        self._orchestrator = orchestrator
        self._judge_manifest = judge_manifest
        self._judge_loader = judge_loader

    # ---------------------------------------------------------------- writing

    def generate(self, source_id: uuid.UUID, sections: List[str], model: LoadedModel) -> Optional[SourceProfile]:
        """
        Writes a source's profile with a model the caller holds: from its whole text when `sections` is one, else
        from one note per section. A profile a person edited is kept. A model reply that can't be used, or a title
        stating a number the source doesn't, falls back to the file name rather than failing ingestion; a
        model-manager failure is raised.
        """
        try:
            existing = self._repo.get(source_id)
            if existing is not None and existing.origin is ProfileOrigin.EDITED:
                return existing
            identity = self._repo.get_identity(source_id)
            if identity is None:
                logger.warning(f"No source {source_id} to profile.")
                return None

            texts = [text for text in sections if text and text.strip()]
            origin = ProfileOrigin.GENERATED
            try:
                if not texts:
                    raise SourceProfileDataError("The source has no text to read.")
                if len(texts) == 1:
                    draft = self._adapter.write_profile(model, identity, texts[0])
                else:
                    notes = [self._adapter.note_section(model, text) for text in _spread(texts, MAX_NOTED_SECTIONS)]
                    material = "\n".join(f"{number}. {note}" for number, note in enumerate(notes, start=1) if note)
                    draft = self._adapter.write_profile(model, identity, material, from_notes=True)
            except ModelManagerError:
                raise
            except SourceProfileError as e:
                logger.warning(f"Profiling source {source_id} from its text failed; titling it by its name. {e}")
                draft, origin = self._named(identity), ProfileOrigin.NAMED

            names = [identity.name, identity.file_name]
            invented, dropped = invented_numbers(draft.title, names, texts), dropped_numbers(draft.title, names)
            if invented or dropped:
                # A wrong or missing number would send questions about this source elsewhere: keep the topic, and
                # title it by its name, whose numbers are right by definition
                logger.warning(f"Generated title '{draft.title}' for source {source_id} invents {invented or 'no'} "
                               f"and drops {dropped or 'no'} numbers; titling it by its name instead.")
                draft = clipped_draft(self._named(identity).title, draft.topic)

            return self._save(identity, draft, origin)
        except Exception as e:
            raise map_source_profile_error(e, context=f"SourceProfileAPI.generate for source {source_id}") from e

    def name_only(self, source_id: uuid.UUID, description: str = "") -> Optional[SourceProfile]:
        """A profile without a model (a standalone image): titled by its file name, its topic the description's
        first sentence."""
        try:
            existing = self._repo.get(source_id)
            if existing is not None and existing.origin is ProfileOrigin.EDITED:
                return existing
            identity = self._repo.get_identity(source_id)
            if identity is None:
                return None
            draft = clipped_draft(title_from_name(identity.name), first_sentence(description) if description else "")
            return self._save(identity, draft, ProfileOrigin.NAMED)
        except Exception as e:
            raise map_source_profile_error(e, context=f"SourceProfileAPI.name_only for source {source_id}") from e

    def edit(self, source_id: uuid.UUID, title: str, topic: str = "") -> SourceProfile:
        """A person's own title and topic. Nothing generated replaces it afterwards."""
        try:
            identity = self._repo.get_identity(source_id)
            if identity is None:
                raise SourceProfileDataError(f"Source {source_id} does not exist.")
            if not (title or "").strip():
                raise SourceProfileDataError("A source's title cannot be empty.")
            return self._save(identity, clipped_draft(title, topic or ""), ProfileOrigin.EDITED)
        except Exception as e:
            raise map_source_profile_error(e, context=f"SourceProfileAPI.edit for source {source_id}") from e

    @staticmethod
    def _named(identity: SourceIdentity) -> ProfileDraft:
        return clipped_draft(title_from_name(identity.name or identity.file_name), "")

    def _save(self, identity: SourceIdentity, draft: ProfileDraft, origin: ProfileOrigin) -> SourceProfile:
        profile = SourceProfile(
            source_id=identity.source_id, title=draft.title, topic=draft.topic, origin=origin,
            language=identity.language or active_language()
        )
        logger.info(f"Source {identity.source_id} profiled ({origin.value}): '{profile.title}' — {profile.topic}")
        return self._repo.save(profile)

    # ---------------------------------------------------------------- reading

    def get(self, source_id: uuid.UUID) -> Optional[SourceProfile]:
        try:
            return self._repo.get(source_id)
        except Exception as e:
            raise map_source_profile_error(e, context=f"SourceProfileAPI.get for source {source_id}") from e

    def list_indexed(self) -> List[ProfiledSource]:
        try:
            return self._repo.list_indexed()
        except Exception as e:
            raise map_source_profile_error(e, context="SourceProfileAPI.list_indexed") from e

    def list_unprofiled(self) -> List[SourceIdentity]:
        try:
            return self._repo.list_unprofiled()
        except Exception as e:
            raise map_source_profile_error(e, context="SourceProfileAPI.list_unprofiled") from e

    # ---------------------------------------------------------------- scope

    def can_resolve_scope(self) -> bool:
        return self._orchestrator is not None and self._judge_manifest is not None

    def resolve_scope(self, question: str) -> Optional[SourceScope]:
        """
        The one source `question` explicitly names ("in lecture 4 of Stat 110"), or None. Sources are shortlisted
        by the words the question shares with their titles and names, without a model; only when some are found
        does the answer model decide which, if any, the question means. It waits for the GPU like every live call,
        so callers run it before any preload of their own.
        """
        if not self.can_resolve_scope():
            return None
        try:
            candidates = shortlist(question, self._repo.list_indexed())
            if not candidates:
                return None

            loaded_model = self._orchestrator.acquire_model(
                self._judge_manifest, self._judge_loader, wait_for_clear=True
            )
            try:
                choice = self._adapter.pick_scope(loaded_model, question, candidates)
            finally:
                self._orchestrator.release_model(loaded_model, self._judge_loader, flush=True)

            if choice is None:
                return None
            picked = candidates[choice]
            logger.info(f"Question scoped to source {picked.source_id} ('{picked.display_title}').")
            return SourceScope(source_id=picked.source_id, title=picked.display_title)
        except Exception as e:
            raise map_source_profile_error(e, context=f"SourceProfileAPI.resolve_scope for '{question}'") from e
