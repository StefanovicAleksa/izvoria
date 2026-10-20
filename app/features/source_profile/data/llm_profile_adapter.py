# Path: app/features/source_profile/data/llm_profile_adapter.py

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.core.language.service.language_api import get_language_pack
from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IChatModel
from app.core.model_manager.utils.llama_cpp_chat import chat_completion, strip_reasoning
from app.features.source_profile.data.prompts import (
    PROFILE_FROM_NOTES_USER_TEMPLATE, PROFILE_SYSTEM_PROMPT, PROFILE_USER_TEMPLATE,
    SCOPE_SYSTEM_PROMPT, SCOPE_USER_TEMPLATE, SECTION_NOTE_SYSTEM_PROMPT
)
from app.features.source_profile.domain.entities import ProfileDraft, ProfiledSource, SourceIdentity
from app.features.source_profile.domain.interfaces import IProfileLanguageAdapter
from app.features.source_profile.exceptions import ProfileGenerationError, ProfileParseError
from app.features.source_profile.utils.error_mapper import map_source_profile_error
from app.features.source_profile.utils.profile_text import clipped_draft

logger = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*}", re.DOTALL)
# Small models escape an apostrophe ("Gambler\'s Ruin"), which JSON does not allow
_ESCAPED_APOSTROPHE = re.compile(r"(?<!\\)\\'")


def _parse_json(reply: str) -> Dict[str, Any]:
    text = strip_reasoning(reply or "").strip()
    match = _JSON_OBJECT.search(text)
    candidate = match.group(0) if match else text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        try:
            parsed = json.loads(_ESCAPED_APOSTROPHE.sub("'", candidate))
        except json.JSONDecodeError:
            raise ProfileParseError(f"Not JSON: {text[:200]!r}", original_exception=e) from e
    if not isinstance(parsed, dict):
        raise ProfileParseError(f"Expected a JSON object, got: {text[:200]!r}")
    return parsed


class LlmProfileAdapter(IProfileLanguageAdapter):
    """
    Runs this feature's prompts on a llama.cpp model (or a llama-server client) that another feature loaded and
    holds. It never loads, keeps or releases a model: the caller's acquisition is the only one.
    """

    PROFILE_MAX_TOKENS = 200
    NOTE_MAX_TOKENS = 120
    SCOPE_MAX_TOKENS = 20

    @staticmethod
    def _complete(model: LoadedModel, system_prompt: str, user_content: str, max_tokens: int,
                  json_reply: bool, thinking_template: bool = False) -> str:
        """
        `thinking_template` is for the answer model (Qwen3.5), whose chat template must be told not to think: a
        20-token reply would otherwise be spent inside <think>. The contextual SLM's template has no such switch
        and runs through llama.cpp's own chat completion, as `SLMContextAdapter` runs it.
        """
        llm: Any = getattr(model, "model_instance", None)
        if llm is None:
            raise ProfileGenerationError("Cannot run a profile prompt: the model is not loaded.")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        kwargs: Dict[str, Any] = {"temperature": 0.0, "max_tokens": max_tokens}
        if json_reply:
            kwargs["response_format"] = {"type": "json_object"}
        if thinking_template or isinstance(llm, IChatModel):
            response = chat_completion(llm, messages, enable_thinking=False, **kwargs)
        else:
            response = llm.create_chat_completion(messages=messages, **kwargs)
        return response["choices"][0]["message"]["content"] or ""

    def write_profile(self, model: LoadedModel, identity: SourceIdentity, material: str,
                      from_notes: bool = False) -> ProfileDraft:
        try:
            template = PROFILE_FROM_NOTES_USER_TEMPLATE if from_notes else PROFILE_USER_TEMPLATE
            reply = self._complete(
                model,
                get_language_pack().localize_prompt(PROFILE_SYSTEM_PROMPT),
                template.format(file_name=identity.file_name or identity.name, material=material),
                self.PROFILE_MAX_TOKENS,
                json_reply=True
            )
            parsed = _parse_json(reply)
            title = parsed.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ProfileParseError(f"The profile reply has no title: {reply[:200]!r}")
            topic = parsed.get("topic")
            return clipped_draft(title, topic if isinstance(topic, str) else "")
        except Exception as e:
            raise map_source_profile_error(e, context=f"Writing the profile of source {identity.source_id}") from e

    def note_section(self, model: LoadedModel, section_text: str) -> str:
        try:
            reply = self._complete(model, SECTION_NOTE_SYSTEM_PROMPT, section_text, self.NOTE_MAX_TOKENS,
                                   json_reply=False)
            return " ".join(strip_reasoning(reply).split())
        except Exception as e:
            raise map_source_profile_error(e, context="Noting one section of a long source") from e

    def pick_scope(self, model: LoadedModel, question: str, candidates: List[ProfiledSource]) -> Optional[int]:
        try:
            listing = "\n".join(
                f"{number}. {_describe(candidate)}" for number, candidate in enumerate(candidates, start=1)
            )
            reply = self._complete(
                model,
                SCOPE_SYSTEM_PROMPT,
                SCOPE_USER_TEMPLATE.format(sources=listing, question=question),
                self.SCOPE_MAX_TOKENS,
                json_reply=True,
                thinking_template=True
            )
            choice = _parse_json(reply).get("source")
            if isinstance(choice, str) and choice.strip().isdigit():
                choice = int(choice.strip())
            if isinstance(choice, bool) or not isinstance(choice, int) or not 1 <= choice <= len(candidates):
                return None
            return choice - 1
        except Exception as e:
            raise map_source_profile_error(e, context=f"Picking the source named in '{question}'") from e


def _describe(candidate: ProfiledSource) -> str:
    """One line per candidate: its title, its file name when that differs, and its topic."""
    parts = [candidate.display_title]
    file_name = candidate.identity.file_name or candidate.identity.name
    if file_name and file_name != candidate.display_title:
        parts.append(f"(file: {file_name})")
    if candidate.profile is not None and candidate.profile.topic:
        parts.append(f"— {candidate.profile.topic}")
    return " ".join(parts)
