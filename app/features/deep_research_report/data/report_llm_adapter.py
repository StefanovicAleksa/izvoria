# Path: app/features/deep_research_report/data/report_llm_adapter.py

import gc
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.utils.llama_cpp_chat import build_llama, chat_completion, strip_reasoning
from app.features.deep_research_report.data.prompts import (
    NO_EVIDENCE_SECTION, OUTLINE_SYSTEM_PROMPT, OUTLINE_USER_TEMPLATE, SCOPE_SYSTEM_PROMPT,
    SCOPE_USER_TEMPLATE, SECTION_CITATION_RETRY_PROMPT, SECTION_SYSTEM_PROMPT, SECTION_USER_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE,
)
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, FollowUpQuery, OutlineAssignment, OutlineSection, ResearchPlan, SubQuestion,
)
from app.features.deep_research_report.domain.interfaces import IReportLlmAdapter
from app.features.deep_research_report.exceptions.pipeline import JSONPlanParseError
from app.features.deep_research_report.utils.error_mapper import map_report_error
from app.features.deep_research_report.utils.evidence_bank import format_timestamp

logger = logging.getLogger(__name__)


class LlamaCppReportAdapter(IReportLlmAdapter):
    """
    Qwen3.5-9B via llama.cpp, serving every report role: scoping, outline optimisation, section writing
    and the executive summary.

    ONE adapter and ONE ModelType on purpose. Residency is keyed on (model_type, model_path), so giving
    the planner its own ModelType would reload the same ~5.7 GB file on every alternation between planning
    and writing -- the cost that already makes QA_GENERATION <-> QA_REASONER switching expensive. With them
    together, a whole report run holds a single acquisition across outline optimisation and every section.
    Splitting this class in two would silently multiply a run's model-loading time by roughly eight.

    Section generation does NOT reuse QA_GENERATION_MAX_ANSWER_TOKENS: a report section is several times
    longer than a chat answer, and 1024 tokens truncates it mid-paragraph.
    """

    TEMPLATE_MARGIN_TOKENS = 64

    # ==========================================
    # MODEL LIFECYCLE
    # ==========================================

    @staticmethod
    def _get_llama_model(model_path: str) -> Any:
        return build_llama(model_path, settings.REPORT_AUTHORING_N_CTX, settings.QA_LLM_KV_CACHE_TYPE)

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            absolute_path = str(Path(manifest.model_path).resolve())
            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"Report model file missing: {absolute_path}")
            llm = self._get_llama_model(absolute_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_report_error(e, context=f"Load Report Adapter: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance") and model.model_instance is not None:
                if hasattr(model.model_instance, "close"):
                    model.model_instance.close()
                del model.model_instance
                model.model_instance = None
            gc.collect()
            return True
        except Exception as e:
            raise map_report_error(e, context="Unload Report Adapter") from e

    def count_tokens(self, model: LoadedModel, text: str) -> int:
        try:
            llm: Any = model.model_instance
            return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        except Exception as e:
            raise map_report_error(e, context="Counting tokens") from e

    def get_section_token_budget(self, model: LoadedModel, section: OutlineSection, topic: str) -> int:
        """
        What is left for a section's excerpts, then capped at REPORT_SECTION_EVIDENCE_TOKENS.

        The cap is the point. The window is 128K, but prefill is what costs wall-clock time on a 3060, and
        a section allowed to fill the window would make an eight-section report take hours.
        """
        try:
            llm: Any = model.model_instance
            fixed = self.count_tokens(model, SECTION_SYSTEM_PROMPT + self._section_user_content(
                topic, section, evidence_block="", allowed_doc_numbers=[], outline_context=""
            ))
            available = llm.n_ctx() - fixed - settings.REPORT_SECTION_MAX_TOKENS - self.TEMPLATE_MARGIN_TOKENS
            return max(min(available, settings.REPORT_SECTION_EVIDENCE_TOKENS), 0)
        except Exception as e:
            raise map_report_error(e, context="Computing the section token budget") from e

    # ==========================================
    # 1. SCOPING
    # ==========================================

    def scope(self, model: LoadedModel, topic: str, corpus: CorpusDigest) -> ResearchPlan:
        try:
            messages = [
                {"role": "system", "content": SCOPE_SYSTEM_PROMPT.format(
                    max_sections=settings.REPORT_MAX_SECTIONS,
                    max_sub_questions=settings.REPORT_MAX_SUB_QUESTIONS,
                )},
                {"role": "user", "content": SCOPE_USER_TEMPLATE.format(
                    topic=topic,
                    corpus_overview=self._corpus_overview(corpus),
                    sample_digests=self._sample_block(corpus),
                )},
            ]
            parsed = self._json_completion(model, messages, settings.REPORT_PLAN_MAX_TOKENS)
            return self._plan_from_json(parsed, topic)

        except Exception as e:
            raise map_report_error(e, context=f"Scoping a research plan for '{topic}'") from e

    @staticmethod
    def _corpus_overview(corpus: CorpusDigest) -> str:
        if not corpus.sources:
            return "(The archive has no indexed recordings.)"
        lines = [
            f"{corpus.indexed_source_count} indexed recording(s), {corpus.indexed_chunk_count} excerpt(s) total."
        ]
        for source in corpus.sources:
            duration = format_timestamp(source.last_end_sec - source.first_start_sec)
            lines.append(f"- {source.name} ({source.chunk_count} excerpts, about {duration} long)")
        return "\n".join(lines)

    @staticmethod
    def _sample_block(corpus: CorpusDigest) -> str:
        if not corpus.sample_digests:
            return "(A first search of this topic returned nothing.)"
        return "\n".join(f"- {digest}" for digest in corpus.sample_digests)

    @staticmethod
    def _plan_from_json(parsed: dict, topic: str) -> ResearchPlan:
        raw_sections = parsed.get("sections")
        if not isinstance(raw_sections, list) or not raw_sections:
            raise JSONPlanParseError(
                "The planner returned no sections. There is no safe default outline to fall back to: "
                "inventing one would put the user's approval on a plan the corpus was never consulted about."
            )

        report_id = uuid.uuid4()
        sections: List[OutlineSection] = []
        seen_keys = set()

        for order_index, raw in enumerate(raw_sections[:settings.REPORT_MAX_SECTIONS]):
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            key = str(raw.get("key") or "").strip() or f"s{order_index + 1}"
            if key in seen_keys:
                key = f"s{order_index + 1}"
            seen_keys.add(key)
            sections.append(OutlineSection(
                key=key,
                title=title,
                order_index=order_index,
                brief=str(raw.get("brief") or "").strip(),
            ))

        if not sections:
            raise JSONPlanParseError("The planner returned sections, but none of them had a usable title.")

        valid_keys = {s.key for s in sections}
        sub_questions: List[SubQuestion] = []
        for order_index, raw in enumerate(parsed.get("sub_questions") or []):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            section_key = str(raw.get("section_key") or "").strip()
            # A sub-question aimed at a section the planner did not actually emit would never be written
            # into anything; attach it to the first section rather than dropping the search entirely.
            if not text:
                continue
            if section_key not in valid_keys:
                section_key = sections[0].key
            sub_questions.append(SubQuestion(text=text, section_key=section_key, order_index=order_index))
            if len(sub_questions) >= settings.REPORT_MAX_SUB_QUESTIONS:
                break

        linked = [
            section.model_copy(update={
                "sub_question_ids": [q.id for q in sub_questions if q.section_key == section.key]
            })
            for section in sections
        ]

        return ResearchPlan(
            report_id=report_id,
            topic=topic,
            scope_note=str(parsed.get("scope_note") or "").strip(),
            sections=linked,
            sub_questions=sub_questions,
            revision=1,
        )

    # ==========================================
    # 2. OUTLINE OPTIMISATION
    # ==========================================

    def optimize_outline(
            self,
            model: LoadedModel,
            plan: ResearchPlan,
            evidence_digest: str,
    ) -> OutlineAssignment:
        try:
            outline_block = "\n".join(
                f"- [{s.key}] {s.title}: {s.brief or '(no brief)'}" for s in plan.sections
            )
            messages = [
                {"role": "system", "content": OUTLINE_SYSTEM_PROMPT},
                {"role": "user", "content": OUTLINE_USER_TEMPLATE.format(
                    topic=plan.topic, outline_block=outline_block, bank_index=evidence_digest
                )},
            ]
            parsed = self._json_completion(model, messages, settings.REPORT_PLAN_MAX_TOKENS)
            return self._assignment_from_json(parsed, plan)

        except Exception as e:
            raise map_report_error(e, context=f"Optimising the outline of report {plan.report_id}") from e

    @staticmethod
    def _assignment_from_json(parsed: dict, plan: ResearchPlan) -> OutlineAssignment:
        """
        Applies the model's assignments to the approved outline.

        Only keys the outline already has are honoured, and a section the model omitted keeps its place
        with no evidence rather than vanishing: dropping a section the user approved without telling them
        would quietly change the report they signed off on. It is reported as uncovered instead.
        """
        by_key = {s.key: s for s in plan.sections}
        assigned: Dict[str, List[int]] = {}
        order: List[str] = []

        for raw in parsed.get("sections") or []:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key") or "").strip()
            if key not in by_key or key in assigned:
                continue
            numbers = [
                int(n) for n in (raw.get("doc_numbers") or [])
                if isinstance(n, (int, str)) and str(n).strip().lstrip("-").isdigit()
            ]
            assigned[key] = sorted({n for n in numbers if n > 0})
            order.append(key)

        # Sections the model never mentioned keep their original relative order, after the ones it did.
        order += [s.key for s in plan.sections if s.key not in assigned]

        sections = [
            by_key[key].model_copy(update={
                "order_index": index,
                "evidence_doc_numbers": assigned.get(key, []),
            })
            for index, key in enumerate(order)
        ]

        # Follow-ups are only meaningful for a section that ended up with nothing; one aimed anywhere
        # else would spend a whole extra round searching for evidence the report already has.
        uncovered = {s.key for s in sections if not s.evidence_doc_numbers}
        follow_ups: List[FollowUpQuery] = []
        seen_keys = set()
        for raw in parsed.get("follow_up_questions") or []:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("section_key") or "").strip()
            text = str(raw.get("text") or "").strip()
            if not text or key not in uncovered or key in seen_keys:
                continue
            seen_keys.add(key)
            follow_ups.append(FollowUpQuery(section_key=key, text=text))
            if len(follow_ups) >= settings.REPORT_FOLLOWUP_QUERIES_PER_ROUND:
                break

        return OutlineAssignment(sections=sections, follow_ups=follow_ups)

    # ==========================================
    # 3. SECTION WRITING
    # ==========================================

    @staticmethod
    def _section_user_content(
            topic: str,
            section: OutlineSection,
            evidence_block: str,
            allowed_doc_numbers: List[int],
            outline_context: str,
    ) -> str:
        return SECTION_USER_TEMPLATE.format(
            topic=topic,
            section_title=section.title,
            section_brief=section.brief or "(No further brief; cover what the excerpts support.)",
            outline_context=outline_context or "(This is the only section.)",
            allowed_doc_numbers=", ".join(f"[Doc {n}]" for n in allowed_doc_numbers) or "(none)",
            evidence_block=evidence_block or "(No excerpts were assigned to this section.)",
        )

    def write_section(
            self,
            model: LoadedModel,
            topic: str,
            section: OutlineSection,
            evidence_block: str,
            allowed_doc_numbers: List[int],
            outline_context: str,
    ) -> str:
        """
        Writes one section and, if the draft cites nothing, re-prompts once within the same acquisition.

        The retry mirrors the chat path's `_cited_or_refused`: an uncited draft is far more often a
        formatting lapse than a genuine absence of evidence, and one extra generation is much cheaper than
        discarding a section the corpus could support. If it still cites nothing, the caller decides -- the
        adapter does not invent a fallback.
        """
        try:
            llm: Any = model.model_instance
            messages = [
                {"role": "system", "content": SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": self._section_user_content(
                    topic, section, evidence_block, allowed_doc_numbers, outline_context
                )},
            ]

            draft = self._complete(model, messages, settings.REPORT_SECTION_MAX_TOKENS)
            if self._cites_something(draft) or self._is_no_evidence(draft):
                return draft

            logger.info(
                f"Section {section.key!r} drafted without citations; re-prompting once within the same "
                f"model acquisition."
            )
            messages += [
                {"role": "assistant", "content": draft},
                {"role": "user", "content": SECTION_CITATION_RETRY_PROMPT},
            ]
            return self._complete(model, messages, settings.REPORT_SECTION_MAX_TOKENS)

        except Exception as e:
            raise map_report_error(e, context=f"Writing section {section.key} of '{topic}'") from e

    _CITATION_TAG = re.compile(r"\[Doc \d+]")

    @classmethod
    def _cites_something(cls, text: str) -> bool:
        return bool(cls._CITATION_TAG.search(text))

    @staticmethod
    def _is_no_evidence(text: str) -> bool:
        return text.strip().upper().startswith(NO_EVIDENCE_SECTION)

    # ==========================================
    # 4. EXECUTIVE SUMMARY
    # ==========================================

    def write_summary(
            self,
            model: LoadedModel,
            topic: str,
            section_digest: str,
            allowed_doc_numbers: List[int],
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(
                    topic=topic,
                    section_digest=section_digest or "(The report has no written sections.)",
                    allowed_doc_numbers=", ".join(f"[Doc {n}]" for n in allowed_doc_numbers) or "(none)",
                )},
            ]
            return self._complete(model, messages, settings.REPORT_SECTION_MAX_TOKENS)

        except Exception as e:
            raise map_report_error(e, context=f"Writing the executive summary of '{topic}'") from e

    # ==========================================
    # COMPLETION HELPERS
    # ==========================================

    def _complete(self, model: LoadedModel, messages: List[Dict[str, str]], max_tokens: int) -> str:
        llm: Any = model.model_instance
        response = chat_completion(
            llm,
            messages,
            enable_thinking=settings.QA_LLM_ENABLE_THINKING,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return strip_reasoning(response["choices"][0]["message"]["content"] or "")

    def _json_completion(self, model: LoadedModel, messages: List[Dict[str, str]], max_tokens: int) -> dict:
        llm: Any = model.model_instance
        response = chat_completion(
            llm,
            messages,
            enable_thinking=settings.QA_LLM_ENABLE_THINKING,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = strip_reasoning(response["choices"][0]["message"]["content"] or "")
        parsed = self._parse_json_object(raw)
        if parsed is None:
            raise JSONPlanParseError(f"Report model returned no usable JSON object. Raw reply: {raw[:400]!r}")
        return parsed

    @staticmethod
    def _parse_json_object(raw: str) -> Optional[dict]:
        match = re.search(r"\{.*}", raw, re.DOTALL)
        candidate = match.group(0) if match else raw
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
