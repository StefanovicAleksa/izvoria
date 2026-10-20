# Path: scripts/backfill_source_profiles.py

"""
Gives every source indexed before source profiles existed a title and topic, and names it in its chunks' indexed
context, without re-ingesting anything.

    venv/bin/python -m scripts.backfill_source_profiles              # every source without a profile
    venv/bin/python -m scripts.backfill_source_profiles --dry-run    # list them only
    venv/bin/python -m scripts.backfill_source_profiles --source <uuid> [--source <uuid> ...]
    venv/bin/python -m scripts.backfill_source_profiles --regenerate  # rewrite every profile a person didn't edit

Two passes, one model each: the contextual SLM reads each source's stored transcript or text and writes its profile
(minutes for a few dozen lectures), then the embedder re-embeds each source's chunks in place under its new
`[Source: …]` line (same chunk ids, so clips, conversations and reports keep pointing at them). A profile a person
edited is never replaced. Run it with the app closed: it uses the GPU. On the hosted version, run it once per firm
database (POSTGRES_DB=onyx_firm_<slug>).
"""

import argparse
import logging
import sys
import uuid
from typing import Callable, Dict, List

from app.core.config import settings
from app.core.database.service.manager import db_manager
from app.core.database.service.migrations import upgrade_to_head
from app.engine.service.bootstrap import IzvoriaBootstrapper
from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.data.slm_adapter import SLMContextAdapter
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.rag_storage.service.storage_api import RAGStorageService
from app.features.source_profile.data.llm_profile_adapter import LlmProfileAdapter
from app.features.source_profile.data.postgres_profile_repo import PostgresSourceProfileRepository
from app.features.source_profile.domain.entities import SourceIdentity
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.service.source_profile_api import SourceProfileAPI

logger = logging.getLogger("backfill_source_profiles")


def _targets(profile_api: SourceProfileAPI, requested: List[uuid.UUID], regenerate: bool) -> List[SourceIdentity]:
    if not requested:
        if regenerate:
            return [source.identity for source in profile_api.list_indexed()
                    if source.profile is None or source.profile.origin is not ProfileOrigin.EDITED]
        return profile_api.list_unprofiled()
    indexed = {source.source_id: source.identity for source in profile_api.list_indexed()}
    missing = [str(source_id) for source_id in requested if source_id not in indexed]
    if missing:
        raise SystemExit(f"Not indexed: {', '.join(missing)}")
    return [indexed[source_id] for source_id in requested]


def _by_language(targets: List[SourceIdentity]) -> Dict[str, List[SourceIdentity]]:
    """Sources grouped by the language they were ingested in (the active one for a source without)."""
    groups: Dict[str, List[SourceIdentity]] = {}
    for identity in targets:
        groups.setdefault(identity.language or settings.APPLIANCE_LANGUAGE, []).append(identity)
    return groups


def _profile_in(language: str, work, contextualizer_for: Callable[[], ContextualizerAPI]) -> int:
    """
    Profiles one language's sources with that language's SLM: a pack can swap it (Serbian uses Gemma, which reads
    Serbian; Llama 3.2 3B doesn't), and the path is fixed in the manifest when the API is built. The language
    the run started in is restored afterwards. A source in a language with no served pack is skipped.
    """
    started_in = settings.APPLIANCE_LANGUAGE
    try:
        settings.use_language(language)
    except ValueError as e:
        logger.error(f"Skipping {len(work)} source(s) in '{language}': {e}")
        return 0
    try:
        logger.info(f"Profiling {len(work)} source(s) in '{language}' with {settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH}.")
        return contextualizer_for().profile_sources(work)
    finally:
        settings.use_language(started_in)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", type=uuid.UUID, default=[],
                        help="Profile (again) only this source; repeatable.")
    parser.add_argument("--regenerate", action="store_true",
                        help="Profile again every source whose profile a person did not edit (only the --source ones, if given).")
    parser.add_argument("--dry-run", action="store_true", help="List the sources that would be profiled.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    upgrade_to_head(db_manager.get_engine())

    kit = IzvoriaBootstrapper.build_model_kit()
    profile_api = SourceProfileAPI(PostgresSourceProfileRepository(db_manager), LlmProfileAdapter())

    def contextualizer_for() -> ContextualizerAPI:
        # Built after the language is switched: its manifest takes the SLM path from the active pack
        return ContextualizerAPI(repository=ContextualRepository(db_manager), orchestrator=kit.orchestrator,
                                 slm_adapter=SLMContextAdapter(), profile_api=profile_api)

    rag_storage = RAGStorageService(rag_repo=PostgresRagRepository(db_manager), orchestrator=kit.orchestrator,
                                    embedder=kit.hf_embedder, profile_api=profile_api)

    targets = _targets(profile_api, args.source, args.regenerate)
    if not targets:
        logger.info("Every indexed source already has a profile.")
        return 0
    for identity in targets:
        logger.info(f"  {identity.source_id}  [{identity.language or '?'}]  {identity.name}")
    if args.dry_run:
        logger.info(f"{len(targets)} source(s) would be profiled.")
        return 0

    try:
        # Pass 1: the contextual SLM, once per language, for every profile
        profiled = 0
        for language, group in _by_language(targets).items():
            work = [(identity.source_id, rag_storage.stored_texts(identity.source_id), identity.language)
                    for identity in group]
            profiled += _profile_in(language, work, contextualizer_for)
        logger.info(f"Profiled {profiled} of {len(targets)} source(s).")

        # Pass 2: the embedder, once, for every source's chunks
        updated = 0
        for identity in targets:
            profile = profile_api.get(identity.source_id)
            if profile is None:
                continue
            chunks = rag_storage.refresh_source_line(identity.source_id, release_after=False)
            updated += chunks
            logger.info(f"'{profile.title}': {chunks} chunk(s) re-indexed.")
        logger.info(f"Done: {updated} chunk(s) re-indexed.")
        return 0
    finally:
        kit.async_service.shutdown()
        kit.async_service.release_all()
        db_manager.dispose()


if __name__ == "__main__":
    sys.exit(main())
