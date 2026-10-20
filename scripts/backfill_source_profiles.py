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
from typing import List

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
    contextualizer = ContextualizerAPI(repository=ContextualRepository(db_manager), orchestrator=kit.orchestrator,
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
        # Pass 1: the contextual SLM, once, for every profile
        work = [(identity.source_id, rag_storage.stored_texts(identity.source_id), identity.language)
                for identity in targets]
        profiled = contextualizer.profile_sources(work)
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
