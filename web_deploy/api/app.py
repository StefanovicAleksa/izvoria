"""
The hosted appliance's HTTP API, on the GPU box. Only the gateway talks to it.

Every /api route is signed by the gateway on behalf of one user of one firm (web_deploy/common/signing.py); the
request then reaches only that firm's engine and database, and only that user's history. Long calls start a run and
return its id; the browser follows the run's events (see runs.py). Ingestion is queued for the ingestion worker,
never run here.

No client file passes through here: the firm's vault and uploads live on the VPS (docs/web_deploy.md, "Where data
lives"). Uploads go to the gateway, and a cited source is served by the gateway from its own vault once this API has
said which file and which span the cited chunk points at (`/api/sources/{chunk}/locate`).

Run: uvicorn web_deploy.api.app:create_production_app --factory --host 127.0.0.1 --port 8095
(reached by the gateway through the box's SSH tunnel)
(one process: the shared models are per process, and concurrency comes from the runner's threads)
"""

import asyncio
import contextlib
import logging
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from web_deploy.api import mapping
from web_deploy.api.config import ApiSettings
from web_deploy.api.engine_port import ApplianceIngesting, IEngineHost
from web_deploy.api.runner import Runner, call_in_scope, error_payload
from web_deploy.api.runs import RunNotFound, RunStore, resume_point, sse_frame
from web_deploy.common import youtube
from web_deploy.common.release import read_release
from web_deploy.common.signing import Principal, SignatureError, verify, verify_service
from web_deploy.common.sources import locate
from web_deploy.common.uploads import BatchRejected, batch_dir as firm_batch_dir, new_batch_id
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, JobOrigin, QueueError
from web_deploy.ingest_worker.staging import pending_markers

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 4000
MAX_RESULT_LIMIT = 50
SSE_POLL_SECONDS = 0.25
SSE_KEEPALIVE_SECONDS = 15.0
_PLAIN_CITATION = re.compile(r"\]\(izvoria://doc/\d+\)")

Handler = Callable[[Request, Principal], Awaitable[Response]]


class BadRequest(ValueError):
    pass


def create_app(host: IEngineHost, settings: ApiSettings, queue: IngestQueue,
               runs: Optional[RunStore] = None, runner: Optional[Runner] = None, start_host: bool = True,
               clock: Callable[[], float] = time.time,
               link_extractor: Optional[youtube.Extractor] = None) -> Starlette:
    runs = runs or RunStore()
    runner = runner or Runner(host, settings.live_workers, settings.report_workers)
    startup: Dict[str, Any] = {"error": None}

    # ---------- plumbing ----------

    def signed(handler: Handler) -> Callable[[Request], Awaitable[Response]]:
        async def wrapper(request: Request) -> Response:
            try:
                principal = verify(settings.secret, request.method, request.url.path, request.url.query,
                                   request.headers, now=clock())
            except SignatureError as exc:
                return JSONResponse({"error": str(exc)}, status_code=401)
            if not host.ready:
                return JSONResponse({"error": "The appliance is still starting.", "kind": "starting"},
                                    status_code=503)
            try:
                return await handler(request, principal)
            except (BadRequest, ValueError, BatchRejected, QueueError) as exc:
                return JSONResponse({"error": str(exc), "kind": "invalid"}, status_code=400)
            except RunNotFound:
                return JSONResponse({"error": "Not found.", "kind": "not_found"}, status_code=404)
            except Exception as exc:
                payload = error_payload(exc)
                status = {"busy": 409, "ingesting": 409, "offline": 503, "invalid": 400,
                          "not_found": 404}.get(payload["kind"], 500)
                if status == 500:
                    logger.exception(f"{request.method} {request.url.path} failed for {principal.firm}")
                return JSONResponse({"error": payload["message"], "kind": payload["kind"]}, status_code=status)
        return wrapper

    async def body(request: Request) -> dict:
        try:
            data = await request.json()
        except Exception:
            raise BadRequest("Expected a JSON object.")
        if not isinstance(data, dict):
            raise BadRequest("Expected a JSON object.")
        return data

    def language_of(data: dict) -> str:
        code = str(data.get("language") or "")
        codes = [lang["code"] for lang in host.languages()]
        if code not in codes:
            raise BadRequest(f"Choose a language for the answer: {', '.join(codes)}.")
        return code

    def prompt_of(data: dict, field: str = "prompt") -> str:
        text = str(data.get(field) or "").strip()
        if not text:
            raise BadRequest("Ask a question first.")
        if len(text) > MAX_PROMPT_CHARS:
            raise BadRequest(f"Keep it under {MAX_PROMPT_CHARS} characters.")
        return text

    def limit_of(data: dict, default: int) -> int:
        try:
            return max(1, min(MAX_RESULT_LIMIT, int(data.get("limit") or default)))
        except (TypeError, ValueError):
            raise BadRequest("'limit' must be a number.")

    async def engine_for(principal: Principal) -> Any:
        return await asyncio.to_thread(host.engine, principal.firm)

    def require_live() -> None:
        """Refuses model work up front while the GPU is handed to ingestion (the run would only fail)."""
        if host.paused:
            raise ApplianceIngesting()

    def started(run) -> JSONResponse:
        return JSONResponse({"run_id": run.id, "events": f"/api/runs/{run.id}/events"}, status_code=202)

    # ---------- health (unsigned: counts and the deployed version only) ----------

    # Read once: a new version is deployed by restarting this process (update_fleet.py)
    release = read_release()

    async def health(_: Request) -> JSONResponse:
        busy = host.busy() if host.ready else {"inflight": 0, "reports_running": 0}
        try:
            depth = await asyncio.to_thread(queue.depth)
        except Exception as exc:
            logger.warning(f"Queue depth unavailable: {exc}")
            depth = {"queued": 0, "running": 0}
        # An ingestion's results not yet pushed to the VPS (staging.py): stopping the box now would strand them
        staging_pending = bool(pending_markers(settings.staging_state_dir)) if settings.staging_state_dir else False
        return JSONResponse({
            "ready": host.ready, "starting_error": startup["error"], "paused": host.ready and host.paused, **busy,
            "queued": depth["queued"], "ingesting": depth["running"], "staging_pending": staging_pending,
            "busy": bool(busy["inflight"] or busy["reports_running"] or depth["queued"] or depth["running"]
                         or staging_pending),
            "release": release,
        })

    # ---------- runs ----------

    async def list_runs(request: Request, principal: Principal) -> Response:
        return JSONResponse({"runs": [r.public() for r in runs.active(principal)]})

    async def run_events(request: Request, principal: Principal) -> Response:
        run = runs.get(request.path_params["run_id"], principal)
        after = resume_point(request.headers.get("last-event-id") or request.query_params.get("after"))

        async def stream():
            nonlocal after
            yield "retry: 3000\n\n"
            quiet = 0.0
            while True:
                events = run.after(after)
                for seq, event, data in events:
                    after = seq
                    yield sse_frame(seq, event, data)
                if run.done and not run.after(after):
                    return
                if events:
                    quiet = 0.0
                elif quiet >= SSE_KEEPALIVE_SECONDS:
                    quiet = 0.0
                    yield ": keepalive\n\n"
                await asyncio.sleep(SSE_POLL_SECONDS)
                quiet += SSE_POLL_SECONDS

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------- live: chat & search ----------

    async def chat(request: Request, principal: Principal) -> Response:
        data = await body(request)
        prompt, language, limit = prompt_of(data), language_of(data), limit_of(data, 8)
        mode = str(data.get("mode") or "chat")
        if mode not in ("chat", "reasoned"):
            raise BadRequest("'mode' must be 'chat' or 'reasoned'.")
        session_id = mapping.parse_uuid(data["session_id"]) if data.get("session_id") else uuid.uuid4()
        require_live()
        engine = await engine_for(principal)
        run = runs.create(principal, f"chat:{mode}", subject=str(session_id))

        def work() -> dict:
            query = host.search_query(prompt, limit)
            with host.live():
                if mode == "reasoned":
                    message = mapping.reasoned_message(host, engine, engine.chat_reasoned(
                        session_id, query, owner_id=principal.user_id))
                else:
                    message = mapping.chat_message(host, engine, engine.chat(
                        query, session_id=session_id, owner_id=principal.user_id))
            return {"session_id": str(session_id), "message": message}

        runner.submit(run, work, language)
        return started(run)

    async def search(request: Request, principal: Principal) -> Response:
        data = await body(request)
        prompt, limit = prompt_of(data), limit_of(data, 10)
        language = language_of(data) if data.get("language") else None
        mode = str(data.get("mode") or "standard")
        if mode not in ("standard", "verified"):
            raise BadRequest("'mode' must be 'standard' or 'verified'.")
        require_live()
        engine = await engine_for(principal)
        run = runs.create(principal, f"search:{mode}")

        def work() -> dict:
            query = host.search_query(prompt, limit)
            call = engine.search_verified if mode == "verified" else engine.search_standard
            with host.live():
                result = call(query, owner_id=principal.user_id)
            return mapping.search_outcome(host, result, mode)

        runner.submit(run, work, language)
        return started(run)

    async def evidence(request: Request, principal: Principal) -> Response:
        data = await body(request)
        raw = data.get("chunk_ids") or []
        if not isinstance(raw, list) or len(raw) > 100:
            raise BadRequest("'chunk_ids' must be a list of at most 100 ids.")
        ids = [mapping.parse_uuid(c) for c in raw]
        engine = await engine_for(principal)

        def read() -> list:
            by_id = {c.chunk_id: c for c in engine.get_chunks(ids)}
            return [mapping.chunk_evidence(host, n, by_id[i]) for n, i in enumerate(ids, start=1) if i in by_id]

        return JSONResponse({"evidence": await asyncio.to_thread(read)})

    # ---------- history ----------

    async def conversations(request: Request, principal: Principal) -> Response:
        engine = await engine_for(principal)
        items = await asyncio.to_thread(engine.list_conversations, 200, principal.user_id)
        return JSONResponse({"conversations": [mapping.conversation_ref(s) for s in items]})

    async def conversation(request: Request, principal: Principal) -> Response:
        session_id = mapping.parse_uuid(request.path_params["session_id"])
        engine = await engine_for(principal)
        if request.method == "DELETE":
            deleted = await asyncio.to_thread(engine.delete_conversation, session_id, principal.user_id)
            return JSONResponse({"deleted": deleted})
        turns = await asyncio.to_thread(engine.get_conversation, session_id, principal.user_id)
        if not turns:
            raise RunNotFound(str(session_id))
        messages = await asyncio.to_thread(mapping.conversation, host, engine, turns)
        return JSONResponse({"session_id": str(session_id), "messages": messages})

    async def searches(request: Request, principal: Principal) -> Response:
        engine = await engine_for(principal)
        items = await asyncio.to_thread(engine.list_searches, 200, principal.user_id)
        return JSONResponse({"searches": [mapping.search_record_ref(r) for r in items]})

    async def search_record(request: Request, principal: Principal) -> Response:
        record_id = mapping.parse_uuid(request.path_params["record_id"])
        engine = await engine_for(principal)
        if request.method == "DELETE":
            deleted = await asyncio.to_thread(engine.delete_search, record_id, principal.user_id)
            return JSONResponse({"deleted": bool(deleted)})
        record = await asyncio.to_thread(engine.get_search, record_id, principal.user_id)
        if record is None:
            raise RunNotFound(str(record_id))
        return JSONResponse(await asyncio.to_thread(mapping.search_record, host, engine, record))

    # ---------- reports ----------

    async def own_report(engine: Any, principal: Principal, report_id: uuid.UUID) -> Any:
        report = await asyncio.to_thread(engine.get_report, report_id, principal.user_id)
        if report is None:
            raise RunNotFound(str(report_id))
        return report

    async def plan_report(request: Request, principal: Principal) -> Response:
        data = await body(request)
        topic, language = prompt_of(data, "topic"), language_of(data)
        session_id = mapping.parse_uuid(data["session_id"]) if data.get("session_id") else None
        require_live()
        engine = await engine_for(principal)
        run = runs.create(principal, "report:plan")

        def work() -> dict:
            with host.live():
                proposal = engine.propose_research_plan(topic, session_id, owner_id=principal.user_id)
            return mapping.proposal(proposal)

        runner.submit(run, work, language)
        return started(run)

    async def reports(request: Request, principal: Principal) -> Response:
        engine = await engine_for(principal)
        items = await asyncio.to_thread(engine.list_reports, None, 200, principal.user_id)
        return JSONResponse({"reports": [mapping.report_ref(r) for r in items]})

    async def report(request: Request, principal: Principal) -> Response:
        report_id = mapping.parse_uuid(request.path_params["report_id"])
        engine = await engine_for(principal)
        stored = await own_report(engine, principal, report_id)
        if stored.status.value == "awaiting_approval":
            expected, minutes = engine.plan_cost(stored.plan)
            corpus = stored.corpus
            return JSONResponse({"stage": "plan", "plan": mapping.plan_draft(
                stored.id, stored.plan,
                indexed_source_count=corpus.indexed_source_count if corpus else stored.indexed_source_count,
                indexed_chunk_count=corpus.indexed_chunk_count if corpus else 0,
                expected_job_count=expected, estimated_minutes=minutes)})
        document = await asyncio.to_thread(call_in_scope, host, stored.language,
                                           lambda: mapping.report_document(host, engine, stored))
        running = any(r.subject == str(report_id) for r in runs.active(principal))
        return JSONResponse({"stage": "document", "document": document, "running": running})

    async def report_markdown(request: Request, principal: Principal) -> Response:
        report_id = mapping.parse_uuid(request.path_params["report_id"])
        engine = await engine_for(principal)
        stored = await own_report(engine, principal, report_id)
        text = await asyncio.to_thread(call_in_scope, host, stored.language, lambda: host.render_report(stored))
        name = mapping.download_name(stored, ".md")
        return PlainTextResponse(_PLAIN_CITATION.sub("]", text), media_type="text/markdown; charset=utf-8",
                                 headers={"Content-Disposition": f"attachment; filename=\"{name}\""})

    async def report_pdf(request: Request, principal: Principal) -> Response:
        report_id = mapping.parse_uuid(request.path_params["report_id"])
        engine = await engine_for(principal)
        stored = await own_report(engine, principal, report_id)
        pdf = await asyncio.to_thread(call_in_scope, host, stored.language, lambda: host.report_pdf(stored))
        name = mapping.download_name(stored, ".pdf")
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=\"{name}\"",
                                 "Cache-Control": "private, no-store"})

    async def run_report(request: Request, principal: Principal) -> Response:
        report_id = mapping.parse_uuid(request.path_params["report_id"])
        data = await body(request)
        mode = str(data.get("mode") or "deep")
        if mode not in ("deep", "exhaustive"):
            raise BadRequest("'mode' must be 'deep' or 'exhaustive'.")
        require_live()
        engine = await engine_for(principal)
        stored = await own_report(engine, principal, report_id)
        approved = mapping.apply_edits(host, stored.plan, data.get("plan") or {})
        run = runs.create(principal, "report:run", subject=str(report_id))

        def work() -> dict:
            execute = (engine.execute_iterative_research_report if mode == "exhaustive"
                       else engine.execute_research_report)
            with host.live():
                report = execute(report_id, approved)
            return mapping.report_document(host, engine, report)

        # A report is written in the language it was scoped in, whatever the user picks later
        runner.submit(run, work, stored.language, long_running=True)
        return started(run)

    async def resume_report(request: Request, principal: Principal) -> Response:
        report_id = mapping.parse_uuid(request.path_params["report_id"])
        require_live()
        engine = await engine_for(principal)
        stored = await own_report(engine, principal, report_id)
        run = runs.create(principal, "report:run", subject=str(report_id))

        def work() -> dict:
            with host.live():
                report = engine.resume_research_report(report_id)
            return mapping.report_document(host, engine, report)

        runner.submit(run, work, stored.language, long_running=True)
        return started(run)

    async def cancel_report(request: Request, principal: Principal) -> Response:
        report_id = mapping.parse_uuid(request.path_params["report_id"])
        engine = await engine_for(principal)
        await own_report(engine, principal, report_id)
        return JSONResponse({"cancelling": bool(engine.cancel_research_report(report_id))})

    async def rename_report(request: Request, principal: Principal) -> Response:
        report_id = mapping.parse_uuid(request.path_params["report_id"])
        data = await body(request)
        title = " ".join(str(data.get("title") or "").split())[:mapping.MAX_TITLE_CHARS]
        engine = await engine_for(principal)
        renamed = await asyncio.to_thread(engine.rename_research_report, report_id, title, principal.user_id)
        if renamed is None:
            raise RunNotFound(str(report_id))
        return JSONResponse({"report_id": str(report_id), "title": renamed.title})

    # ---------- library & sources ----------

    async def library(request: Request, principal: Principal) -> Response:
        engine = await engine_for(principal)
        sources = await asyncio.to_thread(engine.list_sources)
        profiles = {p.source_id: p.profile for p in await asyncio.to_thread(engine.list_source_profiles)}
        return JSONResponse({"sources": [mapping.source_ref(s, profiles.get(s.id)) for s in sources]})

    async def edit_source_profile(request: Request, principal: Principal) -> Response:
        """
        A firm administrator's own title and topic for a source, for the whole firm. Its index is re-embedded with
        them (the embedder only, seconds), so it is model work and waits its turn like a question.
        """
        if principal.role not in ("admin", "owner"):
            return JSONResponse({"error": "Only a firm administrator can rename files in the Library.",
                                 "kind": "forbidden"}, status_code=403)
        source_id = mapping.parse_uuid(request.path_params["source_id"])
        data = await body(request)
        title = " ".join(str(data.get("title") or "").split())
        topic = " ".join(str(data.get("topic") or "").split())
        if not title:
            raise BadRequest("Give the file a title.")
        engine = await engine_for(principal)
        if source_id not in {source.id for source in await asyncio.to_thread(engine.list_sources)}:
            raise RunNotFound(str(source_id))

        def work():
            with host.live():
                return engine.edit_source_profile(source_id, title, topic)

        return JSONResponse(mapping.source_profile(await asyncio.to_thread(work)))

    async def source_location(request: Request, principal: Principal) -> Response:
        """
        Which file and which span a cited chunk points at, for the gateway to serve from the firm's vault on the VPS.
        Re-resolved by chunk id through the firm's own engine; the gateway still checks the path lies in that vault.
        """
        engine = await engine_for(principal)
        chunks = await asyncio.to_thread(engine.get_chunks, [mapping.parse_uuid(request.path_params["chunk_id"])])
        if not chunks:
            raise RunNotFound(request.path_params["chunk_id"])
        return JSONResponse(locate(chunks[0]))

    # ---------- ingestion (uploads, the wait list and cancelling are the gateway's, on the VPS) ----------

    # ---------- YouTube links (video only) ----------

    def requested_links(data: dict) -> list:
        if not settings.link_ingest_enabled:
            raise BadRequest("Adding videos from links is turned off on this appliance.")
        if str(data.get("mode") or "video") != IngestMode.VIDEO.value:
            raise BadRequest("Links can only be added as video.")
        links = data.get("links")
        if not isinstance(links, list) or not any(str(link).strip() for link in links):
            raise BadRequest("Paste at least one YouTube link.")
        return [str(link)[:2000] for link in links[:100]]

    async def resolve_links(links: list) -> youtube.Resolution:
        return await asyncio.to_thread(youtube.resolve, links, settings.link_max_videos, link_extractor)

    async def preview_links(request: Request, principal: Principal) -> Response:
        resolution = await resolve_links(requested_links(await body(request)))
        return JSONResponse(resolution.public())

    async def enqueue_links(request: Request, principal: Principal) -> Response:
        data = await body(request)
        resolution = await resolve_links(requested_links(data))
        # The user may have dropped some of the previewed videos; the links are resolved again, never trusted
        if isinstance(data.get("video_ids"), list):
            chosen = {str(v) for v in data["video_ids"]}
            resolution.videos = [v for v in resolution.videos if v.id in chosen]
        if not resolution.videos:
            raise BadRequest("None of these links has a video that can be downloaded.")
        language = language_of(data)
        # The folder exists only in the box's scratch, made when the job runs; the videos travel in the job row
        folder = firm_batch_dir(settings.tenants_root, principal.firm, principal.user_id, new_batch_id())
        job = await asyncio.to_thread(queue.enqueue, principal.firm, principal.user_id, IngestMode.VIDEO, language,
                                      str(folder), len(resolution.videos), JobOrigin.YOUTUBE,
                                      youtube.new_manifest(resolution.videos))
        positions = await asyncio.to_thread(queue.queue_positions)
        return JSONResponse(job.public(positions.get(job.id)), status_code=201)

    # ---------- handing the GPU to ingestion (box-internal: the ingestion worker only) ----------

    async def gpu_handoff(request: Request) -> Response:
        if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        try:
            verify_service(settings.secret, request.method, request.url.path, request.headers, now=clock())
        except SignatureError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        if not host.ready:
            return JSONResponse({"error": "The appliance is still starting."}, status_code=503)
        action = request.path_params["action"]
        if action == "pause":
            # Blocks until the questions and reports in flight are done and the live models are freed
            await asyncio.to_thread(host.pause)
        elif action == "resume":
            await asyncio.to_thread(host.resume)
        else:
            return JSONResponse({"error": "Unknown action."}, status_code=404)
        return JSONResponse({"paused": host.paused})

    async def languages(request: Request, principal: Principal) -> Response:
        return JSONResponse({"languages": host.languages()})

    # ---------- lifecycle ----------

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        task = None
        if start_host:
            async def boot():
                try:
                    await asyncio.to_thread(host.start)
                except Exception as exc:
                    logger.exception("The appliance failed to start")
                    startup["error"] = str(exc)
            task = asyncio.create_task(boot())
        try:
            yield
        finally:
            if task:
                task.cancel()
            runner.shutdown()
            if start_host:
                await asyncio.to_thread(host.stop)

    r = lambda path, handler, methods: Route(path, signed(handler), methods=methods)
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/internal/gpu/{action}", gpu_handoff, methods=["POST"]),
            r("/api/languages", languages, ["GET"]),
            r("/api/runs", list_runs, ["GET"]),
            r("/api/runs/{run_id}/events", run_events, ["GET"]),
            r("/api/chat", chat, ["POST"]),
            r("/api/search", search, ["POST"]),
            r("/api/evidence", evidence, ["POST"]),
            r("/api/conversations", conversations, ["GET"]),
            r("/api/conversations/{session_id}", conversation, ["GET", "DELETE"]),
            r("/api/searches", searches, ["GET"]),
            r("/api/searches/{record_id}", search_record, ["GET", "DELETE"]),
            r("/api/reports", reports, ["GET"]),
            r("/api/reports/plan", plan_report, ["POST"]),
            r("/api/reports/{report_id}", report, ["GET"]),
            r("/api/reports/{report_id}/markdown", report_markdown, ["GET"]),
            r("/api/reports/{report_id}/pdf", report_pdf, ["GET"]),
            r("/api/reports/{report_id}/run", run_report, ["POST"]),
            r("/api/reports/{report_id}/resume", resume_report, ["POST"]),
            r("/api/reports/{report_id}/cancel", cancel_report, ["POST"]),
            r("/api/reports/{report_id}/rename", rename_report, ["POST"]),
            r("/api/library", library, ["GET"]),
            r("/api/library/{source_id}/profile", edit_source_profile, ["PUT"]),
            r("/api/sources/{chunk_id}/locate", source_location, ["GET"]),
            r("/api/ingest/links/preview", preview_links, ["POST"]),
            r("/api/ingest/links", enqueue_links, ["POST"]),
        ],
        lifespan=lifespan,
    )


def create_production_app() -> Starlette:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from web_deploy.common.box_credentials import install_from_env
    install_from_env()  # the hosted box: the database password comes from the gateway, never from the environment
    from web_deploy.api.engine_host import EngineHost
    from web_deploy.api.progress import job_listener
    settings = ApiSettings.from_env()
    host = EngineHost(settings, progress_listener=job_listener)
    return create_app(host, settings, IngestQueue.connect(settings.control_database_url))
