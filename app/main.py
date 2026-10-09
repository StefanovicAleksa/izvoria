# Canvas: main_v7
# Path: app/main.py

import os
import sys
import uuid
import logging

# CRITICAL FIX: Ensure PyTorch uses expandable segments to prevent fragmentation crashes.
# This MUST be declared at the absolute top before any PyTorch-dependent modules are imported.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from app.core.anchors.domain.enums import AnchorKind
from app.core.anchors.utils.formatting import format_location
from app.core.language.service.language_api import get_language_pack, get_supported_language_packs
from app.engine.service.bootstrap import OnyxBootstrapper
from app.engine.exceptions.base import EngineError
from app.features.rag_retrieval.domain.entities import SearchQuery

# Setup minimal logging for clean CLI output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("onyx_cli")
logger.setLevel(logging.INFO)


def print_header():
    print("\n" + "=" * 60)
    print("          O N Y X   E N T E R P R I S E   A I          ")
    print("=" * 60)
    print("   AIR-GAPPED AI APPLIANCE | VRAM-MANAGED PIPELINES    ")
    print("=" * 60 + "\n")


def language_label(code: str) -> str:
    return f"{get_language_pack(code).native_name} ({code})"


def choose_language(engine, question: str):
    """A supported language code, the active one on a blank answer, or None for an unknown code."""
    options = ", ".join(f"{pack.code} = {pack.native_name}" for pack in get_supported_language_packs())
    answer = input(f"[{question} ({options}); blank keeps {engine.active_language}] > ").strip().lower()
    if not answer:
        return engine.active_language
    if answer not in {pack.code for pack in get_supported_language_packs()}:
        print(f"[ERROR] '{answer}' is not a supported language.")
        return None
    return answer


def switch_language(engine, code: str):
    """The engine to use from now on: rebuilt in `code`, or the same one when it is already active."""
    if code == engine.active_language:
        return engine
    print(f"\n[SYSTEM] Switching to {language_label(code)}...")
    try:
        return OnyxBootstrapper.switch_language(engine, code)
    except EngineError:
        raise
    except Exception as e:
        # The old engine is already stopped; come back up in the language we were in
        print(f"\n[ENGINE ERROR] Could not switch to {language_label(code)}: {e}")
        print(f"[SYSTEM] Restarting in {language_label(engine.active_language)}...")
        restarted = OnyxBootstrapper.build()
        restarted.start()
        return restarted


def ingest_batch(engine, folder_prompt: str, pipeline_name: str, run):
    """Asks for a folder and the language its files are in, switches to that language, and ingests."""
    raw_path = input(f"[{folder_prompt}] > ").strip()
    if not raw_path:
        return engine
    code = choose_language(engine, "Language of these files")
    if code is None:
        return engine
    engine = switch_language(engine, code)
    print(f"\n[SYSTEM] Evicting UI models... Initializing {pipeline_name} ({language_label(code)}).")
    print_report(run(engine, raw_path))
    return engine


def print_report(report):
    print("\n" + "-" * 20 + " INGESTION REPORT " + "-" * 20)
    print(f"Target Path: {report.target_path}")
    print(f"Duration:    {report.total_duration_seconds} seconds")
    print(f"Scanned:     {report.total_files_scanned} files")
    print(f"Success:     {report.successful_ingestions} files")
    print(f"Failed:      {report.failed_ingestions} files")
    print(f"Vectors:     {report.vectors_created}")
    print(f"Clips:       {report.clips_extracted}")
    if report.pages_converted:
        print(f"Pages:       {report.pages_converted} ({report.pages_ocr} by OCR, "
              f"{report.pages_ocr_incomplete} missing lines, {report.pages_unreadable} unreadable)")
    if report.images_described:
        print(f"Images:      {report.images_described} described ({report.images_text_unverified} with text "
              f"unverified, {report.image_text_items_dropped} text items not confirmed by OCR and left out, "
              f"{report.image_ocr_lines_withheld} unreadable OCR lines left out)")
    print("-" * 58)

    if report.errors:
        print("\n[ERRORS RECORDED]")
        for key, err_msg in report.errors.items():
            print(f" ! {key}: {err_msg}")


def print_search_results(results, mode_name: str):
    print(f"\n[{mode_name}] Found {len(results.results)} relevant segments:")
    print("=" * 80)

    for i, chunk in enumerate(results.results, 1):
        physical_path = getattr(chunk, 'physical_path', 'No physical file resolved')

        print(f"RESULT #{i}")
        print(f"  Source:    {physical_path}")
        print(f"  {_location(chunk, 'Timestamp: ', 'Location:  ')}")
        print(f"  Content:   {chunk.text_payload}")
        print("-" * 80)


def _location(item, time_label: str, document_label: str) -> str:
    """A search hit's or citation's place in its source: a recording's seconds, as always, or a page and section."""
    if item.anchor_kind is AnchorKind.TIME:
        return f"{time_label}{item.start_sec}s - {item.end_sec}s"
    return document_label + format_location(item.anchor_kind, item.start_sec, item.end_sec,
                                            item.page_start, item.page_end, item.section_path)


def print_reasoned_result(result):
    if result.qa_response:
        print(f"\n[ONYX AI - REASONED ANSWER]\n{result.qa_response.generated_answer}")
        print(f"\n[CITATIONS USED: {len(result.qa_response.citations)}]")
        for cit in result.qa_response.citations:
            print(f"  {cit.doc_id} {cit.physical_path or 'No physical file resolved'}")
            print(f"         {_location(cit, 'Timestamp: ', 'Location: ')}")
        if result.critique_unresolved:
            print("\n[NOTE] Self-critique had unresolved concerns about this answer's grounding.")
    else:
        print(f"\n[ONYX AI - CLARIFICATION NEEDED]\n{result.clarification_question}")

    print(f"\n[REASONING TRACE: {result.total_iterations} iteration(s)]")
    for step in result.iteration_trace:
        print(f"  Iter {step.iteration}: {step.decision} - {step.interim_analysis[:150]}")
        if step.refined_query:
            print(f"    -> refined query: {step.refined_query}")


def print_chat_response(response):
    print(f"\n[ONYX AI]\n{response.generated_answer}")
    print(f"\n[CITATIONS USED: {len(response.citations)}]")
    for cit in response.citations:
        print(f"  {cit.doc_id} {cit.physical_path or 'No physical file resolved'}")
        print(f"         {_location(cit, 'Timestamp: ', 'Location: ')}")


def run_conversation(engine, session_id):
    """
    Multi-turn chat: each question is a turn of the session's conversation, so follow-ups can refer to earlier
    questions, answers and cited excerpts. Returns the session id to keep using (/new starts a fresh one).
    """
    print("\n[SYSTEM] Conversation mode. Blank line returns to the menu; /new starts a new conversation.")
    while True:
        prompt = input("\n[You] > ").strip()
        if not prompt:
            return session_id
        if prompt.lower() == "/new":
            session_id = uuid.uuid4()
            print("[SYSTEM] Started a new conversation.")
            continue

        print("\n[SYSTEM] Running RAG + QA Generation Pipeline...")
        try:
            response = engine.chat(SearchQuery(prompt=prompt, limit=5), session_id=session_id)
        except EngineError as e:
            print(f"\n[ENGINE ERROR] Task failed: {e}")
            continue
        print_chat_response(response)


def run_reasoned_with_clarifications(engine, session_id, original_question, max_clarification_rounds=3):
    """
    chat_reasoned()'s intra-question state (evidence pool, decision history) is ephemeral
    and discarded once the call returns, so a clarification round can't resume mid-loop — it
    re-asks with the original question plus every clarification given so far appended, and lets
    the Reasoner start its reasoning fresh with that fuller picture.
    """
    clarifications = []
    for round_num in range(max_clarification_rounds + 1):
        conversation = original_question
        for i, clarification in enumerate(clarifications, 1):
            conversation += f"\n\n[Clarification {i}] {clarification}"

        print("\n[SYSTEM] Running the Reasoned Answer loop (this may take a while)...")
        query = SearchQuery(prompt=conversation, limit=5)
        result = engine.chat_reasoned(session_id, query)
        print_reasoned_result(result)

        if result.qa_response is not None:
            return

        if round_num >= max_clarification_rounds:
            print(f"\n[SYSTEM] Reached the clarification limit ({max_clarification_rounds} rounds) without a grounded answer.")
            return

        answer = input("\n[Your clarification] > (leave blank to cancel) ").strip()
        if not answer:
            print("\n[SYSTEM] Cancelled.")
            return
        clarifications.append(answer)


def print_research_plan(proposal):
    """The outline the user is being asked to approve, with what it will cost."""
    plan = proposal.plan
    print("\n" + "=" * 78)
    print(f"PROPOSED RESEARCH PLAN — {plan.display_title}")
    print("=" * 78)
    if plan.title:
        print(f"\nRequest: {plan.topic}")
    if plan.scope_note:
        print(f"\nScope: {plan.scope_note}")
    print(f"\nCorpus: {proposal.corpus.indexed_source_count} indexed recording(s), "
          f"{proposal.corpus.indexed_chunk_count} excerpt(s).")
    print(f"Estimated run time: about {proposal.estimated_minutes} minutes "
          f"({proposal.expected_job_count} steps).\n")

    for section in plan.sections:
        print(f"  [{section.key}] {section.title}")
        if section.brief:
            print(f"        {section.brief}")
        for question in plan.sub_questions_for(section.key):
            print(f"          - {question.text}")
    print("=" * 78)


def edit_research_plan(plan):
    """
    Minimal outline editing at the terminal: drop sections, or rename one.

    Deliberately small. The point of the approval gate is to catch a misread topic before spending half
    an hour on it, not to be a document editor -- the desktop UI is where real editing belongs.
    """
    from app.features.deep_research_report.domain.entities import ResearchPlan

    sections = list(plan.sections)
    while True:
        print("\n  Sections: " + ", ".join(f"{s.key}={s.title}" for s in sections))
        command = input("  [d <key> to drop, r <key> to rename, blank to finish] > ").strip()
        if not command:
            break

        action, _, argument = command.partition(" ")
        key = argument.strip()
        target = next((s for s in sections if s.key == key), None)
        if target is None:
            print(f"  [ERROR] No section {key!r}.")
            continue

        if action == "d":
            if len(sections) == 1:
                print("  [ERROR] A report needs at least one section.")
                continue
            sections = [s for s in sections if s.key != key]
        elif action == "r":
            new_title = input(f"  [New title for {key}] > ").strip()
            if new_title:
                sections = [s.model_copy(update={"title": new_title}) if s.key == key else s
                            for s in sections]
        else:
            print("  [ERROR] Use 'd <key>' or 'r <key>'.")

    kept = {s.key for s in sections}
    return plan.model_copy(update={
        "sections": sections,
        "sub_questions": [q for q in plan.sub_questions if q.section_key in kept],
    })


def run_research_report(engine, session_id):
    """
    Deep Research Report: scope, approve, then the long run.

    Separated into two calls on purpose -- the run holds the GPU for tens of minutes, so the outline is
    shown first and a misread topic costs a few seconds instead of the whole run.
    """
    topic = input("[Enter the research topic] > ").strip()
    if not topic:
        return

    print("\n  [d] Deep       — one sweep of the corpus. Faster.")
    print("  [e] Exhaustive — re-searches for any section the first sweep could not support. Slower.")
    exhaustive = input("[Mode: d/e] > ").strip().lower().startswith("e")

    print("\n[SYSTEM] Scoping the report against the indexed corpus (this takes a minute)...")
    try:
        proposal = engine.propose_research_plan(topic, session_id)
    except Exception as e:
        print(f"\n[ERROR] Could not scope this topic: {e}")
        return

    print_research_plan(proposal)
    plan = proposal.plan

    while True:
        choice = input("\n[a]ccept / [e]dit / [c]ancel > ").strip().lower()
        if choice in ("c", "cancel", ""):
            print("[SYSTEM] Cancelled. Nothing was run.")
            return
        if choice in ("e", "edit"):
            plan = edit_research_plan(plan)
            continue
        if choice in ("a", "accept"):
            break
        print("[ERROR] Choose a, e or c.")

    print(f"\n[SYSTEM] Researching in {'exhaustive' if exhaustive else 'deep'} mode. The appliance is "
          f"locked for roughly {proposal.estimated_minutes} minutes; Ctrl-C stops after the current section.")
    try:
        if exhaustive:
            report = engine.execute_iterative_research_report(proposal.report_id, plan)
        else:
            report = engine.execute_research_report(proposal.report_id, plan)
    except KeyboardInterrupt:
        # The run is on this thread, so Ctrl-C lands here rather than in the cancellation token. Whatever
        # was committed per section is already safe, and resume picks the rest up.
        print("\n[SYSTEM] Interrupted. Sections already written are saved; "
              f"resume with report id {proposal.report_id}.")
        return
    except Exception as e:
        print(f"\n[ERROR] The research run failed: {e}")
        return

    print("\n" + "=" * 78)
    print(f"REPORT COMPLETE — {report.display_title}")
    print("=" * 78)
    print(f"  Sections written : {len(report.written_sections)}")
    print(f"  Not covered      : {len(report.uncovered_sections)}")
    print(f"  Cited excerpts   : {len(report.cited_doc_numbers)}")
    print(f"  Coverage         : swept {report.swept_source_count} of "
          f"{report.indexed_source_count} indexed source(s)")
    print(f"  Gathering rounds : {report.rounds_used} ({report.mode.value} mode)")
    print(f"\n  Written to: {report.markdown_path}")
    print("=" * 78)


def main():
    print_header()
    print("[SYSTEM] Initializing hardware state and preloading UI models...")

    try:
        engine = OnyxBootstrapper.build()
        engine.start()
    except Exception as e:
        print(f"\n[FATAL STARTUP ERROR] Appliance failed to boot: {e}")
        sys.exit(1)

    # Chat (5) and Reasoned Answer (7) share one conversation until /new is typed in chat
    conversation_session_id = uuid.uuid4()

    try:
        while True:
            print(f"\nACTIVE LANGUAGE: {language_label(engine.active_language)} "
                  f"-- new batches are indexed in it, answers are written in it")
            print("\nAVAILABLE OPERATIONS:")
            print("  1. Ingest Folders [Video Pipeline - FULL]")
            print("  2. Ingest Folders [Audio Pipeline - FAST]")
            print("  3. Standard Search [Fast RAG]")
            print("  4. Verified Search [SLM Hallucination Scrubber]")
            print("  5. Chat with Data [Multi-Turn Conversation]")
            print("  6. Shutdown Appliance")
            print("  7. Reasoned Answer [Multi-Pass Reasoning]")
            print("  8. Deep Research Report [Multi-Page Cited Report]")
            print("  9. Ingest Folders [Documents & Books - PDF / DOCX / EPUB / TXT / MD / HTML / RTF]")
            print(" 10. Ingest Folders [Images - JPG / PNG / TIFF / WebP]")
            print(" 11. Switch Answer Language")

            choice = input("\n[Select Option 1-11] > ").strip()

            try:
                if choice == '1':
                    engine = ingest_batch(engine, "Enter Absolute Path to Video Folder", "Video Pipeline",
                                          lambda e, path: e.ingest_videos(path))

                elif choice == '2':
                    engine = ingest_batch(engine, "Enter Absolute Path to Audio Folder", "Audio Pipeline",
                                          lambda e, path: e.ingest_audios(path))

                elif choice == '3':
                    prompt = input("[Enter Search Query] > ").strip()
                    if prompt:
                        query = SearchQuery(prompt=prompt, limit=5)
                        results = engine.search_standard(query)
                        print_search_results(results, "STANDARD RAG")

                elif choice == '4':
                    prompt = input("[Enter High-Precision Query] > ").strip()
                    if prompt:
                        print("\n[SYSTEM] Running SLM Scrubber verification...")
                        query = SearchQuery(prompt=prompt, limit=5)
                        results = engine.search_verified(query)
                        print_search_results(results, "VERIFIED RETRIEVAL")

                elif choice == '5':
                    conversation_session_id = run_conversation(engine, conversation_session_id)

                elif choice == '7':
                    prompt = input("[Enter your research question] > ").strip()
                    if prompt:
                        run_reasoned_with_clarifications(engine, conversation_session_id, prompt)

                elif choice == '8':
                    run_research_report(engine, conversation_session_id)

                elif choice == '9':
                    engine = ingest_batch(engine, "Enter Absolute Path to Document / Book Folder",
                                          "Document Pipeline", lambda e, path: e.ingest_documents(path))

                elif choice == '10':
                    engine = ingest_batch(engine, "Enter Absolute Path to Image Folder", "Image Pipeline",
                                          lambda e, path: e.ingest_images(path))

                elif choice == '11':
                    code = choose_language(engine, "Answer in")
                    if code is not None:
                        engine = switch_language(engine, code)

                elif choice == '6' or choice.lower() in ['exit', 'quit']:
                    print("\n[SYSTEM] Initiating graceful shutdown sequence...")
                    break

                else:
                    print("[ERROR] Invalid selection.")

            except EngineError as e:
                print(f"\n[ENGINE ERROR] Task failed: {e}")
            except Exception as e:
                print(f"\n[CRITICAL FAILURE] Unexpected crash: {e}")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Emergency interrupt detected.")
    finally:
        engine.stop()
        print("[SYSTEM] Onyx Appliance is Offline.")


if __name__ == "__main__":
    main()