import threading

import pytest

from web_deploy.api.progress import job_listener, reporting_to
from web_deploy.api.runner import Runner, error_payload
from web_deploy.api.runs import MAX_PROGRESS_EVENTS, RUN_TTL_SECONDS, RunNotFound, RunStore, resume_point, sse_frame
from web_deploy.common.signing import Principal
from web_deploy.tests.api.fakes import FakeHost

ALICE = Principal("legal-demo", "alice", "member")


class TestRunStore:
    def test_a_run_collects_events_then_one_result(self):
        run = RunStore().create(ALICE, "chat")
        run.emit("progress", {"phase": "rag_retrieval"})
        run.finish({"ok": True})
        run.fail({"kind": "error"})  # after the result: ignored
        run.emit("progress", {})

        assert [(s, e) for s, e, _ in run.events] == [(1, "progress"), (2, "result")]
        assert run.after(1) == [(2, "result", {"ok": True})]

    def test_a_run_is_only_visible_to_its_user(self):
        store = RunStore()
        run = store.create(ALICE, "chat")
        assert store.get(run.id, ALICE) is run
        for other in (Principal("legal-demo", "bob", "member"), Principal("other-firm", "alice", "member")):
            with pytest.raises(RunNotFound):
                store.get(run.id, other)

    def test_finished_runs_expire(self):
        now = [0.0]
        store = RunStore(clock=lambda: now[0])
        run = store.create(ALICE, "chat")
        run.finished_at = 0.0
        now[0] = RUN_TTL_SECONDS + 1
        with pytest.raises(RunNotFound):
            store.get(run.id, ALICE)

    def test_progress_is_capped(self):
        run = RunStore().create(ALICE, "report:run")
        for _ in range(MAX_PROGRESS_EVENTS + 10):
            run.emit("progress", {})
        run.finish({})
        assert len(run.events) == MAX_PROGRESS_EVENTS + 1

    def test_active_lists_unfinished_runs_of_the_user(self):
        store = RunStore()
        a, b = store.create(ALICE, "chat"), store.create(ALICE, "report:run", subject="r1")
        a.finish({})
        assert [r.public() for r in store.active(ALICE)] == [{"id": b.id, "kind": "report:run", "done": False,
                                                              "subject": "r1"}]

    def test_sse_framing_and_resume_point(self):
        assert sse_frame(3, "progress", {"a": 1}) == 'id: 3\nevent: progress\ndata: {"a": 1}\n\n'
        assert (resume_point("7"), resume_point(None), resume_point("x")) == (7, 0, 0)


class TestProgressRouting:
    def test_ticks_reach_only_the_run_of_the_thread_that_made_them(self):
        store = RunStore()
        first, second = store.create(ALICE, "chat"), store.create(ALICE, "chat")
        barrier = threading.Barrier(2)

        def work(run, phase):
            with reporting_to(run):
                barrier.wait()
                job_listener(phase, "completed", None, "detail")

        threads = [threading.Thread(target=work, args=(r, p)) for r, p in ((first, "a"), (second, "b"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        job_listener("stray", "completed", None)

        assert [e[2]["phase"] for e in first.events] == ["a"]
        assert [e[2]["phase"] for e in second.events] == ["b"]


class TestRunner:
    def test_work_runs_in_the_requests_language_and_records_its_result(self, tmp_path):
        host = FakeHost(tmp_path)
        runner = Runner(host, 2, 1)
        run = RunStore().create(ALICE, "chat")
        runner.submit(run, lambda: {"language": host.current_language()}, "sr")
        for _ in range(100):
            if run.done:
                break
            threading.Event().wait(0.02)
        runner.shutdown()
        assert run.events[-1][1:] == ("result", {"language": "sr"})

    @pytest.mark.parametrize("name, kind", [("EngineBusyError", "busy"), ("EngineOfflineError", "offline"),
                                            ("InvalidPlan", "invalid"), ("RuntimeError", "error")])
    def test_errors_are_classified_by_class_name(self, name, kind):
        exc = type(name, (Exception,), {})("why")
        assert error_payload(exc) == {"kind": kind, "message": "why"}
