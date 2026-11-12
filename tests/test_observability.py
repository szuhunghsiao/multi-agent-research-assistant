import io
import json
import logging
import asyncio
import pytest

from langchain_core.exceptions import ModelRateLimitError

from app.graph import build_graph
from app.jobs import JobManager
from app.observability import JsonFormatter, job_id_var
from app.reliability import Resilient, RetryPolicy
from tests.fakes import FakeLLM, FakeSearch, tool_call

class ChattyGraph:
    """Logs from 'inside' the graph, like a node or a retry would."""
    async def ainvoke(self, state):
        logging.getLogger("app.graph").info("node_finished", extra={"node": "writer", "duration_ms": 12})
        return {**state, "draft": "d", "approved": True}


def capture():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    return buf, lambda: root.removeHandler(h)


def test_formatter_emits_json_with_extras():
    rec = logging.LogRecord("x", logging.INFO, "", 0, "hello", None, None)
    rec.duration_ms = 5
    out = json.loads(JsonFormatter().format(rec))
    assert out["event"] == "hello" and out["duration_ms"] == 5 and out["job_id"] is None


async def test_every_line_during_a_job_is_tagged_with_its_job_id():
    buf, detach = capture()
    try:
        m = JobManager(ChattyGraph(), workers=1)
        m.start()
        job = m.submit("topic")
        await m.queue.join()
        await m.stop()
    finally:
        detach()

    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    events = {l["event"]: l for l in lines}
    assert {"job_queued", "job_started", "node_finished", "job_finished"} <= events.keys()
    assert all(l["job_id"] == job.job_id for l in lines)
    assert events["job_finished"]["status"] == "succeeded"
    assert isinstance(events["job_finished"]["duration_ms"], int)
    assert job_id_var.get() is None   # context restored after the job

async def test_each_node_logs_its_duration(caplog):
    caplog.set_level(logging.INFO)
    graph = build_graph(FakeLLM([tool_call("q"), "done"]), FakeLLM(["v1", "v2"]),
                        FakeLLM(["REVISE\nmore", "APPROVED"]), FakeSearch())
    await graph.ainvoke({"topic": "t", "research_notes": "", "research_rounds": 0, "draft": "",
                         "critique": "", "revision_count": 0, "approved": False})

    nodes = [(r.node, r.revision) for r in caplog.records if r.getMessage() == "node_finished"]
    assert nodes == [("researcher", 0), ("writer", 0), ("critic", 0), ("writer", 0), ("critic", 1)]
    assert all(isinstance(r.duration_ms, int) for r in caplog.records if r.getMessage() == "node_finished")


async def test_retry_and_success_are_logged_with_target(caplog):
    caplog.set_level(logging.INFO)

    async def no_sleep(_):
        pass

    r = Resilient(FakeLLM([ModelRateLimitError("429"), "ok"]), "writer",
                  RetryPolicy(max_attempts=3), sleep=no_sleep)
    await r.ainvoke("x")

    events = [(rec.getMessage(), rec.target) for rec in caplog.records if hasattr(rec, "target")]
    assert events == [("call_retry", "writer"), ("call_finished", "writer")]
    assert [rec.attempts for rec in caplog.records if rec.getMessage() == "call_finished"] == [2]

async def test_critic_raw_verdict_line_is_logged(caplog):
    caplog.set_level(logging.INFO)
    graph = build_graph(FakeLLM(["done"]), FakeLLM(["v1"]),
                        FakeLLM(["**APPROVED**\nLooks good."]), FakeSearch(), max_revisions=0)
    await graph.ainvoke({"topic": "t", "research_notes": "", "research_rounds": 0, "draft": "",
                         "critique": "", "revision_count": 0, "approved": False})
    [rec] = [r for r in caplog.records if r.getMessage() == "critic_verdict"]
    assert rec.verdict_line == "**APPROVED**"

async def test_node_cut_off_by_deadline_is_logged_as_cancelled(caplog):
    caplog.set_level(logging.INFO)

    class SlowLLM:
        async def ainvoke(self, *_args, **_kwargs):
            await asyncio.sleep(10)

    graph = build_graph(FakeLLM(["done"]), SlowLLM(), FakeLLM(["APPROVED"]), FakeSearch())
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await graph.ainvoke({"topic": "t", "research_notes": "", "research_rounds": 0, "draft": "",
                                 "critique": "", "revision_count": 0, "approved": False})

    [rec] = [r for r in caplog.records if r.getMessage() == "node_finished" and r.node == "writer"]
    assert rec.status == "cancelled"