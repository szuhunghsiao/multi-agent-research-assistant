"""API tests: real FastAPI app + real JobManager + graph built from scripted fakes."""
import asyncio
import time

from fastapi.testclient import TestClient

from app.graph import build_graph
from app.main import create_app
from tests.fakes import FakeLLM, FakeSearch, tool_call


def fake_graph(critic_replies=("APPROVED\nGood.",), writer_error=None):
    writer = FakeLLM([writer_error] if writer_error else ["draft v1"])
    return build_graph(FakeLLM([tool_call("q"), "done"]), writer,
                       FakeLLM(list(critic_replies)), FakeSearch())


class BlockingGraph:
    """Never finishes -> lets us fill the queue deterministically."""
    async def ainvoke(self, state):
        await asyncio.Event().wait()


def wait_for(client, job_id, statuses, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/research/{job_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {statuses}, last={body['status']}")


def test_health():
    with TestClient(create_app(graph=fake_graph(), workers=3, max_queue=7)) as c:
        assert c.get("/health").json() == {"status": "ok", "workers": 3, "queue_depth": 0}


def test_submit_then_poll_until_succeeded():
    with TestClient(create_app(graph=fake_graph())) as c:
        r = c.post("/research", json={"topic": "humanoid robots"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert r.headers["location"] == f"/research/{job_id}"

        body = wait_for(c, job_id, {"succeeded", "failed"})
        assert body["status"] == "succeeded"
        assert body["result"]["draft"] == "draft v1"
        assert body["result"]["approved"] is True
        assert body["result"]["research_rounds"] == 1
        assert body["started_at"] and body["finished_at"]


def test_graph_exception_marks_job_failed():
    with TestClient(create_app(graph=fake_graph(writer_error=RuntimeError("boom")))) as c:
        job_id = c.post("/research", json={"topic": "robots"}).json()["job_id"]
        body = wait_for(c, job_id, {"succeeded", "failed"})
        assert body["status"] == "failed"
        assert body["error"] == "RuntimeError: boom"
        assert body["result"] is None


def test_validation_and_not_found():
    with TestClient(create_app(graph=fake_graph())) as c:
        assert c.post("/research", json={"topic": "x"}).status_code == 422
        assert c.post("/research", json={}).status_code == 422
        assert c.get("/research/does-not-exist").status_code == 404


def test_full_queue_returns_503_with_retry_after():
    with TestClient(create_app(graph=BlockingGraph(), workers=1, max_queue=1)) as c:
        first = c.post("/research", json={"topic": "first"}).json()["job_id"]
        wait_for(c, first, {"running"})               # worker busy forever
        assert c.post("/research", json={"topic": "second"}).status_code == 202  # fills queue
        r = c.post("/research", json={"topic": "third"})
        assert r.status_code == 503
        assert r.headers["retry-after"] == "30"