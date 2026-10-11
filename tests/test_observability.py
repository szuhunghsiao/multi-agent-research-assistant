import io
import json
import logging

from app.jobs import JobManager
from app.observability import JsonFormatter, job_id_var


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