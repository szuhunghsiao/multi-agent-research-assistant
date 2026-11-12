"""UI client tests against a fake HTTP transport -- no server, no Gradio, no API keys."""
import httpx

from ui.client import run_research


def transport(post_response, get_statuses=()):
    statuses = list(get_statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return post_response
        status = statuses.pop(0) if statuses else "running"
        body = {"job_id": "abc12345xyz", "status": status, "result": None, "error": None}
        if status == "succeeded":
            body["result"] = {"draft": "# Report", "approved": True, "revision_count": 1,
                              "research_rounds": 2, "critique": "Good."}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


async def collect(t, **kw):
    return [u async for u in run_research("Kafka vs RabbitMQ", "http://api", poll_s=0, transport=t, **kw)]


async def test_submit_then_poll_until_succeeded():
    t = transport(httpx.Response(202, json={"job_id": "abc12345xyz", "status": "queued"}),
                  ["queued", "running", "running", "succeeded"])
    updates = await collect(t)
    assert [u.status for u in updates] == ["queued", "queued", "running", "running", "succeeded"]
    assert updates[-1].job["result"]["draft"] == "# Report"


async def test_queue_full_shows_retry_after():
    t = transport(httpx.Response(503, headers={"Retry-After": "30"}, json={"detail": "full"}))
    [u] = await collect(t)
    assert u.status == "error" and "30 s" in u.message


async def test_validation_error_is_explained():
    [u] = await collect(transport(httpx.Response(422, json={"detail": []})))
    assert u.status == "error" and "3–300" in u.message


async def test_api_unreachable():
    def boom(request):
        raise httpx.ConnectError("refused")
    [u] = await collect(httpx.MockTransport(boom))
    assert u.status == "error" and "Cannot reach the API" in u.message


async def test_gives_up_after_timeout_without_crashing():
    t = transport(httpx.Response(202, json={"job_id": "abc12345xyz", "status": "queued"}),
                  ["running"])
    updates = await collect(t, timeout_s=0.01)
    assert updates[-1].status == "error" and "may still finish" in updates[-1].message