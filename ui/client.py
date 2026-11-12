"""Thin async client for the research API: submit a topic, then poll until the job finishes.

No UI code here, so the polling logic can be unit-tested without Gradio.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

TERMINAL = {"succeeded", "failed"}


@dataclass
class Update:
    status: str              # queued | running | succeeded | failed | error
    message: str
    elapsed_s: float
    job: dict | None = None


async def run_research(topic: str, base_url: str, poll_s: float = 2.0, timeout_s: float = 360.0,
                       transport: httpx.AsyncBaseTransport | None = None) -> AsyncIterator[Update]:
    t0 = time.monotonic()
    elapsed = lambda: round(time.monotonic() - t0, 1)

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0, transport=transport) as client:
        try:
            r = await client.post("/research", json={"topic": topic})
        except httpx.HTTPError as e:
            yield Update("error", f"Cannot reach the API at {base_url} ({type(e).__name__}).", elapsed())
            return
        if r.status_code == 422:
            yield Update("error", "Topic must be 3–300 characters.", elapsed())
            return
        if r.status_code == 503:
            retry = r.headers.get("retry-after", "a few")
            yield Update("error", f"Server is busy (job queue full). Try again in {retry} s.", elapsed())
            return
        if r.status_code != 202:
            yield Update("error", f"Unexpected response from API: HTTP {r.status_code}.", elapsed())
            return

        job_id = r.json()["job_id"]
        yield Update("queued", f"Job {job_id[:8]} queued.", elapsed())

        while time.monotonic() - t0 < timeout_s:
            await asyncio.sleep(poll_s)
            try:
                job = (await client.get(f"/research/{job_id}")).json()
            except httpx.HTTPError as e:
                yield Update("error", f"Lost connection while polling ({type(e).__name__}).", elapsed())
                return
            if job["status"] in TERMINAL:
                yield Update(job["status"], f"Job {job_id[:8]} {job['status']}.", elapsed(), job)
                return
            yield Update(job["status"], f"Job {job_id[:8]} {job['status']}…", elapsed(), job)

        yield Update("error", f"Stopped waiting after {timeout_s:.0f} s; job {job_id[:8]} may still "
                              "finish on the server.", elapsed())