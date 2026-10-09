"""In-process job manager: bounded asyncio.Queue + N worker tasks + in-memory store.

Single-process only. To scale horizontally, swap the queue for Redis/SQS and the
store for Redis/Postgres; submit()/get() stay the same.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from app.graph import initial_state
from app.schemas import JobStatus, JobView, ResearchResult


class QueueFullError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    def __init__(self, graph: Any, workers: int = 2, max_queue: int = 20,
                 job_timeout_s: float = 300.0):
        self.graph = graph
        self.job_timeout_s = job_timeout_s
        self.jobs: dict[str, JobView] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue)
        self.n_workers = workers
        self._workers: list[asyncio.Task] = []

    def submit(self, topic: str) -> JobView:
        job = JobView(job_id=uuid.uuid4().hex, status=JobStatus.queued,
                      topic=topic, created_at=_now())
        try:
            self.queue.put_nowait(job.job_id)
        except asyncio.QueueFull:
            raise QueueFullError from None
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobView | None:
        return self.jobs.get(job_id)

    async def _run(self, job: JobView) -> None:
        job.status, job.started_at = JobStatus.running, _now()
        deadline = asyncio.timeout(self.job_timeout_s)   # whole-job budget
        try:
            async with deadline:
                final = await self.graph.ainvoke(initial_state(job.topic))
            job.result = ResearchResult(
                draft=final["draft"], approved=final["approved"],
                revision_count=final["revision_count"],
                research_rounds=final["research_rounds"], critique=final["critique"])
            job.status = JobStatus.succeeded
        except Exception as e:  # CancelledError is BaseException -> not swallowed
            if isinstance(e, TimeoutError) and deadline.expired():
                job.error = f"JobTimeout: exceeded {self.job_timeout_s:g}s"
            else:
                job.error = f"{type(e).__name__}: {e}"
            job.status = JobStatus.failed
        finally:
            job.finished_at = _now()

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._run(self.jobs[job_id])
            finally:
                self.queue.task_done()

    def start(self) -> None:
        self._workers = [asyncio.create_task(self._worker(), name=f"worker-{i}")
                         for i in range(self.n_workers)]

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)