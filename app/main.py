"""FastAPI service: submit research jobs (202 + job_id), poll status, health check."""
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response

from app.jobs import JobManager, QueueFullError
from app.schemas import JobCreated, JobView, ResearchRequest


def create_app(graph: Any = None, workers: int | None = None,
               max_queue: int | None = None, job_timeout_s: float | None = None) -> FastAPI:
    """App factory. Tests pass a fake graph; production builds the real one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        g, w, q, t = graph, workers, max_queue, job_timeout_s
        if g is None:
            from app.config import get_settings
            from app.graph import build_default_graph
            s = get_settings()
            g, w, q, t = (build_default_graph(), w or s.workers, q or s.max_queue,
                          t or s.job_timeout_s)
        app.state.jobs = JobManager(g, workers=w or 2, max_queue=q or 20,
                                    job_timeout_s=t or 300.0)
        app.state.jobs.start()
        yield
        await app.state.jobs.stop()

    app = FastAPI(title="Multi-Agent Research Assistant", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict:
        jobs: JobManager = request.app.state.jobs
        return {"status": "ok", "workers": jobs.n_workers, "queue_depth": jobs.queue.qsize()}

    @app.post("/research", status_code=202, response_model=JobCreated)
    async def submit_research(body: ResearchRequest, request: Request, response: Response):
        try:
            job = request.app.state.jobs.submit(body.topic)
        except QueueFullError:
            raise HTTPException(503, "Job queue is full, retry later",
                                headers={"Retry-After": "30"}) from None
        response.headers["Location"] = f"/research/{job.job_id}"
        return JobCreated(job_id=job.job_id, status=job.status)

    @app.get("/research/{job_id}", response_model=JobView)
    async def get_research(job_id: str, request: Request):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    return app


app = create_app()