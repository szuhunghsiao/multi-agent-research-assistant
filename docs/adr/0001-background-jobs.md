# ADR 0001: Background jobs with a bounded in-process queue

- **Status:** Accepted
- **Date:** 2026-10-04

## Context

A research job runs a Researcher → Writer → Critic graph with up to ~11 LLM calls and 3 searches. Measured jobs take 9–52 s. Running the graph inside the HTTP request would hit common 30–60 s proxy and load-balancer timeouts, hold connections open, and give clients no progress signal.

The binding constraint is not CPU but the Gemini free tier: **15 requests/minute, shared across all API keys** (quota is per project per model). One job alone can approach that rate, so unbounded concurrency would turn bursts of requests into bursts of 429s.

## Decision

- `POST /research` validates input, enqueues a `job_id`, and returns **`202 Accepted`** with a `Location: /research/{job_id}` header. Clients poll `GET /research/{job_id}`.
- Jobs go into a **bounded `asyncio.Queue`** (`MAX_QUEUE`, default 20) drained by a **fixed pool of worker tasks** (`WORKERS`, default 2). The worker count is effectively the knob for concurrent LLM pressure.
- When the queue is full, `POST` returns **`503` with `Retry-After`** (backpressure) instead of queueing without limit. `503` rather than `429` because the limit is global capacity, not a per-client quota.
- Job state lives in an in-memory store behind a `JobManager` that exposes only `submit()` and `get()`.
- One failing job never kills a worker: errors are recorded on the job (`status=failed`, `error=...`) and the worker continues.

## Alternatives considered

- **Synchronous request:** simplest, but breaks on long jobs (timeouts) and blocks the client.
- **FastAPI `BackgroundTasks`:** fire-and-forget with no concurrency limit, no handle to cancel, no backpressure. Intended for short side effects such as sending an email.
- **`asyncio.create_task` per request:** gives a handle, but still unbounded, and tasks must be referenced explicitly or they can be garbage-collected.
- **External queue now (Celery/arq + Redis, or SQS):** durable and horizontally scalable, but adds infrastructure that a single-instance, quota-bound service does not yet need.

## Consequences

- Concurrency against the LLM quota is bounded and configurable at deploy time (`WORKERS=1` in `docker-compose.yml`).
- Requests return in milliseconds. Measured: `/health` answered in 0.02 s while a job was running, confirming the event loop is not blocked.
- **Single process only:** a restart loses queued and running jobs; multiple instances would not share state; the job store grows without eviction.
- **Scale path:** queue → Redis/SQS with at-least-once delivery, job store → Redis/Postgres with TTL, stateless API instances, workers scaled separately. `main.py` would not change because it only depends on `submit()` and `get()`.
- Shutdown currently cancels in-flight jobs. A graceful drain (stop accepting work, finish or time out running jobs) is a known gap.