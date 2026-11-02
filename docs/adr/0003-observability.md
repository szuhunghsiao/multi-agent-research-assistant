# ADR 0003: Structured JSON events with `contextvars` for correlation

- **Status:** Accepted
- **Date:** 2026-10-16

## Context

Final job status alone cannot answer "why was this job slow?": which node, whether retries happened, whether Gemini or Tavily was the bottleneck. Several jobs run concurrently on one event loop, so log lines must be attributable to a job without threading a `job_id` through every function signature and into the graph.

## Decision

- **JSON lines on stdout** via a standard-library `logging.Formatter` (no extra dependency). Collection and storage are left to the platform (12-factor). Uvicorn's own logs use the same formatter.
- **Fixed event names** with typed fields rather than sentences: `job_queued`, `job_started`, `job_finished`, `node_finished`, `call_finished`, `call_retry`, `call_failed`, `rate_limit_wait`, `research_done`, `critic_verdict`.
- **`job_id` in a `ContextVar`**, set when a worker starts a job and reset afterwards. Each asyncio task gets its own copy of the context, so concurrent jobs do not mix.
- **Timing via wrappers:** `timed_node()` around each graph node and `Resilient` around each outbound call, measured with `time.perf_counter()` (monotonic) rather than wall-clock time.

## Alternatives considered

- **Plain-text logs:** readable, but analysis needs regexes that break when wording changes.
- **Passing `job_id` explicitly:** couples graph code to the job system and changes every signature.
- **`structlog`:** a good library; the standard library was sufficient and kept dependencies minimal.
- **OpenTelemetry tracing now:** the right long-term tool for cross-service traces; overkill for one process today.

## Consequences

- One `jq` query reconstructs a job's timeline. A measured job: Researcher 33%, Writer 40%, Critic 26%; per-node times sum to within 16 ms of the job total, so time goes to external calls, not orchestration.
- The **eval harness counts API calls from these same events**, so evaluation and monitoring share one source of truth.
- Logging the Critic's raw verdict line disproved a suspected parsing bug before any code was changed.
- `research_done` (`notes_chars`, `notes_source`) exposed that the Researcher's direct answer was being dropped whenever it skipped search; the fix is covered by a regression test.
- No metrics or traces yet. `duration_ms` fields are a ready source for Prometheus histograms or OpenTelemetry spans.