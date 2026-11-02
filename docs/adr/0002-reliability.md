# ADR 0002: One retry layer, client-side rate limiting, and a per-job deadline

- **Status:** Accepted
- **Date:** 2026-10-09 (updated 2026-10-30)

## Context

Outbound calls fail in four ways: hangs/timeouts, 429 rate limits, 5xx/"high demand" errors, and permanent errors such as 401. The first three are transient; the last is not.

Reading the libraries' source showed two traps:

1. `langchain-tavily` **returns** `{"error": e}` on HTTP errors instead of raising. The pipeline then continued with zero search results and produced a normal-looking report with no research behind it — a silent failure.
2. The same tool **raises** `ToolException` when a query has zero results, which failed the entire job for what is a normal outcome.

The Gemini SDK can retry on its own. Retrying in the SDK *and* in our code multiplies attempts (N per layer → N^layers in the worst case) against a 15 RPM quota. Worst case without an overall budget: ~14 calls × 4 attempts × 35 s ≈ 33 minutes holding one of two workers.

## Decision

- **`TavilyAdapter`** normalizes the tool's failure modes: 429/5xx → `RetryableError`; other HTTP errors → `RuntimeError`; zero results → an empty result set.
- **`Resilient`** wraps any object with `ainvoke()` (LLMs, bound LLMs, the search tool), so graph code is unchanged:
  - per-attempt timeout (`asyncio.wait_for`), set 5 s above the SDK timeout so the SDK's clearer error normally fires first;
  - retries **only** transient errors (timeouts, transport errors, 408/429/5xx, or `is_retryable` on LangChain model errors);
  - exponential backoff with **full jitter**, capped at 20 s;
  - honors a server `retryDelay` up to 60 s; fails fast on daily-quota errors (`PerDay`), since waiting seconds cannot help.
- **Retries live in exactly one layer:** the SDK is configured with `max_retries=1`, which in google-genai means one attempt in total.
- **Shared client-side `RateLimiter`** (sliding window, 14 starts/60 s) used by all three Gemini roles, because the quota is shared across keys. Every attempt, including retries, takes a slot.
- **Per-job deadline** (`JOB_TIMEOUT_S`, 300 s) with `asyncio.timeout()`. `.expired()` distinguishes "the job ran out of time" from "one call timed out after its retries", so errors are not misreported.

## Alternatives considered

- **Just raise the SDK's `max_retries`:** cannot handle Tavily's behavior, cannot be tested deterministically, and stacks with any other retry layer.
- **`tenacity`:** a solid choice; a ~40-line wrapper was preferred to keep the policy explicit and to inject `sleep`/`clock` for fast tests. The wrapper is the only place to change if switching.
- **Token bucket instead of sliding window:** smoother, but a sliding window maps directly onto the provider's "requests per minute" wording.

## Consequences

- Failures are loud: a bad Tavily key now fails the job with `RuntimeError: Tavily Error 401` instead of silently succeeding.
- In the eval, a 5xx on the Writer was retried after 0.83 s and the case passed.
- **Found by the eval:** a 429 asked for roughly 40 s, more than the 20 s backoff cap, and the call was abandoned after one attempt, failing the whole case. The fix separated "our own backoff cap" (20 s) from "how long we honor a server-specified delay" (60 s), and `server_delay_s` is now logged.
- The limiter is per process and allows bursts within its window, which may explain 429s despite the limiter. That is a hypothesis to confirm with the new log fields.
- The deadline is not propagated to individual calls, so a call may start even when too little time is left for it to finish.