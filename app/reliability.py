"""Per-call timeout + exponential backoff (full jitter) for LLM and search calls.

Wraps anything exposing `ainvoke()` so graph.py stays unaware of retry policy.
"""
import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from langchain_core.tools import ToolException

log = logging.getLogger(__name__)

RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")
_HTTP_CODE = re.compile(r"Error (\d{3})")


class RetryableError(Exception):
    """Transient failure detected by an adapter (the SDK did not classify it)."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    attempt_timeout_s: float = 30.0

    def backoff(self, attempt: int) -> float:
        """Full jitter: uniform(0, min(cap, base * 2**attempt))."""
        return random.uniform(0, min(self.max_delay_s, self.base_delay_s * 2 ** attempt))


def _text(exc: BaseException) -> str:
    return f"{exc} {exc.__cause__ or ''}"


def is_retryable(exc: BaseException) -> bool:
    if "PerDay" in _text(exc):          # daily quota: waiting seconds won't help
        return False
    if isinstance(exc, (TimeoutError, RetryableError, httpx.TransportError)):
        return True
    flag = getattr(exc, "is_retryable", None)   # langchain_core ModelError family
    if flag is not None:
        return bool(flag)
    return getattr(exc, "code", None) in RETRYABLE_HTTP


def server_retry_delay(exc: BaseException) -> float | None:
    """Gemini 429s carry RetryInfo, e.g. "retryDelay": "17s"."""
    m = _RETRY_DELAY.search(_text(exc))
    return float(m.group(1)) if m else None


class Resilient:
    def __init__(self, inner: Any, name: str, policy: RetryPolicy = RetryPolicy(),
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.inner, self.name, self.policy, self._sleep = inner, name, policy, sleep

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        p, t0 = self.policy, time.perf_counter()
        for attempt in range(1, p.max_attempts + 1):
            try:
                result = await asyncio.wait_for(self.inner.ainvoke(input, **kwargs),
                                                p.attempt_timeout_s)
            except Exception as e:
                hint = server_retry_delay(e)
                give_up = (attempt == p.max_attempts or not is_retryable(e)
                           or (hint is not None and hint > p.max_delay_s))
                if give_up:
                    log.warning("call_failed", extra={
                        "target": self.name, "attempts": attempt, "error": type(e).__name__,
                        "retryable": is_retryable(e), "duration_ms": _ms_since(t0)})
                    raise
                delay = max(p.backoff(attempt - 1), hint or 0.0)
                log.warning("call_retry", extra={
                    "target": self.name, "attempt": attempt, "error": type(e).__name__,
                    "sleep_s": round(delay, 2)})
                await self._sleep(delay)
            else:
                log.info("call_finished", extra={
                    "target": self.name, "attempts": attempt, "duration_ms": _ms_since(t0)})
                return result


def _ms_since(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class TavilyAdapter:
    """Normalize TavilySearch failure modes before they reach the retry wrapper.

    TavilySearch returns {"error": e} on HTTP errors (does not raise) and raises
    ToolException when a query has zero results.
    """

    def __init__(self, tool: Any):
        self.tool = tool

    async def ainvoke(self, args: dict, **kwargs: Any) -> dict:
        try:
            out = await self.tool.ainvoke(args, **kwargs)
        except ToolException as e:
            return {"query": args.get("query"), "results": [], "note": str(e)}
        if isinstance(out, dict) and "error" in out:
            msg = str(out["error"])
            m = _HTTP_CODE.search(msg)
            if m and int(m.group(1)) in RETRYABLE_HTTP:
                raise RetryableError(f"Tavily {msg}")
            raise RuntimeError(f"Tavily {msg}")
        return out