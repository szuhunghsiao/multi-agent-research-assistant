import asyncio

import pytest
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from langchain_core.tools import ToolException

from app.reliability import Resilient, RetryableError, RetryPolicy, TavilyAdapter
from tests.fakes import FakeLLM

POLICY = RetryPolicy(max_attempts=3, base_delay_s=1.0, max_delay_s=20.0, attempt_timeout_s=0.1)


def make(replies, policy=POLICY):
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    inner = FakeLLM(replies)
    return Resilient(inner, "test", policy, sleep=fake_sleep), inner, slept


async def test_429_is_retried_then_succeeds():
    r, inner, slept = make([ModelRateLimitError("429"), "ok"])
    assert (await r.ainvoke("x")).content == "ok"
    assert len(inner.calls) == 2 and len(slept) == 1


async def test_non_retryable_fails_fast():
    r, inner, slept = make([ModelAuthenticationError("401 bad key")])
    with pytest.raises(ModelAuthenticationError):
        await r.ainvoke("x")
    assert len(inner.calls) == 1 and slept == []


async def test_gives_up_after_max_attempts():
    r, inner, slept = make([ModelRateLimitError("429")] * 5)
    with pytest.raises(ModelRateLimitError):
        await r.ainvoke("x")
    assert len(inner.calls) == 3 and len(slept) == 2


async def test_server_retry_delay_is_respected():
    r, _, slept = make([ModelRateLimitError('429 "retryDelay": "7s"'), "ok"])
    await r.ainvoke("x")
    assert slept[0] >= 7.0


async def test_daily_quota_is_not_retried():
    r, inner, _ = make([ModelRateLimitError("429 GenerateRequestsPerDayPerProjectPerModel")])
    with pytest.raises(ModelRateLimitError):
        await r.ainvoke("x")
    assert len(inner.calls) == 1


async def test_hung_call_times_out_and_is_retried():
    class Hangs:
        calls = 0
        async def ainvoke(self, _):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(10)
            return "ok"

    async def no_sleep(_):
        pass

    inner = Hangs()
    assert await Resilient(inner, "t", POLICY, sleep=no_sleep).ainvoke("x") == "ok"
    assert inner.calls == 2


class FakeTool:
    def __init__(self, outcome):
        self.outcome = outcome

    async def ainvoke(self, args, **_):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


async def test_tavily_429_becomes_retryable():
    with pytest.raises(RetryableError):
        await TavilyAdapter(FakeTool({"error": Exception("Error 429: Too Many Requests")})).ainvoke({"query": "q"})


async def test_tavily_401_is_not_retryable():
    with pytest.raises(RuntimeError):
        await TavilyAdapter(FakeTool({"error": Exception("Error 401: Unauthorized")})).ainvoke({"query": "q"})


async def test_tavily_no_results_is_empty_not_crash():
    out = await TavilyAdapter(FakeTool(ToolException("No search results found"))).ainvoke({"query": "q"})
    assert out["results"] == []