import asyncio
import logging

from evals.checks import Case
from evals.runner import percentile, run_eval


class FakeGraph:
    """Emits the same call_* events as Resilient, then returns a canned final state."""

    def __init__(self, drafts: dict[str, str], fail: set[str] = frozenset()):
        self.drafts, self.fail = drafts, fail
        self.active = self.max_active = 0

    async def ainvoke(self, state):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            log = logging.getLogger("app.reliability")
            log.info("call_finished", extra={"target": "researcher", "attempts": 1})
            log.info("call_finished", extra={"target": "tavily", "attempts": 1})
            log.warning("call_retry", extra={"target": "writer", "attempt": 1})
            log.info("call_finished", extra={"target": "writer", "attempts": 2})
            if state["topic"] in self.fail:
                raise RuntimeError("boom")
            return {**state, "draft": self.drafts[state["topic"]], "approved": True,
                    "revision_count": 1, "research_rounds": 1}
        finally:
            self.active -= 1


def case(cid, topic, **checks):
    return Case(id=cid, category=cid[0], topic=topic, checks=checks)


async def test_metrics_counts_and_errors():
    cases = [case("a1", "A", include_all=["alpha"]),
             case("b1", "B", include_all=["beta"]),
             case("c1", "C")]
    graph = FakeGraph({"A": "alpha text", "B": "no match"}, fail={"C"})

    results, s = await run_eval(graph, cases, concurrency=2)

    by_id = {r.id: r for r in results}
    assert by_id["a1"].passed and not by_id["b1"].passed
    assert by_id["c1"].status == "error" and "boom" in by_id["c1"].error
    assert by_id["a1"].llm_calls == 3          # researcher 1 + writer 2 (retry counted)
    assert by_id["a1"].search_calls == 1 and by_id["a1"].retries == 1
    assert s["cases"] == 3 and s["errors"] == 1
    assert s["pass_rate"] == round(1 / 3, 3)
    assert s["critic_approval_rate"] == 1.0
    assert s["failed_checks"] == {"include_all": 1}
    assert graph.max_active <= 2               # concurrency respected


async def test_concurrency_one_runs_serially():
    cases = [case(f"x{i}", f"T{i}") for i in range(4)]
    graph = FakeGraph({f"T{i}": "d" for i in range(4)})
    await run_eval(graph, cases, concurrency=1)
    assert graph.max_active == 1


def test_percentile_nearest_rank():
    xs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(xs, 50) == 50
    assert percentile(xs, 95) == 100
    assert percentile([], 50) is None