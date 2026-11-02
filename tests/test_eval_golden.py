"""Live eval against real Gemini + Tavily. Excluded from plain `pytest`; run with:

    pytest -m eval -s                         # smoke set (default)
    EVAL_MODE=full EVAL_CONCURRENCY=2 pytest -m eval -s
"""
import asyncio
import os

import pytest

from evals.checks import load_golden
from evals.runner import compare, load_baseline, run_eval, save, select

pytestmark = pytest.mark.eval


def test_golden_set_has_no_regression():
    from app.config import get_settings
    from app.graph import build_default_graph

    mode = os.getenv("EVAL_MODE", "smoke")
    concurrency = int(os.getenv("EVAL_CONCURRENCY", "1"))
    cases = select(load_golden(), smoke=(mode == "smoke"), ids=None, limit=None)

    results, summary = asyncio.run(run_eval(build_default_graph(), cases, concurrency,
                                            get_settings().job_timeout_s))
    print("\nresults ->", save(results, summary, mode))
    print(summary)

    baseline = load_baseline(mode)
    if baseline is None:
        pytest.skip(f"no [{mode}] baseline yet: python -m evals.runner "
                    f"{'--smoke ' if mode == 'smoke' else ''}--save-baseline")
    failures, warnings = compare(summary, baseline, len(cases))
    for w in warnings:
        print("WARN", w)
    assert not failures, "\n".join(failures)