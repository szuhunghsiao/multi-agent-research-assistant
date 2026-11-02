# ADR 0004: A deterministic, behavior-based evaluation harness with a noise-aware regression gate

- **Status:** Accepted
- **Date:** 2026-10-19 (updated 2026-10-30)

## Context

Unit tests with fake LLMs verify orchestration, not output quality. Prompt or model changes need a repeatable way to tell "better" from "worse". Constraints: LLM output is non-deterministic, and the free tier allows **500 requests/day**. A full run of 18 cases costs roughly 130–200 calls.

## Decision

- **Golden dataset (`evals/golden.yaml`)** of 18 cases grouped by **behavior**, not topic: current events that must search, evergreen explanations, comparisons, practical guidance, edge inputs (one ambiguous word, Chinese, a non-existent term), and robustness (prompt injection, a long topic, a contested topic). A 5-case **smoke** subset covers each group.
- **Deterministic checks only:** word/character counts, required and forbidden terms, minimum searches, minimum headings. No checks on fast-changing facts. Every check returns a reason (e.g. `missing ['MongoDB']`).
- **The grader is unit-tested** and **fails loudly** on unknown check names or duplicate ids, because a silently ignored misspelled check would inflate the pass rate.
- **Runner metrics:** pass rate (overall, per category, and most-failed checks), Critic approval rate, revision histogram, Critic format-violation rate, average LLM calls including retries (cost), and p50/p95 latency (nearest-rank). Concurrency is bounded with a semaphore; the shared rate limiter still applies.
- **Regression gate (`pytest -m eval`)** against a **baseline committed to git**. Any errored case fails. Pass rate may drop by at most one case. Average LLM calls may grow at most 30%. p95 latency only warns. Baselines from runs with errors are refused. The eval is excluded from plain `pytest`.

## Alternatives considered

- **LLM-as-judge:** captures subtler quality, but costs 1–2 extra calls per case, is itself non-deterministic, and would need its own validation (e.g. agreement with human labels). A candidate next step on a sample.
- **Exact or snapshot matching:** meaningless for generated text.
- **Per-case pass/fail assertions:** flaky; one stochastic flip would fail CI.

## Consequences

- Findings so far:
  - The same input took 1 or 3 revision rounds, so cost can nearly double. A suspected verdict-parsing bug was ruled out by logging raw verdicts (0% format violations); the cause is the Critic's non-determinism.
  - The Critic approved a vague current-events report that named no companies and searched only once. The check was kept rather than loosened, since loosening it would be moving the goalposts.
  - A real bug in retry handling of long server delays (see ADR 0002).
  - The first baseline was polluted by an errored case, which led to the "no baselines with errors" rule.
- Rule-based checks catch clear failures, not subtle quality differences.
- 18 cases (5 in smoke) are enough to detect regressions, not to compare close variants: one case is 5.6% (20% in smoke). The baseline comes from a single run; running several times and taking the median would reduce noise at extra quota cost.
- Risk of overfitting prompts to the golden set; mitigate with held-out cases and by adding real failure cases over time.