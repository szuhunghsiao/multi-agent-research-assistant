"""Run the golden set through the real pipeline and report quality + latency + cost metrics.

    python -m evals.runner --smoke                 # 5 cases, ~35 Gemini calls
    python -m evals.runner --concurrency 2         # all cases
    python -m evals.runner --ids cap-theorem,attention  # specific cases
"""
import argparse
import asyncio
import json
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph import initial_state
from app.observability import job_id_var
from evals.checks import Case, grade, load_golden

RESULTS_DIR = Path(__file__).with_name("results")
BASELINE_PATH = Path(__file__).with_name("baseline.json")
LLM_TARGETS = {"researcher", "writer", "critic"}
VERDICT_WORDS = ("APPROVED", "REVISE")


class CallCounter(logging.Handler):
    """Count real API requests per case from the same `call_*` log events production emits.
    `attempts` includes retries, so this is what the quota actually saw."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.llm: Counter[str] = Counter()
        self.search: Counter[str] = Counter()
        self.retries: Counter[str] = Counter()
        self.verdicts: Counter[str] = Counter()
        self.bad_verdicts: Counter[str] = Counter()   # first line not exactly APPROVED/REVISE

    def emit(self, record: logging.LogRecord) -> None:
        cid = job_id_var.get()
        event, target = record.getMessage(), getattr(record, "target", None)
        if cid is not None and event == "critic_verdict":
            self.verdicts[cid] += 1
            if not record.verdict_line.strip().upper().startswith(VERDICT_WORDS):
                self.bad_verdicts[cid] += 1
            return
        if cid is None or target is None:
            return
        if event in ("call_finished", "call_failed"):
            bucket = self.llm if target in LLM_TARGETS else self.search
            bucket[cid] += record.attempts
        elif event == "call_retry":
            self.retries[cid] += 1


@dataclass
class CaseResult:
    id: str
    category: str
    status: str                      # "ok" | "error"
    passed: bool
    latency_s: float
    approved: bool | None = None
    revision_count: int | None = None
    research_rounds: int | None = None
    llm_calls: int = 0
    search_calls: int = 0
    retries: int = 0
    critic_verdicts: int = 0
    critic_format_violations: int = 0
    checks: list[dict] = field(default_factory=list)
    error: str | None = None
    draft: str = ""


async def run_case(graph: Any, case: Case, sem: asyncio.Semaphore, counter: CallCounter,
                   timeout_s: float) -> CaseResult:
    async with sem:
        token = job_id_var.set(case.id)         # tags every log line -> CallCounter key
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                final = await graph.ainvoke(initial_state(case.topic))
            results = grade(case, final["draft"], final["research_rounds"])
            res = CaseResult(
                id=case.id, category=case.category, status="ok",
                passed=all(r.passed for r in results), latency_s=time.perf_counter() - t0,
                approved=final["approved"], revision_count=final["revision_count"],
                research_rounds=final["research_rounds"],
                checks=[asdict(r) for r in results], draft=final["draft"])
        except Exception as e:
            res = CaseResult(id=case.id, category=case.category, status="error", passed=False,
                             latency_s=time.perf_counter() - t0,
                             error=f"{type(e).__name__}: {e}")
        finally:
            job_id_var.reset(token)
        res.llm_calls, res.search_calls = counter.llm[case.id], counter.search[case.id]
        res.retries = counter.retries[case.id]
        res.critic_verdicts = counter.verdicts[case.id]
        res.critic_format_violations = counter.bad_verdicts[case.id]
        mark = "PASS" if res.passed else ("ERROR" if res.status == "error" else "FAIL")
        print(f"  {mark:5} {case.id:22} {res.latency_s:5.1f}s  llm={res.llm_calls}", flush=True)
        return res


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (no interpolation): honest for small samples."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def summarize(results: list[CaseResult]) -> dict:
    ok = [r for r in results if r.status == "ok"]
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r.passed)
    lat = [r.latency_s for r in ok]
    failed_checks = Counter(c["name"] for r in ok for c in r.checks if not c["passed"])
    verdicts = sum(r.critic_verdicts for r in results)
    return {
        "cases": len(results),
        "errors": len(results) - len(ok),
        "pass_rate": round(sum(r.passed for r in results) / len(results), 3),
        "critic_approval_rate": round(sum(bool(r.approved) for r in ok) / len(ok), 3) if ok else None,
        "avg_revisions": mean([r.revision_count for r in ok]),
        "revisions_histogram": dict(sorted(Counter(r.revision_count for r in ok).items())),
        "critic_format_violation_rate": (round(sum(r.critic_format_violations for r in results)
                                               / verdicts, 3) if verdicts else None),
        "avg_research_rounds": mean([r.research_rounds for r in ok]),
        "avg_llm_calls": mean([r.llm_calls for r in results]),
        "total_llm_calls": sum(r.llm_calls for r in results),
        "total_retries": sum(r.retries for r in results),
        "latency_p50_s": round(percentile(lat, 50), 1) if lat else None,
        "latency_p95_s": round(percentile(lat, 95), 1) if lat else None,
        "pass_rate_by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "failed_checks": dict(failed_checks),
    }


async def run_eval(graph: Any, cases: list[Case], concurrency: int = 1,
                   timeout_s: float = 300.0) -> tuple[list[CaseResult], dict]:
    counter = CallCounter()
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(min(old_level or logging.WARNING, logging.INFO))  # INFO events must reach counter
    root.addHandler(counter)
    try:
        sem = asyncio.Semaphore(concurrency)
        results = await asyncio.gather(*(run_case(graph, c, sem, counter, timeout_s) for c in cases))
    finally:
        root.removeHandler(counter)
        root.setLevel(old_level)
    return list(results), summarize(list(results))


def save(results: list[CaseResult], summary: dict, tag: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{stamp}_{tag}.json"
    path.write_text(json.dumps({"summary": summary, "results": [asdict(r) for r in results]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_baseline(mode: str, path: Path = BASELINE_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(mode)


def save_baseline(mode: str, summary: dict, path: Path = BASELINE_PATH) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[mode] = summary
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compare(current: dict, baseline: dict, n_cases: int) -> tuple[list[str], list[str]]:
    """Regression gate. Tolerances are sized to the noise of a small, non-deterministic eval:
    one case flipping is allowed; cost may grow 30%; latency only warns (network-dependent)."""
    failures, warnings = [], []
    if current["errors"] > 0:
        failures.append(f"{current['errors']} case(s) errored")
    allowed_drop = 1 / n_cases
    if current["pass_rate"] < baseline["pass_rate"] - allowed_drop - 1e-9:
        failures.append(f"pass_rate {current['pass_rate']} < baseline {baseline['pass_rate']} "
                        f"- {allowed_drop:.3f} (one case)")
    if baseline.get("avg_llm_calls") and current["avg_llm_calls"] > baseline["avg_llm_calls"] * 1.3:
        failures.append(f"avg_llm_calls {current['avg_llm_calls']} > 1.3x baseline "
                        f"{baseline['avg_llm_calls']}")
    if (baseline.get("latency_p95_s") and current.get("latency_p95_s")
            and current["latency_p95_s"] > baseline["latency_p95_s"] * 1.5):
        warnings.append(f"latency_p95_s {current['latency_p95_s']} > 1.5x baseline "
                        f"{baseline['latency_p95_s']}")
    return failures, warnings


def select(cases: list[Case], smoke: bool, ids: str | None, limit: int | None) -> list[Case]:
    if ids:
        wanted = set(ids.split(","))
        cases = [c for c in cases if c.id in wanted]
    elif smoke:
        cases = [c for c in cases if c.smoke]
    return cases[:limit] if limit else cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="only cases marked smoke: true")
    ap.add_argument("--ids", help="comma-separated case ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--save-baseline", action="store_true",
                    help="record this run's summary as the baseline for its mode")
    args = ap.parse_args()

    from app.config import get_settings
    from app.graph import build_default_graph
    from app.observability import configure_logging

    s = get_settings()
    configure_logging("INFO")
    logging.getLogger().handlers[0].setLevel(logging.WARNING)  # console: retries/failures only
    cases = select(load_golden(), args.smoke, args.ids, args.limit)
    print(f"Running {len(cases)} case(s), concurrency={args.concurrency}")
    results, summary = asyncio.run(run_eval(build_default_graph(), cases, args.concurrency,
                                            s.job_timeout_s))
    mode = "smoke" if args.smoke else "full"
    partial = bool(args.ids or args.limit)       # ad-hoc subsets are never compared/baselined
    path = save(results, summary, "adhoc" if partial else mode)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved -> {path}")
    if partial:
        print("(partial run: baseline not compared)")
    elif args.save_baseline:
        if summary["errors"]:
            print(f"Refusing to save baseline: {summary['errors']} case(s) errored. "
                  "A baseline must come from a clean run.")
        else:
            save_baseline(mode, summary)
            print(f"Baseline [{mode}] updated -> {BASELINE_PATH}")
    elif (baseline := load_baseline(mode)) is not None:
        failures, warnings = compare(summary, baseline, len(cases))
        for w in warnings:
            print("WARN ", w)
        print("REGRESSION:\n  " + "\n  ".join(failures) if failures else "No regression vs baseline")


if __name__ == "__main__":
    main()