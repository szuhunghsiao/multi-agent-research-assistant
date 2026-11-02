"""Load the golden dataset and grade one pipeline output against its checks.

Pure functions only -- no LLM calls -- so the grader itself is unit-tested and free to run.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

GOLDEN_PATH = Path(__file__).with_name("golden.yaml")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    topic: str
    checks: dict[str, Any]
    smoke: bool = False
    notes: str = ""


# --- individual checks: (arg, draft, research_rounds) -> (passed, detail) ----------------

def _contains(draft: str, term: str) -> bool:
    return term.lower() in draft.lower()


def _word_count(arg, draft, _):
    lo, hi = arg
    n = len(draft.split())
    return lo <= n <= hi, f"{n} words (want {lo}-{hi})"


def _min_chars(arg, draft, _):
    n = len(draft.strip())
    return n >= arg, f"{n} chars (want >= {arg})"


def _include_all(arg, draft, _):
    missing = [t for t in arg if not _contains(draft, t)]
    return not missing, f"missing {missing}" if missing else "all present"


def _include_any(arg, draft, _):
    hits = [t for t in arg if _contains(draft, t)]
    return bool(hits), f"found {hits}" if hits else f"none of {arg}"


def _exclude(arg, draft, _):
    found = [t for t in arg if _contains(draft, t)]
    return not found, f"found forbidden {found}" if found else "clean"


def _min_research_rounds(arg, _, rounds):
    return rounds >= arg, f"{rounds} searches (want >= {arg})"


def _min_headings(arg, draft, _):
    n = len(_HEADING.findall(draft))
    return n >= arg, f"{n} headings (want >= {arg})"


CHECKS: dict[str, Callable[[Any, str, int], tuple[bool, str]]] = {
    "word_count": _word_count,
    "min_chars": _min_chars,
    "include_all": _include_all,
    "include_any": _include_any,
    "exclude": _exclude,
    "min_research_rounds": _min_research_rounds,
    "min_headings": _min_headings,
}


# --- loading -------------------------------------------------------------------------------

def load_golden(path: Path = GOLDEN_PATH) -> list[Case]:
    """Parse golden.yaml, merge defaults, and fail loudly on typos or duplicate ids."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = data.get("defaults") or {}
    cases, seen = [], set()
    for raw in data["cases"]:
        cid = raw["id"]
        if cid in seen:
            raise ValueError(f"duplicate case id: {cid}")
        seen.add(cid)
        merged = {**defaults, **(raw.get("checks") or {})}
        checks = {k: v for k, v in merged.items() if v is not None}   # null = disable default
        unknown = set(checks) - CHECKS.keys()
        if unknown:
            raise ValueError(f"case {cid}: unknown check(s) {sorted(unknown)}; "
                             f"known: {sorted(CHECKS)}")
        cases.append(Case(id=cid, category=raw["category"], topic=raw["topic"].strip(),
                          checks=checks, smoke=bool(raw.get("smoke")),
                          notes=raw.get("notes", "")))
    return cases


def grade(case: Case, draft: str, research_rounds: int) -> list[CheckResult]:
    return [CheckResult(name, *CHECKS[name](arg, draft, research_rounds))
            for name, arg in case.checks.items()]