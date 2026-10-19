import pytest

from evals.checks import Case, grade, load_golden


def case(**checks):
    return Case(id="t", category="x", topic="t", checks=checks)


def by_name(results):
    return {r.name: r for r in results}


DRAFT = """# Title
## Section
PostgreSQL is relational. MongoDB stores documents. However, it depends.
"""


def test_include_all_and_any_are_case_insensitive():
    r = by_name(grade(case(include_all=["postgresql", "MONGODB"], include_any=["nope", "Depends"]), DRAFT, 0))
    assert r["include_all"].passed and r["include_any"].passed


def test_include_all_reports_what_is_missing():
    r = by_name(grade(case(include_all=["PostgreSQL", "Redis"]), DRAFT, 0))
    assert not r["include_all"].passed and "Redis" in r["include_all"].detail


def test_exclude_word_count_headings_rounds():
    r = by_name(grade(case(exclude=["As an AI"], word_count=[5, 50], min_headings=2,
                           min_research_rounds=1), DRAFT, 0))
    assert r["exclude"].passed
    assert r["word_count"].passed
    assert r["min_headings"].passed
    assert not r["min_research_rounds"].passed and "0 searches" in r["min_research_rounds"].detail


def test_min_chars_for_chinese():
    assert grade(case(min_chars=5), "台灣電動機車", 0)[0].passed
    assert not grade(case(min_chars=50), "台灣電動機車", 0)[0].passed


def test_defaults_merge_and_null_disables(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("""
defaults:
  word_count: [200, 900]
  exclude: ["As an AI"]
cases:
  - id: a
    category: x
    topic: "A"
  - id: b
    category: x
    topic: "B"
    checks:
      word_count: null
      min_chars: 10
""", encoding="utf-8")
    a, b = load_golden(p)
    assert a.checks == {"word_count": [200, 900], "exclude": ["As an AI"]}
    assert b.checks == {"exclude": ["As an AI"], "min_chars": 10}


def test_typo_in_check_name_fails_loudly(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("cases:\n  - id: a\n    category: x\n    topic: A\n    checks:\n      inculde_any: [x]\n")
    with pytest.raises(ValueError, match="unknown check"):
        load_golden(p)


def test_duplicate_id_fails_loudly(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("cases:\n  - {id: a, category: x, topic: A}\n  - {id: a, category: x, topic: B}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(p)


def test_real_golden_file_is_valid():
    cases = load_golden()
    assert len(cases) >= 15
    assert sum(c.smoke for c in cases) == 5