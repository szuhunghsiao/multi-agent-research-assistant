import pytest

from app.graph import build_graph, initial_state, parse_verdict
from tests.fakes import FakeLLM, FakeSearch, tool_call


def make(researcher, writer, critic, search=None, **kw):
    return build_graph(researcher, writer, critic, search or FakeSearch(), **kw)


@pytest.mark.parametrize("text,approved,feedback", [
    ("APPROVED\nGood.", True, "Good."),
    ("  approved: fine", True, ""),
    ("REVISE\nAdd sources.", False, "Add sources."),
    ("Looks good overall", False, ""),   # contract violation -> treated as REVISE
])
def test_parse_verdict(text, approved, feedback):
    assert parse_verdict(text) == (approved, feedback)


async def test_happy_path_one_search_approved_first_try():
    search = FakeSearch()
    researcher = FakeLLM([tool_call("robots"), "enough info"])
    writer = FakeLLM(["draft v1"])
    critic = FakeLLM(["APPROVED\nSolid."])

    final = await make(researcher, writer, critic, search).ainvoke(initial_state("robots"))

    assert final["approved"] is True
    assert final["revision_count"] == 0
    assert final["research_rounds"] == 1
    assert "[Source A] Fact A." in final["research_notes"]
    assert final["draft"] == "draft v1"
    assert search.calls == [{"query": "robots"}]


async def test_revise_loop_feeds_critique_and_draft_back_to_writer():
    writer = FakeLLM(["draft v1", "draft v2"])
    critic = FakeLLM(["REVISE\nCite a source.", "APPROVED\nNow good."])

    final = await make(FakeLLM(["no search needed"]), writer, critic).ainvoke(initial_state("t"))

    assert final["approved"] is True
    assert final["revision_count"] == 1
    assert final["draft"] == "draft v2"
    assert "Cite a source." in writer.calls[1] and "draft v1" in writer.calls[1]


async def test_max_revisions_caps_the_loop():
    writer = FakeLLM([f"draft v{i}" for i in range(1, 5)])
    critic = FakeLLM(["REVISE\nno"] * 4)

    final = await make(FakeLLM(["x"]), writer, critic, max_revisions=2).ainvoke(initial_state("t"))

    assert final["approved"] is False
    assert final["revision_count"] == 2
    assert len(writer.calls) == 3 and len(critic.calls) == 3


async def test_researcher_stops_at_max_iterations():
    search = FakeSearch()
    researcher = FakeLLM([tool_call("q", id=f"c{i}") for i in range(5)])

    final = await make(researcher, FakeLLM(["d"]), FakeLLM(["APPROVED"]), search,
                       max_iterations=2).ainvoke(initial_state("t"))

    assert final["research_rounds"] == 2
    assert len(researcher.calls) == 2