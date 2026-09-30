import json
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
import ast

class ResearchState(TypedDict):
  topic: str            # User input topic
  research_notes: str   # Now: Clean plain text after extration (from extract_research_notes)
                        # Pre:Researcher generate info clearn up
  research_rounds: int  # Adding Researcher actual search round (from coutn_research_rounds)
  draft: str            # Writer current draft
  critique: str         # Critic latest judge opinion
  revision_count: int   # Current correctness times, for controling loop ending
  approved: bool        # Critic passing or not

def initial_state(topic: str) -> ResearchState:
    return ResearchState(topic=topic, research_notes="", research_rounds=0, draft="",
                         critique="", revision_count=0, approved=False)

RESEARCHER_SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research assistant. You have a web_search tool that can "
    "check real-time data. If the information you have is enough to "
    "answer the user's question, answer directly with text -- do not "
    "call the tool again. Only call the tool again if the current "
    "information is clearly insufficient, conflicting, or missing key details."
))

def extract_text(response) -> str:
  """
    Dealing with Gemini return format (new version could be list of dict, or number)
  """
  content = response.content
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block,dict)
    )
  return str(content)

def extract_research_notes(messages: list) -> str:
    """
    Extract all ToolMessage's search result from researcher_loop returns message.
    Only keep title and content with plain text for Writer.
    """
    notes_parts = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        raw = msg.content
        # ToolMessage.content is Tavily return dict.
        try:
            data = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            # If extract fail, it will be kept as plain text
            notes_parts.append(str(raw))
            continue

        results = data.get("results", [])
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            notes_parts.append(f"[{title}] {content}")

    return "\n\n".join(notes_parts)

def count_research_rounds(messages: list) -> int:
    """
    Calculate the actual web_search tool calling count
    The value is same as ToolMessage's number
    The nubmer will be store in State["research_rounds"] for further usage.    
    """
    return sum(1 for msg in messages if isinstance(msg, ToolMessage))

def parse_verdict(text: str) -> tuple[bool, str]:
    """Critic contract: first line APPROVED or REVISE, rest is feedback."""
    first, _, rest = text.strip().partition("\n")
    return first.strip().upper().startswith("APPROVED"), rest.strip()

def build_graph(researcher_llm: Any, writer_llm: Any, critic_llm: Any, search: Any,
                max_iterations: int = 3, max_revisions: int = 3):
    """researcher_llm must already have the search tool bound (.bind_tools)."""

    async def researcher_loop(question: str) -> list:
        messages = [RESEARCHER_SYSTEM_PROMPT, HumanMessage(content=question)]
        for _ in range(max_iterations):
            response = await researcher_llm.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return messages
            for tc in response.tool_calls:
                result = await search.ainvoke(tc["args"])
                messages.append(ToolMessage(content=json.dumps(result, default=str),
                                            tool_call_id=tc["id"]))
        return messages

    async def researcher_node(state: ResearchState) -> dict:
        messages = await researcher_loop(state["topic"])
        return {"research_notes": extract_research_notes(messages),
                "research_rounds": count_research_rounds(messages)}

    async def writer_node(state: ResearchState) -> dict:
        if state.get("critique"):
            prompt = (
                f"This is the draft you wrote on the topic '{state['topic']}':\n{state['draft']}\n\n"
                f"The critic gave this feedback:\n{state['critique']}\n\n"
                "Revise the draft and output the complete revised draft, not just the changes."
            )
        else:
            prompt = (
                f"You are a professional writer. On the topic '{state['topic']}', write a clear, "
                f"structured 300-500 word first draft based on these notes:\n\n{state['research_notes']}"
            )
        response = await writer_llm.ainvoke(prompt)
        return {"draft": extract_text(response),
                "revision_count": state["revision_count"] + (1 if state.get("critique") else 0)}

    async def critic_node(state: ResearchState) -> dict:
        prompt = (
            f"You are a strict reviewer. Check this report draft on the topic '{state['topic']}':\n\n"
            f"{state['draft']}\n\n"
            "Check accuracy, clarity, and any obvious missing points or logic issues.\n"
            "Reply in this format. The first line must be exactly APPROVED or REVISE:\n"
            "APPROVED or REVISE\n"
            "<your feedback; if APPROVED, briefly explain why>"
        )
        approved, feedback = parse_verdict(extract_text(await critic_llm.ainvoke(prompt)))
        return {"approved": approved, "critique": feedback}

    def route_after_critic(state: ResearchState) -> str:
        if state["approved"] or state["revision_count"] >= max_revisions:
            return "end"
        return "revise"

    g = StateGraph(ResearchState)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"end": END, "revise": "writer"})
    return g.compile()


def build_default_graph():
    """Wire real Gemini + Tavily clients from Settings."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_tavily import TavilySearch

    from app.config import get_settings

    s = get_settings()

    def llm(key):
        return ChatGoogleGenerativeAI(model=s.model_name, google_api_key=key.get_secret_value(),
                                      timeout=s.llm_timeout_s, max_retries=1)

    search = TavilySearch(max_results=s.search_max_results,
                          tavily_api_key=s.tavily_api_key.get_secret_value())
    return build_graph(llm(s.researcher_api_key).bind_tools([search]),
                       llm(s.writer_api_key), llm(s.critic_api_key), search,
                       max_iterations=s.max_iterations, max_revisions=s.max_revisions)


if __name__ == "__main__":
    import asyncio
    import sys

    topic = " ".join(sys.argv[1:]) or "Latest developments in humanoid robotics 2026"
    final = asyncio.run(build_default_graph().ainvoke(initial_state(topic)))
    print("approved:", final["approved"], "| revision_count:", final["revision_count"],
          "| research_rounds:", final["research_rounds"])
    print(final["draft"][:500])