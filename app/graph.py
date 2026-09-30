from typing import TypedDict, List
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
import json
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

DEFAULT_SYSTEM_PROMPT = SystemMessage(content=(
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

