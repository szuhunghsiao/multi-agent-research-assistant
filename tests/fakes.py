"""Scripted fakes: each ainvoke() pops the next scripted reply. No network, no keys."""
from langchain_core.messages import AIMessage


class FakeLLM:
    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list = []

    async def ainvoke(self, input, **kwargs):
        self.calls.append(input)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)


def tool_call(query: str, id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "tavily_search", "args": {"query": query}, "id": id}])


class FakeSearch:
    def __init__(self, results=None):
        self.results = results or [{"title": "Source A", "content": "Fact A."}]
        self.calls: list = []

    async def ainvoke(self, args, **kwargs):
        self.calls.append(args)
        return {"query": args.get("query"), "results": self.results}