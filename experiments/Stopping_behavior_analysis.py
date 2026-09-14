"""
Experiment: Researcher tool-calling stopping behavior analysis
================================================================

GOAL
----
Determine whether the Researcher agent (Gemini 3.5 Flash-Lite bound to a
web_search tool) can reliably decide, on its own, when it has "enough"
information and should stop calling the tool. This determines whether we
can rely purely on LLM self-judgment (no explicit stopping logic), or
whether we need an engineering safety net (max_iterations cap).

METHODOLOGY NOTE
-----------------
Early trials changed multiple variables at once (language, phrasing, units)
across single runs, which made it impossible to separate "does system
prompt help" from "is this just LLM sampling randomness". This was
corrected in Trial 3 by holding the question fixed and running 5 repeated
trials per condition (with / without system prompt) to average out
sampling noise -- same logic as cross-validation when evaluating a model,
rather than trusting a single train/test split.


TRIAL 1 -- No system prompt, single run
-----------------------------------------
messages = [HumanMessage(content="What's the weather in San Jose today?")]
response = llm_with_tools.invoke(messages)

OUTPUT:
    Tool calls: [{'name': 'web_search', 'args': {'query': 'weather in San Jose today'}, ...}]
    Content: []
# LLM correctly triggers the tool for a query needing live data. This just
# validates the tool-calling mechanism itself works (see Step 1).


TRIAL 2 -- No system prompt, feed back a fake tool result, check round 2
---------------------------------------------------------------------------
messages = [HumanMessage(content="What's the weather in San Jose today?")]
# ... round 1 tool_call happens, then a fake result is fed back:
fake_tool_result = "San Jose current temperature 21.9 C, Sunny, Humidity: 36%"

OUTPUT:
    Round 1 tool_calls: [{'name': 'web_search', 'args': {'query': 'weather in San Jose today'}, ...}]
    Round 2 tool_calls: [{'name': 'web_search', 'args': {'query': 'San Jose CA weather today forecast'}, ...}]
    Round 2 content: []
# Model re-queries with a reworded query even though the fake result already
# answers the question. First evidence that pure self-judgment is unreliable.


TRIAL 3 -- Add explicit SystemMessage instructing when to stop, single run
------------------------------------------------------------------------------
system_prompt = SystemMessage(content=(
    "You are a research assistant, you have a web_search tool can check "
    "real time data. If the data you have is enough to answer user's "
    "question, answer the user's prompt directly with text, don't need "
    "to call the tool again. If the information is clearly not enough, "
    "has conflict, or missing some key points, call the tool and search again."
))
messages = [system_prompt, HumanMessage(content="San Jose temperature today in C")]
# ... same fake-result feedback as Trial 2

OUTPUT (varied across ad-hoc manual runs with different phrasing):
    - "San Jose temperature today in C"  -> Round 2: STOPPED
    - "San Jose temperature today in F"  -> Round 2: queried again
    - Original Chinese phrasing          -> Round 2: queried again
# Results looked inconsistent across manual single-shot tests. Could not
# tell whether the system prompt was helping, or whether the differences
# were driven by phrasing/language, or just sampling noise. This motivated
# the controlled experiment below.


TRIAL 4 -- Controlled experiment: fixed question, 5 repeated trials per condition
--------------------------------------------------------------------------------------
def run_trial(use_system_prompt: bool, question: str):
    msgs = [system_prompt] if use_system_prompt else []
    msgs.append(HumanMessage(content=question))
    r1 = llm_with_tools.invoke(msgs)
    if not r1.tool_calls:
        return "no_tool_call_round1"
    msgs.append(r1)
    tool_call_id = r1.tool_calls[0]["id"]
    fake_result = "San Jose current temperature 21.9 C, Sunny, Humidity: 36%"
    msgs.append(ToolMessage(content=fake_result, tool_call_id=tool_call_id))
    r2 = llm_with_tools.invoke(msgs)
    return "stopped" if not r2.tool_calls else "queried_again"

question = "San Jose temperature today in C"
# 5x with system prompt, 5x without, same fixed question

OUTPUT:
    With system prompt (5 trials):    stopped, stopped, stopped, stopped, stopped
    Without system prompt (5 trials): stopped, stopped, stopped, stopped, stopped

# CONCLUSION: Both conditions are 100% stable for this specific question.
# The system prompt had zero measurable effect here -- the question itself
# (specific units, narrow scope) was already easy for the model to judge as
# "answered", regardless of the system prompt. This means the system prompt
# is NOT the dominant factor in stopping behavior; question specificity (and
# possibly other factors -- see Trial 5) matters more.
#
# This also means the single-run "queried_again" result for the same
# question with "in F" phrasing (from Trial 3) cannot be trusted as a
# real finding -- n=1 is not enough to draw conclusions, consistent with
# how a single test case shouldn't be used to judge a model's behavior.


TRIAL 5 -- Multi-round loop with REAL Tavily search results (not fake string)
------------------------------------------------------------------------------
def run_multi_round(question, use_system_prompt=True, max_safety=8):
    msgs = [system_prompt] if use_system_prompt else []
    msgs.append(HumanMessage(content=question))
    round_num = 0
    while round_num < max_safety:
        round_num += 1
        response = llm_with_tools.invoke(msgs)
        msgs.append(response)
        if not response.tool_calls:
            return round_num  # stopped
        for tc in response.tool_calls:
            result = web_search.invoke(tc["args"])  # REAL Tavily call
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return round_num  # hit safety limit

run_multi_round("What's the weather in San Jose today?")   # open-ended, Chinese in orig test
run_multi_round("San Jose temperature today in C")          # specific

OUTPUT:
    Open-ended question:
        Round 1: called with query = "San Jose weather today"
        Round 2: STOPPED (no tool call)

    Specific question:
        Round 1: called with query = "San Jose temperature today Celsius"
        Round 2: STOPPED (no tool call)

# Both stopped at round 2 with REAL search results, including the
# open-ended question that previously triggered re-querying when fed a
# FAKE result (Trial 2). This surfaces a variable we hadn't controlled for:
# the richness/quality of the tool result itself. Tavily's real response
# (multiple sources, relevance scores, fuller content) appears to give the
# model more confidence to stop than a single thin, hand-written fake
# string did. This is a plausible explanation, not a proven causal claim --
# sample size here is still n=1 per condition.


FINAL CONCLUSIONS
------------------
1. LLM self-judgment on "is this enough?" is NOT reliably stable across
   conditions we tested (system prompt presence, phrasing, mock vs real
   data all produced different outcomes at various points).
2. The clearest controlled result (Trial 4) shows system prompt wording
   alone does not reliably change stopping behavior for this model size
   (Gemini 3.5 Flash-Lite).
3. Tool result quality/richness (Trial 5) appears to be an underexplored
   but possibly significant factor -- worth flagging as a system fragility:
   if a lower-quality search backend is swapped in later, stopping behavior
   may become less stable again.
4. Because we cannot guarantee every downstream query (from Writer, or a
   real user) will land in the "easy to judge" regime, self-judgment alone
   is not a sufficient safety mechanism.

DECISION: Keep the system prompt (low cost, no observed harm, may help in
some regimes) AND enforce a hard max_iterations cap as the actual safety
mechanism. Based on real-data trials consistently stopping at round 2,
max_iterations = 3 was chosen as a one-round buffer above the observed
common case, rather than cutting it exactly at the observed minimum.
"""