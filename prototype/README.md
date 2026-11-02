# Prototype (Colab, Aug–Sep 2026)

The original single-notebook prototype that the service in `app/` was refactored from.
Kept for history; **not maintained**. Use `app/` (see the top-level README) instead.

| File | What it is |
|---|---|
| `research_assistant.ipynb` | Researcher → Writer → Critic LangGraph pipeline + Gradio UI, synchronous, single file |
| `stopping_behavior_analysis.py` | Experiments on when the tool-calling Researcher stops searching (led to `MAX_ITERATIONS=3` and the "answer directly when sufficient" system prompt) |

## Known issues fixed in `app/`
- Revise prompt did not include the previous draft, so the Writer "revised" text it could not see.
- When the Researcher answered without searching, its answer was dropped and the Writer got empty notes.
- Tavily HTTP errors were returned as `{"error": ...}` instead of raised, producing reports with no research behind them.
- Synchronous calls, module-level clients (importing the file called the APIs), no retries, timeouts, or logs.