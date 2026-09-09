# Multi-Agent Research Assistant

Use LangGraph to organize three LLM agents (Researcher / Writer / Critic) to auto-generate a research report
Built-in revision loop: The Critic will check whether the draft is approved; if not, it will loop back to the Writer to rewrite until the maximum loop count is met or the report is approved.

[![Open In Colab](https://colab.research.google.com/drive/1veCeAS4SZp2iaZea32_Lh6kuimwgMbBx#scrollTo=sS7mkR3AFhVk)

## Design

**State-driven coordination**: All state nodes share one State (TypedDict), not point-to-point, so the Critic can access the original topic and the Writer's draft.

**Conditional loop**: Use LangGraph's `add_conditional_edges` to state "If Critic not approve -> loop back to Writer" loop, not using a while loop.
Also include `MAX_REVISIONS` to prevent an infinite loop.

**Model tiering**: Use three different versions of the Gemini models
- Researcher: 3.5-glash-lite
  - Information collection
- Writer: 3.5-flash
  - Generate content
- Critic: 3.5-flash-lite
  - Checking
 
## How to execute
1. Click the Colab badge
2. In Google AI Studio, create Gemini API Key
3. At Colab left side bar "Secrets" create three API key with names (Researcher, Writer, Critic)
4. Run all cells

## Tech Stack
LangGraph, LangChain, Google Gemini API, Gradio
