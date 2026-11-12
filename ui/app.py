"""Gradio front end. It is a plain client of the HTTP API (POST -> 202 -> poll), exactly like
curl or the eval runner -- it never imports the pipeline.

    API_URL=http://localhost:8000 python -m ui.app      # then open http://localhost:7860
"""
import os

import gradio as gr

from ui.client import Update, run_research

API_URL = os.getenv("API_URL", "http://localhost:8000")

STATUS_ICON = {"queued": "⏳", "running": "🔄", "succeeded": "✅", "failed": "❌", "error": "⚠️"}


def render(u: Update) -> tuple[str, str, str, str]:
    """Map one progress update to (status line, summary line, report, critic feedback)."""
    status = f"{STATUS_ICON.get(u.status, '')} **{u.status}** — {u.message} ({u.elapsed_s:.0f} s)"
    job = u.job or {}
    result = job.get("result") or {}
    if u.status == "succeeded":
        verdict = ("Approved by the Critic" if result.get("approved")
                   else "Revision limit reached without Critic approval")
        summary = (f"{verdict} · revisions: {result.get('revision_count')} · "
                   f"web searches: {result.get('research_rounds')}")
    elif u.status == "failed":
        summary = f"Job failed: `{job.get('error')}`"
    else:
        summary = ""
    return status, summary, result.get("draft", ""), result.get("critique", "")


async def on_submit(topic: str):
    async for update in run_research(topic.strip(), API_URL):
        yield render(update)


with gr.Blocks(title="Research Assistant") as demo:
    gr.Markdown("# Multi-Agent Research Assistant\n"
                "Researcher (web-search agent) → Writer → Critic, served by an async FastAPI job API.")
    with gr.Row():
        topic = gr.Textbox(label="Research topic", scale=5,
                           placeholder="e.g. Kafka vs RabbitMQ for event-driven microservices")
        run = gr.Button("Research", variant="primary", scale=1)
    status = gr.Markdown()
    summary = gr.Markdown()
    report = gr.Markdown()
    with gr.Accordion("Critic's final feedback", open=False):
        critique = gr.Markdown()

    outputs = [status, summary, report, critique]
    run.click(on_submit, inputs=topic, outputs=outputs)
    topic.submit(on_submit, inputs=topic, outputs=outputs)


if __name__ == "__main__":
    demo.queue().launch(server_name=os.getenv("UI_HOST", "127.0.0.1"),
                        server_port=int(os.getenv("UI_PORT", "7860")))