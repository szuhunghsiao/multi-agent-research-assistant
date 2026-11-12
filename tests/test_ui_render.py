"""Rendering of progress updates. Skipped when Gradio isn't installed (e.g. the API image)."""
import pytest

pytest.importorskip("gradio")

from ui.app import render  # noqa: E402
from ui.client import Update  # noqa: E402


def test_succeeded_shows_verdict_report_and_critique():
    job = {"result": {"draft": "# Report", "approved": False, "revision_count": 3,
                      "research_rounds": 0, "critique": "Too vague."}}
    status, summary, report, critique = render(Update("succeeded", "done", 41.2, job))
    assert "succeeded" in status and "41 s" in status
    assert "Revision limit reached" in summary and "revisions: 3" in summary
    assert report == "# Report" and critique == "Too vague."


def test_failed_shows_error_and_no_report():
    _, summary, report, _ = render(Update("failed", "x", 5, {"error": "RuntimeError: boom"}))
    assert "RuntimeError: boom" in summary and report == ""


def test_in_progress_has_status_only():
    status, summary, report, critique = render(Update("running", "Job abc running…", 4.2))
    assert "running" in status and (summary, report, critique) == ("", "", "")