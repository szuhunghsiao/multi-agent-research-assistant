# syntax=docker/dockerfile:1

# ---------- base ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False
WORKDIR /app

# ---------- builder: pinned UI deps only (no LangGraph, no Gemini SDK, no API keys) ----------
FROM base AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY ui/requirements.lock .
RUN pip install -r requirements.lock

# ---------- runtime ----------
FROM base AS runtime
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    UI_HOST=0.0.0.0 \
    UI_PORT=7860
COPY ui ./ui
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/', timeout=2).status == 200 else 1)"
CMD ["python", "-m", "ui.app"]