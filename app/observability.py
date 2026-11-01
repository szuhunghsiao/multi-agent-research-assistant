"""Structured JSON logging. job_id is carried in a contextvar so every log line
emitted while a job runs (graph nodes, retries) is tagged without passing it around."""
import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)

# Attributes every LogRecord has; anything else came from `extra=` and is ours to emit.
_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName", "color_message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "job_id": job_id_var.get(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STD_ATTRS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):  # make server logs JSON too
        lg = logging.getLogger(name)
        lg.handlers[:] = [handler]
        lg.propagate = False
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # AFC advisory noise
    logging.getLogger("httpx").setLevel(logging.WARNING)              # per-request INFO noise