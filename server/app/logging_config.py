"""Logging setup for the app's own loggers (uvicorn only configures its own)."""

import contextvars
import logging

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Attach one stderr handler to the `app` logger tree. Idempotent."""
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level.upper())
    if any(getattr(h, "_lawweb", False) for h in app_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler._lawweb = True  # type: ignore[attr-defined]
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
        )
    )
    app_logger.addHandler(handler)
    app_logger.propagate = False
