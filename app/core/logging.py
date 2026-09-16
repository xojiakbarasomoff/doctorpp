"""Logging setup that emits what this codebase logs, and nothing else.

Nearly every log call here passes its detail through `extra=` -- the tenant
id on a webhook, the account id on an unresolved one, the length and preview
of a message. The stdlib's default formatter renders only the message text,
so all of that was discarded silently: the calls looked like structured
logging while the deployment's log stream showed bare event names.

The handler is attached to the `app` logger, deliberately not to the root.
Turning root up to INFO switches on every third-party library's INFO output
too, and two of those are actively harmful here:

* httpx logs each request's full URL, and app.channels.instagram.client
  authenticates by query parameter -- so an INFO root logger writes a live
  Instagram access token into the log stream on every reply sent.
* arq installs its own handler on the `arq` logger via dictConfig without
  `propagate: False`, so a root handler makes every worker line print twice.

Scoping to `app` keeps both at their own defaults while this project's own
logs come through in full.
"""

import logging

# Attributes the stdlib puts on every LogRecord. Anything outside this set
# arrived via `extra=` and is what the call site actually wanted to say.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

# The package whose loggers this configures. Every logger in this codebase is
# named after its module, so they all sit under this one.
_APP_LOGGER = "app"

_CHARS_NEEDING_QUOTES = frozenset(" \t\r\n\"'=")


def _render_value(value: object) -> str:
    """A single `extra=` value, safe to put in a log line.

    Values include user-controlled text (app.core.redaction.preview truncates
    a patient's message but does not strip newlines), so an unescaped value
    lets whoever sent that message inject a line into the log stream that
    reads exactly like a genuine record. repr() escapes the newline and makes
    the boundaries of a value containing spaces unambiguous.
    """
    text = str(value)
    if any(char in _CHARS_NEEDING_QUOTES for char in text):
        return repr(text)
    return text


class ExtraFieldFormatter(logging.Formatter):
    """Appends a record's `extra=` fields to the formatted message."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
        }
        if not extras:
            return super().format(record)

        rendered = " ".join(
            f"{key}={_render_value(value)}" for key, value in sorted(extras.items())
        )
        # Appended to the message rather than to the formatted output, so the
        # fields stay on the event line instead of being glued to the last
        # line of a traceback when exc_info is set.
        annotated = logging.makeLogRecord(record.__dict__)
        annotated.msg = f"{record.getMessage()} {rendered}"
        annotated.args = ()
        return super().format(annotated)


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single handler on the `app` logger.

    Replaces a handler this function installed previously rather than adding
    to it, so a second call (a test, a reimport) cannot double every line.
    Leaves the root logger and third-party loggers untouched -- see the
    module docstring for why that matters.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ExtraFieldFormatter("%(levelname)s %(name)s %(message)s"))
    handler.set_name("app-stream")

    logger = logging.getLogger(_APP_LOGGER)
    for existing in [h for h in logger.handlers if h.get_name() == "app-stream"]:
        logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.setLevel(level)
    # Without this an app record reaching root would also hit whatever the
    # host process put there (uvicorn's handler, arq's), printing twice.
    logger.propagate = False
