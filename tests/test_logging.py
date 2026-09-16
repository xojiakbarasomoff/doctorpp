import logging
from collections.abc import Iterator

import pytest

from app.core.logging import ExtraFieldFormatter, configure_logging


@pytest.fixture
def _restore_logging() -> Iterator[None]:
    """configure_logging mutates process-wide logging state, which would
    otherwise leak into every test that runs after this module.
    """
    logger = logging.getLogger("app")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _record(msg: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def _format(msg: str, **extra: object) -> str:
    return ExtraFieldFormatter("%(levelname)s %(name)s %(message)s").format(_record(msg, **extra))


def test_extra_fields_are_rendered() -> None:
    """The whole point: this codebase logs its detail through extra=, and
    the stdlib formatter drops all of it.
    """
    assert (
        _format("webhook_unknown_ig_account", ig_account_id="1784143")
        == "INFO app.test webhook_unknown_ig_account ig_account_id=1784143"
    )


def test_message_without_extras_is_unchanged() -> None:
    assert _format("startup_complete") == "INFO app.test startup_complete"


def test_fields_are_ordered_so_lines_are_comparable() -> None:
    # Sorted, not insertion-ordered: two lines for the same event should
    # diff against each other cleanly.
    formatter = ExtraFieldFormatter("%(message)s")
    assert formatter.format(_record("event", zebra=1, alpha=2)) == "event alpha=2 zebra=1"


def test_newline_in_a_value_cannot_forge_a_log_line() -> None:
    """message_preview carries patient text verbatim (redaction.preview
    truncates, it does not strip newlines). Without escaping, whoever sent
    the message chooses what the next log line says.
    """
    output = _format("webhook_message_received", message_preview="hi\nWARNING forged entry")
    assert "\n" not in output
    assert "\\n" in output


def test_value_with_spaces_stays_one_field() -> None:
    # Unquoted, everything after the first space would read as a separate
    # key=value pair and the line stops being parseable.
    output = _format("event", message_preview="hello there", tenant_id="abc")
    assert output.endswith("message_preview='hello there' tenant_id=abc")


def test_extras_stay_on_the_event_line_when_a_traceback_follows() -> None:
    """logger.exception(..., extra={...}) must not bury its fields under the
    traceback, where no grep on the event name finds them.
    """
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record("provisioning_failed", ig_account_id="1784143")
        record.exc_info = sys.exc_info()
        output = ExtraFieldFormatter("%(message)s").format(record)

    first_line = output.splitlines()[0]
    assert first_line == "provisioning_failed ig_account_id=1784143"
    assert "ValueError: boom" in output


@pytest.mark.usefixtures("_restore_logging")
def test_handler_is_installed_on_app_not_root() -> None:
    """Root at INFO would switch on httpx's request logging, and
    app.channels.instagram.client authenticates by query parameter -- the
    access token would land in the log stream on every reply.
    """
    root_handlers_before = logging.getLogger().handlers[:]

    configure_logging()

    assert logging.getLogger("app").handlers
    assert logging.getLogger().handlers == root_handlers_before
    assert logging.getLogger("httpx").getEffectiveLevel() > logging.INFO


@pytest.mark.usefixtures("_restore_logging")
def test_configure_logging_is_idempotent() -> None:
    """Called from both app.main and app.workers.tasks, and a reimport must
    not start doubling every line.
    """
    configure_logging()
    configure_logging()

    # Counts only this function's own handler: pytest's logging plugin
    # attaches its capture handlers to the same logger, and configure_logging
    # deliberately leaves handlers it did not install alone.
    installed = [h for h in logging.getLogger("app").handlers if h.get_name() == "app-stream"]
    assert len(installed) == 1
