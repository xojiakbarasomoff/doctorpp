"""What somebody typed into a dashboard search box, as SQL conditions.

Two things every list's search needs and none had:

* "%" and "_" are wildcards to LIKE. A search for "a_b" matched "axb", and
  one for "100%" matched everything starting "100". Escaped here.
* A telephone number is written a dozen ways -- "+998 90 123-45-67",
  "998901234567", "90 1234567" -- so it is compared on its digits alone,
  on both sides.
"""

import re

from sqlalchemy import ColumnElement, func
from sqlalchemy.orm import InstrumentedAttribute

ESCAPE = "\\"
# Fewer digits than this is part of a name or a date, not a number to match.
MIN_PHONE_DIGITS = 4

_NON_DIGITS = re.compile(r"\D")


def like_pattern(text: str) -> str:
    """'%text%', lower-cased, with LIKE's own wildcards taken literally."""
    escaped = (
        text.strip()
        .lstrip("@")
        .lower()
        .replace(ESCAPE, ESCAPE * 2)
        .replace("%", ESCAPE + "%")
        .replace("_", ESCAPE + "_")
    )
    return f"%{escaped}%"


def contains(column: InstrumentedAttribute[str | None], text: str) -> ColumnElement[bool]:
    """Case-insensitive "column contains text"."""
    return func.lower(column).like(like_pattern(text), escape=ESCAPE)


def phone_digits(text: str) -> str | None:
    """The digits of a search that looks like a telephone number, or None."""
    digits = _NON_DIGITS.sub("", text)
    return digits if len(digits) >= MIN_PHONE_DIGITS else None


def phone_contains(column: InstrumentedAttribute[str | None], digits: str) -> ColumnElement[bool]:
    """The column's digits contain these digits, however either was written."""
    return func.regexp_replace(column, "[^0-9]", "", "g").like(f"%{digits}%")
