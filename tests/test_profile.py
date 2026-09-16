"""Putting a handle on a patient row.

The dashboard used to label every conversation with the platform id, which
an operator cannot recognise and cannot search for. These cover the two ways
a handle arrives -- Telegram hands it over, Instagram has to be asked -- and,
more importantly, every way the asking fails: a patient must still be
answered, and still be visible, when Meta says no.
"""

import pytest

from app.services.profile import normalise_username


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("asomov", "asomov"),
        # What an operator pastes, out of habit.
        ("@asomov", "asomov"),
        ("  @asomov  ", "asomov"),
        ("  asomov\n", "asomov"),
        # Nothing worth storing: a row with "" in it looks like a handle in
        # every listing and matches nothing in the search.
        ("", None),
        ("   ", None),
        ("@", None),
        (None, None),
    ],
)
def test_a_handle_is_stored_one_way(raw: str | None, expected: str | None) -> None:
    assert normalise_username(raw) == expected
