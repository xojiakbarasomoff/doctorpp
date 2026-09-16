"""The clinic's admin setting a rule from their phone.

This is a door into how the assistant answers patients, opened by a direct
message, so the interesting cases are the ones where it must stay shut: a
patient quoting the keyword, a handle that is nearly the admin's, a command
with nothing after it.
"""

import pytest

from app.services.admin_commands import MAX_RULE_LENGTH, is_admin, parse_rule

KEYWORD = "Aiadm1in:"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Aiadm1in: har doim shanba qabulini eslat", "har doim shanba qabulini eslat"),
        # Phone keyboards capitalise, and people put spaces where they like.
        ("aiadm1in: EKO haqida gapirma", "EKO haqida gapirma"),
        ("AIADM1IN:   EKO haqida gapirma  ", "EKO haqida gapirma"),
        ("  Aiadm1in: EKO haqida gapirma", "EKO haqida gapirma"),
        # A second colon is punctuation, not part of the rule.
        ("Aiadm1in:: EKO haqida gapirma", "EKO haqida gapirma"),
    ],
)
def test_the_rule_is_what_follows_the_keyword(message: str, expected: str) -> None:
    assert parse_rule(message, KEYWORD) == expected


@pytest.mark.parametrize(
    "message",
    [
        # Nothing to store.
        "Aiadm1in:",
        "Aiadm1in:    ",
        # An ordinary message.
        "narxi qancha",
        "",
        # The keyword mentioned rather than used. A patient who was told the
        # word, or who is quoting the clinic back at itself, must not be able
        # to set a rule -- so it only counts at the very start.
        "menga Aiadm1in: nima degani ayting",
        "salom Aiadm1in: EKO haqida gapirma",
    ],
)
def test_anything_else_is_not_a_command(message: str) -> None:
    assert parse_rule(message, KEYWORD) is None


def test_a_rule_is_a_sentence_not_a_paste() -> None:
    """It goes in front of every reply from then on, so an essay pasted here
    is an essay the model reads before answering every patient.
    """
    rule = parse_rule(KEYWORD + " " + "x" * 5000, KEYWORD)
    assert rule is not None
    assert len(rule) == MAX_RULE_LENGTH


def test_no_keyword_configured_means_no_command() -> None:
    """The empty default. A deployment that has not set one must not have
    every message read as a possible command.
    """
    assert parse_rule("Aiadm1in: EKO haqida gapirma", "") is None


@pytest.mark.parametrize(
    ("username", "expected"),
    [
        ("meduza.ii", True),
        # Both sides are typed by a person: the variable, and the handle.
        ("Meduza.II", True),
        ("@meduza.ii", True),
        ("  meduza.ii  ", True),
        # Near misses. A handle that merely contains the admin's is not it.
        ("meduza", False),
        ("meduza.iii", False),
        ("xmeduza.ii", False),
        ("", False),
        (None, False),
    ],
)
def test_only_a_nominated_handle_is_an_admin(username: str | None, expected: bool) -> None:
    assert is_admin(username, ["@meduza.ii"]) is expected


def test_nobody_is_an_admin_by_default() -> None:
    """The empty list is the shipped default, and it has to mean nobody --
    this is a feature whose entire surface is "somebody messages the clinic".
    """
    assert is_admin("meduza.ii", []) is False
    assert is_admin("meduza.ii", ["", "   "]) is False
