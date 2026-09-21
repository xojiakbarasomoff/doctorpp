"""Reading an admin's instruction: what the model may say, and what is refused."""

import json
from datetime import datetime

import pytest

from app.rag.llm import ChatMessage, LLMProvider
from app.services.rule_interpreter import (
    MAX_FACTS_PER_MESSAGE,
    MAX_RULE_LENGTH,
    MAX_RULES_PER_MESSAGE,
    InterpretationError,
    build_prompt,
    interpret,
    parse,
    verbatim,
)

NOW = datetime(2026, 9, 21, 15, 0)


def _reply(**fields: object) -> str:
    body: dict[str, object] = {
        "kind": "rules",
        "summary": "",
        "rules": [],
        "remove": [],
        "facts": [],
        "question": "",
    }
    body.update(fields)
    return json.dumps(body, ensure_ascii=False)


def test_a_rule_a_removal_and_a_fact_are_read_as_written() -> None:
    result = parse(
        _reply(
            summary="Shanba yopiq.",
            rules=["Shanba kuni vaqt taklif qilmang."],
            remove=[2],
            facts=[{"question": "Shanba ishlaysizmi?", "answer": "Yo'q, dam olamiz."}],
        ),
        existing_count=3,
    )

    assert result.kind == "rules"
    assert result.rules == ("Shanba kuni vaqt taklif qilmang.",)
    assert result.remove == (2,)
    assert result.facts[0].question == "Shanba ishlaysizmi?"
    assert result.understood is True


def test_a_code_fence_and_a_sentence_around_the_json_are_tolerated() -> None:
    raw = "Mana javob:\n```json\n" + _reply(rules=["Qoida."]) + "\n```"

    assert parse(raw, existing_count=0).rules == ("Qoida.",)


@pytest.mark.parametrize(
    "raw",
    [
        "tushunmadim",
        "{not json}",
        "[1, 2]",
        _reply(kind="chatting"),
        _reply(kind="clarify", question=""),
        # Removing a rule that is not there.
        _reply(rules=["Qoida."], remove=[4]),
        _reply(rules=["Qoida."], remove=[0]),
        # Says it changes rules and changes nothing, without a word of why.
        _reply(kind="rules"),
    ],
)
def test_an_unusable_answer_is_an_error_not_a_guess(raw: str) -> None:
    with pytest.raises(InterpretationError):
        parse(raw, existing_count=3)


def test_already_covered_is_a_legitimate_answer_when_it_says_so() -> None:
    result = parse(_reply(summary="Bu allaqachon 1-qoidada bor."), existing_count=1)

    assert not result.rules
    assert result.summary == "Bu allaqachon 1-qoidada bor."


def test_the_other_kinds_are_read() -> None:
    assert parse(_reply(kind="list"), 0).kind == "list"
    assert parse(_reply(kind="clarify", question="Qaysi kun?"), 0).question == "Qaysi kun?"
    assert parse(_reply(kind="refused", summary="Dori aytolmayman."), 0).kind == "refused"


def test_what_one_message_may_write_is_bounded() -> None:
    result = parse(
        _reply(
            rules=[f"Qoida {n}" for n in range(20)] + ["x" * 5000],
            facts=[{"question": f"Savol {n}", "answer": "Javob"} for n in range(20)],
        ),
        existing_count=0,
    )

    assert len(result.rules) == MAX_RULES_PER_MESSAGE
    assert len(result.facts) == MAX_FACTS_PER_MESSAGE
    long = parse(_reply(rules=["x" * 5000]), 0).rules[0]
    assert len(long) == MAX_RULE_LENGTH


def test_blank_and_mistyped_entries_are_dropped() -> None:
    result = parse(
        _reply(
            rules=["", "  ", 5, "Qoida."],
            remove=["1", True, 1, 1],
            facts=[{"question": "Savol", "answer": ""}, "nima", {"question": "S", "answer": "J"}],
        ),
        existing_count=1,
    )

    assert result.rules == ("Qoida.",)
    assert result.remove == (1,)
    assert [(f.question, f.answer) for f in result.facts] == [("S", "J")]


def test_the_model_is_shown_the_rules_in_force_and_the_date() -> None:
    prompt = build_prompt(["Birinchi qoida.", "Ikkinchi qoida."], NOW)

    assert "1. Birinchi qoida.\n2. Ikkinchi qoida." in prompt
    assert "2026-09-21 15:00" in prompt
    assert "(none yet)" in build_prompt([], NOW)
    # The format braces in the examples survived formatting.
    assert '"kind": "rules"' in prompt


def test_the_admins_own_words_stand_in_when_nothing_better_is_possible() -> None:
    result = verbatim("  shanba yopiq  ")

    assert result.rules == ("shanba yopiq",)
    assert result.understood is False


class _Scripted(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[tuple[str, list[ChatMessage]]] = []

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.seen.append((system_prompt, messages))
        return self.reply


async def test_interpret_sends_the_admins_words_and_reads_the_answer() -> None:
    provider = _Scripted(_reply(rules=["Qoida."], remove=[1]))

    result = await interpret("shanba yopiq", ["Eski qoida."], provider, now=NOW)

    assert result.rules == ("Qoida.",)
    assert result.remove == (1,)
    prompt, messages = provider.seen[0]
    assert "1. Eski qoida." in prompt
    assert messages == [{"role": "user", "content": "shanba yopiq"}]
