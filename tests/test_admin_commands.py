"""The clinic's admin setting a rule from their phone.

This is a door into how the assistant answers patients, opened by a direct
message, so the interesting cases are the ones where it must stay shut: a
patient quoting the keyword, a handle that is nearly the admin's, a command
with nothing after it -- and, now that the instruction is read by a model
before anything is stored, the ones where the model is wrong or absent.
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.models.tenant import Tenant
from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.services.admin_commands import (
    FACT_CATEGORY,
    MAX_INSTRUCTION_LENGTH,
    MAX_RULES,
    handle_instruction,
    is_admin,
    parse_rule,
)
from app.services.knowledge_base import RULE_CATEGORY
from tests.conftest import Seed

KEYWORD = "//:"
LEGACY_KEYWORD = "Aiadm1in:"
NOW = datetime(2026, 9, 21, 15, 0)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("//: shanba kuni ishlamaymiz", "shanba kuni ishlamaymiz"),
        ("  //:   shanba kuni ishlamaymiz  ", "shanba kuni ishlamaymiz"),
        # A second colon is punctuation, not part of the instruction.
        ("//:: shanba kuni ishlamaymiz", "shanba kuni ishlamaymiz"),
        ("//:shanba kuni ishlamaymiz", "shanba kuni ishlamaymiz"),
    ],
)
def test_the_instruction_is_what_follows_the_keyword(message: str, expected: str) -> None:
    assert parse_rule(message, KEYWORD) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # Phone keyboards capitalise, and people put spaces where they like.
        ("aiadm1in: EKO haqida gapirma", "EKO haqida gapirma"),
        ("AIADM1IN:   EKO haqida gapirma  ", "EKO haqida gapirma"),
    ],
)
def test_a_word_keyword_is_matched_however_it_is_capitalised(message: str, expected: str) -> None:
    assert parse_rule(message, LEGACY_KEYWORD) == expected


@pytest.mark.parametrize(
    "message",
    [
        # Nothing to store.
        "//:",
        "//:    ",
        # An ordinary message.
        "narxi qancha",
        "",
        # The keyword mentioned rather than used. A patient quoting the clinic
        # back at itself must not be able to set a rule, so it only counts at
        # the very start.
        "menga //: nima degani ayting",
        "salom //: shanba ishlamaymiz",
    ],
)
def test_anything_else_is_not_a_command(message: str) -> None:
    assert parse_rule(message, KEYWORD) is None


def test_an_instruction_is_a_message_not_a_document() -> None:
    rule = parse_rule(KEYWORD + " " + "x" * 5000, KEYWORD)

    assert rule is not None
    assert len(rule) == MAX_INSTRUCTION_LENGTH


def test_no_keyword_configured_means_no_command() -> None:
    """The empty default. A deployment that has not set one must not have
    every message read as a possible command.
    """
    assert parse_rule("//: shanba yopiq", "") is None


@pytest.mark.parametrize(
    ("username", "expected"),
    [
        ("as.1.ed", True),
        ("med.uz.ai", True),
        # Both sides are typed by a person: the variable, and the handle.
        ("As.1.Ed", True),
        ("@med.uz.ai", True),
        ("  med.uz.ai  ", True),
        # Near misses. A handle that merely contains the admin's is not it.
        ("med.uz", False),
        ("med.uz.aii", False),
        ("xmed.uz.ai", False),
        ("", False),
        (None, False),
    ],
)
def test_only_a_nominated_handle_is_an_admin(username: str | None, expected: bool) -> None:
    assert is_admin(username, ["as.1.ed", "@med.uz.ai"]) is expected


def test_nobody_is_an_admin_by_default() -> None:
    """The empty list is the shipped default, and it has to mean nobody --
    this is a feature whose entire surface is "somebody messages the clinic".
    """
    assert is_admin("as.1.ed", []) is False
    assert is_admin("as.1.ed", ["", "   "]) is False


# --- understanding the instruction ------------------------------------------


class _Scripted(LLMProvider):
    def __init__(self, reply: str | Exception) -> None:
        self._reply = reply
        self.calls = 0

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.calls += 1
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


class _Embeddings(EmbeddingProvider):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding service down")
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1) for _ in texts]


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


async def _rules(session: AsyncSession, seed: Seed) -> list[str]:
    tenant = await session.get(Tenant, seed.tenant_a.id)
    assert tenant is not None
    return list(tenant.settings.get("strict_rules", []))


async def _rows(session: AsyncSession, seed: Seed, category: str) -> list[KnowledgeBase]:
    stmt = select(KnowledgeBase).where(
        KnowledgeBase.tenant_id == seed.tenant_a.id, KnowledgeBase.category == category
    )
    return list((await session.execute(stmt)).scalars())


async def _handle(
    session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    instruction: str,
    provider: LLMProvider,
    *,
    embeddings: EmbeddingProvider | None = None,
) -> str:
    with as_tenant(seed.tenant_a.id):
        return await handle_instruction(
            session,
            tenant_id=seed.tenant_a.id,
            instruction=instruction,
            provider=provider,
            embedding_provider=embeddings or _Embeddings(),
            now=NOW,
        )


async def test_the_rule_stored_is_what_the_model_understood_not_what_was_typed(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    understood = "Shanba kuni klinika yopiq. Shanbaga vaqt taklif qilmang."
    provider = _Scripted(
        _reply(
            summary="Klinika doim shanba kuni yopiq.",
            rules=[understood],
            facts=[{"question": "Shanba kuni ishlaysizmi?", "answer": "Yo'q, shanba dam olamiz."}],
        )
    )

    reply = await _handle(db_session, seed, as_tenant, "shanba kuni ishlamaymiz", provider)

    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == [understood]
        [mirror] = await _rows(db_session, seed, RULE_CATEGORY)
        [fact] = await _rows(db_session, seed, FACT_CATEGORY)
    # The rule is visible on the knowledge-base screen and can never be
    # retrieved for a patient; the fact is live, so a question can find it.
    assert mirror.is_active is False
    assert fact.is_active is True
    assert fact.answer == "Yo'q, shanba dam olamiz."
    assert "Tushundim: Klinika doim shanba kuni yopiq." in reply
    assert f"- {understood}" in reply
    assert "Shanba kuni ishlaysizmi? → Yo'q, shanba dam olamiz." in reply
    assert "Jami 1 ta qoida" in reply
    assert provider.calls == 1


async def test_an_instruction_that_replaces_a_rule_removes_the_old_one(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        seed.tenant_a.settings = {
            **seed.tenant_a.settings,
            "strict_rules": ["Yakshanba yopiq.", "EKO qilmaymiz."],
        }
        await db_session.flush()
    provider = _Scripted(
        _reply(
            summary="Yakshanba ham ishlaysiz.",
            rules=["Yakshanba kuni ham ishlaymiz; yakshanbaga vaqt taklif qilish mumkin."],
            remove=[1],
        )
    )

    reply = await _handle(db_session, seed, as_tenant, "yakshanba ham ishlaymiz", provider)

    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == [
            "EKO qilmaymiz.",
            "Yakshanba kuni ham ishlaymiz; yakshanbaga vaqt taklif qilish mumkin.",
        ]
    assert "Endi amal qilmayman:\n- Yakshanba yopiq." in reply


async def test_a_cancelling_instruction_removes_and_adds_nothing(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        seed.tenant_a.settings = {**seed.tenant_a.settings, "strict_rules": ["A.", "B."]}
        await db_session.flush()
    provider = _Scripted(_reply(summary="1-qoida bekor qilindi.", remove=[1]))

    await _handle(db_session, seed, as_tenant, "1-qoidani o'chir", provider)

    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == ["B."]


async def test_a_rule_that_is_already_there_is_not_written_twice(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        seed.tenant_a.settings = {**seed.tenant_a.settings, "strict_rules": ["Shanba yopiq."]}
        await db_session.flush()
    provider = _Scripted(_reply(rules=["shanba YOPIQ."]))

    reply = await _handle(db_session, seed, as_tenant, "shanba yopiq", provider)

    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == ["Shanba yopiq."]
    assert "allaqachon qoidalarda bor" in reply


async def test_a_cancellation_the_model_forgot_to_carry_out_is_not_reported_as_done(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The model said "cancelled" and left "remove" empty. The list did not
    change, and the admin must be told that, not that it was understood."""
    with as_tenant(seed.tenant_a.id):
        seed.tenant_a.settings = {**seed.tenant_a.settings, "strict_rules": ["A.", "B."]}
        await db_session.flush()
    provider = _Scripted(_reply(summary="1-qoida bekor qilindi."))

    reply = await _handle(db_session, seed, as_tenant, "1-qoidani o'chir", provider)

    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == ["A.", "B."]
    assert reply.startswith("Hech narsa o'zgarmadi. 1-qoida bekor qilindi.")
    assert "Tushundim" not in reply


async def test_a_question_back_stores_nothing(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = _Scripted(_reply(kind="clarify", question="Faqat shu yakshanbami yoki doimmi?"))

    reply = await _handle(db_session, seed, as_tenant, "yakshanba ishlaymiz", provider)

    assert reply == "Faqat shu yakshanbami yoki doimmi?"
    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == []


@pytest.mark.parametrize("kind", ["refused", "not_an_instruction"])
async def test_a_refused_or_pointless_instruction_stores_nothing_and_says_why(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    kind: str,
) -> None:
    provider = _Scripted(_reply(kind=kind, summary="Buni qila olmayman: dori aytish mumkin emas."))

    reply = await _handle(db_session, seed, as_tenant, "bemorlarga dori ayt", provider)

    assert reply == "Buni qila olmayman: dori aytish mumkin emas."
    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == []
        assert await _rows(db_session, seed, FACT_CATEGORY) == []


async def test_the_admin_can_ask_what_the_rules_are(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        seed.tenant_a.settings = {**seed.tenant_a.settings, "strict_rules": ["A.", "B."]}
        await db_session.flush()

    reply = await _handle(
        db_session, seed, as_tenant, "qoidalarni ko'rsat", _Scripted(_reply(kind="list"))
    )

    assert reply == "Hozirgi qoidalar (2 ta):\n1. A.\n2. B."


@pytest.mark.parametrize(
    "provider",
    [_Scripted("bu json emas"), _Scripted(RuntimeError("model down"))],
    ids=["nonsense", "down"],
)
async def test_an_instruction_is_never_lost_to_a_model_that_could_not_be_read(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    provider: LLMProvider,
) -> None:
    reply = await _handle(db_session, seed, as_tenant, "shanba kuni ishlamaymiz", provider)

    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == ["shanba kuni ishlamaymiz"]
    # ...and the admin is told the truth about what "understood" meant.
    assert "tahlil qila olmadim" in reply
    assert "Tushundim" not in reply


async def test_a_fact_that_cannot_be_filed_does_not_lose_the_rule_beside_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = _Scripted(
        _reply(
            rules=["UZI narxini telefon orqali ayting."],
            facts=[{"question": "UZI qancha?", "answer": "200 000"}],
        )
    )

    reply = await _handle(
        db_session, seed, as_tenant, "UZI 200 ming", provider, embeddings=_Embeddings(fail=True)
    )

    with as_tenant(seed.tenant_a.id):
        assert await _rules(db_session, seed) == ["UZI narxini telefon orqali ayting."]
        assert await _rows(db_session, seed, FACT_CATEGORY) == []
    assert "Bilimlar bazasiga yozib bo'lmadi" in reply


async def test_the_list_of_rules_has_a_ceiling(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        seed.tenant_a.settings = {
            **seed.tenant_a.settings,
            "strict_rules": [f"Qoida {n}." for n in range(MAX_RULES)],
        }
        await db_session.flush()

    reply = await _handle(
        db_session, seed, as_tenant, "yana bir narsa", _Scripted(_reply(rules=["Yangi qoida."]))
    )

    with as_tenant(seed.tenant_a.id):
        assert len(await _rules(db_session, seed)) == MAX_RULES
    assert f"{MAX_RULES} tadan oshmasligi kerak" in reply


async def test_the_reply_fits_in_one_direct_message(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    long_rules = [f"{n}. " + "juda uzun qoida " * 25 for n in range(5)]

    reply = await _handle(
        db_session, seed, as_tenant, "ko'p narsa", _Scripted(_reply(rules=long_rules))
    )

    assert len(reply) <= 900
