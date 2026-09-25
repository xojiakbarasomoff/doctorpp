from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_base import KnowledgeBase
from app.models.tenant import Tenant
from app.rag.embeddings import EMBEDDING_MODEL, EmbeddingProvider, get_embedding_provider
from app.repositories.knowledge_base import KnowledgeBaseRepository


class FAQImport(BaseModel):
    """One row of an FAQ import. Rejects missing/blank question or answer;
    category is optional since not every clinic buckets its FAQs.
    """

    question: str
    answer: str
    category: str | None = None

    @field_validator("question", "answer")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


async def ingest_faqs(
    session: AsyncSession,
    faqs: Sequence[FAQImport | dict[str, Any]],
    embedding_provider: EmbeddingProvider | None = None,
    *,
    reembed_existing: bool = False,
) -> list[KnowledgeBase]:
    """Load a clinic's FAQ list into knowledge_base, tenant-scoped via
    KnowledgeBaseRepository (which stamps tenant_id from the current
    request/task context — see set_current_tenant()).

    We embed the question only, not question+answer. Retrieval later matches
    an incoming user question against these vectors, so the embedding should
    capture question intent as precisely as possible; folding the answer text
    in would dilute the vector with wording the user's message will never
    contain, and would force a re-embed of the same question every time an
    answer is copyedited even though its retrieval target hasn't changed.

    Idempotent: a FAQ is treated as "the same" as an existing one when its
    question matches an existing row's question exactly (post-strip) for
    this tenant. On a match the existing row's answer/category/embedding are
    updated in place instead of inserting a duplicate. Exact match is a
    deliberate simplification for now — it won't catch a reworded question,
    which will insert a new row rather than update.

    All rows are validated before any embedding call is made, so a bad row
    fails fast without spending API calls on the valid ones ahead of it.

    Only questions that are new to this tenant are embedded. The embedding is
    of the question alone and the question is also the match key, so a row
    that is already here already holds the right vector for it; its answer and
    category are still updated in place, which is free. This matters because
    the same file is re-ingested on every deploy (see SEED_FAQS_FROM) and the
    clinic's file has grown past 2500 rows: re-embedding all of them to change
    two was minutes of startup and an API bill for work already done.

    A row whose stored embedding_model is not the model now configured is
    re-embedded even so. Vectors from two different models are not comparable,
    so such a row is not stale, it is meaningless -- and the failure is silent
    without this check: retrieval returns nothing, and the assistant answers
    "I don't know" to questions it has an exact row for.

    reembed_existing forces every row through the provider again, whatever the
    stored model says. It is what scripts/ingest_faqs.py passes, for the case
    where the model kept its name but the vectors still need rebuilding.
    """
    validated = [
        faq if isinstance(faq, FAQImport) else FAQImport.model_validate(faq) for faq in faqs
    ]
    if not validated:
        return []

    repo = KnowledgeBaseRepository(session)
    existing_rows = {faq.question: await repo.get_by_question(faq.question) for faq in validated}

    # A row is embedded again when it is new, when the caller asked for it,
    # or when the vector it holds was made by a different model from the one
    # configured now. That last case is the one that used to be silent: the
    # old vectors stayed, every distance came back too far, and the clinic
    # found out through a bot that had stopped recognising its own FAQ.
    needs_embedding = [
        faq
        for faq in validated
        if reembed_existing
        or existing_rows[faq.question] is None
        or existing_rows[faq.question].embedding_model != EMBEDDING_MODEL
    ]
    provider = embedding_provider or get_embedding_provider()
    fresh = dict(
        zip(
            (faq.question for faq in needs_embedding),
            await provider.embed([faq.question for faq in needs_embedding]),
            strict=True,
        )
    )

    results: list[KnowledgeBase] = []
    for faq in validated:
        existing = existing_rows[faq.question]
        if existing is not None:
            values: dict[str, Any] = {"answer": faq.answer, "category": faq.category}
            if faq.question in fresh:
                values["embedding"] = fresh[faq.question]
                values["embedding_model"] = EMBEDDING_MODEL
            row = await repo.update(existing, **values)
        else:
            row = await repo.create(
                question=faq.question,
                answer=faq.answer,
                category=faq.category,
                embedding=fresh[faq.question],
                embedding_model=EMBEDDING_MODEL,
            )
        results.append(row)
    return results


# Every rule is read before every reply. Past this the list is a manual, and
# the newest rules are the ones a model attends to least.
MAX_RULES = 40


# What the clinic's own rules are filed under in the knowledge base, so an
# operator scanning the list can see at a glance that these are instructions
# to the assistant rather than answers to a patient.
RULE_CATEGORY = "Klinika qoidasi"


async def record_rule_in_knowledge_base(
    session: AsyncSession,
    *,
    rule: str,
    position: int,
    embedding_provider: EmbeddingProvider | None = None,
) -> KnowledgeBase:
    """Show a rule the admin set on the screen the clinic actually reads.

    Inactive on purpose, and that is the whole design of this function.
    KnowledgeBaseRepository.search() only ever returns active rows, so an
    inactive one is visible in the dashboard's list -- marked "o'chirilgan"
    -- and can never be retrieved for a patient. An instruction filed among
    the answers is an instruction a question can land on, and "never say we
    do IVF" read back to somebody asking about IVF is worse than not showing
    the rule at all.

    It is still embedded rather than stored with a dummy vector: the column
    is not nullable, an operator may switch a row on from the dashboard, and
    a row that is live with a meaningless vector would match arbitrary
    questions -- which is the failure this whole week has been about.
    """
    provider = embedding_provider or get_embedding_provider()
    [vector] = await provider.embed([rule])
    return await KnowledgeBaseRepository(session).create(
        question=f"Qoida {position}: {rule[:120]}",
        answer=rule,
        category=RULE_CATEGORY,
        embedding=vector,
        embedding_model=EMBEDDING_MODEL,
        is_active=False,
    )


def _same_rule(a: str | None, b: str | None) -> bool:
    return " ".join((a or "").split()) == " ".join((b or "").split())


@dataclass(frozen=True)
class RuleEntry:
    """One rule as the dashboard lists it.

    Online: in tenants.settings.strict_rules, so the assistant reads it on
    every reply. Offline: only a copy left in the knowledge base -- a rule
    that was later replaced or removed, which the assistant no longer reads.
    """

    number: int
    text: str
    online: bool


async def list_rules(session: AsyncSession, tenant: Tenant) -> list[RuleEntry]:
    """Every rule the clinic has, online ones first in the order they apply."""
    live = [rule for rule in tenant.settings.get("strict_rules") or [] if str(rule).strip()]
    copies = (
        await session.execute(
            select(KnowledgeBase.answer)
            .where(KnowledgeBase.tenant_id == tenant.id, KnowledgeBase.category == RULE_CATEGORY)
            .order_by(KnowledgeBase.question)
        )
    ).scalars()
    texts: list[tuple[str, bool]] = [(rule, True) for rule in live]
    for copy in copies:
        if not any(_same_rule(copy, text) for text, _ in texts):
            texts.append((copy, False))
    return [
        RuleEntry(number=index, text=text, online=online)
        for index, (text, online) in enumerate(texts, start=1)
    ]


async def remove_rule(session: AsyncSession, tenant: Tenant, text: str) -> bool:
    """Take a rule away everywhere: off the live list and out of the
    knowledge base. False when there was no such rule."""
    live = list(tenant.settings.get("strict_rules") or [])
    kept = [rule for rule in live if not _same_rule(rule, text)]
    copies = (
        await session.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.tenant_id == tenant.id, KnowledgeBase.category == RULE_CATEGORY
            )
        )
    ).scalars()
    doomed = [copy for copy in copies if _same_rule(copy.answer, text)]
    if len(kept) == len(live) and not doomed:
        return False
    if len(kept) != len(live):
        tenant.settings = {**tenant.settings, "strict_rules": kept}
    for copy in doomed:
        await session.delete(copy)
    await session.flush()
    return True


class TooManyRulesError(Exception):
    """Switching this rule on would take the live list past MAX_RULES."""


async def set_rule_online(
    session: AsyncSession,
    tenant: Tenant,
    text: str,
    online: bool,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> bool:
    """Switch a rule on (onto the live list) or off (a copy that stays listed).

    False when there is no such rule. Switching off keeps the rule in the
    dashboard's list, as a knowledge-base copy -- one is made for a rule typed
    straight into Settings, which never had one -- so it can be switched back.
    """
    live = list(tenant.settings.get("strict_rules") or [])
    is_live = any(_same_rule(rule, text) for rule in live)
    copies = (
        (
            await session.execute(
                select(KnowledgeBase).where(
                    KnowledgeBase.tenant_id == tenant.id, KnowledgeBase.category == RULE_CATEGORY
                )
            )
        )
        .scalars()
        .all()
    )
    copy = next((c for c in copies if _same_rule(c.answer, text)), None)
    if not is_live and copy is None:
        return False

    if online and not is_live:
        if len(live) >= MAX_RULES:
            raise TooManyRulesError
        tenant.settings = {**tenant.settings, "strict_rules": [*live, copy.answer]}
    elif not online and is_live:
        if copy is None:
            await record_rule_in_knowledge_base(
                session,
                rule=next(rule for rule in live if _same_rule(rule, text)),
                position=len(copies) + 1,
                embedding_provider=embedding_provider,
            )
        tenant.settings = {
            **tenant.settings,
            "strict_rules": [rule for rule in live if not _same_rule(rule, text)],
        }
    await session.flush()
    return True
